# Methodology

A detailed explanation of every design decision in this forecasting system and the reasoning behind each one. Written for technical reviewers — hiring managers, senior analysts, and anyone who wants to understand what the model actually does.

---

## The core problem framing

Most football ML projects frame the problem as: *predict the result of the next match.* This leads to a classification model, typically trained on accuracy, that produces a single label (H/D/A) and discards all uncertainty.

This project frames the problem differently: *what is the probability distribution over all possible season outcomes?* This requires:

1. A match-level model that outputs calibrated probabilities, not labels
2. A simulation layer that propagates those probabilities into season-level outcomes
3. A validation framework that measures calibration, not just prediction rate

The distinction matters. A model that says "Arsenal will win the title" has nothing useful to say about how certain that outcome is. A model that says "Arsenal win the title in 99.3% of simulated seasons given the current table" tells you both the outcome and how much uncertainty remains.

---

## Data

**Source:** [football-data.co.uk](https://www.football-data.co.uk/englandm.php) — free, well-maintained, and used in serious sports analytics research.

**Coverage:** 2010/11 through 2025/26 — approximately 6,009 completed matches.

**Training restriction:** Model training uses only 2019/20 onwards (6 seasons). Earlier seasons load for ELO warmup but are excluded from the training pool. Reason: the bookmaker odds columns (`AvgH`, `AvgD`, `AvgA`) — the model's most important feature at 18.2% importance — do not exist in pre-2019 data. A flat prior (0.45/0.27/0.28) was substituted for those rows. Training on 8 seasons of a partially-fabricated primary feature degraded performance by approximately 4 percentage points of accuracy. Less data, better quality, improved results.

**Key columns used:**

| Column | Description |
|--------|-------------|
| `FTHG`, `FTAG` | Full-time goals |
| `FTR` | Full-time result (H/D/A) |
| `HST`, `AST` | Shots on target (xG proxy) |
| `AvgH`, `AvgD`, `AvgA` | Market-average bookmaker odds |
| `Date`, `HomeTeam`, `AwayTeam` | Match identity |

---

## Temporal integrity

The most common mistake in sports ML is temporal data leakage — using future information to predict the past.

**Every rolling and EWMA feature uses `.shift(1)` before the window function.** This means the feature value for match N is computed from matches 1 through N-1 only. Match N's result does not influence its own features.

Example:

```python
# WRONG — includes the current match result
df['HomeForm'] = df.groupby('HomeTeam')['HomePoints'].transform(
    lambda x: x.ewm(span=5).mean()
)

# CORRECT — only uses results up to and including the previous match
df['HomeForm'] = df.groupby('HomeTeam')['HomePoints'].transform(
    lambda x: x.shift(1).ewm(span=5, min_periods=1).mean()
)
```

**Validation uses walk-forward cross-validation, not random split.** A random 80/20 split on time-series data allows the model to see 2024/25 results while predicting 2020/21 — this is data leakage disguised as validation. Walk-forward validation enforces the constraint the model faces in production: only the past informs predictions about the future.

---

## ELO ratings

Standard Elo with two football-specific calibrations.

**K-factor (K=20):** Controls how much a single match shifts ratings. K=20 is standard for football — lower than chess (K=32) because football results are noisier and single matches are less informative.

**Home advantage offset (100 points):** When computing expected scores, the home team is treated as 100 Elo points stronger. This models structural home advantage in win probability calculations without embedding it in the stored ratings.

**Seasonal decay (10% toward 1500):** At the start of each season, every team's rating moves 10% toward the league mean. Formula: `new_elo = old_elo × 0.9 + 1500 × 0.1`. Rationale: summer transfers, manager changes, and promotion/relegation mean a team's August quality is not identical to their May quality. Without decay, Man City's 2022/23 dominant season (ELO ~1800) inflates their 2025/26 ratings even after two weaker seasons. The 10% value was chosen to balance stability against responsiveness — small enough to preserve multi-season signal, large enough to allow meaningful pre-season adjustment.

---

## Feature engineering — the three-layer form system

The central insight of this model is that **structural quality** (who a team is over multiple seasons) and **current form** (what they're doing right now) are two different signals that should be weighted differently depending on how far into the season you are.

**Layer 1 — Long-run baseline (HomeForm_Long)**

EWMA with span=10 across all seasons. This is slow-moving and relatively stable. It answers "how good is this team as a franchise?" A newly promoted club that wins their first 5 games has a strong Layer 1 historical signal from their Championship days, but their Layer 1 Premier League signal needs time to build.

**Layer 2 — Current-season form (HomeForm_Season)**

EWMA with span=5, grouped by `[team, season]`. Resets to NaN at the start of each new season. This answers "how are they playing right now, this year?" A team that was dominant last season but has lost their manager and three key players starts each August from scratch in Layer 2.

**Layer 3 — Progressive blend (HomeForm_Blended)**

```python
HomeForm_Blended = SeasonProgress × HomeForm_Season
                 + (1 - SeasonProgress) × HomeForm_Long
```

Where `SeasonProgress = MatchNumber / 380`.

| Gameweek | Progress | Season weight | Historical weight |
|----------|----------|---------------|-------------------|
| GW1      | 0.03     | 3%            | 97%               |
| GW10     | 0.26     | 26%           | 74%               |
| GW20     | 0.53     | 53%           | 47%               |
| GW31     | 0.82     | 82%           | 18%               |
| GW38     | 1.00     | 100%          | 0%                |

This is a continuous adaptive prior — the model doesn't need to be told "it's late season, trust the table more." The weighting adjusts automatically.

---

## Model selection

**Algorithm:** `HistGradientBoostingClassifier` from scikit-learn.

**Why not XGBoost?** `HistGradientBoostingClassifier` handles NaN natively — no imputation step needed for early-season rows where rolling features haven't filled yet. It ships with scikit-learn (zero additional dependencies) and performs comparably to XGBoost on tabular data at this scale. Occam's razor.

**Why not Logistic Regression?** Football match outcomes depend on non-linear interactions between features. The relationship between ELO and form is multiplicative, not additive — a team with strong ELO and terrible form is more dangerous than form alone suggests, but less so than their ELO suggests. Gradient boosting captures these interactions without manually engineering them.

**Hyperparameters:**

```python
HistGradientBoostingClassifier(
    max_iter=300,       # trees — sufficient for convergence
    max_depth=4,        # shallow enough to prevent overfit
    learning_rate=0.04, # conservative shrinkage
    random_state=42     # reproducibility
)
```

The conservative learning rate (0.04) combined with 300 trees provides good generalisation. Tested against 0.1/100 trees — performance was similar, but the conservative setting showed more stability across validation folds.

---

## Why log loss, not accuracy

Accuracy is the right metric when all errors cost the same. In probabilistic forecasting, they don't.

A model that predicts 90% home win, and the away team wins, made a badly calibrated prediction. A model that predicted 55% home win and the away team wins made a reasonable prediction — it gave the event a 45% chance.

Accuracy treats both identically. Log loss penalises the first model heavily and the second model lightly. It measures whether the stated probabilities match the true outcome frequencies — exactly what matters for a calibration-focused system.

Our mean log loss of 1.105 means the model's match probabilities are reasonably calibrated against real outcomes over 4 validation seasons. The steady improvement from 1.2002 (2021/22) to 1.0283 (2024/25) reflects the model benefiting from more odds-era training data.

---

## Monte Carlo simulation

For each remaining fixture, the model produces a probability triplet (p_H, p_D, p_A). The simulation runs as follows:

```python
for sim in range(5000):
    points = dict(base_points)  # actual points already locked in
    for fixture in remaining_fixtures:
        outcome = random.choice([H, D, A], p=[p_H, p_D, p_A])
        update_points(points, outcome)
    record_winner(points)

title_probability = wins[team] / 5000
```

5,000 simulations gives stable estimates to within ±1.4% at 95% confidence (for a 50% probability event). For extreme probabilities like Arsenal's 99.3%, the standard error is negligible.

Key property: each fixture is simulated independently. This is a simplification — in reality, a team that loses GW32 has slightly different fatigue, morale, and rotation patterns in GW33. The simulation treats each remaining match as exchangeable. For late-season forecasting where the sample of remaining games is small and specific, this is a reasonable approximation.

---

## The neutral odds limitation

When predicting fixtures for which no bookmaker odds are available (upcoming matches not yet in the data file), the model falls back to `ProbH=0.45, ProbD=0.27, ProbA=0.28` — the rough Premier League average split.

This is a known limitation that produces counter-intuitive results for fixtures where current form strongly favours one side. Chelsea vs Man City at GW32 reads Chelsea 56% with the neutral prior — clearly wrong given City's form and ELO advantage.

With real market odds (estimated: Chelsea 2.80 / Draw 3.40 / City 2.60), the prediction reads Chelsea 34%, Draw 28%, City 37% — which reflects the actual balance of evidence.

The fix: pass real odds via `predict_match()` parameters. Football-data.co.uk publishes odds for upcoming fixtures — they can be scraped or copied manually for high-priority predictions.

---

## What the model cannot know

- Injury information (no team sheet data)
- Manager tactical instructions
- European competition fatigue (no fixture schedule cross-referencing)
- Transfer window changes not yet reflected in recent results
- Weather and pitch conditions

The bookmaker odds feature partially compensates for some of these — markets move significantly on injury news. But the model has no direct injury signal. For high-stakes predictions, odds injection is recommended precisely because it captures this information indirectly.

---

*This document is intended for technical reviewers evaluating the quality of the analytical methodology.*
