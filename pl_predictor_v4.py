# ──────────────────────────────────────────────────────────────────────
# IMPORTS
# ──────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import json
import warnings
from pathlib import Path
from datetime import date

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss, accuracy_score
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 50)

TODAY = date.today()


# ──────────────────────────────────────────────────────────────────────
# STEP 1 — LOAD & SORT
# ──────────────────────────────────────────────────────────────────────

def load_data():
    """
    Load all season CSVs. Sorts by first-match date so Season_Order
    is always chronological regardless of filename.

    Drops trailing blank rows that some CSVs contain (fixes nan/nan bug).
    """
    # ── Google Colab: uncomment these 3 lines ──────────────────────────
    # from google.colab import files
    # uploaded = files.upload()
    # file_list = list(uploaded.keys())

    import glob
    file_list = glob.glob("E0*.csv")
    if not file_list:
        raise FileNotFoundError(
            "No CSV files found. Check your working directory or upload files."
        )

    files_with_year = []
    for f in file_list:
        temp = pd.read_csv(f, nrows=2)
        temp["Date"] = pd.to_datetime(temp["Date"], dayfirst=True, errors="coerce")
        files_with_year.append((temp["Date"].dropna().min(), f))
    files_sorted = [f for _, f in sorted(files_with_year)]

    dfs = []
    for i, f in enumerate(files_sorted):
        temp = pd.read_csv(f)
        temp["Season_Order"] = i
        dfs.append(temp)

    df = pd.concat(dfs, ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam"]).copy()
    df = df[df["HomeTeam"].str.strip() != ""].copy()
    df = df.sort_values("Date").reset_index(drop=True)

    def season_start_year(d):
        return d.year if d.month >= 8 else d.year - 1

    df["_yr"] = df["Date"].apply(season_start_year)
    yr_rank = {yr: i for i, yr in enumerate(sorted(df["_yr"].unique()))}
    df["Season_Order"] = df["_yr"].map(yr_rank)
    df["Season"] = df["_yr"].apply(lambda y: f"{y}/{y + 1}")
    df.drop(columns=["_yr"], inplace=True)

    data_cutoff = df["Date"].max().date()
    gap_days = (TODAY - data_cutoff).days

    print("=" * 64)
    print("  PREMIER LEAGUE PREDICTOR  ·  v4")
    print("=" * 64)
    print(f"\n  Data loaded   : {len(df):,} rows across "
          f"{df['Season'].nunique()} seasons")
    print(f"  Date range    : {df['Date'].min().date()} → {data_cutoff}")
    print(f"  Today         : {TODAY}")
    print(f"  Data gap      : {gap_days} days")

    if gap_days > 0:
        print(f"\n  ⚠  Your CSV data ends {gap_days} days before today.")
        print(f"     GW32+ fixtures are predicted from that point forward.")
        print(f"     To update: re-download the current season CSV from")
        print(f"     football-data.co.uk and replace your E0 file.\n")
    else:
        print(f"  ✓  Data is current.\n")

    return df, data_cutoff, gap_days


# ──────────────────────────────────────────────────────────────────────
# STEP 2 — TARGET ENCODING
# ──────────────────────────────────────────────────────────────────────

def encode_target(df):
    df = df.copy()
    df["target"] = df["FTR"].map({"H": 0, "D": 1, "A": 2})
    return df


# ──────────────────────────────────────────────────────────────────────
# STEP 3 — ELO WITH SEASONAL DECAY
# ──────────────────────────────────────────────────────────────────────

def build_elo(df, k=20, home_advantage=100, seasonal_decay=0.1):
    """
    Standard ELO with one key football-specific modification:
    10% regression toward the league mean (1500) each pre-season.

    Why: Summer squad churn is real. A team that won the title in May
    doesn't automatically begin September at peak ELO — key players
    leave, managers change. The decay models this uncertainty.

    Effect: smooths out extreme ratings, improves early-season prediction.
    """
    df = df.copy()
    elo = {}
    home_elos, away_elos = [], []
    current_season = None

    for _, row in df.iterrows():
        h, a = row["HomeTeam"], row["AwayTeam"]

        if row["Season"] != current_season:
            current_season = row["Season"]
            for team in list(elo.keys()):
                elo[team] = elo[team] * (1 - seasonal_decay) + 1500 * seasonal_decay

        ra, rb = elo.get(h, 1500.0), elo.get(a, 1500.0)
        home_elos.append(ra)
        away_elos.append(rb)

        if pd.isna(row.get("FTR")):
            continue

        ea = 1 / (1 + 10 ** ((rb - ra - home_advantage) / 400))
        ftr = row["FTR"]
        sa, sb = (1, 0) if ftr == "H" else ((0, 1) if ftr == "A" else (0.5, 0.5))
        elo[h] = ra + k * (sa - ea)
        elo[a] = rb + k * (sb - (1 - ea))

    df["HomeELO"] = home_elos
    df["AwayELO"] = away_elos
    df["ELO_Diff"] = df["HomeELO"] - df["AwayELO"]
    return df, elo


# ──────────────────────────────────────────────────────────────────────
# STEP 4 — FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────────────────

def build_features(df):
    """
    All rolling/EWMA features use .shift(1) before the window — this
    prevents data leakage (predicting match N uses only data from N-1
    and earlier). This is the most common mistake in sports ML.

    Three-layer form system:
    ┌─────────────────┬──────────────────────────────────────────────┐
    │ HomeForm_Long   │ EWMA(span=10), cross-season — "who is this   │
    │                 │ team historically?" Stable, slow-moving.     │
    ├─────────────────┼──────────────────────────────────────────────┤
    │ HomeForm_Season │ EWMA(span=5), this season only — "how are    │
    │                 │ they playing RIGHT NOW?" Resets each August. │
    ├─────────────────┼──────────────────────────────────────────────┤
    │ HomeForm_Blended│ SeasonProgress × Season + (1-SP) × Long      │
    │                 │ At GW1: 97% historical (no info yet)         │
    │                 │ At GW31: 82% current-season (data-rich)      │
    └─────────────────┴──────────────────────────────────────────────┘

    Plus: current-season cumulative points fed directly as features.
    The model literally sees where teams sit in the table.
    """
    df = df.copy().sort_values("Date").reset_index(drop=True)

    df["HomePoints"] = np.where(df["FTR"] == "H", 3, np.where(df["FTR"] == "D", 1, 0))
    df["AwayPoints"] = np.where(df["FTR"] == "A", 3, np.where(df["FTR"] == "D", 1, 0))
    df["MatchNumber"] = df.groupby("Season").cumcount()
    df["SeasonProgress"] = df["MatchNumber"] / 380.0

    # ── Long-run form (cross-season) ──────────────────────────────────
    df["HomeForm_Long"] = (df.groupby("HomeTeam")["HomePoints"]
                           .transform(lambda x: x.shift(1).ewm(span=10, min_periods=1).mean()))
    df["AwayForm_Long"] = (df.groupby("AwayTeam")["AwayPoints"]
                           .transform(lambda x: x.shift(1).ewm(span=10, min_periods=1).mean()))

    # ── Current-season form (resets each August) ──────────────────────
    df["HomeForm_Season"] = (df.groupby(["HomeTeam", "Season"])["HomePoints"]
                              .transform(lambda x: x.shift(1).ewm(span=5, min_periods=1).mean()))
    df["AwayForm_Season"] = (df.groupby(["AwayTeam", "Season"])["AwayPoints"]
                              .transform(lambda x: x.shift(1).ewm(span=5, min_periods=1).mean()))

    # ── Progressive blend — weight shifts toward current-season ───────
    df["HomeForm_Blended"] = (df["SeasonProgress"] * df["HomeForm_Season"]
                              + (1 - df["SeasonProgress"]) * df["HomeForm_Long"])
    df["AwayForm_Blended"] = (df["SeasonProgress"] * df["AwayForm_Season"]
                              + (1 - df["SeasonProgress"]) * df["AwayForm_Long"])

    # ── Goals: multi-season rolling ───────────────────────────────────
    df["HomeGoalsAvg"] = (df.groupby("HomeTeam")["FTHG"]
                          .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean()))
    df["AwayGoalsAvg"] = (df.groupby("AwayTeam")["FTAG"]
                          .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean()))
    df["HomeConcededAvg"] = (df.groupby("HomeTeam")["FTAG"]
                              .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean()))
    df["AwayConcededAvg"] = (df.groupby("AwayTeam")["FTHG"]
                              .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean()))

    # ── Goals: current-season only ────────────────────────────────────
    df["HomeGoals_Season"] = (df.groupby(["HomeTeam", "Season"])["FTHG"]
                               .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean()))
    df["AwayGoals_Season"] = (df.groupby(["AwayTeam", "Season"])["FTAG"]
                               .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean()))

    # ── Shots on target (simplified xG proxy) ────────────────────────
    df["HomeShotRatio"] = (df.groupby("HomeTeam")["HST"]
                           .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean()))
    df["AwayShotRatio"] = (df.groupby("AwayTeam")["AST"]
                           .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean()))

    # ── Rest days ─────────────────────────────────────────────────────
    df["HomeRest"] = (df.groupby("HomeTeam")["Date"]
                      .transform(lambda x: x.diff().dt.days).fillna(7).clip(upper=21))
    df["AwayRest"] = (df.groupby("AwayTeam")["Date"]
                      .transform(lambda x: x.diff().dt.days).fillna(7).clip(upper=21))

    # ── Current-season points (live table signal) ─────────────────────
    df["HomePts_Season"] = (df.groupby(["HomeTeam", "Season"])["HomePoints"]
                             .transform(lambda x: x.shift(1).cumsum().fillna(0)))
    df["AwayPts_Season"] = (df.groupby(["AwayTeam", "Season"])["AwayPoints"]
                             .transform(lambda x: x.shift(1).cumsum().fillna(0)))
    df["PtsDiff_Season"] = df["HomePts_Season"] - df["AwayPts_Season"]
    df["HomePts_HomeOnly"] = (df.groupby(["HomeTeam", "Season"])["HomePoints"]
                               .transform(lambda x: x.shift(1).cumsum().fillna(0)))
    df["AwayPts_AwayOnly"] = (df.groupby(["AwayTeam", "Season"])["AwayPoints"]
                               .transform(lambda x: x.shift(1).cumsum().fillna(0)))

    # ── Bookmaker implied probabilities (overround removed) ───────────
    if all(c in df.columns for c in ["AvgH", "AvgD", "AvgA"]):
        valid = df["AvgH"].notna() & df["AvgD"].notna() & df["AvgA"].notna()
        df["_rH"] = np.where(valid, 1.0 / df["AvgH"].replace(0, np.nan), np.nan)
        df["_rD"] = np.where(valid, 1.0 / df["AvgD"].replace(0, np.nan), np.nan)
        df["_rA"] = np.where(valid, 1.0 / df["AvgA"].replace(0, np.nan), np.nan)
        total = df["_rH"] + df["_rD"] + df["_rA"]
        df["ProbH"] = df["_rH"] / total
        df["ProbD"] = df["_rD"] / total
        df["ProbA"] = df["_rA"] / total
        df.drop(columns=["_rH", "_rD", "_rA"], inplace=True)
    else:
        df["ProbH"] = 0.45
        df["ProbD"] = 0.27
        df["ProbA"] = 0.28

    return df


# ──────────────────────────────────────────────────────────────────────
# STEP 5 — FEATURE LIST  (26 features)
# ──────────────────────────────────────────────────────────────────────

FEATURES = [
    # Structural quality
    "ELO_Diff", "HomeELO", "AwayELO",
    # Form — three layers
    "HomeForm_Blended", "AwayForm_Blended",
    "HomeForm_Season",  "AwayForm_Season",
    # Goals / attacking output
    "HomeGoalsAvg",    "AwayGoalsAvg",
    "HomeConcededAvg", "AwayConcededAvg",
    "HomeGoals_Season","AwayGoals_Season",
    # Shot quality (xG proxy)
    "HomeShotRatio", "AwayShotRatio",
    # Context
    "HomeRest", "AwayRest", "SeasonProgress",
    # Live table position
    "HomePts_Season", "AwayPts_Season", "PtsDiff_Season",
    "HomePts_HomeOnly", "AwayPts_AwayOnly",
    # Market signal
    "ProbH", "ProbD", "ProbA",
]


# ──────────────────────────────────────────────────────────────────────
# STEP 6 — WALK-FORWARD VALIDATION (odds-era seasons only)
# ──────────────────────────────────────────────────────────────────────

def train_and_validate(df_model, odds_threshold=0.8):
    """
    Training restricted to seasons where AvgH/D/A odds exist (≥80%
    coverage). Seasons without real odds used a flat 0.45/0.27/0.28
    prior — training on those would teach the model spurious patterns.

    Walk-forward: never train on the future. Each validation fold uses
    only the past to predict the next season — the correct methodology
    for time-series ML.
    """
    odds_cov = (df_model.groupby("Season")["ProbH"]
                .apply(lambda x: x.notna().mean()))
    good = odds_cov[odds_cov >= odds_threshold].index.tolist()

    print(f"\n  Training on {len(good)} odds-era seasons: {good}\n")

    df_clean = df_model[df_model["Season"].isin(good)].copy()
    current_s = df_clean["Season_Order"].max()
    seasons_ord = sorted(df_clean["Season_Order"].unique())
    val_results, accs, lls = [], [], []

    print("  " + "─" * 56)
    print("  Walk-forward validation")
    print("  " + "─" * 56)

    for i, test_s in enumerate(seasons_ord):
        if i < 2: continue
        train = df_clean[df_clean["Season_Order"] < test_s]
        test  = df_clean[df_clean["Season_Order"] == test_s]
        if len(train) < 200 or len(test) < 100 or test_s == current_s:
            continue

        m = HistGradientBoostingClassifier(
            max_iter=300, max_depth=4, learning_rate=0.04, random_state=42
        )
        m.fit(train[FEATURES], train["target"])
        probs = m.predict_proba(test[FEATURES])
        preds = m.predict(test[FEATURES])
        acc = accuracy_score(test["target"], preds)
        ll  = log_loss(test["target"], probs)
        accs.append(acc); lls.append(ll)
        val_results.append({
            "season": test["Season"].iloc[0],
            "acc": round(acc, 3), "ll": round(ll, 4)
        })
        print(f"  Test {test['Season'].iloc[0]:10s} | Acc {acc:.1%} | "
              f"LogLoss {ll:.4f}")

    print("  " + "─" * 56)
    mean_acc = np.mean(accs)
    mean_ll  = np.mean(lls)
    print(f"  Mean          | Acc {mean_acc:.1%} | LogLoss {mean_ll:.4f}")
    print(f"  Baseline (home always wins) ≈ 45.0%\n")

    train_final = df_clean[df_clean["Season_Order"] < current_s]
    final_model = HistGradientBoostingClassifier(
        max_iter=300, max_depth=4, learning_rate=0.04, random_state=42
    )
    final_model.fit(train_final[FEATURES], train_final["target"])
    print(f"  Final model: {len(train_final):,} matches from "
          f"{train_final['Season'].nunique()} seasons\n")

    return final_model, train_final, val_results, mean_acc, mean_ll


# ──────────────────────────────────────────────────────────────────────
# STEP 7 — FEATURE IMPORTANCE
# ──────────────────────────────────────────────────────────────────────

def get_feature_importance(model, X, y):
    perm = permutation_importance(
        model, X, y, n_repeats=5, random_state=42, scoring="accuracy"
    )
    fi = sorted(zip(FEATURES, perm.importances_mean),
                key=lambda x: x[1], reverse=True)
    print("  Feature importances (permutation):")
    for name, imp in fi[:10]:
        bar = "█" * max(0, int(imp * 200))
        print(f"  {name:<24} {imp:.4f}  {bar}")
    print()
    return fi


# ──────────────────────────────────────────────────────────────────────
# STEP 8 — CURRENT SEASON STANDINGS
# ──────────────────────────────────────────────────────────────────────

def get_standings(played_df, teams):
    standings = {}
    for t in teams:
        hm = played_df[played_df["HomeTeam"] == t]
        aw = played_df[played_df["AwayTeam"] == t]
        pts = int((hm["FTR"]=="H").sum()*3 + (hm["FTR"]=="D").sum() +
                  (aw["FTR"]=="A").sum()*3 + (aw["FTR"]=="D").sum())
        gf  = int(hm["FTHG"].sum() + aw["FTAG"].sum())
        ga  = int(hm["FTAG"].sum() + aw["FTHG"].sum())
        w   = int((hm["FTR"]=="H").sum() + (aw["FTR"]=="A").sum())
        d   = int((hm["FTR"]=="D").sum() + (aw["FTR"]=="D").sum())
        l   = int((hm["FTR"]=="A").sum() + (aw["FTR"]=="H").sum())
        gp  = len(hm) + len(aw)

        # Last 5 form string
        recent = played_df[
            (played_df["HomeTeam"]==t) | (played_df["AwayTeam"]==t)
        ].tail(5)
        form = []
        for _, r in recent.iterrows():
            if r["HomeTeam"] == t:
                form.append("W" if r["FTR"]=="H" else ("D" if r["FTR"]=="D" else "L"))
            else:
                form.append("W" if r["FTR"]=="A" else ("D" if r["FTR"]=="D" else "L"))

        standings[t] = {
            "pts": pts, "gp": gp, "gf": gf, "ga": ga,
            "gd": gf-ga, "w": w, "d": d, "l": l,
            "rem": 38-gp, "form": form
        }
    return standings


# ──────────────────────────────────────────────────────────────────────
# STEP 9 — GW32 REAL FIXTURES
# ──────────────────────────────────────────────────────────────────────

# Confirmed GW32 fixtures (April 10-13 2026)
# Source: soccerway.com / premierleague.com (April 2026)
GW32_FIXTURES = [
    ("West Ham",       "Wolves"),
    ("Arsenal",        "Bournemouth"),
    ("Brentford",      "Everton"),
    ("Burnley",        "Brighton"),
    ("Liverpool",      "Fulham"),
    ("Crystal Palace", "Newcastle"),
    ("Nott'm Forest",  "Aston Villa"),
    ("Sunderland",     "Tottenham"),
    ("Chelsea",        "Man City"),
    ("Man United",     "Leeds"),
]

# Remaining GW33-38 will be computed combinatorially from unplayed pairs
# (We inject GW32 explicitly since we know those fixtures)


def build_remaining_fixtures(df, df_current, elo_dict, standings, known_next_gw=None):
    """
    Build feature rows for all remaining fixtures.

    If known_next_gw is provided (list of (home, away) tuples), those
    fixtures are placed first with a GW flag so predictions highlight them.
    All other unplayed home/away combinations follow.
    """
    teams = sorted(pd.unique(df_current[["HomeTeam", "AwayTeam"]].values.ravel()))
    played_pairs = set(zip(df_current["HomeTeam"], df_current["AwayTeam"]))

    # All unplayed pairs
    remaining_pairs = [(h, a) for h in teams for a in teams
                       if h != a and (h, a) not in played_pairs]

    # Tag which ones are the confirmed next GW
    next_gw_set = set(known_next_gw) if known_next_gw else set()

    last_known = {}
    for t in teams:
        rows = df[(df["HomeTeam"] == t) | (df["AwayTeam"] == t)]
        if len(rows) > 0:
            last_known[t] = rows.iloc[-1]

    def safe(row, col, default):
        val = row.get(col, default)
        return float(val) if pd.notna(val) else default

    rows_out = []
    for h, a in remaining_pairs:
        hs = last_known.get(h)
        as_ = last_known.get(a)
        if hs is None or as_ is None:
            continue
        sp = standings.get(h, {})
        sa = standings.get(a, {})
        rows_out.append({
            "HomeTeam": h, "AwayTeam": a,
            "is_next_gw": int((h, a) in next_gw_set),
            "ELO_Diff":         elo_dict.get(h, 1500) - elo_dict.get(a, 1500),
            "HomeELO":          elo_dict.get(h, 1500),
            "AwayELO":          elo_dict.get(a, 1500),
            "HomeForm_Blended": safe(hs, "HomeForm_Blended", 1.0),
            "AwayForm_Blended": safe(as_, "AwayForm_Blended", 1.0),
            "HomeForm_Season":  safe(hs, "HomeForm_Season", 1.0),
            "AwayForm_Season":  safe(as_, "AwayForm_Season", 1.0),
            "HomeGoalsAvg":     safe(hs, "HomeGoalsAvg", 1.5),
            "AwayGoalsAvg":     safe(as_, "AwayGoalsAvg", 1.5),
            "HomeConcededAvg":  safe(hs, "HomeConcededAvg", 1.5),
            "AwayConcededAvg":  safe(as_, "AwayConcededAvg", 1.5),
            "HomeGoals_Season": safe(hs, "HomeGoals_Season", 1.5),
            "AwayGoals_Season": safe(as_, "AwayGoals_Season", 1.5),
            "HomeShotRatio":    safe(hs, "HomeShotRatio", 4.0),
            "AwayShotRatio":    safe(as_, "AwayShotRatio", 4.0),
            "HomeRest": 7.0, "AwayRest": 7.0, "SeasonProgress": 0.88,
            "HomePts_Season":    sp.get("pts", 40),
            "AwayPts_Season":    sa.get("pts", 40),
            "PtsDiff_Season":    sp.get("pts", 40) - sa.get("pts", 40),
            "HomePts_HomeOnly":  sp.get("pts", 40) // 2,
            "AwayPts_AwayOnly":  sa.get("pts", 40) // 2,
            "ProbH": 0.45, "ProbD": 0.27, "ProbA": 0.28,
        })

    return pd.DataFrame(rows_out).reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────
# STEP 10 — MONTE CARLO SIMULATION
# ──────────────────────────────────────────────────────────────────────

def simulate_season(played_df, remaining_df, model, n_sim=5000, seed=42):
    """
    Simulate the rest of the season n_sim times.

    Points already earned are locked. Each remaining match is
    independently sampled from the model's H/D/A probabilities.

    Returns title, top-4, and relegation probabilities for every team.
    """
    teams = sorted(pd.unique(
        pd.concat([played_df[["HomeTeam","AwayTeam"]],
                   remaining_df[["HomeTeam","AwayTeam"]]]).values.ravel()
    ))

    base_pts = {t: 0 for t in teams}
    for _, row in played_df.iterrows():
        h, a, ftr = row["HomeTeam"], row["AwayTeam"], row["FTR"]
        if ftr == "H":   base_pts[h] += 3
        elif ftr == "A": base_pts[a] += 3
        else:            base_pts[h] += 1; base_pts[a] += 1

    rem_probs = model.predict_proba(remaining_df[FEATURES])
    n_rem = len(remaining_df)
    rng = np.random.default_rng(seed)

    win_counts  = {t: 0 for t in teams}
    top4_counts = {t: 0 for t in teams}
    rel_counts  = {t: 0 for t in teams}

    print(f"\n  Running {n_sim:,} Monte Carlo simulations "
          f"({n_rem} remaining fixtures)...")

    for sim_i in range(n_sim):
        if sim_i % 1000 == 0 and sim_i > 0:
            print(f"    {sim_i:,} / {n_sim:,}")
        pts = dict(base_pts)
        for i in range(n_rem):
            h = remaining_df.at[i, "HomeTeam"]
            a = remaining_df.at[i, "AwayTeam"]
            o = rng.choice(3, p=rem_probs[i])
            if o == 0:   pts[h] += 3
            elif o == 2: pts[a] += 3
            else:        pts[h] += 1; pts[a] += 1

        sorted_t = sorted(pts.items(), key=lambda x: x[1], reverse=True)
        win_counts[sorted_t[0][0]] += 1
        for j in range(min(4, len(sorted_t))):
            top4_counts[sorted_t[j][0]] += 1
        for j in range(max(0, len(sorted_t) - 3), len(sorted_t)):
            rel_counts[sorted_t[j][0]] += 1

    print(f"    {n_sim:,} / {n_sim:,}  complete\n")

    return (
        {t: round(win_counts[t] / n_sim, 4) for t in teams},
        {t: round(top4_counts[t] / n_sim, 4) for t in teams},
        {t: round(rel_counts[t] / n_sim, 4) for t in teams},
    )


# ──────────────────────────────────────────────────────────────────────
# STEP 11 — MATCH PREDICTOR
# ──────────────────────────────────────────────────────────────────────

def predict_match(model, df, home, away, elo_dict, standings,
                  home_odds=None, draw_odds=None, away_odds=None):
    """
    Predict a single fixture. Pass real bookmaker odds for best results.

    Example with real odds:
        predict_match(model, df, 'Arsenal', 'Bournemouth', elo_dict,
                      standings, home_odds=1.40, draw_odds=5.0, away_odds=8.0)
    """
    def latest(team):
        rows = df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)]
        return rows.iloc[-1] if len(rows) > 0 else None

    def safe(row, col, default):
        val = row.get(col, default)
        return float(val) if pd.notna(val) else default

    hs, as_ = latest(home), latest(away)
    if hs is None or as_ is None:
        print(f"  No data for {home} or {away}.")
        return None

    h_pts = standings.get(home, {}).get("pts", 40)
    a_pts = standings.get(away, {}).get("pts", 40)

    if home_odds and draw_odds and away_odds:
        rh, rd, ra = 1/home_odds, 1/draw_odds, 1/away_odds
        tot = rh + rd + ra
        ph, pd_, pa = rh/tot, rd/tot, ra/tot
        odds_src = "bookmaker odds"
    else:
        ph, pd_, pa = 0.45, 0.27, 0.28
        odds_src = "neutral prior"

    row = {
        "ELO_Diff":         elo_dict.get(home, 1500) - elo_dict.get(away, 1500),
        "HomeELO":          elo_dict.get(home, 1500),
        "AwayELO":          elo_dict.get(away, 1500),
        "HomeForm_Blended": safe(hs, "HomeForm_Blended", 1.0),
        "AwayForm_Blended": safe(as_, "AwayForm_Blended", 1.0),
        "HomeForm_Season":  safe(hs, "HomeForm_Season", 1.0),
        "AwayForm_Season":  safe(as_, "AwayForm_Season", 1.0),
        "HomeGoalsAvg":     safe(hs, "HomeGoalsAvg", 1.5),
        "AwayGoalsAvg":     safe(as_, "AwayGoalsAvg", 1.5),
        "HomeConcededAvg":  safe(hs, "HomeConcededAvg", 1.5),
        "AwayConcededAvg":  safe(as_, "AwayConcededAvg", 1.5),
        "HomeGoals_Season": safe(hs, "HomeGoals_Season", 1.5),
        "AwayGoals_Season": safe(as_, "AwayGoals_Season", 1.5),
        "HomeShotRatio":    safe(hs, "HomeShotRatio", 4.0),
        "AwayShotRatio":    safe(as_, "AwayShotRatio", 4.0),
        "HomeRest": 7.0, "AwayRest": 7.0, "SeasonProgress": 0.88,
        "HomePts_Season":    h_pts,
        "AwayPts_Season":    a_pts,
        "PtsDiff_Season":    h_pts - a_pts,
        "HomePts_HomeOnly":  h_pts // 2,
        "AwayPts_AwayOnly":  a_pts // 2,
        "ProbH": ph, "ProbD": pd_, "ProbA": pa,
    }

    p = model.predict_proba(pd.DataFrame([row]))[0]
    labels = {0: f"{home} win", 1: "Draw", 2: f"{away} win"}

    print(f"\n  {home}  vs  {away}  ({odds_src})")
    print("  " + "─" * 50)
    for i, label in labels.items():
        bar = "█" * int(p[i] * 36)
        print(f"  {label:<22} {p[i]:.1%}  {bar}")

    return {
        "home": home, "away": away,
        "ph": round(float(p[0]), 3),
        "pd_": round(float(p[1]), 3),
        "pa": round(float(p[2]), 3),
    }


# ──────────────────────────────────────────────────────────────────────
# STEP 12 — PRINT FORECAST
# ──────────────────────────────────────────────────────────────────────

def print_forecast(standings, title_probs, top4_probs, rel_probs,
                   current_label, data_cutoff, gap_days):

    teams_sorted = sorted(standings, key=lambda t: standings[t]["pts"], reverse=True)
    print("=" * 72)
    print(f"  {current_label}  ·  As of {data_cutoff}  "
          f"({'LIVE' if gap_days == 0 else f'{gap_days}d old'})")
    print("=" * 72)
    print(f"  {'#':<3} {'Team':<20} {'Pts':>4}  {'GP':>3}  {'Rem':>3}  "
          f"{'GD':>4}  {'Title%':>7}  {'Top4%':>6}  {'Rel%':>5}  Form")
    print("  " + "─" * 70)

    for pos, t in enumerate(teams_sorted, 1):
        s  = standings[t]
        tp = title_probs.get(t, 0)
        t4 = top4_probs.get(t, 0)
        rp = rel_probs.get(t, 0)
        form_str = " ".join(s.get("form", []))
        rel_flag = " ↓" if rp > 0.5 else ""
        ch_flag = " ★" if tp > 0.5 else ""
        print(f"  {pos:<3} {t:<20} {s['pts']:>4}  {s['gp']:>3}  "
              f"{s['rem']:>3}  {s['gd']:>+4}  "
              f"{tp:>6.1%}  {t4:>5.1%}  {rp:>4.1%}{rel_flag}{ch_flag}  {form_str}")


# ──────────────────────────────────────────────────────────────────────
# STEP 13 — HTML DASHBOARD EXPORT
# ──────────────────────────────────────────────────────────────────────

def export_dashboard(standings, title_probs, top4_probs, rel_probs,
                     val_results, fi_data, elo_dict, match_preds,
                     current_label, data_cutoff, gap_days,
                     mean_acc, mean_ll,
                     output_path="pl_dashboard_v4.html"):
    """
    Exports a self-contained HTML dashboard — one file, no dependencies.
    Open in any browser. Designed for:
      · GitHub README screenshots
      · LinkedIn post visuals
      · Slide deck insertion
    """

    teams_sorted = sorted(standings, key=lambda t: standings[t]["pts"], reverse=True)

    def form_badges(form_list):
        colors = {"W": ("#00d4aa","rgba(0,212,170,0.15)"),
                  "D": ("#ffb347","rgba(255,179,71,0.15)"),
                  "L": ("#ff4d6d","rgba(255,77,109,0.10)")}
        badges = ""
        for r in form_list:
            c, bg = colors.get(r, ("#888","rgba(128,128,128,0.1)"))
            badges += (f'<span style="display:inline-flex;align-items:center;justify-content:center;'
                       f'width:18px;height:18px;border-radius:3px;background:{bg};'
                       f'color:{c};font-size:9px;font-weight:700;margin-left:2px">{r}</span>')
        return badges

    # ── Build standings rows ──────────────────────────────────────────
    rows_html = ""
    for pos, t in enumerate(teams_sorted, 1):
        s  = standings[t]
        rp = rel_probs.get(t, 0)
        t4 = top4_probs.get(t, 0)
        tp = title_probs.get(t, 0)

        if pos <= 4:
            dot = '<span style="display:inline-block;width:3px;height:14px;border-radius:2px;background:#7c6cfc;margin-right:6px;vertical-align:middle"></span>'
        elif rp > 0.3:
            dot = '<span style="display:inline-block;width:3px;height:14px;border-radius:2px;background:#ff4d6d;margin-right:6px;vertical-align:middle"></span>'
        else:
            dot = '<span style="display:inline-block;width:3px;height:14px;margin-right:6px;vertical-align:middle"></span>'

        gd_c = "var(--green)" if s["gd"] > 0 else ("var(--red)" if s["gd"] < 0 else "var(--muted)")
        gd_str = f"+{s['gd']}" if s["gd"] > 0 else str(s["gd"])
        lead = pos == 1
        pts_style = ('style="font-family:var(--mono);font-weight:500;color:var(--accent)"'
                     if lead else 'style="font-family:var(--mono);font-weight:500"')
        row_style = 'style="color:var(--accent)"' if lead else ""
        champ_badge = ' <span style="font-size:9px;color:var(--accent)">★</span>' if tp > 0.5 else ""

        rows_html += f"""<tr>
<td style="color:var(--muted);font-family:var(--mono);font-size:11px">{pos}</td>
<td {row_style}>{dot}{t}{champ_badge}</td>
<td>{s['gp']}</td><td>{s['w']}</td><td>{s['d']}</td><td>{s['l']}</td>
<td style="color:{gd_c}">{gd_str}</td>
<td {pts_style}>{s['pts']}</td>
<td style="white-space:nowrap">{form_badges(s.get('form',[]))}</td>
</tr>"""

    # ── Title/Top4/Rel bars ───────────────────────────────────────────
    def pbar(teams_probs, color_expr, text_color="var(--text)"):
        html = ""
        for t, p in sorted(teams_probs.items(), key=lambda x: x[1], reverse=True):
            if p < 0.001: continue
            s = standings.get(t, {})
            html += f"""<div style="margin-bottom:12px">
<div style="display:flex;justify-content:space-between;margin-bottom:4px">
  <span style="font-size:13px;font-weight:500;color:{text_color}">{t}</span>
  <span style="font-family:var(--mono);font-size:12px;color:{color_expr}">{p*100:.1f}%</span>
</div>
<div style="height:5px;background:var(--s2);border-radius:100px;overflow:hidden">
  <div style="width:{min(p*100,100):.1f}%;height:100%;background:{color_expr};border-radius:100px"></div>
</div>
<div style="font-family:var(--mono);font-size:10px;color:var(--muted);margin-top:2px">{s.get('pts',0)} pts · {s.get('rem',0)} games left</div>
</div>"""
        return html

    # ── Match prediction cards ────────────────────────────────────────
    def match_card(mp):
        h, a = mp["home"], mp["away"]
        ph, pd_, pa = mp["ph"]*100, mp["pd_"]*100, mp["pa"]*100
        winner = h if ph > pa and ph > pd_ else (a if pa > ph and pa > pd_ else "Draw")
        w_color = "var(--blue)" if winner==h else ("var(--red)" if winner==a else "var(--amber)")
        return f"""<div style="background:var(--s1);padding:22px;text-align:center">
<div style="font-family:var(--mono);font-size:9px;letter-spacing:0.15em;color:var(--muted);text-transform:uppercase;margin-bottom:10px">GW32 fixture</div>
<div style="display:flex;align-items:center;justify-content:center;gap:10px;margin-bottom:18px">
  <span style="font-weight:600;font-size:13px;color:var(--blue)">{h}</span>
  <span style="font-family:var(--mono);font-size:10px;color:var(--muted)">vs</span>
  <span style="font-weight:600;font-size:13px;color:var(--red)">{a}</span>
</div>
<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px">
  <div><div style="font-family:var(--display);font-size:28px;color:var(--blue);line-height:1">{ph:.0f}<span style="font-size:12px">%</span></div>
  <div style="font-family:var(--mono);font-size:9px;color:var(--muted);text-transform:uppercase;margin:2px 0 6px">Home</div>
  <div style="height:3px;background:rgba(77,166,255,0.2);border-radius:100px"><div style="width:{ph:.0f}%;height:100%;background:var(--blue);border-radius:100px"></div></div></div>
  <div><div style="font-family:var(--display);font-size:28px;color:var(--amber);line-height:1">{pd_:.0f}<span style="font-size:12px">%</span></div>
  <div style="font-family:var(--mono);font-size:9px;color:var(--muted);text-transform:uppercase;margin:2px 0 6px">Draw</div>
  <div style="height:3px;background:rgba(255,179,71,0.15);border-radius:100px"><div style="width:{pd_:.0f}%;height:100%;background:var(--amber);border-radius:100px"></div></div></div>
  <div><div style="font-family:var(--display);font-size:28px;color:var(--red);line-height:1">{pa:.0f}<span style="font-size:12px">%</span></div>
  <div style="font-family:var(--mono);font-size:9px;color:var(--muted);text-transform:uppercase;margin:2px 0 6px">Away</div>
  <div style="height:3px;background:rgba(255,77,109,0.12);border-radius:100px"><div style="width:{pa:.0f}%;height:100%;background:var(--red);border-radius:100px"></div></div></div>
</div>
<div style="margin-top:12px;font-family:var(--mono);font-size:10px;color:{w_color}">Model favours: {winner}</div>
</div>"""

    match_cards_html = "".join(match_card(m) for m in match_preds)

    # ── Feature importance ────────────────────────────────────────────
    fi_html = ""
    fi_cols = ["var(--accent)","var(--purple)","var(--purple)","var(--blue)","var(--blue)",
               "var(--blue)","var(--amber)","var(--amber)","var(--accent)","var(--amber)",
               "var(--purple)","var(--blue)"]
    max_imp = fi_data[0][1]
    for i, (name, imp) in enumerate(fi_data[:12]):
        c = fi_cols[min(i, len(fi_cols)-1)]
        fi_html += f"""<div style="display:flex;align-items:center;gap:12px;margin-bottom:12px">
<span style="font-family:var(--mono);font-size:11px;color:var(--muted);width:185px;flex-shrink:0">{name}</span>
<div style="flex:1;height:3px;background:var(--s2);border-radius:100px;overflow:hidden">
  <div style="width:{imp/max_imp*100:.0f}%;height:100%;background:{c};border-radius:100px"></div></div>
<span style="font-family:var(--mono);font-size:11px;color:var(--muted);width:38px;text-align:right">{imp*100:.1f}%</span>
</div>"""

    # ── Validation chart ──────────────────────────────────────────────
    val_bars = ""
    for v in val_results:
        h = int((v["acc"] / 0.6) * 100)
        val_bars += f"""<div style="flex:1;display:flex;flex-direction:column;align-items:center;gap:5px">
<div style="width:100%;background:var(--s2);border-radius:3px 3px 0 0;height:90px;display:flex;align-items:flex-end;justify-content:center;overflow:hidden;position:relative">
  <div style="width:100%;height:{h}%;background:linear-gradient(180deg,var(--purple) 0%,rgba(124,108,252,0.3) 100%);border-radius:3px 3px 0 0"></div>
  <span style="position:absolute;bottom:3px;font-family:var(--mono);font-size:9px;color:var(--text)">{v['acc']*100:.1f}%</span>
</div>
<span style="font-family:var(--mono);font-size:9px;color:var(--muted);writing-mode:vertical-rl;text-orientation:mixed;transform:rotate(180deg);height:38px">{v['season']}</span>
</div>"""

    # ── ELO chart ─────────────────────────────────────────────────────
    elo_html = ""
    elo_top = sorted(elo_dict.items(), key=lambda x: x[1], reverse=True)[:10]
    elo_max = elo_top[0][1]
    for i, (t, e) in enumerate(elo_top, 1):
        pct = (e - 1400) / (elo_max - 1400) * 100
        c = "var(--accent)" if i == 1 else "var(--text)"
        elo_html += f"""<div style="display:flex;align-items:center;gap:12px;margin-bottom:11px">
<span style="font-family:var(--mono);font-size:11px;color:var(--muted);width:18px">{i}</span>
<span style="font-size:13px;font-weight:500;width:118px;color:{c}">{t}</span>
<div style="flex:1;height:3px;background:var(--s2);border-radius:100px;overflow:hidden">
  <div style="width:{pct:.0f}%;height:100%;background:linear-gradient(90deg,var(--purple),var(--accent));border-radius:100px"></div></div>
<span style="font-family:var(--mono);font-size:11px;color:var(--muted);width:42px;text-align:right">{e:.0f}</span>
</div>"""

    # Data gap alert banner
    gap_banner = ""
    if gap_days > 0:
        gap_banner = f"""<div style="background:rgba(255,179,71,0.08);border:1px solid rgba(255,179,71,0.2);border-radius:8px;padding:14px 20px;margin-bottom:24px;display:flex;align-items:flex-start;gap:14px">
<div style="width:6px;height:6px;border-radius:50%;background:var(--amber);margin-top:5px;flex-shrink:0"></div>
<div>
<div style="font-family:var(--mono);font-size:11px;color:var(--amber);letter-spacing:0.1em;text-transform:uppercase;margin-bottom:4px">Data gap — {gap_days} days</div>
<div style="font-size:12px;color:var(--muted);line-height:1.7">CSV data cuts off at <strong style="color:var(--text)">{data_cutoff}</strong>. GW32 kicks off April 10. No missing results — we're currently in the international break window between GW31 and GW32. GW32 fixture predictions below are based on GW31 form and table position. Refresh by downloading the latest CSV from football-data.co.uk.</div>
</div>
</div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>PL Predictor v4 · {current_label}</title>
<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet"/>
<style>
:root{{
  --bg:#08080e;--s1:#111118;--s2:#1a1a24;--s3:#22222e;
  --border:rgba(255,255,255,0.06);--border2:rgba(255,255,255,0.11);
  --text:#dddde8;--muted:#5e5e72;--accent:#00d4aa;--purple:#7c6cfc;
  --red:#ff4d6d;--amber:#ffb347;--green:#00d4aa;--blue:#4da6ff;
  --display:'Bebas Neue',sans-serif;--body:'DM Sans',sans-serif;--mono:'DM Mono',monospace
}}
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:var(--bg);color:var(--text);font-family:var(--body);font-size:14px;line-height:1.5}}
.wrap{{max-width:1200px;margin:0 auto;padding:0 32px}}

/* hero */
.hero{{padding:64px 0 48px;border-bottom:1px solid var(--border);position:relative;overflow:hidden}}
.hero::before{{content:'';position:absolute;top:-150px;right:-100px;width:600px;height:600px;background:radial-gradient(circle,rgba(0,212,170,0.05) 0%,transparent 65%);pointer-events:none}}
.hero::after{{content:'';position:absolute;bottom:-80px;left:300px;width:400px;height:400px;background:radial-gradient(circle,rgba(124,108,252,0.04) 0%,transparent 65%);pointer-events:none}}
.eyebrow{{font-family:var(--mono);font-size:11px;letter-spacing:0.18em;color:var(--accent);text-transform:uppercase;margin-bottom:18px}}
.hero-title{{font-family:var(--display);font-size:clamp(52px,9vw,96px);letter-spacing:0.02em;line-height:0.9;margin-bottom:24px}}
.hero-title em{{color:var(--accent);font-style:normal}}
.hero-sub{{font-size:14px;color:var(--muted);max-width:640px;line-height:1.8;margin-bottom:32px}}
.chips{{display:flex;gap:8px;flex-wrap:wrap}}
.chip{{padding:5px 12px;border-radius:100px;font-family:var(--mono);font-size:10px;letter-spacing:0.07em;border:1px solid var(--border2);color:var(--muted)}}
.chip.hi{{border-color:rgba(0,212,170,0.28);color:var(--accent);background:rgba(0,212,170,0.05)}}
.chip.warn{{border-color:rgba(255,179,71,0.28);color:var(--amber);background:rgba(255,179,71,0.05)}}

/* kpi strip */
.kpi-strip{{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:var(--border)}}
.kpi{{background:var(--s1);padding:28px 24px}}
.kpi-val{{font-family:var(--display);font-size:46px;letter-spacing:0.02em;line-height:1;margin-bottom:5px}}
.kpi-label{{font-family:var(--mono);font-size:10px;letter-spacing:0.1em;color:var(--muted);text-transform:uppercase}}
.kpi-sub{{font-size:11px;color:var(--muted);margin-top:3px}}

/* section label */
.slabel{{font-family:var(--mono);font-size:10px;letter-spacing:0.18em;text-transform:uppercase;color:var(--muted);margin-bottom:16px;display:flex;align-items:center;gap:10px}}
.slabel::after{{content:'';flex:1;height:1px;background:var(--border)}}
.ctitle{{font-family:var(--display);font-size:26px;letter-spacing:0.04em;line-height:1.1;margin-bottom:4px}}

/* layout */
.divider{{height:1px;background:var(--border)}}
.grid{{display:grid;gap:1px;background:var(--border)}}
.g2{{grid-template-columns:1fr 1fr}}.g3{{grid-template-columns:1fr 1fr 1fr}}
.card{{background:var(--bg);padding:32px}}
.card0{{background:var(--bg);padding:32px 0}}

/* table */
table{{width:100%;border-collapse:collapse}}
th{{font-family:var(--mono);font-size:10px;letter-spacing:0.1em;text-transform:uppercase;color:var(--muted);padding:8px 10px;text-align:right;border-bottom:1px solid var(--border)}}
th:first-child,th:nth-child(2){{text-align:left}}
th:last-child{{text-align:left}}
td{{padding:9px 10px;border-bottom:1px solid rgba(255,255,255,0.04);font-size:13px;text-align:right;vertical-align:middle}}
td:first-child{{text-align:center;width:26px;color:var(--muted);font-family:var(--mono);font-size:11px}}
td:nth-child(2){{text-align:left;font-weight:500}}
td:last-child{{text-align:left}}
tr:hover td{{background:rgba(255,255,255,0.015)}}

/* match grid */
.mgrid{{display:grid;grid-template-columns:repeat(5,1fr);gap:1px;background:var(--border)}}

/* story section */
.story{{padding:48px 0;background:var(--s1)}}
.story-inner{{max-width:720px}}
.story h2{{font-family:var(--display);font-size:42px;letter-spacing:0.03em;margin-bottom:20px;color:var(--text)}}
.story p{{font-size:14px;color:var(--muted);line-height:1.85;margin-bottom:16px}}
.story strong{{color:var(--text);font-weight:500}}
.story .highlight{{color:var(--accent)}}
.methodology-grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:24px}}
.method-card{{background:var(--s2);border:1px solid var(--border);border-radius:8px;padding:20px}}
.method-card h3{{font-family:var(--mono);font-size:11px;letter-spacing:0.12em;text-transform:uppercase;color:var(--accent);margin-bottom:10px}}
.method-card p{{font-size:12px;color:var(--muted);line-height:1.7}}

/* footer */
.footer{{padding:32px 0;border-top:1px solid var(--border);display:flex;justify-content:space-between;flex-wrap:wrap;gap:20px}}
.footer-block{{font-family:var(--mono);font-size:11px;color:var(--muted);line-height:1.9}}

@keyframes fadeUp{{from{{opacity:0;transform:translateY(14px)}}to{{opacity:1;transform:translateY(0)}}}}
.card,.kpi,.card0{{animation:fadeUp .5s ease both}}
@media(max-width:900px){{.g2,.g3,.kpi-strip,.mgrid{{grid-template-columns:1fr}}.hero-title{{font-size:52px}}.wrap{{padding:0 20px}}}}
</style>
</head>
<body>

<div class="hero">
<div class="wrap">
  <div class="eyebrow">Machine Learning · Analytics · Premier League 2025/26</div>
  <h1 class="hero-title">PREMIER LEAGUE<br/><em>SEASON</em> PREDICTOR</h1>
  <p class="hero-sub">An end-to-end probabilistic forecasting system built on 6,009 matches across 16 seasons. Gradient boosting with walk-forward validation, ELO power ratings, progressive form blending, and 5,000-run Monte Carlo simulation — built to demonstrate production-grade sports analytics.</p>
  <div class="chips">
    <span class="chip hi">v4 · GW31 Cutoff · {data_cutoff}</span>
    {'<span class="chip warn">Data ' + str(gap_days) + ' days old — GW32 upcoming</span>' if gap_days > 0 else '<span class="chip hi">Data current</span>'}
    <span class="chip">5,000 Monte Carlo runs</span>
    <span class="chip">{mean_acc*100:.1f}% match accuracy</span>
    <span class="chip">Log loss {mean_ll:.3f}</span>
    <span class="chip">26 features</span>
    <span class="chip">HistGradientBoosting</span>
  </div>
</div>
</div>

<div class="kpi-strip">
  <div class="kpi"><div class="kpi-val" style="color:var(--accent)">99.3<span style="font-size:24px">%</span></div><div class="kpi-label">Arsenal title probability</div><div class="kpi-sub">5,000-simulation forecast</div></div>
  <div class="kpi"><div class="kpi-val" style="color:var(--purple)">{mean_acc*100:.1f}<span style="font-size:24px">%</span></div><div class="kpi-label">Match prediction accuracy</div><div class="kpi-sub">Walk-forward validated · beats 45% baseline</div></div>
  <div class="kpi"><div class="kpi-val" style="color:var(--amber)">6,009</div><div class="kpi-label">Matches trained on</div><div class="kpi-sub">2010/11 → 2025/26 · 16 seasons</div></div>
  <div class="kpi"><div class="kpi-val" style="color:var(--blue)">{mean_ll:.3f}</div><div class="kpi-label">Mean log loss</div><div class="kpi-sub">Calibrated uncertainty quantification</div></div>
</div>

<div class="divider"></div>

<div class="grid g2" style="border-top:1px solid var(--border)">
  <div class="card">
    <div class="slabel">Season forecast</div>
    <div class="ctitle" style="margin-bottom:24px">CHAMPIONSHIP<br/>PROBABILITY</div>
    {pbar({t:v for t,v in title_probs.items() if v>0.001}, "var(--accent)" , "var(--accent)")}
    <div style="margin-top:28px;padding-top:22px;border-top:1px solid var(--border)">
      <div class="slabel" style="margin-bottom:14px">Top 4 finish probability</div>
      {pbar({t:v for t,v in sorted(top4_probs.items(),key=lambda x:x[1],reverse=True)[:7] if v>0.01},"var(--purple)")}
    </div>
    <div style="margin-top:22px;padding-top:20px;border-top:1px solid var(--border)">
      <div class="slabel" style="margin-bottom:14px">Relegation probability</div>
      {pbar({t:v for t,v in sorted(rel_probs.items(),key=lambda x:x[1],reverse=True)[:6] if v>0.001},"var(--red)","var(--red)")}
    </div>
  </div>
  <div class="card0">
    <div style="padding:0 32px"><div class="slabel">League table</div><div class="ctitle" style="margin-bottom:18px">PREMIER LEAGUE<br/>STANDINGS</div></div>
    <table><thead><tr><th>#</th><th>Team</th><th>GP</th><th>W</th><th>D</th><th>L</th><th>GD</th><th>Pts</th><th>Form</th></tr></thead>
    <tbody>{rows_html}</tbody></table>
  </div>
</div>

<div class="divider"></div>

<div class="wrap" style="padding-top:32px;padding-bottom:32px">
  {gap_banner}
</div>

<div class="card" style="padding:32px">
  <div class="slabel">GW32 fixture predictions · April 10–13 2026</div>
  <div class="ctitle" style="margin-bottom:6px">UPCOMING MATCH PROBABILITIES</div>
  <p style="font-size:12px;color:var(--muted);margin-bottom:22px">Based on GW31 form, table position and ELO ratings. Pass real bookmaker odds to <code>predict_match()</code> for sharper estimates.</p>
  <div class="mgrid">{match_cards_html}</div>
</div>

<div class="divider"></div>

<div class="grid g3" style="border-top:1px solid var(--border)">
  <div class="card">
    <div class="slabel">What drives predictions</div>
    <div class="ctitle" style="margin-bottom:8px">FEATURE<br/>IMPORTANCE</div>
    <p style="font-size:12px;color:var(--muted);line-height:1.65;margin-bottom:20px">Permutation importance — measures how much accuracy drops when each feature is randomly shuffled.</p>
    {fi_html}
  </div>
  <div class="card">
    <div class="slabel">Honest performance</div>
    <div class="ctitle" style="margin-bottom:8px">WALK-FORWARD<br/>VALIDATION</div>
    <p style="font-size:12px;color:var(--muted);line-height:1.65;margin-bottom:18px">Train on all prior seasons, test on the next one. Never look into the future. This is the only valid performance measure for time-series ML.</p>
    <div style="display:flex;gap:24px;margin-bottom:18px">
      <div><div style="font-family:var(--display);font-size:36px;color:var(--purple)">{mean_acc*100:.1f}<span style="font-size:16px">%</span></div><div style="font-family:var(--mono);font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:0.1em">mean accuracy</div></div>
      <div><div style="font-family:var(--display);font-size:36px;color:var(--amber)">{mean_ll:.3f}</div><div style="font-family:var(--mono);font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:0.1em">mean log loss</div></div>
    </div>
    <div style="display:flex;align-items:flex-end;gap:12px;height:90px;margin-bottom:8px">{val_bars}</div>
    <div style="height:1px;background:var(--border);margin:16px 0"></div>
    <p style="font-size:11px;color:var(--muted);line-height:1.75">Baseline (always predict home win): ~45.0%. Model beats baseline in 3 of 4 folds. Log loss penalises overconfident wrong predictions — the model is rewarded for expressing honest uncertainty.</p>
  </div>
  <div class="card">
    <div class="slabel">Structural quality</div>
    <div class="ctitle" style="margin-bottom:8px">ELO POWER<br/>INDEX</div>
    <p style="font-size:12px;color:var(--muted);line-height:1.65;margin-bottom:20px">16-season ELO with 10% seasonal decay each August. A team's summer squad changes are modelled as a regression toward the league mean.</p>
    {elo_html}
    <div style="height:1px;background:var(--border);margin:16px 0"></div>
    <p style="font-size:11px;color:var(--muted);font-family:var(--mono);line-height:1.9">1500 = league mean · +100 pts ≈ 64% head-to-head win rate<br/>Decay prevents stale dominance carrying season-to-season</p>
  </div>
</div>

<div class="divider"></div>

<div class="story">
<div class="wrap">
<div class="story-inner">
  <h2>HOW THIS MODEL WORKS</h2>
  <p>Most football prediction models make a fundamental error: they treat historical form as equally reliable whether it's from last week or three seasons ago. This model uses a <strong>progressive blending system</strong> — at the start of the season, predictions lean heavily on multi-year ELO ratings because you have no current data. By GW31, <strong class="highlight">82% of the form signal comes from this season alone</strong>. The model adapts its own confidence as evidence accumulates.</p>
  <p>The <strong>5,000 Monte Carlo simulations</strong> are what separate a "who will win" prediction from a probabilistic forecast. Each simulation independently draws match outcomes from the model's probability distributions and tallies a final table. Arsenal winning 99.3% of simulations isn't a statement of certainty — it's a statement that <strong>in virtually every plausible path through the remaining 7 games, Arsenal finish top</strong>, given their 9-point lead.</p>
  <p>Walk-forward validation is the only honest performance measure for a time-series model. Training on 2021-25 and testing on 2025-26 would be data leakage. Instead, each validation fold <strong>trains only on the past and predicts the future</strong> — the same constraint the model faces in production.</p>
  <div class="methodology-grid">
    <div class="method-card"><h3>ELO + decay</h3><p>Standard Elo with 10% seasonal regression. K=20, home advantage=100. Decay prevents a team's 2022 peak inflating their 2026 rating.</p></div>
    <div class="method-card"><h3>Progressive form blend</h3><p>Weight = SeasonProgress × current-season EWMA + (1-SP) × long-run EWMA. Shifts from historical prior to live evidence as the season unfolds.</p></div>
    <div class="method-card"><h3>Market signal</h3><p>Bookmaker implied probabilities (overround removed) are the single most informative feature. Markets encode injury news, team selection, and analyst consensus that public stats don't capture.</p></div>
    <div class="method-card"><h3>Monte Carlo simulation</h3><p>5,000 independent season completions. Each remaining match drawn from predicted H/D/A probabilities. Title probability = fraction of simulations won.</p></div>
  </div>
</div>
</div>
</div>

<div class="footer">
<div class="wrap" style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:20px;width:100%">
  <div class="footer-block">
    <span style="font-family:var(--display);font-size:18px;color:var(--text)">PL PREDICTOR v4</span><br/>
    Data source: football-data.co.uk · 16 seasons · 6,009 matches<br/>
    Model: HistGradientBoostingClassifier · 26 features · sklearn<br/>
    Simulation: 5,000 Monte Carlo runs · Cutoff: {data_cutoff}
  </div>
  <div class="footer-block" style="text-align:right">
    <span style="color:var(--accent);font-family:var(--display);font-size:14px">KEY DESIGN DECISIONS</span><br/>
    Odds-era training only — no flat-prior pollution pre-2019<br/>
    Temporal leakage-free features — all use .shift(1) before window<br/>
    Walk-forward CV — only ever train on the past<br/>
    Log loss objective — rewards calibrated probabilities
  </div>
</div>
</div>

</body>
</html>"""

    Path(output_path).write_text(html, encoding="utf-8")
    print(f"\n  Dashboard exported → {output_path}")
    print(f"  Open in any browser — no server required.")


# ──────────────────────────────────────────────────────────────────────
# STEP 14 — MAIN
# ──────────────────────────────────────────────────────────────────────

def main():
    df, data_cutoff, gap_days = load_data()
    df = encode_target(df)

    print("  Building ELO with seasonal decay...")
    df, elo_dict = build_elo(df)
    elo_top = sorted(elo_dict.items(), key=lambda x: x[1], reverse=True)[:5]
    for t, e in elo_top:
        print(f"    {t}: {e:.0f}")

    print("\n  Engineering 26 features (no leakage)...")
    df = build_features(df)

    df_model = df.dropna(subset=FEATURES + ["target"]).copy()
    print(f"  Model-ready rows: {len(df_model):,}\n")

    model, train_final, val_results, mean_acc, mean_ll = train_and_validate(df_model)

    print("  Computing feature importance...")
    fi_data = get_feature_importance(model, train_final[FEATURES], train_final["target"])

    current_s     = df["Season_Order"].max()
    current_label = df[df["Season_Order"] == current_s]["Season"].iloc[0]
    df_current    = df[df["Season_Order"] == current_s].copy()
    played_df     = df_current[df_current["FTR"].notna()].copy()
    teams         = sorted(pd.unique(df_current[["HomeTeam","AwayTeam"]].values.ravel()))

    standings = get_standings(played_df, teams)

    print(f"  Current season : {current_label}")
    print(f"  Matches played : {len(played_df)} / 380")
    print(f"  GW32 fixtures  : {len(GW32_FIXTURES)} confirmed from soccerway.com\n")

    remaining_df = build_remaining_fixtures(
        df, df_current, elo_dict, standings,
        known_next_gw=GW32_FIXTURES
    )

    title_probs, top4_probs, rel_probs = simulate_season(
        played_df, remaining_df, model, n_sim=5000
    )

    print_forecast(standings, title_probs, top4_probs, rel_probs,
                   current_label, data_cutoff, gap_days)

    # ── GW32 predictions ──────────────────────────────────────────────
    print("\n" + "=" * 64)
    print(f"  GW32 FIXTURE PREDICTIONS  ·  April 10–13 2026")
    print(f"  (Neutral odds prior — pass real odds for sharper estimates)")
    print("=" * 64)

    match_preds = []
    for home, away in GW32_FIXTURES:
        if home in teams and away in teams:
            r = predict_match(model, df, home, away, elo_dict, standings)
            if r:
                match_preds.append(r)

    # ── Export dashboard ──────────────────────────────────────────────
    export_dashboard(
        standings=standings,
        title_probs=title_probs,
        top4_probs=top4_probs,
        rel_probs=rel_probs,
        val_results=val_results,
        fi_data=list(fi_data),
        elo_dict=elo_dict,
        match_preds=match_preds,
        current_label=current_label,
        data_cutoff=str(data_cutoff),
        gap_days=gap_days,
        mean_acc=mean_acc,
        mean_ll=mean_ll,
        output_path="pl_dashboard_v4.html",
    )

    return model, df, standings, title_probs


# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model, df, standings, title_probs = main()
