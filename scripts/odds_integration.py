"""
Vegas Odds Benchmarking Pipeline
=================================
Benchmarks our LightGBM win probability model (Layer 4) against Vegas closing
lines to determine if the model has betting value.

5 Steps:
  1. Ingest odds from JSON (2021-2025) and Excel (2010-2021) into SQLite
  2. Benchmark model vs. Vegas on 2025 test set
  3. 80/20 ensemble (Vegas + model)
  4. Value betting backtest with flat and Kelly staking
  5. Calibration / reliability check

Usage:
  python scripts/odds_integration.py            # Run all steps
  python scripts/odds_integration.py --step 1    # Run specific step
"""

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ODDS_DIR = DATA_DIR / "odds"
DB_PATH = DATA_DIR / "mlb_analytics.db"
FEATURES_PATH = DATA_DIR / "features" / "game_features.csv"
MODEL_PATH = ROOT / "models" / "win_probability_lgbm.txt"
META_PATH = ROOT / "models" / "win_probability_meta.json"

# ---------------------------------------------------------------------------
# Team-name mappings
# ---------------------------------------------------------------------------

# JSON fullName → DB name (only entries that differ)
JSON_TO_DB = {
    "Oakland Athletics": "Athletics",
    "Athletics Athletics": "Athletics",  # odd variant in dataset
}

# SBRO 3-letter codes → DB team name
# Cleveland was "Indians" before 2022 rebrand
SBRO_TEAM_MAP = {
    "ARI": "Arizona Diamondbacks",
    "ATL": "Atlanta Braves",
    "BAL": "Baltimore Orioles",
    "BOS": "Boston Red Sox",
    "CIN": "Cincinnati Reds",
    "CLE": "Cleveland Guardians",   # overridden for pre-2022 below
    "COL": "Colorado Rockies",
    "CUB": "Chicago Cubs",
    "CWS": "Chicago White Sox",
    "DET": "Detroit Tigers",
    "HOU": "Houston Astros",
    "KAN": "Kansas City Royals",
    "LAA": "Los Angeles Angels",
    "LAD": "Los Angeles Dodgers",
    "MIA": "Miami Marlins",
    "MIL": "Milwaukee Brewers",
    "MIN": "Minnesota Twins",
    "NYM": "New York Mets",
    "NYY": "New York Yankees",
    "OAK": "Athletics",
    "PHI": "Philadelphia Phillies",
    "PIT": "Pittsburgh Pirates",
    "SDG": "San Diego Padres",
    "SEA": "Seattle Mariners",
    "SFO": "San Francisco Giants",
    "STL": "St. Louis Cardinals",
    "TAM": "Tampa Bay Rays",
    "TEX": "Texas Rangers",
    "TOR": "Toronto Blue Jays",
    "WAS": "Washington Nationals",
}

# Teams that changed names — for pre-2022 Excel files
HISTORICAL_NAMES = {
    ("CLE", 2021): "Cleveland Guardians",  # Guardians rebranded mid-2021 offseason
    # For 2010-2021 files CLE played as Indians, but our DB only has "Cleveland Guardians"
    # We keep the mapping to Guardians so it joins correctly to the DB
}


def map_json_team(full_name: str) -> str:
    """Map a JSON fullName to the canonical DB team name."""
    return JSON_TO_DB.get(full_name, full_name)


def map_sbro_team(code: str, year: int) -> str | None:
    """Map a 3-letter SBRO code + year to DB team name."""
    return SBRO_TEAM_MAP.get(code)


# ---------------------------------------------------------------------------
# Helper: American odds → implied probability
# ---------------------------------------------------------------------------

def american_to_prob(ml: float) -> float:
    """Convert American moneyline to raw implied probability."""
    if ml < 0:
        return abs(ml) / (abs(ml) + 100)
    elif ml > 0:
        return 100 / (ml + 100)
    else:
        return 0.5  # pick'em edge case


def remove_vig(home_raw: float, away_raw: float) -> tuple[float, float]:
    """Remove vig to get fair probabilities that sum to 1."""
    total = home_raw + away_raw
    if total == 0:
        return 0.5, 0.5
    return home_raw / total, away_raw / total


# ---------------------------------------------------------------------------
# STEP 1 — Odds Ingestion
# ---------------------------------------------------------------------------

def parse_json_odds() -> pd.DataFrame:
    """Parse the JSON odds file (2021-2025) into a DataFrame.

    The JSON is keyed by date string ("YYYY-MM-DD"), each value is a list of
    game objects. We extract the FanDuel closing moneyline for each game.
    """
    json_path = ODDS_DIR / "mlb_odds_dataset.json"
    print(f"  Loading {json_path.name} …")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    skipped_no_ml = 0
    skipped_allstar = 0

    for date_str, games in data.items():
        for game in games:
            gv = game["gameView"]

            # Skip All-Star games and spring training
            game_type = gv.get("gameType", "R")
            if game_type != "R":
                continue

            home_full = gv["homeTeam"]["fullName"]
            away_full = gv["awayTeam"]["fullName"]

            # Skip All-Star entries
            if "All-Stars" in home_full or "All-Stars" in away_full:
                skipped_allstar += 1
                continue

            home_team = map_json_team(home_full)
            away_team = map_json_team(away_full)
            home_score = gv.get("homeTeamScore")
            away_score = gv.get("awayTeamScore")

            # Find moneyline: prefer FanDuel, fallback to first available
            moneylines = game["odds"].get("moneyline", [])
            home_ml, away_ml = None, None
            sportsbook_used = None

            # Try FanDuel first
            for ml_entry in moneylines:
                if ml_entry.get("sportsbook") == "fanduel":
                    cl = ml_entry.get("currentLine", {})
                    home_ml = cl.get("homeOdds")
                    away_ml = cl.get("awayOdds")
                    sportsbook_used = "fanduel"
                    break

            # Fallback: first sportsbook with valid data
            if home_ml is None and moneylines:
                for ml_entry in moneylines:
                    cl = ml_entry.get("currentLine", {})
                    if cl.get("homeOdds") is not None:
                        home_ml = cl["homeOdds"]
                        away_ml = cl.get("awayOdds")
                        sportsbook_used = ml_entry.get("sportsbook", "unknown")
                        break

            if home_ml is None:
                skipped_no_ml += 1
                continue

            rows.append({
                "date": date_str,
                "home_team": home_team,
                "away_team": away_team,
                "home_ml": home_ml,
                "away_ml": away_ml,
                "home_score": home_score,
                "away_score": away_score,
                "sportsbook": sportsbook_used,
                "source": "json",
            })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    print(f"  JSON: {len(df)} regular-season games parsed "
          f"(skipped {skipped_no_ml} no-ML, {skipped_allstar} All-Star)")
    return df


def parse_excel_odds() -> pd.DataFrame:
    """Parse SBRO Excel files (2010-2021) into a DataFrame.

    Each Excel file has paired rows: visitor (VH='V') then home (VH='H').
    Date column is MMDD integer; year comes from the filename.
    """
    excel_files = sorted(ODDS_DIR.glob("mlb-odds-*.xlsx"))
    if not excel_files:
        print("  No Excel odds files found — skipping.")
        return pd.DataFrame()

    all_rows = []

    for fpath in excel_files:
        # Extract year from filename like "mlb-odds-2021.xlsx"
        year = int(fpath.stem.split("-")[-1])
        print(f"  Parsing {fpath.name} (year={year}) …")

        df = pd.read_excel(fpath, header=0)

        # Pair visitor + home rows
        i = 0
        while i < len(df) - 1:
            row_v = df.iloc[i]
            row_h = df.iloc[i + 1]

            # Validate V/H pairing
            if str(row_v.get("VH", "")).strip() != "V" or str(row_h.get("VH", "")).strip() != "H":
                i += 1
                continue

            # Parse date: MMDD integer → datetime.date
            mmdd = row_v["Date"]
            try:
                mmdd = int(mmdd)
            except (ValueError, TypeError):
                i += 2
                continue

            month = mmdd // 100
            day = mmdd % 100
            if month < 1 or month > 12 or day < 1 or day > 31:
                i += 2
                continue

            try:
                from datetime import date
                game_date = date(year, month, day)
            except ValueError:
                i += 2
                continue

            # Team mapping
            away_code = str(row_v["Team"]).strip()
            home_code = str(row_h["Team"]).strip()
            away_team = map_sbro_team(away_code, year)
            home_team = map_sbro_team(home_code, year)

            if away_team is None or home_team is None:
                i += 2
                continue

            # Moneyline close
            try:
                away_ml = int(row_v["Close"])
                home_ml = int(row_h["Close"])
            except (ValueError, TypeError):
                i += 2
                continue

            # Final scores
            away_score = row_v.get("Final")
            home_score = row_h.get("Final")
            try:
                away_score = int(away_score)
                home_score = int(home_score)
            except (ValueError, TypeError):
                away_score = None
                home_score = None

            all_rows.append({
                "date": game_date,
                "home_team": home_team,
                "away_team": away_team,
                "home_ml": home_ml,
                "away_ml": away_ml,
                "home_score": home_score,
                "away_score": away_score,
                "sportsbook": "SBRO",
                "source": "excel",
            })

            i += 2

    df = pd.DataFrame(all_rows)
    print(f"  Excel: {len(df)} games parsed from {len(excel_files)} files")
    return df


def add_implied_probs(df: pd.DataFrame) -> pd.DataFrame:
    """Add fair implied probabilities (vig removed) to the odds DataFrame."""
    df["home_raw_prob"] = df["home_ml"].apply(american_to_prob)
    df["away_raw_prob"] = df["away_ml"].apply(american_to_prob)

    # Remove vig → fair probabilities summing to 1.0
    fair = df.apply(
        lambda r: remove_vig(r["home_raw_prob"], r["away_raw_prob"]), axis=1
    )
    df["vegas_home_prob"] = fair.apply(lambda x: x[0])
    df["vegas_away_prob"] = fair.apply(lambda x: x[1])

    return df


def match_to_db(odds_df: pd.DataFrame) -> pd.DataFrame:
    """Match odds rows to games in our SQLite DB by date + home team name.

    Returns the odds DataFrame with a `game_id` column (NaN if unmatched).
    """
    import sqlite3

    conn = sqlite3.connect(str(DB_PATH))
    # Load games joined with team names
    games_df = pd.read_sql_query(
        """
        SELECT g.id AS game_id, g.date, g.season,
               ht.name AS db_home_team, at.name AS db_away_team
        FROM games g
        JOIN teams ht ON g.home_team_id = ht.id
        JOIN teams at ON g.away_team_id = at.id
        """,
        conn,
    )
    conn.close()

    games_df["date"] = pd.to_datetime(games_df["date"]).dt.date
    odds_df["date"] = pd.to_datetime(odds_df["date"]).dt.date

    # Merge on date + home team name
    merged = odds_df.merge(
        games_df,
        left_on=["date", "home_team"],
        right_on=["date", "db_home_team"],
        how="left",
    )

    matched = merged["game_id"].notna().sum()
    total = len(merged)
    print(f"  Matched {matched}/{total} odds rows to DB games "
          f"({total - matched} unmatched)")

    return merged


def step1_ingest():
    """STEP 1: Parse all odds sources, compute fair probs, match to DB, save."""
    print("\n" + "=" * 60)
    print("STEP 1 — Odds Ingestion")
    print("=" * 60)

    # 1a. Parse JSON (2021-2025)
    json_df = parse_json_odds()

    # 1b. Parse Excel (2010-2021)
    excel_df = parse_excel_odds()

    # Combine
    odds_df = pd.concat([json_df, excel_df], ignore_index=True)
    print(f"\n  Combined: {len(odds_df)} total odds rows")

    # 1c. Implied probabilities
    odds_df = add_implied_probs(odds_df)

    # 1d. Match to DB
    odds_df = match_to_db(odds_df)

    # 1e. Store in SQLite
    import sqlite3

    # Keep only the columns we need
    save_cols = [
        "game_id", "date", "season", "home_team", "away_team",
        "home_ml", "away_ml", "vegas_home_prob", "vegas_away_prob",
        "sportsbook", "source",
    ]
    # Only save rows that have a game_id match
    save_df = odds_df.dropna(subset=["game_id"])[save_cols].copy()
    save_df["game_id"] = save_df["game_id"].astype(int)

    conn = sqlite3.connect(str(DB_PATH))
    save_df.to_sql("odds", conn, if_exists="replace", index=False)
    conn.close()

    print(f"  Saved {len(save_df)} matched odds rows to 'odds' table")

    # Season breakdown
    print("\n  Season breakdown:")
    for season, group in save_df.groupby("season"):
        print(f"    {int(season)}: {len(group)} games")

    return save_df


# ---------------------------------------------------------------------------
# STEP 2 — Vegas Benchmark (2025 test set)
# ---------------------------------------------------------------------------

def step2_benchmark():
    """STEP 2: Compare model vs. Vegas log loss on 2025 matched games."""
    print("\n" + "=" * 60)
    print("STEP 2 — Vegas Benchmark (2025)")
    print("=" * 60)

    # Load model
    model = lgb.Booster(model_file=str(MODEL_PATH))
    with open(META_PATH) as f:
        meta = json.load(f)
    features = meta["features"]
    print(f"  Model loaded: {len(features)} features, "
          f"reported log loss = {meta['log_loss']:.4f}")

    # Load features and filter to 2025
    feat_df = pd.read_csv(FEATURES_PATH)
    feat_2025 = feat_df[feat_df["season"] == 2025].copy()
    print(f"  2025 feature rows: {len(feat_2025)}")

    # Model predictions
    X = feat_2025[features].values
    model_probs = model.predict(X)  # LightGBM Booster.predict gives probabilities directly
    feat_2025 = feat_2025.copy()
    feat_2025["model_home_prob"] = model_probs
    feat_2025["home_win"] = (feat_2025["home_score"] > feat_2025["away_score"]).astype(int)

    # Load odds table
    import sqlite3
    conn = sqlite3.connect(str(DB_PATH))
    odds_df = pd.read_sql_query(
        "SELECT * FROM odds WHERE season = 2025", conn
    )
    conn.close()

    # Join model predictions with odds on game_id
    merged = feat_2025.merge(odds_df, on="game_id", how="inner")
    print(f"  Matched model+odds rows for 2025: {len(merged)}")

    if len(merged) == 0:
        print("  ERROR: No matched games — cannot benchmark.")
        return None

    y_true = merged["home_win"]
    vegas_probs = merged["vegas_home_prob"]
    model_probs_matched = merged["model_home_prob"]

    # Clip probabilities to avoid log(0)
    eps = 1e-6
    vegas_clipped = vegas_probs.clip(eps, 1 - eps)
    model_clipped = model_probs_matched.clip(eps, 1 - eps)

    # Metrics
    vegas_ll = log_loss(y_true, vegas_clipped)
    model_ll = log_loss(y_true, model_clipped)
    gap_bp = (model_ll - vegas_ll) * 10000  # basis points

    vegas_acc = accuracy_score(y_true, (vegas_probs > 0.5).astype(int))
    model_acc = accuracy_score(y_true, (model_probs_matched > 0.5).astype(int))

    vegas_brier = brier_score_loss(y_true, vegas_clipped)
    model_brier = brier_score_loss(y_true, model_clipped)

    print(f"\n  {'Metric':<20} {'Model':>10} {'Vegas':>10} {'Gap':>12}")
    print(f"  {'-'*52}")
    print(f"  {'Log Loss':<20} {model_ll:>10.4f} {vegas_ll:>10.4f} "
          f"{gap_bp:>+10.0f} bp")
    print(f"  {'Accuracy':<20} {model_acc:>10.4f} {vegas_acc:>10.4f} "
          f"{(model_acc - vegas_acc)*100:>+10.1f} pp")
    print(f"  {'Brier Score':<20} {model_brier:>10.4f} {vegas_brier:>10.4f} "
          f"{(model_brier - vegas_brier)*10000:>+10.0f} bp")
    print(f"\n  Games evaluated: {len(merged)}")
    print(f"  Home win rate: {y_true.mean():.3f}")

    return merged


# ---------------------------------------------------------------------------
# STEP 3 — 80/20 Ensemble
# ---------------------------------------------------------------------------

def step3_ensemble(merged: pd.DataFrame):
    """STEP 3: Blend model and Vegas probabilities (80% Vegas, 20% model)."""
    print("\n" + "=" * 60)
    print("STEP 3 — 80/20 Ensemble (Vegas 80% / Model 20%)")
    print("=" * 60)

    y_true = merged["home_win"]
    vegas = merged["vegas_home_prob"]
    model = merged["model_home_prob"]
    ensemble = 0.80 * vegas + 0.20 * model

    eps = 1e-6

    results = {}
    for name, probs in [("Model", model), ("Vegas", vegas), ("Ensemble", ensemble)]:
        clipped = probs.clip(eps, 1 - eps)
        results[name] = {
            "log_loss": log_loss(y_true, clipped),
            "accuracy": accuracy_score(y_true, (probs > 0.5).astype(int)),
            "brier": brier_score_loss(y_true, clipped),
        }

    print(f"\n  {'System':<12} {'Log Loss':>10} {'Accuracy':>10} {'Brier':>10}")
    print(f"  {'-'*42}")
    for name, m in results.items():
        print(f"  {name:<12} {m['log_loss']:>10.4f} {m['accuracy']:>10.4f} "
              f"{m['brier']:>10.4f}")

    # Find best
    best = min(results, key=lambda k: results[k]["log_loss"])
    print(f"\n  Best log loss: {best} ({results[best]['log_loss']:.4f})")

    return ensemble


# ---------------------------------------------------------------------------
# STEP 4 — Value Betting Backtest
# ---------------------------------------------------------------------------

def ml_payout(stake: float, ml: float, won: bool) -> float:
    """Calculate profit from a moneyline bet.

    Args:
        stake: amount wagered
        ml: American moneyline odds
        won: whether the bet won

    Returns:
        Profit (positive) or loss (negative, = -stake).
    """
    if not won:
        return -stake
    if ml < 0:
        return stake * (100 / abs(ml))
    else:
        return stake * (ml / 100)


def step4_value_betting(merged: pd.DataFrame):
    """STEP 4: Backtest value betting strategies on 2025 data."""
    print("\n" + "=" * 60)
    print("STEP 4 — Value Betting Backtest (2025)")
    print("=" * 60)

    thresholds = [0.03, 0.05, 0.07, 0.10]
    stake = 100.0

    y_true = merged["home_win"].values
    model_prob = merged["model_home_prob"].values
    vegas_prob = merged["vegas_home_prob"].values
    home_ml = merged["home_ml"].values
    away_ml = merged["away_ml"].values

    # --- Flat betting ---
    print(f"\n  FLAT BETTING ($100/bet)")
    print(f"  {'Threshold':>10} {'Bets':>6} {'Wins':>6} {'Win%':>7} "
          f"{'Profit':>10} {'ROI':>8}")
    print(f"  {'-'*50}")

    for thresh in thresholds:
        total_profit = 0.0
        bets = 0
        wins = 0

        for i in range(len(merged)):
            # Check home side edge
            home_edge = model_prob[i] - vegas_prob[i]
            # Check away side edge
            away_edge = (1 - model_prob[i]) - (1 - vegas_prob[i])

            if home_edge >= thresh:
                bets += 1
                won = bool(y_true[i])
                profit = ml_payout(stake, home_ml[i], won)
                total_profit += profit
                if won:
                    wins += 1
            elif away_edge >= thresh:
                bets += 1
                won = not bool(y_true[i])
                profit = ml_payout(stake, away_ml[i], won)
                total_profit += profit
                if won:
                    wins += 1

        if bets > 0:
            roi = total_profit / (bets * stake) * 100
            win_pct = wins / bets * 100
            print(f"  {thresh:>9.0%} {bets:>6} {wins:>6} {win_pct:>6.1f}% "
                  f"  ${total_profit:>+8.0f} {roi:>+7.1f}%")
        else:
            print(f"  {thresh:>9.0%} {'no bets':>6}")

    # --- Kelly criterion ---
    print(f"\n  KELLY CRITERION (quarter-Kelly, unit bankroll = $10,000)")
    print(f"  {'Threshold':>10} {'Bets':>6} {'End $':>10} {'Max DD':>10} "
          f"{'ROI':>8}")
    print(f"  {'-'*50}")

    for thresh in thresholds:
        bankroll = 10_000.0
        peak = bankroll
        max_drawdown = 0.0
        bets = 0

        for i in range(len(merged)):
            home_edge = model_prob[i] - vegas_prob[i]
            away_edge = (1 - model_prob[i]) - (1 - vegas_prob[i])

            # Determine which side to bet
            if home_edge >= thresh:
                edge = home_edge
                ml = home_ml[i]
                won = bool(y_true[i])
            elif away_edge >= thresh:
                edge = away_edge
                ml = away_ml[i]
                won = not bool(y_true[i])
            else:
                continue

            # Decimal odds for Kelly calculation
            if ml < 0:
                decimal_odds = 1 + 100 / abs(ml)
            else:
                decimal_odds = 1 + ml / 100

            # Quarter-Kelly: (edge / (decimal_odds - 1)) * 0.25
            denom = decimal_odds - 1
            if denom <= 0:
                continue
            kelly_f = (edge / denom) * 0.25  # quarter-Kelly
            kelly_f = max(kelly_f, 0.0)

            bet_size = bankroll * kelly_f
            profit = ml_payout(bet_size, ml, won)
            bankroll += profit
            bets += 1

            peak = max(peak, bankroll)
            drawdown = (peak - bankroll) / peak
            max_drawdown = max(max_drawdown, drawdown)

        roi = (bankroll - 10_000) / 10_000 * 100
        print(f"  {thresh:>9.0%} {bets:>6} ${bankroll:>9,.0f} "
              f"{max_drawdown:>9.1%} {roi:>+7.1f}%")


# ---------------------------------------------------------------------------
# STEP 5 — Calibration Check
# ---------------------------------------------------------------------------

def step5_calibration(merged: pd.DataFrame, ensemble_probs: pd.Series):
    """STEP 5: Check calibration of model, Vegas, and ensemble predictions."""
    print("\n" + "=" * 60)
    print("STEP 5 — Calibration Check")
    print("=" * 60)

    y_true = merged["home_win"].values
    systems = {
        "Model": merged["model_home_prob"].values,
        "Vegas": merged["vegas_home_prob"].values,
        "Ensemble": ensemble_probs.values,
    }

    # Bucket predictions into 5% intervals
    bin_edges = np.arange(0, 1.05, 0.05)
    bin_labels = [f"{int(lo*100):>2}-{int(hi*100):<2}%"
                  for lo, hi in zip(bin_edges[:-1], bin_edges[1:])]

    for sys_name, probs in systems.items():
        print(f"\n  {sys_name} Calibration:")
        print(f"  {'Bucket':<10} {'Count':>6} {'Predicted':>10} {'Actual':>10} "
              f"{'Gap':>8}")
        print(f"  {'-'*46}")

        indices = np.digitize(probs, bin_edges) - 1
        indices = np.clip(indices, 0, len(bin_labels) - 1)

        for b in range(len(bin_labels)):
            mask = indices == b
            count = mask.sum()
            if count == 0:
                continue
            pred_mean = probs[mask].mean()
            actual_mean = y_true[mask].mean()
            gap = actual_mean - pred_mean
            print(f"  {bin_labels[b]:<10} {count:>6} {pred_mean:>10.3f} "
                  f"{actual_mean:>10.3f} {gap:>+7.3f}")

    # Overall Brier scores
    print(f"\n  Overall Brier Scores:")
    for sys_name, probs in systems.items():
        brier = brier_score_loss(y_true, np.clip(probs, 1e-6, 1 - 1e-6))
        print(f"    {sys_name:<12}: {brier:.4f}")

    # Confidence assessment
    print("\n  Confidence Assessment:")
    for sys_name, probs in systems.items():
        # Mean absolute calibration error
        indices = np.digitize(probs, bin_edges) - 1
        indices = np.clip(indices, 0, len(bin_labels) - 1)
        abs_errors = []
        for b in range(len(bin_labels)):
            mask = indices == b
            if mask.sum() < 10:
                continue
            pred_mean = probs[mask].mean()
            actual_mean = y_true[mask].mean()
            abs_errors.append(abs(actual_mean - pred_mean))
        if abs_errors:
            mace = np.mean(abs_errors)
            bias = np.mean(probs) - np.mean(y_true)
            direction = "overconfident" if bias > 0.005 else \
                        "underconfident" if bias < -0.005 else "well-calibrated"
            print(f"    {sys_name:<12}: MACE={mace:.4f}, bias={bias:+.4f} "
                  f"({direction})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Vegas Odds Benchmarking Pipeline")
    parser.add_argument("--step", type=int, choices=[1, 2, 3, 4, 5],
                        help="Run a specific step (default: all)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Vegas Odds Benchmarking Pipeline")
    print("=" * 60)

    if args.step is None or args.step == 1:
        step1_ingest()

    if args.step is None or args.step >= 2:
        merged = step2_benchmark()
        if merged is None:
            print("\n  Cannot proceed without matched data. Exiting.")
            sys.exit(1)

    if args.step is None or args.step >= 3:
        ensemble_probs = step3_ensemble(merged)

    if args.step is None or args.step >= 4:
        step4_value_betting(merged)

    if args.step is None or args.step >= 5:
        step5_calibration(merged, ensemble_probs)

    print("\n" + "=" * 60)
    print("  Pipeline complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
