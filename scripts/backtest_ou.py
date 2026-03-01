"""
backtest_ou.py -- Over/Under Backtest for MLB Run Total Model
=============================================================
Evaluates the run total model against Vegas O/U lines using historical
2025 data. For each game with a Vegas line, computes:
  - Model predicted total (home + away LightGBM regressors)
  - Negative binomial P(over) and P(under)
  - Edge vs. Vegas line
  - Simulated flat-bet P&L at various edge thresholds

Usage:
  python scripts/backtest_ou.py                    # Default 2025 test
  python scripts/backtest_ou.py --test-season 2025
  python scripts/backtest_ou.py --edge-thresholds 0.03 0.05 0.10

Author: Loko
"""

import argparse
import json
import sys
import warnings
from pathlib import Path
from statistics import median

if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("LightGBM not installed. Install with: pip install lightgbm")
    sys.exit(1)

# Add project root to path so we can import src modules
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.run_distribution import RunDistribution

# ===================================================================
# PATHS
# ===================================================================

FEATURES_PATH = PROJECT_ROOT / "data" / "features" / "game_features.csv"
ODDS_PATH = PROJECT_ROOT / "data" / "odds" / "mlb_odds_dataset.json"
MODELS_DIR = PROJECT_ROOT / "models"
HOME_MODEL_PATH = MODELS_DIR / "run_home_lgbm.txt"
AWAY_MODEL_PATH = MODELS_DIR / "run_away_lgbm.txt"
META_PATH = MODELS_DIR / "run_model_meta.json"
MEDIANS_PATH = MODELS_DIR / "run_feature_medians.json"

# ===================================================================
# TEAM NAME MAPPING
# ===================================================================
#
# The odds JSON uses shortName (e.g. "NYY"), while the features CSV
# uses numeric team IDs from the database. We map JSON shortName
# to DB team_id so we can join the two datasets.
#
# DB team table:
#   1=Athletics, 2=Pirates, 3=Padres, 4=Mariners, 5=Giants,
#   6=Cardinals, 7=Rays, 8=Rangers, 9=Blue Jays, 10=Twins,
#   11=Phillies, 12=Braves, 13=White Sox, 14=Marlins, 15=Yankees,
#   16=Brewers, 17=Angels, 18=Diamondbacks, 19=Orioles, 20=Red Sox,
#   21=Cubs, 22=Reds, 23=Guardians, 24=Rockies, 25=Tigers,
#   26=Astros, 27=Royals, 28=Dodgers, 29=Nationals, 30=Mets

JSON_SHORT_TO_DB_ID = {
    "OAK": 1,   "ATH": 1,   # Athletics (Oakland -> Sacramento)
    "PIT": 2,
    "SD":  3,
    "SEA": 4,
    "SF":  5,
    "STL": 6,
    "TB":  7,
    "TEX": 8,
    "TOR": 9,
    "MIN": 10,
    "PHI": 11,
    "ATL": 12,
    "CHW": 13,
    "MIA": 14,
    "NYY": 15,
    "MIL": 16,
    "LAA": 17,
    "ARI": 18,  "AZ": 18,   # Arizona uses both codes
    "BAL": 19,
    "BOS": 20,
    "CHC": 21,
    "CIN": 22,
    "CLE": 23,
    "COL": 24,
    "DET": 25,
    "HOU": 26,
    "KC":  27,
    "LAD": 28,
    "WAS": 29,  "WSH": 29,  # Washington uses both codes
    "NYM": 30,
}

# Reverse mapping for display
DB_ID_TO_ABBR = {
    1: "ATH", 2: "PIT", 3: "SD", 4: "SEA", 5: "SF", 6: "STL",
    7: "TB", 8: "TEX", 9: "TOR", 10: "MIN", 11: "PHI", 12: "ATL",
    13: "CWS", 14: "MIA", 15: "NYY", 16: "MIL", 17: "LAA", 18: "ARI",
    19: "BAL", 20: "BOS", 21: "CHC", 22: "CIN", 23: "CLE", 24: "COL",
    25: "DET", 26: "HOU", 27: "KC", 28: "LAD", 29: "WSH", 30: "NYM",
}

# ===================================================================
# HELPER FUNCTIONS
# ===================================================================

def sanitize_american_odds(odds: float) -> float:
    """
    Ensure American odds are valid. The median of mixed positive/negative
    American odds can produce nonsensical values near zero (e.g., -0.5, +2)
    which break decimal conversions. Clamp to standard -110 if invalid.

    Valid American odds: <= -100 or >= +100.
    """
    if -100 < odds < 100:
        return -110.0  # Standard vig line as fallback
    return odds


def american_to_decimal(american: float) -> float:
    """
    Convert American odds to decimal odds.
    -150 -> 1.667 (bet $150 to win $100, total payout $250, decimal = 250/150)
    +130 -> 2.30  (bet $100 to win $130, total payout $230, decimal = 230/100)
    """
    american = sanitize_american_odds(american)
    if american < 0:
        return 1 + 100 / abs(american)
    else:
        return 1 + american / 100


def implied_probability(american: float) -> float:
    """
    Convert American odds to implied probability (no-vig).
    -150 -> 60.0%
    +130 -> 43.5%
    """
    american = sanitize_american_odds(american)
    if american < 0:
        return abs(american) / (abs(american) + 100)
    else:
        return 100 / (american + 100)


def get_consensus_line(totals: list) -> dict:
    """
    Extract consensus (median) total line from sportsbook totals array.

    Returns dict with:
      - opening_total: median opening line
      - closing_total: median closing line
      - closing_over_odds: median over odds (American)
      - closing_under_odds: median under odds (American)

    Uses only major books: fanduel, draftkings, caesars, bet365.
    """
    target_books = {"fanduel", "draftkings", "caesars", "bet365"}

    opening_totals = []
    closing_totals = []
    over_odds = []
    under_odds = []

    for entry in totals:
        book = entry.get("sportsbook", "").lower()
        if book not in target_books:
            continue

        # Opening line
        opening = entry.get("openingLine", {})
        if opening.get("total") is not None:
            opening_totals.append(opening["total"])

        # Closing line
        closing = entry.get("currentLine", {})
        if closing.get("total") is not None:
            closing_totals.append(closing["total"])
            if closing.get("overOdds") is not None:
                over_odds.append(closing["overOdds"])
            if closing.get("underOdds") is not None:
                under_odds.append(closing["underOdds"])

    if not closing_totals:
        return None

    result = {
        "opening_total": median(opening_totals) if opening_totals else None,
        "closing_total": median(closing_totals),
        "closing_over_odds": median(over_odds) if over_odds else -110,
        "closing_under_odds": median(under_odds) if under_odds else -110,
    }

    return result


def load_odds(season: int) -> pd.DataFrame:
    """
    Parse the odds JSON and return a DataFrame with one row per game
    for the specified season, including consensus O/U lines.

    Columns: date, home_team_id, away_team_id, opening_total, closing_total,
             closing_over_odds, closing_under_odds, actual_home_score, actual_away_score
    """
    print(f"Loading odds from {ODDS_PATH}...")

    with open(ODDS_PATH, "r") as f:
        raw = json.load(f)

    rows = []
    skipped_no_totals = 0
    skipped_no_teams = 0
    skipped_no_consensus = 0

    for date_str, games in raw.items():
        # Filter to target season
        if not date_str.startswith(str(season)):
            continue

        for game in games:
            gv = game.get("gameView", {})

            # Skip non-regular-season games
            if gv.get("gameType") != "R":
                continue

            # Skip games that aren't final
            status = gv.get("gameStatusText", "")
            if "Final" not in status:
                continue

            # Map team shortNames to DB IDs
            home_short = gv.get("homeTeam", {}).get("shortName", "")
            away_short = gv.get("awayTeam", {}).get("shortName", "")

            home_id = JSON_SHORT_TO_DB_ID.get(home_short)
            away_id = JSON_SHORT_TO_DB_ID.get(away_short)

            if home_id is None or away_id is None:
                skipped_no_teams += 1
                continue

            # Extract totals
            odds = game.get("odds", {})
            totals = odds.get("totals", [])
            if not totals:
                skipped_no_totals += 1
                continue

            consensus = get_consensus_line(totals)
            if consensus is None:
                skipped_no_consensus += 1
                continue

            rows.append({
                "date": date_str,
                "home_team_id": home_id,
                "away_team_id": away_id,
                "opening_total": consensus["opening_total"],
                "closing_total": consensus["closing_total"],
                "closing_over_odds": consensus["closing_over_odds"],
                "closing_under_odds": consensus["closing_under_odds"],
                "odds_home_score": gv.get("homeTeamScore"),
                "odds_away_score": gv.get("awayTeamScore"),
            })

    df = pd.DataFrame(rows)

    print(f"  Parsed {len(df)} games with O/U lines for {season}")
    if skipped_no_totals:
        print(f"  Skipped {skipped_no_totals} games (no totals in odds)")
    if skipped_no_teams:
        print(f"  Skipped {skipped_no_teams} games (unmapped team shortName)")
    if skipped_no_consensus:
        print(f"  Skipped {skipped_no_consensus} games (no consensus line)")

    return df


def load_features_and_models(season: int):
    """
    Load the feature matrix, saved LightGBM models, metadata, and medians.

    Returns: features_df, home_model, away_model, meta, medians
    """
    print(f"\nLoading features from {FEATURES_PATH}...")
    df = pd.read_csv(FEATURES_PATH)
    df = df[df["season"] == season].copy()
    print(f"  {len(df)} games in features for {season}")

    print(f"Loading models from {MODELS_DIR}...")
    home_model = lgb.Booster(model_file=str(HOME_MODEL_PATH))
    away_model = lgb.Booster(model_file=str(AWAY_MODEL_PATH))

    with open(META_PATH) as f:
        meta = json.load(f)

    with open(MEDIANS_PATH) as f:
        medians = json.load(f)

    print(f"  Features: {len(meta['features'])}")
    print(f"  Phi home: {meta['phi_home']:.2f}, Phi away: {meta['phi_away']:.2f}")

    return df, home_model, away_model, meta, medians


def merge_features_with_odds(features_df: pd.DataFrame, odds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join the features matrix with the odds data on (date, home_team_id, away_team_id).

    The features CSV has home_team_id and away_team_id columns.
    The odds DF has the same. We join on date + team IDs.
    """
    # Normalize dates
    features_df["date"] = pd.to_datetime(features_df["date"]).dt.strftime("%Y-%m-%d")
    odds_df["date"] = pd.to_datetime(odds_df["date"]).dt.strftime("%Y-%m-%d")

    merged = features_df.merge(
        odds_df,
        on=["date", "home_team_id", "away_team_id"],
        how="inner"
    )

    print(f"\nMerged: {len(merged)} games (features + odds)")
    print(f"  Features only (no odds): {len(features_df) - len(merged)}")
    print(f"  Odds only (no features): {len(odds_df) - len(merged)}")

    return merged


def predict_and_evaluate(merged_df: pd.DataFrame, home_model, away_model,
                         meta: dict, medians: dict,
                         edge_thresholds: list) -> None:
    """
    For each game in the merged dataset:
      1. Predict home/away runs
      2. Compute NB P(over) and P(under)
      3. Compare to Vegas line
      4. Simulate betting at various edge thresholds
      5. Print comprehensive results
    """
    features = meta["features"]
    phi_home = meta["phi_home"]
    phi_away = meta["phi_away"]

    # Initialize the negative binomial distribution
    rd = RunDistribution(phi_home=phi_home, phi_away=phi_away)

    # Prepare feature matrix — fill NaN with training medians
    X = merged_df[features].copy()
    for col in features:
        if col in medians:
            X[col] = X[col].fillna(medians[col])
        else:
            X[col] = X[col].fillna(0)

    # Predict home and away runs
    # Booster.predict expects the raw feature matrix
    pred_home = home_model.predict(X)
    pred_away = away_model.predict(X)
    pred_total = pred_home + pred_away

    # Actual outcomes
    actual_home = merged_df["home_score"].values
    actual_away = merged_df["away_score"].values
    actual_total = merged_df["total_runs"].values
    vegas_line = merged_df["closing_total"].values
    opening_line = merged_df["opening_total"].values

    # ── 1. Raw MAE comparison ──
    model_mae = np.abs(pred_total - actual_total).mean()
    vegas_mae = np.abs(vegas_line - actual_total).mean()
    home_mae = np.abs(pred_home - actual_home).mean()
    away_mae = np.abs(pred_away - actual_away).mean()

    print("\n" + "=" * 65)
    print("  OVER/UNDER BACKTEST RESULTS")
    print("=" * 65)

    print(f"\n--- MAE Comparison ---")
    print(f"  Model total MAE:  {model_mae:.3f}")
    print(f"  Vegas total MAE:  {vegas_mae:.3f}")
    print(f"  Model home MAE:   {home_mae:.3f}")
    print(f"  Model away MAE:   {away_mae:.3f}")
    print(f"  Model bias:       {(pred_total - actual_total).mean():+.3f} runs")
    print(f"  Vegas bias:       {(vegas_line - actual_total).mean():+.3f} runs")

    # ── 2. Compute P(over) and P(under) for each game ──
    print(f"\nComputing NB probabilities for {len(merged_df)} games...")

    p_overs = []
    p_unders = []
    for i in range(len(merged_df)):
        p_over, p_under = rd.over_under_probability(
            mu_home=pred_home[i],
            mu_away=pred_away[i],
            line=vegas_line[i]
        )
        p_overs.append(p_over)
        p_unders.append(p_under)

    p_overs = np.array(p_overs)
    p_unders = np.array(p_unders)

    # Model's predicted side and edge
    model_side = np.where(p_overs > p_unders, "over", "under")
    model_confidence = np.maximum(p_overs, p_unders)

    # Actual outcome (did game go over or under the Vegas line?)
    actual_over = actual_total > vegas_line
    actual_under = actual_total < vegas_line
    actual_push = actual_total == vegas_line

    # For accuracy, assign push as a loss (conservative)
    model_correct = np.where(
        model_side == "over",
        actual_over,
        actual_under
    )

    # ── 3. Overall accuracy ──
    non_push = ~actual_push
    accuracy = model_correct[non_push].mean() if non_push.sum() > 0 else 0

    print(f"\n--- Model Accuracy ---")
    print(f"  Total games:        {len(merged_df)}")
    print(f"  Pushes:             {actual_push.sum()}")
    print(f"  Non-push games:     {non_push.sum()}")
    print(f"  Raw accuracy:       {accuracy:.1%}")

    # ── 4. Over vs Under breakdown ──
    over_picks = model_side == "over"
    under_picks = model_side == "under"

    over_correct = (model_correct & over_picks)[non_push & over_picks].mean() if (non_push & over_picks).sum() > 0 else 0
    under_correct = (model_correct & under_picks)[non_push & under_picks].mean() if (non_push & under_picks).sum() > 0 else 0

    print(f"\n--- Over vs Under Breakdown ---")
    print(f"  Over picks:  {over_picks.sum():>5}  accuracy: {over_correct:.1%}")
    print(f"  Under picks: {under_picks.sum():>5}  accuracy: {under_correct:.1%}")

    # ── 5. Betting simulation at edge thresholds ──
    print(f"\n--- Betting Simulation (flat $100 per bet) ---")
    print(f"  {'Threshold':>10}  {'Bets':>5}  {'W':>4}  {'L':>4}  {'P':>3}  {'Win%':>6}  {'Profit':>9}  {'ROI':>7}")
    print(f"  {'-'*10}  {'-'*5}  {'-'*4}  {'-'*4}  {'-'*3}  {'-'*6}  {'-'*9}  {'-'*7}")

    over_odds_arr = merged_df["closing_over_odds"].values
    under_odds_arr = merged_df["closing_under_odds"].values

    for threshold in edge_thresholds:
        # Calculate edge: model probability minus implied probability from odds
        # For each game, determine if we bet over or under
        edges = []
        bet_sides = []
        bet_results = []
        bet_odds = []

        for i in range(len(merged_df)):
            # Over edge
            implied_over = implied_probability(int(over_odds_arr[i]))
            over_edge = p_overs[i] - implied_over

            # Under edge
            implied_under = implied_probability(int(under_odds_arr[i]))
            under_edge = p_unders[i] - implied_under

            # Take the side with the bigger edge (if it exceeds threshold)
            if over_edge > under_edge and over_edge >= threshold:
                edges.append(over_edge)
                bet_sides.append("over")
                bet_odds.append(int(over_odds_arr[i]))
                if actual_push[i]:
                    bet_results.append("push")
                elif actual_over[i]:
                    bet_results.append("win")
                else:
                    bet_results.append("loss")
            elif under_edge >= threshold:
                edges.append(under_edge)
                bet_sides.append("under")
                bet_odds.append(int(under_odds_arr[i]))
                if actual_push[i]:
                    bet_results.append("push")
                elif actual_under[i]:
                    bet_results.append("win")
                else:
                    bet_results.append("loss")

        if not bet_results:
            print(f"  {threshold:>9.0%}  {'0':>5}  {'-':>4}  {'-':>4}  {'-':>3}  {'-':>6}  {'-':>9}  {'-':>7}")
            continue

        wins = sum(1 for r in bet_results if r == "win")
        losses = sum(1 for r in bet_results if r == "loss")
        pushes = sum(1 for r in bet_results if r == "push")
        total_bets = len(bet_results)

        # Calculate actual profit using odds
        profit = 0.0
        for j, result in enumerate(bet_results):
            if result == "win":
                dec = american_to_decimal(bet_odds[j])
                profit += 100 * (dec - 1)  # Win amount on a $100 bet
            elif result == "loss":
                profit -= 100
            # Push: $0

        win_pct = wins / (wins + losses) if (wins + losses) > 0 else 0
        roi = profit / (total_bets * 100) if total_bets > 0 else 0

        print(f"  {threshold:>9.0%}  {total_bets:>5}  {wins:>4}  {losses:>4}  {pushes:>3}  {win_pct:>5.1%}  ${profit:>+8.0f}  {roi:>+6.1%}")

    # ── 6. Edge distribution ──
    all_edges = np.maximum(
        p_overs - np.array([implied_probability(int(o)) for o in over_odds_arr]),
        p_unders - np.array([implied_probability(int(u)) for u in under_odds_arr])
    )

    print(f"\n--- Edge Distribution ---")
    print(f"  Mean edge (all games):    {all_edges.mean():+.1%}")
    print(f"  Median edge:              {np.median(all_edges):+.1%}")
    print(f"  Games with >0% edge:      {(all_edges > 0).sum()}")
    print(f"  Games with >3% edge:      {(all_edges > 0.03).sum()}")
    print(f"  Games with >5% edge:      {(all_edges > 0.05).sum()}")
    print(f"  Games with >10% edge:     {(all_edges > 0.10).sum()}")

    # ── 7. CLV Analysis (opening vs closing line) ──
    has_opening = ~np.isnan(opening_line.astype(float))
    if has_opening.sum() > 0:
        line_movement = vegas_line[has_opening] - opening_line[has_opening]

        # Did our bet side align with line movement?
        # If we bet OVER and line moved UP, that's positive CLV
        # If we bet UNDER and line moved DOWN, that's positive CLV
        print(f"\n--- Closing Line Value (CLV) ---")
        print(f"  Games with opening lines: {has_opening.sum()}")
        print(f"  Avg line movement:        {line_movement.mean():+.2f} runs")

        # For games where model had >3% edge
        for threshold in [0.03, 0.05, 0.10]:
            clv_values = []
            for i in range(len(merged_df)):
                if not has_opening[i]:
                    continue

                implied_over = implied_probability(int(over_odds_arr[i]))
                over_edge = p_overs[i] - implied_over
                implied_under = implied_probability(int(under_odds_arr[i]))
                under_edge = p_unders[i] - implied_under

                best_edge = max(over_edge, under_edge)
                if best_edge < threshold:
                    continue

                move = vegas_line[i] - opening_line[i]
                if over_edge > under_edge:
                    # Bet over: positive CLV if line moved up
                    clv_values.append(move)
                else:
                    # Bet under: positive CLV if line moved down
                    clv_values.append(-move)

            if clv_values:
                avg_clv = np.mean(clv_values)
                pos_clv = sum(1 for c in clv_values if c > 0)
                print(f"  Edge >{threshold:.0%}: avg CLV = {avg_clv:+.3f} runs "
                      f"({pos_clv}/{len(clv_values)} positive = {pos_clv/len(clv_values):.1%})")

    # ── 8. Monthly breakdown ──
    print(f"\n--- Monthly Breakdown ---")
    merged_df_copy = merged_df.copy()
    merged_df_copy["month"] = pd.to_datetime(merged_df_copy["date"]).dt.strftime("%Y-%m")
    merged_df_copy["pred_total"] = pred_total
    merged_df_copy["model_correct"] = model_correct
    merged_df_copy["actual_push"] = actual_push

    for month, group in merged_df_copy.groupby("month"):
        non_push_m = ~group["actual_push"]
        if non_push_m.sum() == 0:
            continue
        acc = group.loc[non_push_m, "model_correct"].mean()
        mae = np.abs(group["pred_total"] - group["total_runs"]).mean()
        print(f"  {month}:  {len(group):>4} games  accuracy={acc:.1%}  MAE={mae:.2f}")

    print(f"\n{'=' * 65}")


# ===================================================================
# MAIN
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="O/U Backtest for Run Total Model")
    parser.add_argument("--test-season", type=int, default=2025,
                        help="Season to backtest (default: 2025)")
    parser.add_argument("--edge-thresholds", type=float, nargs="+",
                        default=[0.0, 0.03, 0.05, 0.10, 0.15],
                        help="Edge thresholds for betting simulation")
    args = parser.parse_args()

    season = args.test_season
    print(f"O/U Backtest — Season {season}")
    print("=" * 40)

    # 1. Load models and features
    features_df, home_model, away_model, meta, medians = load_features_and_models(season)

    # 2. Load and parse Vegas odds
    odds_df = load_odds(season)

    if odds_df.empty:
        print(f"No odds data found for {season}. Exiting.")
        return

    # 3. Merge features with odds
    merged = merge_features_with_odds(features_df, odds_df)

    if merged.empty:
        print("No games matched between features and odds. Exiting.")
        return

    # 4. Run predictions and evaluation
    predict_and_evaluate(
        merged_df=merged,
        home_model=home_model,
        away_model=away_model,
        meta=meta,
        medians=medians,
        edge_thresholds=args.edge_thresholds,
    )


if __name__ == "__main__":
    main()
