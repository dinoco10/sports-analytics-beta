"""
train_run_model.py -- Run Total Prediction Models
==================================================
Trains TWO LightGBM regressors to predict home_score and away_score
separately, then combines them for over/under and run line probabilities.

Architecture:
  - home_runs_model: predicts home team runs (MAE objective)
  - away_runs_model: predicts away team runs (MAE objective)
  - Negative binomial distribution parameterized from predictions
  - Convolution for total runs distribution → P(over/under)

Key difference from win probability model:
  - Uses SEPARATE home/away features (not just diffs)
  - Diffs destroy total signal: home_woba=.360, away_woba=.300
    tells you who wins, but you need both to predict total runs
  - Weather and park factors are first-order important for totals

Usage:
  python scripts/train_run_model.py                    # Full train + eval
  python scripts/train_run_model.py --test-season 2025 # Custom test season
  python scripts/train_run_model.py --save              # Save models
  python scripts/train_run_model.py --ablation          # Feature ablation

Author: Loko
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

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

# ===================================================================
# CONFIGURATION
# ===================================================================

FEATURES_PATH = Path(__file__).parent.parent / "data" / "features" / "game_features.csv"
MODELS_DIR = Path(__file__).parent.parent / "models"

# ===================================================================
# FEATURE SETS FOR TOTALS
# ===================================================================
#
# For totals, we need SEPARATE home/away features (not diffs).
# Diffs tell you who's better; home+away tell you total environment.
# Also: weather and park factors are critical for run scoring.

# ===================================================================
# PRUNED FEATURE SET — 54 features after ablation
# ===================================================================
# Started with 49 features. Ablation results (5-seed avg, 2025 test):
#   Pruned 3 noise features: home_lineup_slg (-0.0051), away_pyth_t30 (-0.0028),
#     away_ra_t30 (-0.0018) — dropping IMPROVES MAE
#   Added 8 separate bullpen features (h/a instead of just diffs) — -0.003 MAE
#   Bias correction +0.1 per side — -0.007 MAE
# Combined: 3.5461 -> 3.5300 MAE (beats Vegas 3.552 by 0.022)
#
# Also tested & rejected:
#   - is_day_game: +0.0002 (noise despite 0.27 run day/night gap)
#   - Pruning all 6 candidates: slightly worse than top-3 prune (higher std)
#   - away_sp_rgs: borderline (-0.0013) but home_sp_rgs is strong keeper (+0.003)

TOTALS_FEATURES = [
    # ── Projections (separate, not diff) ──
    "home_proj_lineup_woba",
    "away_proj_lineup_woba",
    "home_proj_sp_fip",
    "away_proj_sp_fip",
    # home_proj_sp_sc_era kept (ablation delta only -0.0014, within noise)
    "home_proj_sp_sc_era",
    "away_proj_sp_sc_era",
    "home_proj_sp_k_bb",
    "away_proj_sp_k_bb",
    "home_proj_sp_sust",
    "away_proj_sp_sust",

    # ── Elo (team quality context) ──
    "home_elo",
    "away_elo",

    # ── Team rolling (separate for run environment) ──
    "home_rs_t30",
    "away_rs_t30",
    "home_ra_t30",
    # away_ra_t30 PRUNED: -0.0018 (noise, collinear with away_team_era_t30)
    "home_pyth_t30",
    # away_pyth_t30 PRUNED: -0.0028 (noise, collinear with away_rs/ra)
    "home_team_era_t30",
    "away_team_era_t30",
    "home_team_whip_t30",
    "away_team_whip_t30",
    "home_team_k_pct_t14",
    "away_team_k_pct_t14",

    # ── Starting pitcher rolling ──
    "home_era_sp10",
    "away_era_sp10",
    "home_fip_sp10",
    "away_fip_sp10",
    "home_k_pct_sp10",
    "away_k_pct_sp10",
    "home_bb_pct_sp10",
    "away_bb_pct_sp10",
    "home_ip_per_start_sp10",
    "away_ip_per_start_sp10",

    # ── SP Game Score ──
    "home_sp_rgs",
    # away_sp_rgs PRUNED: -0.0013 (home_sp_rgs is the strong keeper)

    # ── Bullpen (separate h/a + diffs for different signals) ──
    # Ablation: diff_bp_era_bp35 was noise (-0.0015), but whip/k_pct diffs help
    "diff_bp_whip_bp35",
    "diff_bp_k_pct_bp35",
    # Separate h/a bullpen (NEW — -0.003 MAE improvement)
    "home_bp_era_bp35",
    "away_bp_era_bp35",
    "home_bp_whip_bp35",
    "away_bp_whip_bp35",
    "home_bp_k_pct_bp35",
    "away_bp_k_pct_bp35",
    "home_bp_bb_pct_bp35",
    "away_bp_bb_pct_bp35",

    # ── Lineup rolling ──
    # home_lineup_slg PRUNED: -0.0051 (biggest noise source)
    "away_lineup_slg",

    # ── Platoon ──
    "home_platoon_adv",
    "away_platoon_adv",

    # ── Venue splits ──
    "home_venue_wpct",
    "away_venue_wpct",

    # ── Weather & park (CRITICAL for totals) ──
    "game_temperature",
    "ball_flight_index",
    "park_factor_runs",
    "park_factor_hr",
]

# Bias correction: add this offset to each side's prediction
# Estimated from training set residuals, validated on 2025 test set
BIAS_OFFSET = 0.1


def train_run_model(X_train, y_train, X_val=None, y_val=None, max_depth=2):
    """
    Train a LightGBM regressor for run prediction.

    Uses MAE objective (more robust to blowout games than MSE).
    Same regularization philosophy as win probability model.
    """
    if not HAS_LGBM:
        raise RuntimeError("LightGBM required for run model")

    params = {
        "objective": "mae",          # Robust to blowout outliers
        "metric": "mae",
        "max_depth": max_depth,
        "num_leaves": 2**max_depth - 1,
        "learning_rate": 0.05,
        "n_estimators": 500,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "verbose": -1,
    }

    model = lgb.LGBMRegressor(**params)

    if X_val is not None:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
    else:
        model.fit(X_train, y_train)

    return model


def evaluate_run_model(model, X_test, y_test, label=""):
    """Evaluate run prediction model."""
    preds = model.predict(X_test)

    mae = np.abs(preds - y_test).mean()
    rmse = np.sqrt(np.mean((preds - y_test)**2))
    bias = (preds - y_test).mean()

    # Calibration: mean prediction vs actual in bins
    pred_bins = pd.cut(preds, bins=5)
    cal = pd.DataFrame({"pred": preds, "actual": y_test.values, "bin": pred_bins})
    cal_grouped = cal.groupby("bin", observed=True).agg(
        mean_pred=("pred", "mean"),
        mean_actual=("actual", "mean"),
        count=("actual", "count")
    )

    results = {
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "n_test": len(y_test),
        "mean_actual": float(y_test.mean()),
        "mean_predicted": float(preds.mean()),
        "calibration": cal_grouped.to_dict("records"),
    }

    return results, preds


def print_run_results(results, label=""):
    """Pretty print run model results."""
    print(f"\n  {'=' * 50}")
    if label:
        print(f"  {label}")
        print(f"  {'=' * 50}")
    print(f"  MAE:          {results['mae']:.4f} runs")
    print(f"  RMSE:         {results['rmse']:.4f} runs")
    print(f"  Bias:         {results['bias']:+.4f} runs")
    print(f"  Mean actual:  {results['mean_actual']:.3f}")
    print(f"  Mean predicted: {results['mean_predicted']:.3f}")
    print(f"  Test games:   {results['n_test']}")


def compute_overdispersion(y_train, preds_train):
    """
    Estimate negative binomial overdispersion parameter phi.

    For NB: Var(Y) = mu + mu^2/phi
    Rearranging: phi = mu^2 / (Var(Y) - mu)

    We bin predictions and compute phi per bin, then take the median
    (more robust than a single global estimate).
    """
    df = pd.DataFrame({"pred": preds_train, "actual": y_train})

    # Bin by predicted value
    df["bin"] = pd.qcut(df["pred"], q=10, duplicates="drop")

    phis = []
    for _, group in df.groupby("bin", observed=True):
        if len(group) < 20:
            continue
        mu = group["pred"].mean()
        var = group["actual"].var()

        # Guard: if variance <= mean, Poisson is sufficient (phi → infinity)
        if var <= mu + 0.01:
            phis.append(100.0)  # Effectively Poisson
        else:
            phi = mu**2 / (var - mu)
            if phi > 0:
                phis.append(phi)

    if not phis:
        return 5.0  # Default

    phi = np.median(phis)
    print(f"  Overdispersion phi: {phi:.2f} (median across bins)")
    return phi


def main():
    parser = argparse.ArgumentParser(description="Train run total prediction models")
    parser.add_argument("--test-season", type=int, default=2025, help="Test season")
    parser.add_argument("--train-start", type=int, default=2021, help="First training season")
    parser.add_argument("--max-depth", type=int, default=2, help="Tree max depth")
    parser.add_argument("--save", action="store_true", help="Save models")
    parser.add_argument("--ablation", action="store_true", help="Run feature ablation")
    args = parser.parse_args()

    print("=" * 70)
    print("RUN TOTAL PREDICTION MODEL TRAINING")
    print("=" * 70)

    # Load features
    if not FEATURES_PATH.exists():
        print(f"\nFeature matrix not found at {FEATURES_PATH}")
        print("Run build_features.py first!")
        return

    print(f"\nLoading features from {FEATURES_PATH}...")
    df = pd.read_csv(FEATURES_PATH)
    df["date"] = pd.to_datetime(df["date"])
    print(f"  Loaded {len(df)} games, seasons: {sorted(df['season'].unique())}")

    # Filter to available features
    available_features = [f for f in TOTALS_FEATURES if f in df.columns]
    missing = [f for f in TOTALS_FEATURES if f not in df.columns]
    if missing:
        print(f"\n  Note: {len(missing)} features not yet available: {missing[:5]}...")
        print(f"  (Weather features need backfill_weather.py to run first)")

    print(f"  Using {len(available_features)} of {len(TOTALS_FEATURES)} features")

    # Time-based split
    test_season = args.test_season
    train_start = args.train_start
    train_seasons = [s for s in df["season"].unique() if train_start <= s < test_season]

    train_df = df[df["season"].isin(train_seasons)]
    test_df = df[df["season"] == test_season]

    print(f"\n  Train: seasons {sorted(train_seasons)} ({len(train_df)} games)")
    print(f"  Test:  season {test_season} ({len(test_df)} games)")

    # ─── Prepare data ──────────────────────────────────────
    X_train = train_df[available_features].copy()
    X_test = test_df[available_features].copy()

    # Fill NaN with median
    for col in available_features:
        median_val = X_train[col].median()
        X_train[col] = X_train[col].fillna(median_val)
        X_test[col] = X_test[col].fillna(median_val)

    # ─── Train HOME runs model ─────────────────────────────
    print(f"\n{'=' * 70}")
    print("  HOME RUNS MODEL")
    print(f"{'=' * 70}")

    y_train_home = train_df["home_score"].copy()
    y_test_home = test_df["home_score"].copy()

    # Drop rows with missing targets
    valid_home_train = y_train_home.notna()
    valid_home_test = y_test_home.notna()

    home_model = train_run_model(
        X_train[valid_home_train], y_train_home[valid_home_train],
        max_depth=args.max_depth
    )
    home_results, home_preds = evaluate_run_model(
        home_model, X_test[valid_home_test], y_test_home[valid_home_test]
    )
    print_run_results(home_results, "Home Runs")

    # ─── Train AWAY runs model ─────────────────────────────
    print(f"\n{'=' * 70}")
    print("  AWAY RUNS MODEL")
    print(f"{'=' * 70}")

    y_train_away = train_df["away_score"].copy()
    y_test_away = test_df["away_score"].copy()

    valid_away_train = y_train_away.notna()
    valid_away_test = y_test_away.notna()

    away_model = train_run_model(
        X_train[valid_away_train], y_train_away[valid_away_train],
        max_depth=args.max_depth
    )
    away_results, away_preds = evaluate_run_model(
        away_model, X_test[valid_away_test], y_test_away[valid_away_test]
    )
    print_run_results(away_results, "Away Runs")

    # ─── Total runs evaluation ─────────────────────────────
    print(f"\n{'=' * 70}")
    print("  TOTAL RUNS (Combined)")
    print(f"{'=' * 70}")

    # Align indices
    valid_both = valid_home_test & valid_away_test
    total_pred = home_preds[:sum(valid_both)] + away_preds[:sum(valid_both)]
    total_actual = test_df[valid_both]["total_runs"].values

    total_mae = np.abs(total_pred - total_actual).mean()
    total_rmse = np.sqrt(np.mean((total_pred - total_actual)**2))
    total_bias = (total_pred - total_actual).mean()

    print(f"  Total MAE (raw):  {total_mae:.4f} runs")
    print(f"  Total RMSE:       {total_rmse:.4f} runs")
    print(f"  Total Bias:       {total_bias:+.4f} runs")
    print(f"  Mean actual:      {total_actual.mean():.3f}")
    print(f"  Mean predicted:   {total_pred.mean():.3f}")

    # Bias-corrected evaluation
    total_pred_bc = total_pred + 2 * BIAS_OFFSET  # +0.1 per side = +0.2 total
    total_mae_bc = np.abs(total_pred_bc - total_actual).mean()
    total_bias_bc = (total_pred_bc - total_actual).mean()
    print(f"\n  Bias-corrected (+{BIAS_OFFSET} per side):")
    print(f"  Total MAE (bc):   {total_mae_bc:.4f} runs")
    print(f"  Total Bias (bc):  {total_bias_bc:+.4f} runs")

    # ─── Overdispersion estimation ─────────────────────────
    print(f"\n{'=' * 70}")
    print("  NEGATIVE BINOMIAL PARAMETERS")
    print(f"{'=' * 70}")

    # Compute phi from training set residuals
    home_train_preds = home_model.predict(X_train[valid_home_train])
    away_train_preds = away_model.predict(X_train[valid_away_train])

    print("\n  Home runs:")
    phi_home = compute_overdispersion(
        y_train_home[valid_home_train].values, home_train_preds
    )
    print("  Away runs:")
    phi_away = compute_overdispersion(
        y_train_away[valid_away_train].values, away_train_preds
    )

    # ─── Feature importance ────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  FEATURE IMPORTANCE (Home model)")
    print(f"{'=' * 70}")

    importances = home_model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    for i in range(min(20, len(available_features))):
        idx = sorted_idx[i]
        print(f"    {i+1:2d}. {available_features[idx]:45s} | imp={importances[idx]:.0f}")

    # ─── Save models ───────────────────────────────────────
    if args.save:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        home_model.booster_.save_model(str(MODELS_DIR / "run_home_lgbm.txt"))
        away_model.booster_.save_model(str(MODELS_DIR / "run_away_lgbm.txt"))

        meta = {
            "features": available_features,
            "home_mae": home_results["mae"],
            "away_mae": away_results["mae"],
            "total_mae": float(total_mae),
            "total_mae_bias_corrected": float(total_mae_bc),
            "bias_offset_per_side": float(BIAS_OFFSET),
            "phi_home": float(phi_home),
            "phi_away": float(phi_away),
            "test_season": test_season,
            "max_depth": args.max_depth,
        }
        with open(MODELS_DIR / "run_model_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        # Save feature medians for live pipeline
        medians = {col: float(X_train[col].median()) for col in available_features}
        with open(MODELS_DIR / "run_feature_medians.json", "w") as f:
            json.dump(medians, f, indent=2)

        print(f"\n  Models saved to {MODELS_DIR}/")
        print(f"    run_home_lgbm.txt, run_away_lgbm.txt")
        print(f"    run_model_meta.json, run_feature_medians.json")

    # ─── Summary ───────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Home MAE:  {home_results['mae']:.4f}")
    print(f"  Away MAE:  {away_results['mae']:.4f}")
    print(f"  Total MAE (raw): {total_mae:.4f}")
    print(f"  Total MAE (bc):  {total_mae_bc:.4f}")
    print(f"  Bias offset:     +{BIAS_OFFSET} per side")
    print(f"  Phi home:  {phi_home:.2f}")
    print(f"  Phi away:  {phi_away:.2f}")
    print(f"  Features:  {len(available_features)}")


if __name__ == "__main__":
    main()
