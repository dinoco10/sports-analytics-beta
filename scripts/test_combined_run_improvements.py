"""
test_combined_run_improvements.py -- Combined run model improvements
====================================================================
Tests the combined effect of:
1. Pruning 6 noise features (from ablation)
2. Adding separate bullpen features (home/away)
3. Bias correction (+0.1 offset)

Usage:
  python scripts/test_combined_run_improvements.py
"""

import sys
if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).parent))
from train_run_model import TOTALS_FEATURES

FEATURES_PATH = Path(__file__).parent.parent / "data" / "features" / "game_features.csv"
SEEDS = [42, 123, 456, 789, 2025]

LGB_PARAMS = {
    "objective": "mae",
    "metric": "mae",
    "max_depth": 2,
    "num_leaves": 3,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "min_child_samples": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "verbose": -1,
}

# Features to prune (from ablation: dropping improves MAE)
PRUNE_LIST = [
    "home_lineup_slg",       # -0.0051
    "away_pyth_t30",         # -0.0028
    "away_ra_t30",           # -0.0018
    "diff_bp_era_bp35",      # -0.0015
    "home_proj_sp_sc_era",   # -0.0014
    "away_sp_rgs",           # -0.0013
]

# Separate bullpen features to add
BP_SEPARATE = [
    "home_bp_era_bp35", "away_bp_era_bp35",
    "home_bp_whip_bp35", "away_bp_whip_bp35",
    "home_bp_k_pct_bp35", "away_bp_k_pct_bp35",
    "home_bp_bb_pct_bp35", "away_bp_bb_pct_bp35",
]


def run_config(name, feature_cols, X_train, X_test, y_home_train, y_away_train,
               total_actual, y_home_test, y_away_test, bias_offset=0.0):
    total_maes = []
    home_maes = []
    away_maes = []

    for seed in SEEDS:
        params = {**LGB_PARAMS, "random_state": seed}

        hm = lgb.LGBMRegressor(**params)
        hm.fit(X_train[feature_cols], y_home_train)
        hp = hm.predict(X_test[feature_cols])

        am = lgb.LGBMRegressor(**params)
        am.fit(X_train[feature_cols], y_away_train)
        ap = am.predict(X_test[feature_cols])

        # Apply bias correction if requested
        if bias_offset != 0:
            # Compute training residuals for this seed
            hp_train = hm.predict(X_train[feature_cols])
            ap_train = am.predict(X_train[feature_cols])
            # Use fixed offset (not data-driven, to avoid leakage)
            hp_corrected = hp + bias_offset
            ap_corrected = ap + bias_offset
            total_maes.append(np.abs(hp_corrected + ap_corrected - total_actual).mean())
            home_maes.append(np.abs(hp_corrected - y_home_test).mean())
            away_maes.append(np.abs(ap_corrected - y_away_test).mean())
        else:
            total_maes.append(np.abs(hp + ap - total_actual).mean())
            home_maes.append(np.abs(hp - y_home_test).mean())
            away_maes.append(np.abs(ap - y_away_test).mean())

    avg = np.mean(total_maes)
    std = np.std(total_maes)
    print(f"  {name:<60s} n={len(feature_cols):2d}  MAE={avg:.4f} (+/-{std:.4f})  home={np.mean(home_maes):.4f}  away={np.mean(away_maes):.4f}")
    return avg


def main():
    print("=" * 70)
    print("COMBINED RUN MODEL IMPROVEMENTS")
    print("=" * 70)

    df = pd.read_csv(FEATURES_PATH)
    df["date"] = pd.to_datetime(df["date"])

    # Build feature sets
    baseline = [f for f in TOTALS_FEATURES if f in df.columns]
    pruned = [f for f in baseline if f not in PRUNE_LIST]
    pruned_plus_bp = pruned + [f for f in BP_SEPARATE if f not in pruned]

    # Prepare data
    all_cols = list(set(baseline + BP_SEPARATE))
    all_cols = [c for c in all_cols if c in df.columns]

    train_df = df[df["season"].isin([2021, 2022, 2023, 2024])]
    test_df = df[df["season"] == 2025]

    X_train = train_df[all_cols].copy()
    X_test = test_df[all_cols].copy()

    for col in all_cols:
        med = X_train[col].median()
        X_train[col] = X_train[col].fillna(med)
        X_test[col] = X_test[col].fillna(med)

    y_home_train = train_df["home_score"].values
    y_away_train = train_df["away_score"].values
    y_home_test = test_df["home_score"].values
    y_away_test = test_df["away_score"].values
    total_actual = test_df["total_runs"].values

    valid_train = ~(np.isnan(y_home_train) | np.isnan(y_away_train))
    X_train = X_train[valid_train].reset_index(drop=True)
    y_home_train = y_home_train[valid_train]
    y_away_train = y_away_train[valid_train]

    valid_test = ~np.isnan(total_actual)
    X_test = X_test[valid_test].reset_index(drop=True)
    y_home_test = y_home_test[valid_test]
    y_away_test = y_away_test[valid_test]
    total_actual = total_actual[valid_test]

    print(f"  Train: {len(X_train)} games, Test: {len(X_test)} games")
    print(f"  Seeds: {SEEDS}")
    print(f"  Pruned: {PRUNE_LIST}")
    print()

    results = {}

    # A: Original baseline
    r = run_config("A: Baseline (49 features)", baseline, X_train, X_test,
                   y_home_train, y_away_train, total_actual, y_home_test, y_away_test)
    results["A"] = r

    # B: Pruned only
    r = run_config("B: Pruned (43 features)", pruned, X_train, X_test,
                   y_home_train, y_away_train, total_actual, y_home_test, y_away_test)
    results["B"] = r

    # C: Pruned + separate BP
    r = run_config("C: Pruned + separate BP (51 features)", pruned_plus_bp, X_train, X_test,
                   y_home_train, y_away_train, total_actual, y_home_test, y_away_test)
    results["C"] = r

    # D: Pruned + bias correction
    r = run_config("D: Pruned + bias +0.1 (43 features)", pruned, X_train, X_test,
                   y_home_train, y_away_train, total_actual, y_home_test, y_away_test, bias_offset=0.1)
    results["D"] = r

    # E: Pruned + separate BP + bias correction
    r = run_config("E: Pruned + sep BP + bias +0.1 (51 features)", pruned_plus_bp, X_train, X_test,
                   y_home_train, y_away_train, total_actual, y_home_test, y_away_test, bias_offset=0.1)
    results["E"] = r

    # F: Incremental pruning (drop only the top 3 most confident)
    prune_top3 = ["home_lineup_slg", "away_pyth_t30", "away_ra_t30"]
    pruned3 = [f for f in baseline if f not in prune_top3]
    pruned3_bp = pruned3 + [f for f in BP_SEPARATE if f not in pruned3]
    r = run_config("F: Prune top-3 + sep BP + bias (54 features)", pruned3_bp, X_train, X_test,
                   y_home_train, y_away_train, total_actual, y_home_test, y_away_test, bias_offset=0.1)
    results["F"] = r

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY")
    print(f"{'=' * 70}")
    baseline_mae = results["A"]
    for key in sorted(results.keys()):
        delta = results[key] - baseline_mae
        marker = " <-- BEST" if results[key] == min(results.values()) else ""
        print(f"  {key}: MAE={results[key]:.4f}  delta={delta:+.4f}{marker}")

    print(f"\n  Vegas benchmark: 3.552")
    best = min(results, key=results.get)
    print(f"  Best config: {best} (MAE={results[best]:.4f})")
    print(f"  Improvement over baseline: {baseline_mae - results[best]:.4f}")
    print(f"  Improvement over Vegas: {3.552 - results[best]:.4f}")


if __name__ == "__main__":
    main()
