"""
ablation_run_wave2.py -- Test Wave 2 features on run total model
================================================================
Tests: (1) separate bullpen features, (2) day/night, (3) both combined

Usage:
  python scripts/ablation_run_wave2.py
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


def run_config(name, feature_cols, X_train, X_test, y_home_train, y_away_train,
               total_actual, y_home_test, y_away_test):
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

        total_maes.append(np.abs(hp + ap - total_actual).mean())
        home_maes.append(np.abs(hp - y_home_test).mean())
        away_maes.append(np.abs(ap - y_away_test).mean())

    avg = np.mean(total_maes)
    print(f"  {name:<55s}  n={len(feature_cols):2d}  total_MAE={avg:.4f}  home={np.mean(home_maes):.4f}  away={np.mean(away_maes):.4f}")
    return avg


def main():
    print("=" * 70)
    print("RUN TOTAL WAVE 2 FEATURE TESTS")
    print("=" * 70)

    df = pd.read_csv(FEATURES_PATH)
    df["date"] = pd.to_datetime(df["date"])

    # Create is_day_game from day_night column
    df["is_day_game"] = (df["day_night"] == "day").astype(int)

    available_base = [f for f in TOTALS_FEATURES if f in df.columns]
    print(f"  Baseline features: {len(available_base)}")

    # Separate bullpen features (already exist in data)
    bp_separate = [
        "home_bp_era_bp35", "away_bp_era_bp35",
        "home_bp_whip_bp35", "away_bp_whip_bp35",
        "home_bp_k_pct_bp35", "away_bp_k_pct_bp35",
        "home_bp_bb_pct_bp35", "away_bp_bb_pct_bp35",
    ]

    # Current diff bullpen features in TOTALS_FEATURES
    bp_diffs = ["diff_bp_era_bp35", "diff_bp_whip_bp35", "diff_bp_k_pct_bp35"]

    # Configs to test
    # A: baseline (has bp diffs)
    # B: replace bp diffs with separate home/away
    # C: keep bp diffs AND add separate
    # D: baseline + is_day_game
    # E: replace bp diffs with separate + is_day_game
    # F: keep both + is_day_game

    base_no_bp = [f for f in available_base if f not in bp_diffs]

    configs = {
        "A: Baseline (49 features, bp diffs)": available_base,
        "B: Replace bp diffs with separate h/a (54 feat)": base_no_bp + bp_separate,
        "C: Keep diffs + add separate (57 feat)": available_base + bp_separate,
        "D: Baseline + is_day_game (50 feat)": available_base + ["is_day_game"],
        "E: Separate bp + is_day_game (55 feat)": base_no_bp + bp_separate + ["is_day_game"],
        "F: All bp + is_day_game (58 feat)": available_base + bp_separate + ["is_day_game"],
    }

    # Check all features exist
    all_needed = set()
    for feats in configs.values():
        all_needed.update(feats)
    missing = [f for f in all_needed if f not in df.columns]
    if missing:
        print(f"  WARNING: Missing columns: {missing}")

    # Prepare data
    train_df = df[df["season"].isin([2021, 2022, 2023, 2024])]
    test_df = df[df["season"] == 2025]

    all_cols = list(all_needed)
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

    valid_test = ~(np.isnan(total_actual))
    X_test = X_test[valid_test].reset_index(drop=True)
    y_home_test = y_home_test[valid_test]
    y_away_test = y_away_test[valid_test]
    total_actual = total_actual[valid_test]

    print(f"  Train: {len(X_train)} games, Test: {len(X_test)} games")
    print(f"  Seeds: {SEEDS}\n")

    # Day/night stats
    dn = test_df[valid_test]["day_night"] if "day_night" in test_df.columns else None
    if dn is not None:
        day_mask = dn.values == "day"
        night_mask = dn.values == "night"
        day_total = total_actual[day_mask].mean()
        night_total = total_actual[night_mask].mean()
        print(f"  Day games avg total: {day_total:.2f} ({day_mask.sum()} games)")
        print(f"  Night games avg total: {night_total:.2f} ({night_mask.sum()} games)")
        print(f"  Difference: {day_total - night_total:+.2f} runs\n")

    results = {}
    for name, feats in configs.items():
        avg = run_config(name, feats, X_train, X_test, y_home_train, y_away_train,
                         total_actual, y_home_test, y_away_test)
        results[name] = avg

    baseline = results[list(configs.keys())[0]]
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY")
    print(f"{'=' * 70}")
    print(f"  {'Config':<55s} {'MAE':>8s} {'Delta':>8s}")
    print(f"  {'-'*55} {'-'*8} {'-'*8}")
    for name, mae in results.items():
        delta = mae - baseline
        marker = "<-- BEST" if mae == min(results.values()) else ""
        print(f"  {name:<55s} {mae:8.4f} {delta:+8.4f} {marker}")

    print(f"\n  Vegas benchmark: 3.552")


if __name__ == "__main__":
    main()
