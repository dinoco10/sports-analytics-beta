"""
test_bias_correction.py -- Test bias correction on run total model
=================================================================
Trains the run model, computes training-set median residuals,
applies as offset to test predictions, reports before/after MAE.

Usage:
  python scripts/test_bias_correction.py
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


def main():
    print("=" * 70)
    print("RUN TOTAL BIAS CORRECTION TEST")
    print("=" * 70)

    df = pd.read_csv(FEATURES_PATH)
    df["date"] = pd.to_datetime(df["date"])

    available = [f for f in TOTALS_FEATURES if f in df.columns]
    print(f"  Using {len(available)} features")

    train_df = df[df["season"].isin([2021, 2022, 2023, 2024])]
    test_df = df[df["season"] == 2025]
    print(f"  Train: {len(train_df)} games, Test: {len(test_df)} games")

    X_train = train_df[available].copy()
    X_test = test_df[available].copy()

    for col in available:
        med = X_train[col].median()
        X_train[col] = X_train[col].fillna(med)
        X_test[col] = X_test[col].fillna(med)

    y_home_train = train_df["home_score"].values
    y_away_train = train_df["away_score"].values
    total_actual_test = test_df["total_runs"].values

    valid_train = ~(np.isnan(y_home_train) | np.isnan(y_away_train))
    X_train_v = X_train[valid_train].reset_index(drop=True)
    y_home_train_v = y_home_train[valid_train]
    y_away_train_v = y_away_train[valid_train]

    valid_test = ~np.isnan(total_actual_test)
    X_test_v = X_test[valid_test].reset_index(drop=True)
    y_home_test = test_df["home_score"].values[valid_test]
    y_away_test = test_df["away_score"].values[valid_test]
    total_actual = total_actual_test[valid_test]

    # Test different offset strategies
    offsets_to_test = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, "median", "mean"]

    print(f"\n  Testing offset strategies across 5 seeds...\n")

    for offset_type in offsets_to_test:
        total_maes_raw = []
        total_maes_corrected = []
        home_maes_raw = []
        away_maes_raw = []

        for seed in SEEDS:
            params = {**LGB_PARAMS, "random_state": seed}

            home_model = lgb.LGBMRegressor(**params)
            home_model.fit(X_train_v, y_home_train_v)

            away_model = lgb.LGBMRegressor(**params)
            away_model.fit(X_train_v, y_away_train_v)

            # Training set predictions for bias estimation
            home_train_preds = home_model.predict(X_train_v)
            away_train_preds = away_model.predict(X_train_v)

            # Compute bias from training residuals
            home_residuals = y_home_train_v - home_train_preds
            away_residuals = y_away_train_v - away_train_preds

            if offset_type == "median":
                home_offset = np.median(home_residuals)
                away_offset = np.median(away_residuals)
            elif offset_type == "mean":
                home_offset = np.mean(home_residuals)
                away_offset = np.mean(away_residuals)
            else:
                home_offset = float(offset_type)
                away_offset = float(offset_type)

            # Test predictions
            home_preds = home_model.predict(X_test_v)
            away_preds = away_model.predict(X_test_v)

            # Raw MAE
            total_raw = home_preds + away_preds
            mae_raw = np.abs(total_raw - total_actual).mean()
            total_maes_raw.append(mae_raw)
            home_maes_raw.append(np.abs(home_preds - y_home_test).mean())
            away_maes_raw.append(np.abs(away_preds - y_away_test).mean())

            # Corrected MAE
            total_corrected = (home_preds + home_offset) + (away_preds + away_offset)
            mae_corrected = np.abs(total_corrected - total_actual).mean()
            total_maes_corrected.append(mae_corrected)

        avg_raw = np.mean(total_maes_raw)
        avg_corrected = np.mean(total_maes_corrected)
        delta = avg_corrected - avg_raw

        label = f"offset={offset_type}" if isinstance(offset_type, str) else f"offset=+{offset_type:.1f}"
        print(f"  {label:<20s}  raw={avg_raw:.4f}  corrected={avg_corrected:.4f}  delta={delta:+.4f}")

    # Also report per-side stats
    print(f"\n  --- Per-side MAE (raw, no correction) ---")
    print(f"  Home MAE (5-seed avg): {np.mean(home_maes_raw):.4f}")
    print(f"  Away MAE (5-seed avg): {np.mean(away_maes_raw):.4f}")
    print(f"  Total MAE (5-seed avg): {np.mean(total_maes_raw):.4f}")
    print(f"  Vegas benchmark: 3.552")


if __name__ == "__main__":
    main()
