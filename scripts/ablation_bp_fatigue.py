"""
ablation_bp_fatigue.py -- Test bullpen fatigue features on win prob model
========================================================================
Tests whether adding bullpen fatigue features to the 48-feature pruned
win probability model improves log loss.

Usage:
  python scripts/ablation_bp_fatigue.py
"""

import sys
if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import log_loss
import lightgbm as lgb

# Import pruned features from win model
sys.path.insert(0, str(Path(__file__).parent))
from train_win_model import PRUNED_FEATURES

FEATURES_PATH = Path(__file__).parent.parent / "data" / "features" / "game_features.csv"
SEEDS = [42, 123, 456, 789, 2025]

LGB_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
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


def train_eval(X_train, y_train, X_test, y_test, seed):
    params = {**LGB_PARAMS, "random_state": seed}
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    return log_loss(y_test, probs)


def run_config(name, feature_cols, df, train_df, test_df):
    available = [f for f in feature_cols if f in df.columns]
    missing = [f for f in feature_cols if f not in df.columns]
    if missing:
        print(f"  [{name}] Missing: {missing}")

    X_train = train_df[available].copy()
    y_train = train_df["home_win"].copy()
    X_test = test_df[available].copy()
    y_test = test_df["home_win"].copy()

    for col in available:
        med = X_train[col].median()
        X_train[col] = X_train[col].fillna(med)
        X_test[col] = X_test[col].fillna(med)

    valid_train = y_train.notna()
    X_train, y_train = X_train[valid_train], y_train[valid_train].astype(int)
    valid_test = y_test.notna()
    X_test, y_test = X_test[valid_test], y_test[valid_test].astype(int)

    losses = []
    for seed in SEEDS:
        ll = train_eval(X_train, y_train, X_test, y_test, seed)
        losses.append(ll)

    avg_ll = np.mean(losses)
    std_ll = np.std(losses)
    print(f"  [{name:40s}] n_feat={len(available):3d}  avg_ll={avg_ll:.6f}  std={std_ll:.6f}")
    return avg_ll, std_ll, len(available)


def main():
    print("=" * 70)
    print("BULLPEN FATIGUE ABLATION — Win Probability Model")
    print("=" * 70)

    df = pd.read_csv(FEATURES_PATH)
    df["date"] = pd.to_datetime(df["date"])

    train_df = df[df["season"].isin([2021, 2022, 2023, 2024])]
    test_df = df[df["season"] == 2025]
    print(f"  Train: {len(train_df)} games, Test: {len(test_df)} games")
    print(f"  Seeds: {SEEDS}\n")

    # Fatigue feature groups
    bp_ip_features = [
        "home_bp_ip_last1", "away_bp_ip_last1",
        "home_bp_ip_last3", "away_bp_ip_last3",
    ]
    bp_pitches_features = [
        "home_bp_pitches_last1", "away_bp_pitches_last1",
        "home_bp_pitches_last3", "away_bp_pitches_last3",
    ]
    bp_fatigue_diff = [
        "diff_bp_fatigue_last1", "diff_bp_fatigue_last3",
    ]
    bp_all_ip_pitches = [
        "home_bp_ip_last1", "away_bp_ip_last1",
        "home_bp_ip_last2", "away_bp_ip_last2",
        "home_bp_ip_last3", "away_bp_ip_last3",
        "home_bp_pitches_last1", "away_bp_pitches_last1",
        "home_bp_pitches_last2", "away_bp_pitches_last2",
        "home_bp_pitches_last3", "away_bp_pitches_last3",
    ]
    bp_all = bp_all_ip_pitches + bp_fatigue_diff + ["diff_bp_fatigue_last2"]

    configs = [
        ("Baseline (48 pruned)", PRUNED_FEATURES),
        ("+ BP IP (last 1g/3g, h/a)", list(PRUNED_FEATURES) + bp_ip_features),
        ("+ BP Pitches (last 1g/3g, h/a)", list(PRUNED_FEATURES) + bp_pitches_features),
        ("+ BP Fatigue Diff (1g/3g)", list(PRUNED_FEATURES) + bp_fatigue_diff),
        ("+ BP IP + Pitches (all windows)", list(PRUNED_FEATURES) + bp_all_ip_pitches),
        ("+ ALL BP fatigue features", list(PRUNED_FEATURES) + bp_all),
    ]

    results = []
    for name, feats in configs:
        avg_ll, std_ll, n = run_config(name, feats, df, train_df, test_df)
        results.append((name, n, avg_ll, std_ll))

    # Summary
    baseline_ll = results[0][2]
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY")
    print(f"{'=' * 70}")
    print(f"  {'Config':<42s} {'N':>4s} {'AvgLL':>10s} {'Delta BP':>10s}")
    print(f"  {'-'*42} {'-'*4} {'-'*10} {'-'*10}")
    for name, n, avg_ll, std_ll in results:
        delta_bp = (avg_ll - baseline_ll) * 10000
        marker = "<-- HELPS" if delta_bp < -1 else ("HURTS" if delta_bp > 1 else "flat")
        print(f"  {name:<42s} {n:4d} {avg_ll:10.6f} {delta_bp:+8.1f} bp  {marker}")

    print(f"\n  Baseline log loss: {baseline_ll:.6f}")


if __name__ == "__main__":
    main()
