"""
ablation_run_model.py -- Drop-One Feature Ablation for Run Total Model
"""

import sys
import time
import warnings
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
import lightgbm as lgb

FEATURES_PATH = Path(__file__).parent.parent / "data" / "features" / "game_features.csv"
SEEDS = [42, 123, 456, 789, 2025]
TRAIN_START = 2021
TEST_SEASON = 2025
MAX_DEPTH = 2

sys.path.insert(0, str(Path(__file__).parent))
from train_run_model import TOTALS_FEATURES


def train_model_with_seed(X_train, y_train, seed):
    model = lgb.LGBMRegressor(
        objective="mae", metric="mae",
        max_depth=MAX_DEPTH, num_leaves=2**MAX_DEPTH - 1,
        learning_rate=0.05, n_estimators=500,
        min_child_samples=50, subsample=0.8,
        colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        random_state=seed, verbose=-1,
    )
    model.fit(X_train, y_train)
    return model


def compute_total_mae(hm, am, X_t, yh, ya):
    return np.abs(hm.predict(X_t) + am.predict(X_t) - yh.values - ya.values).mean()


def run_ablation_trial(Xtr, yth, yta, Xte, yteh, ytea, seeds):
    maes = []
    for seed in seeds:
        hm = train_model_with_seed(Xtr, yth, seed)
        am = train_model_with_seed(Xtr, yta, seed)
        maes.append(compute_total_mae(hm, am, Xte, yteh, ytea))
    return np.mean(maes), np.std(maes)


def main():
    t0 = time.time()
    print("=" * 70)
    print("RUN TOTAL MODEL -- DROP-ONE FEATURE ABLATION")
    print("=" * 70)
    print(f"  Seeds: {SEEDS}")
    print(f"  Train: {TRAIN_START}-{TEST_SEASON-1}, Test: {TEST_SEASON}")
    print(f"  Max depth: {MAX_DEPTH}, num_leaves: {2**MAX_DEPTH-1}")

    if not FEATURES_PATH.exists():
        print("Feature matrix not found"); return

    df = pd.read_csv(FEATURES_PATH)
    df["date"] = pd.to_datetime(df["date"])
    available = [f for f in TOTALS_FEATURES if f in df.columns]
    missing = [f for f in TOTALS_FEATURES if f not in df.columns]
    if missing:
        print(f"  Skipping {len(missing)} unavailable: {missing}")
    print(f"  Using {len(available)} features")

    train_seas = [s for s in df["season"].unique() if TRAIN_START <= s < TEST_SEASON]
    train_df = df[df["season"].isin(train_seas)].copy()
    test_df = df[df["season"] == TEST_SEASON].copy()
    print(f"  Train: {sorted(train_seas)} ({len(train_df)} games)")
    print(f"  Test:  {TEST_SEASON} ({len(test_df)} games)")

    tv = train_df["home_score"].notna() & train_df["away_score"].notna()
    ev = test_df["home_score"].notna() & test_df["away_score"].notna()
    train_df = train_df[tv]; test_df = test_df[ev]

    yth = train_df["home_score"]; yta = train_df["away_score"]
    yteh = test_df["home_score"]; ytea = test_df["away_score"]

    Xtr = train_df[available].copy()
    Xte = test_df[available].copy()
    for col in available:
        med = Xtr[col].median()
        Xtr[col] = Xtr[col].fillna(med)
        Xte[col] = Xte[col].fillna(med)

    print(f"\n  Training BASELINE (all {len(available)} features, {len(SEEDS)} seeds)...")
    bmae, bstd = run_ablation_trial(Xtr, yth, yta, Xte, yteh, ytea, SEEDS)
    print(f"  Baseline total MAE: {bmae:.4f} (+/- {bstd:.4f})")

    print(f"\n  Running drop-one ablation for {len(available)} features...")
    print(f"  ({len(available)} x {2*len(SEEDS)} fits)\n")

    results = []
    for i, feat in enumerate(available):
        cols = [f for f in available if f != feat]
        mae, std = run_ablation_trial(Xtr[cols], yth, yta, Xte[cols], yteh, ytea, SEEDS)
        delta = mae - bmae
        results.append({"feature_dropped": feat, "avg_mae": mae, "std_mae": std, "delta": delta})
        tag = "IMPROVES" if delta < -0.001 else ("hurts" if delta > 0.001 else "neutral")
        print(f"  [{i+1:2d}/{len(available)}] Drop {feat:45s} MAE={mae:.4f} delta={delta:+.4f} {tag}")

    rdf = pd.DataFrame(results).sort_values("delta")

    print(f"\n{'='*80}")
    print(f"  ABLATION RESULTS -- Sorted by impact (negative delta = feature hurts)")
    print(f"  Baseline MAE: {bmae:.4f} (std: {bstd:.4f})")
    print(f"{'='*80}")
    print(f"  {'Feature Dropped':<45s} {'Avg MAE':>8s} {'Delta':>8s} {'Std':>7s} {'Verdict'}")
    print(f"  {'-'*45} {'-'*8} {'-'*8} {'-'*7} {'-'*10}")

    for _, r in rdf.iterrows():
        d = r["delta"]
        if d < -0.005: v = "** DROP **"
        elif d < -0.001: v = "drop?"
        elif d > 0.005: v = "KEEP"
        elif d > 0.001: v = "keep"
        else: v = "neutral"
        print(f"  {r['feature_dropped']:<45s} {r['avg_mae']:>8.4f} {d:>+.4f} {r['std_mae']:>7.4f} {v}")

    cand = rdf[rdf["delta"] < -0.001]
    print(f"\n  PRUNE CANDIDATES ({len(cand)} features where dropping improves MAE):")
    if len(cand) > 0:
        for _, r in cand.iterrows():
            print(f"    - {r['feature_dropped']:40s} delta={r['delta']:+.4f}")
    else:
        print("    None -- all features contribute positively")

    keep = rdf[rdf["delta"] > 0.005]
    print(f"\n  STRONG KEEPERS ({len(keep)} features where dropping hurts MAE > 0.005):")
    for _, r in keep.sort_values("delta", ascending=False).iterrows():
        print(f"    + {r['feature_dropped']:40s} delta={r['delta']:+.4f}")

    out = Path(__file__).parent.parent / "data" / "ablation_run_model_results.csv"
    rdf.to_csv(out, index=False)
    print(f"\n  Results saved to {out}")
    print(f"\n  Total time: {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
