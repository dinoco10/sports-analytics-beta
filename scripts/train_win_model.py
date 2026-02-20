"""
train_win_model.py -- Game Win Probability Model
=================================================
Trains a LightGBM classifier on the feature matrix from build_features.py.

Follows Numeristical's proven methodology:
  - Gradient boosted trees, max_depth=2
  - Time-based train/test split (NEVER random)
  - Log loss as primary metric
  - Incremental feature layers to measure each improvement
  - SHAP analysis for feature importance

Usage:
  python scripts/train_win_model.py                          # Full training + evaluation
  python scripts/train_win_model.py --layer 1                # Only team hitting features
  python scripts/train_win_model.py --layer 4                # All layers
  python scripts/train_win_model.py --all-layers             # Train each layer incrementally
  python scripts/train_win_model.py --test-season 2025       # Custom test season

Author: Loko
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

# Fix Windows cp1252 encoding crashes when printing special characters
if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, brier_score_loss

warnings.filterwarnings("ignore")

# Try importing LightGBM, fall back to sklearn GBT
try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    HAS_LGBM = False
    print("LightGBM not installed. Using sklearn GradientBoosting (slower).")
    print("Install with: pip install lightgbm")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

# ===================================================================
# CONFIGURATION
# ===================================================================

FEATURES_PATH = Path(__file__).parent.parent / "data" / "features" / "game_features.csv"
MODELS_DIR = Path(__file__).parent.parent / "models"


def get_feature_layers():
    """
    Define feature layers in order of proven predictive value.
    Each layer is a list of column name patterns (prefix matching).
    """
    layers = {
        1: {
            "name": "Team Hitting",
            "description": "Rolling team runs scored, runs allowed, Pythagorean win%",
            "patterns": ["rs_t", "ra_t", "pyth_t", "diff_rs", "diff_pyth",
                         "team_era_t", "team_whip_t", "team_k_pct_t"],
        },
        2: {
            "name": "+ Starting Pitcher",
            "description": "SP ERA, K%, BB%, WHIP, FIP (rolling over last 5-10 starts)",
            "patterns": ["era_sp", "k_pct_sp", "bb_pct_sp", "whip_sp", "fip_sp", "ip_per_start"],
        },
        3: {
            "name": "+ Bullpen",
            "description": "Bullpen diff features (bp35 only — bp10 too noisy, raw cols redundant with team stats)",
            "patterns": ["diff_bp_era_bp35", "diff_bp_whip_bp35", "diff_bp_k_pct_bp35", "diff_bp_bb_pct_bp35"],
        },
        4: {
            "name": "+ Lineup",
            "description": "Average OBP and SLG of starting 9 batters",
            "patterns": ["lineup_obp", "lineup_slg"],
        },
        5: {
            "name": "+ Projections",
            "description": "Marcel+Statcast SP WAR/FIP/ERA/sustainability/K-BB and lineup projected wOBA",
            "patterns": ["proj_sp_war", "proj_sp_fip", "proj_sp_sc_era",
                         "proj_sp_sust", "proj_sp_breakout", "proj_sp_k_bb",
                         "proj_lineup_woba", "proj_lineup_bb_score"],
        },
    }
    return layers


def get_layer_features(df, layer_num):
    """Get feature columns for layers 1 through layer_num (inclusive)."""
    layers = get_feature_layers()
    all_patterns = []
    for i in range(1, layer_num + 1):
        if i in layers:
            all_patterns.extend(layers[i]["patterns"])

    feature_cols = []
    for col in df.columns:
        for pattern in all_patterns:
            if any(col.startswith(f"{prefix}{pattern}") or col.startswith(f"{prefix}_{pattern}")
                   for prefix in ["home_", "away_", "diff_"]):
                feature_cols.append(col)
                break
            if col.startswith(pattern):
                feature_cols.append(col)
                break

    return sorted(set(feature_cols))


# ===================================================================
# MODEL TRAINING
# ===================================================================

def train_model(X_train, y_train, X_val=None, y_val=None, max_depth=2):
    """Train a gradient boosted tree model."""

    if HAS_LGBM:
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
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

        model = lgb.LGBMClassifier(**params)

        if X_val is not None:
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
        else:
            model.fit(X_train, y_train)

    else:
        model = GradientBoostingClassifier(
            n_estimators=300,
            max_depth=max_depth,
            learning_rate=0.05,
            min_samples_leaf=50,
            subsample=0.8,
            verbose=0,
        )
        model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test, label=""):
    """Evaluate model on test set."""
    probs = model.predict_proba(X_test)[:, 1]  # P(home_win)

    ll = log_loss(y_test, probs)
    brier = brier_score_loss(y_test, probs)
    accuracy = ((probs > 0.5) == y_test).mean()

    # Calibration: how close are predicted probs to actual win rates?
    bins = np.linspace(0, 1, 11)
    bin_indices = np.digitize(probs, bins) - 1
    calibration = []
    for i in range(len(bins) - 1):
        mask = bin_indices == i
        if mask.sum() > 0:
            predicted_avg = probs[mask].mean()
            actual_avg = y_test.values[mask].mean()
            calibration.append({
                "bin": f"{bins[i]:.1f}-{bins[i+1]:.1f}",
                "count": int(mask.sum()),
                "predicted": predicted_avg,
                "actual": actual_avg,
                "error": abs(predicted_avg - actual_avg),
            })

    results = {
        "log_loss": ll,
        "brier_score": brier,
        "accuracy": accuracy,
        "n_test": len(y_test),
        "home_win_rate": float(y_test.mean()),
        "calibration": calibration,
    }

    return results, probs


def print_results(results, label=""):
    """Pretty print model evaluation results."""
    print(f"\n  {'=' * 50}")
    if label:
        print(f"  {label}")
        print(f"  {'=' * 50}")
    print(f"  Log Loss:     {results['log_loss']:.6f}")
    print(f"  Brier Score:  {results['brier_score']:.6f}")
    print(f"  Accuracy:     {results['accuracy']:.4f}")
    print(f"  Test games:   {results['n_test']}")
    print(f"  Home win rate: {results['home_win_rate']:.4f}")

    # Basis points from naive (always predict home_win_rate)
    home_rate = results["home_win_rate"]
    naive_ll = -(home_rate * np.log(home_rate) + (1 - home_rate) * np.log(1 - home_rate))
    bp_from_naive = (results["log_loss"] - naive_ll) * 10000
    print(f"  Naive log loss: {naive_ll:.6f}")
    print(f"  BP from naive:  {bp_from_naive:.1f}")

    print(f"\n  Calibration:")
    for cal in results.get("calibration", []):
        marker = "+" if cal["error"] < 0.05 else "-"
        print(f"    {cal['bin']:10s} | n={cal['count']:4d} | pred={cal['predicted']:.3f} | actual={cal['actual']:.3f} | {marker}")


def shap_analysis(model, X_test, feature_names, top_n=15):
    """Run SHAP analysis to understand feature importance."""
    if not HAS_SHAP:
        print("\n  SHAP not installed. Install with: pip install shap")
        importances = model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        print(f"\n  Top {top_n} features (by split importance):")
        for i in range(min(top_n, len(feature_names))):
            idx = sorted_idx[i]
            print(f"    {i+1:2d}. {feature_names[idx]:45s} | importance={importances[idx]:.0f}")
        return

    print(f"\n  Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    sorted_idx = np.argsort(mean_abs_shap)[::-1]

    print(f"\n  Top {top_n} features (by mean |SHAP|):")
    for i in range(min(top_n, len(feature_names))):
        idx = sorted_idx[i]
        print(f"    {i+1:2d}. {feature_names[idx]:45s} | SHAP={mean_abs_shap[idx]:.6f}")


# ===================================================================
# MAIN
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="Train win probability model")
    parser.add_argument("--layer", type=int, default=5, help="Max feature layer (1-5)")
    parser.add_argument("--test-season", type=int, default=2025, help="Test season")
    parser.add_argument("--all-layers", action="store_true", help="Train each layer incrementally")
    parser.add_argument("--max-depth", type=int, default=2, help="Tree max depth")
    parser.add_argument("--shap", action="store_true", help="Run SHAP analysis")
    parser.add_argument("--save", action="store_true", help="Save best model")
    args = parser.parse_args()

    print("=" * 70)
    print("WIN PROBABILITY MODEL TRAINING")
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

    # Time-based split
    test_season = args.test_season
    train_seasons = [s for s in df["season"].unique() if s < test_season]

    if not train_seasons:
        print(f"\nNo training data before season {test_season}!")
        print(f"Available seasons: {sorted(df['season'].unique())}")
        return

    train_df = df[df["season"].isin(train_seasons)]
    test_df = df[df["season"] == test_season]

    if len(test_df) == 0:
        print(f"\nNo test data for season {test_season}!")
        return

    print(f"\n  Train: seasons {sorted(train_seasons)} ({len(train_df)} games)")
    print(f"  Test:  season {test_season} ({len(test_df)} games)")

    target = "home_win"

    # Determine which layers to train
    if args.all_layers:
        layers_to_train = list(range(1, args.layer + 1))
    else:
        layers_to_train = [args.layer]

    layers = get_feature_layers()
    best_ll = float("inf")
    best_model = None
    best_features = None
    all_results = []

    for layer_num in layers_to_train:
        feature_cols = get_layer_features(df, layer_num)

        if not feature_cols:
            print(f"\n  Layer {layer_num}: No matching features found in data!")
            continue

        # Build description
        layer_names = " + ".join(layers[i]["name"] for i in range(1, layer_num + 1) if i in layers)

        print(f"\n{'=' * 70}")
        print(f"  LAYER {layer_num}: {layer_names}")
        print(f"  Features: {len(feature_cols)}")
        print(f"{'=' * 70}")

        # Prepare data
        X_train = train_df[feature_cols].copy()
        y_train = train_df[target].copy()
        X_test = test_df[feature_cols].copy()
        y_test = test_df[target].copy()

        # Handle missing values
        for col in feature_cols:
            median_val = X_train[col].median()
            X_train[col] = X_train[col].fillna(median_val)
            X_test[col] = X_test[col].fillna(median_val)

        # Drop any rows where target is missing
        valid_train = y_train.notna()
        X_train = X_train[valid_train]
        y_train = y_train[valid_train].astype(int)

        valid_test = y_test.notna()
        X_test = X_test[valid_test]
        y_test = y_test[valid_test].astype(int)

        # Train
        print(f"  Training (max_depth={args.max_depth})...")
        model = train_model(X_train, y_train, max_depth=args.max_depth)

        # Evaluate
        results, probs = evaluate_model(model, X_test, y_test)
        print_results(results, label=f"Layer {layer_num}: {layer_names}")

        all_results.append({
            "layer": layer_num,
            "name": layer_names,
            "log_loss": results["log_loss"],
            "accuracy": results["accuracy"],
            "n_features": len(feature_cols),
        })

        # Track best
        if results["log_loss"] < best_ll:
            best_ll = results["log_loss"]
            best_model = model
            best_features = feature_cols

        # Feature importance for this layer
        importances = model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        print(f"\n  Top features:")
        for i in range(min(10, len(feature_cols))):
            idx = sorted_idx[i]
            print(f"    {i+1:2d}. {feature_cols[idx]:45s} | imp={importances[idx]:.0f}")

    # SHAP analysis on best model
    if args.shap and best_model is not None:
        X_test_best = test_df[best_features].copy()
        for col in best_features:
            median_val = train_df[col].median()
            X_test_best[col] = X_test_best[col].fillna(median_val)

        shap_analysis(best_model, X_test_best, best_features)

    # Save best model
    if args.save and best_model is not None:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        if HAS_LGBM:
            model_path = MODELS_DIR / "win_probability_lgbm.txt"
            best_model.booster_.save_model(str(model_path))
        else:
            import joblib
            model_path = MODELS_DIR / "win_probability_sklearn.pkl"
            joblib.dump(best_model, str(model_path))

        # Save feature list
        meta = {
            "features": best_features,
            "log_loss": best_ll,
            "test_season": test_season,
            "max_depth": args.max_depth,
        }
        meta_path = MODELS_DIR / "win_probability_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        print(f"\n  Model saved to: {model_path}")
        print(f"  Metadata saved to: {meta_path}")

    # Summary table
    if len(all_results) > 1:
        print(f"\n{'=' * 70}")
        print(f"  LAYER-BY-LAYER SUMMARY")
        print(f"{'=' * 70}")
        print(f"  {'Layer':<8} {'Name':<40} {'LogLoss':>10} {'Accuracy':>10} {'Features':>10}")
        print(f"  {'-'*8} {'-'*40} {'-'*10} {'-'*10} {'-'*10}")
        prev_ll = None
        for r in all_results:
            delta = ""
            if prev_ll is not None:
                diff_bp = (r["log_loss"] - prev_ll) * 10000
                delta = f" ({diff_bp:+.1f} bp)"
            print(f"  {r['layer']:<8} {r['name']:<40} {r['log_loss']:>10.6f}{delta:<12} {r['accuracy']:>8.4f} {r['n_features']:>10}")
            prev_ll = r["log_loss"]

    print(f"\n{'=' * 70}")
    print(f"  BEST LOG LOSS: {best_ll:.6f}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
