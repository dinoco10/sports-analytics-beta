"""
compare_projections.py — Marcel vs FanGraphs Projection Comparison
==================================================================
Compares our Marcel projections against Steamer, ZiPS, and other
FanGraphs systems to identify systematic biases and opportunities.

Outputs:
  - Correlation and bias statistics for key metrics
  - Top 20 biggest disagreements (where Marcel differs most from consensus)
  - Ensemble exploration (Marcel + Steamer blend)

Usage:
  python scripts/compare_projections.py               # Full comparison
  python scripts/compare_projections.py --year 2026   # Specific year
  python scripts/compare_projections.py --top 30      # More disagreements

Requires:
  - FanGraphs CSVs in data/projections/fangraphs/
  - Marcel snapshots in data/features/snapshots/ or data/features/
"""

import argparse
import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.fangraphs_loader import (
    load_fangraphs_projections, match_players, _build_player_map
)


FEATURES_DIR = Path(__file__).parent.parent / "data" / "features"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "projections"


def load_marcel_hitters(year: int) -> pd.DataFrame:
    """Load Marcel hitter projections for the given year."""
    # Try snapshot first, then production
    snapshot = FEATURES_DIR / "snapshots" / f"marcel_hitters_{year}.csv"
    prod = FEATURES_DIR / f"hitter_projections_{year}.csv"

    path = snapshot if snapshot.exists() else prod
    if not path.exists():
        print(f"  Marcel hitters not found for {year}")
        return pd.DataFrame()

    df = pd.read_csv(path)
    print(f"  Marcel hitters: {len(df)} from {path.name}")
    return df


def load_marcel_pitchers(year: int) -> pd.DataFrame:
    """Load Marcel pitcher projections for the given year."""
    snapshot = FEATURES_DIR / "snapshots" / f"marcel_pitchers_{year}.csv"
    prod = FEATURES_DIR / f"pitcher_projections_{year}.csv"

    path = snapshot if snapshot.exists() else prod
    if not path.exists():
        print(f"  Marcel pitchers not found for {year}")
        return pd.DataFrame()

    df = pd.read_csv(path)
    print(f"  Marcel pitchers: {len(df)} from {path.name}")
    return df


def compare_hitters(marcel_df, fg_df, system_name, top_n=20):
    """Compare Marcel vs FanGraphs hitter projections."""
    print(f"\n{'=' * 70}")
    print(f"  HITTER COMPARISON: Marcel vs {system_name}")
    print(f"{'=' * 70}")

    # Match on mlb_player_id
    player_map = _build_player_map()
    fg_matched = match_players(fg_df, player_map)

    # Merge
    fg_cols = ["mlb_player_id", "name"]
    marcel_cols = ["mlb_player_id"]

    for col in ["woba", "war", "obp", "slg", "k_pct", "bb_pct"]:
        if col in fg_matched.columns:
            fg_cols.append(col)

    for col in ["statcast_adjusted_woba", "proj_war", "proj_obp", "proj_slg",
                "proj_k_pct", "proj_bb_pct"]:
        if col in marcel_df.columns:
            marcel_cols.append(col)

    fg_sub = fg_matched[fg_cols].dropna(subset=["mlb_player_id"])
    marcel_sub = marcel_df[marcel_cols].dropna(subset=["mlb_player_id"])

    merged = fg_sub.merge(marcel_sub, on="mlb_player_id", how="inner")
    print(f"\n  Matched players: {len(merged)}")

    if len(merged) < 10:
        print("  Too few matched players for meaningful comparison")
        return

    # Compare wOBA
    if "woba" in merged.columns and "statcast_adjusted_woba" in merged.columns:
        fg_woba = merged["woba"]
        marcel_woba = merged["statcast_adjusted_woba"]

        corr = fg_woba.corr(marcel_woba)
        bias = (marcel_woba - fg_woba).mean()
        mae = (marcel_woba - fg_woba).abs().mean()

        print(f"\n  wOBA comparison (n={len(merged)}):")
        print(f"    Correlation:  {corr:.4f}")
        print(f"    Bias:         {bias:+.4f} (Marcel - {system_name})")
        print(f"    MAE:          {mae:.4f}")
        print(f"    Marcel mean:  {marcel_woba.mean():.4f}")
        print(f"    {system_name:8s} mean:  {fg_woba.mean():.4f}")

        # Top disagreements
        merged["woba_diff"] = marcel_woba - fg_woba
        merged["abs_woba_diff"] = merged["woba_diff"].abs()
        top_diff = merged.nlargest(top_n, "abs_woba_diff")

        print(f"\n  Top {top_n} wOBA disagreements:")
        for _, row in top_diff.iterrows():
            direction = "Marcel higher" if row["woba_diff"] > 0 else f"{system_name} higher"
            print(f"    {row['name']:25s} | Marcel={row['statcast_adjusted_woba']:.3f} "
                  f"| {system_name}={row['woba']:.3f} | diff={row['woba_diff']:+.3f} ({direction})")

    # Compare WAR
    if "war" in merged.columns and "proj_war" in merged.columns:
        fg_war = merged["war"]
        marcel_war = merged["proj_war"]

        corr = fg_war.corr(marcel_war)
        bias = (marcel_war - fg_war).mean()

        print(f"\n  WAR comparison (n={len(merged)}):")
        print(f"    Correlation:  {corr:.4f}")
        print(f"    Bias:         {bias:+.2f} (Marcel - {system_name})")
        print(f"    Marcel mean:  {marcel_war.mean():.2f}")
        print(f"    {system_name:8s} mean:  {fg_war.mean():.2f}")


def compare_pitchers(marcel_df, fg_df, system_name, top_n=20):
    """Compare Marcel vs FanGraphs pitcher projections."""
    print(f"\n{'=' * 70}")
    print(f"  PITCHER COMPARISON: Marcel vs {system_name}")
    print(f"{'=' * 70}")

    player_map = _build_player_map()
    fg_matched = match_players(fg_df, player_map)

    fg_cols = ["mlb_player_id", "name"]
    marcel_cols = ["mlb_player_id"]

    for col in ["era", "fip", "war", "k_bb_pct", "whip"]:
        if col in fg_matched.columns:
            fg_cols.append(col)

    for col in ["proj_era", "proj_fip", "proj_war", "proj_k_bb_pct", "proj_whip"]:
        if col in marcel_df.columns:
            marcel_cols.append(col)

    fg_sub = fg_matched[fg_cols].dropna(subset=["mlb_player_id"])
    marcel_sub = marcel_df[marcel_cols].dropna(subset=["mlb_player_id"])

    merged = fg_sub.merge(marcel_sub, on="mlb_player_id", how="inner")
    print(f"\n  Matched pitchers: {len(merged)}")

    if len(merged) < 10:
        print("  Too few matched pitchers for meaningful comparison")
        return

    # Compare ERA
    if "era" in merged.columns and "proj_era" in merged.columns:
        fg_era = merged["era"]
        marcel_era = merged["proj_era"]

        corr = fg_era.corr(marcel_era)
        bias = (marcel_era - fg_era).mean()
        mae = (marcel_era - fg_era).abs().mean()

        print(f"\n  ERA comparison (n={len(merged)}):")
        print(f"    Correlation:  {corr:.4f}")
        print(f"    Bias:         {bias:+.4f} (Marcel - {system_name})")
        print(f"    MAE:          {mae:.4f}")

    # Compare FIP
    if "fip" in merged.columns and "proj_fip" in merged.columns:
        fg_fip = merged["fip"]
        marcel_fip = merged["proj_fip"]

        corr = fg_fip.corr(marcel_fip)
        bias = (marcel_fip - fg_fip).mean()

        print(f"\n  FIP comparison (n={len(merged)}):")
        print(f"    Correlation:  {corr:.4f}")
        print(f"    Bias:         {bias:+.4f} (Marcel - {system_name})")

    # Compare K-BB%
    if "k_bb_pct" in merged.columns and "proj_k_bb_pct" in merged.columns:
        fg_kbb = merged["k_bb_pct"]
        marcel_kbb = merged["proj_k_bb_pct"]

        corr = fg_kbb.corr(marcel_kbb)
        bias = (marcel_kbb - fg_kbb).mean()

        print(f"\n  K-BB% comparison (n={len(merged)}):")
        print(f"    Correlation:  {corr:.4f}")
        print(f"    Bias:         {bias:+.2f} (Marcel - {system_name})")


def main():
    parser = argparse.ArgumentParser(description="Compare Marcel vs FanGraphs projections")
    parser.add_argument("--year", type=int, default=2026, help="Projection year")
    parser.add_argument("--top", type=int, default=20, help="Top N disagreements")
    args = parser.parse_args()

    print("=" * 70)
    print(f"PROJECTION COMPARISON — Marcel vs FanGraphs ({args.year})")
    print("=" * 70)

    # Load Marcel
    print("\n[1] Loading Marcel projections...")
    marcel_hitters = load_marcel_hitters(args.year)
    marcel_pitchers = load_marcel_pitchers(args.year)

    # Load FanGraphs
    print("\n[2] Loading FanGraphs projections...")
    fg_batting, fg_pitching = load_fangraphs_projections(args.year)

    if not fg_batting and not fg_pitching:
        print("\nNo FanGraphs data found. Download CSVs first:")
        print("  1. Go to https://www.fangraphs.com/projections")
        print("  2. Export Steamer/ZiPS batting and pitching")
        print("  3. Save to data/projections/fangraphs/")
        print("  4. Name as: steamer_batting_2026.csv, etc.")
        return

    # Compare hitters
    if len(marcel_hitters) > 0:
        for system, fg_df in fg_batting.items():
            compare_hitters(marcel_hitters, fg_df, system.capitalize(), args.top)

    # Compare pitchers
    if len(marcel_pitchers) > 0:
        for system, fg_df in fg_pitching.items():
            compare_pitchers(marcel_pitchers, fg_df, system.capitalize(), args.top)

    print(f"\n{'=' * 70}")
    print("  DONE")
    print(f"{'=' * 70}")
    print("  Download FanGraphs CSVs and re-run for full comparison.")
    print("  Once compared, consider ensemble: Marcel + Steamer (50/50)")


if __name__ == "__main__":
    main()
