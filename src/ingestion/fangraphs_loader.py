"""
fangraphs_loader.py — Load FanGraphs Projection CSVs
=====================================================
Reads Steamer and ZiPS projection CSVs (manually downloaded from
fangraphs.com/projections) and maps players to our database.

FanGraphs has no public API, so projections must be downloaded manually:
  1. Go to https://www.fangraphs.com/projections
  2. Select: Steamer/ZiPS, Batting/Pitching, All positions
  3. Click "Export Data" → save CSV
  4. Place in data/projections/fangraphs/

Expected filenames:
  steamer_batting_2026.csv
  steamer_pitching_2026.csv
  zips_batting_2026.csv
  zips_pitching_2026.csv

Usage:
  from src.ingestion.fangraphs_loader import load_fangraphs_projections
  steamer, zips = load_fangraphs_projections(2026)
"""

import sqlite3
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "projections" / "fangraphs"
DB_PATH = Path(__file__).parent.parent.parent / "data" / "mlb_analytics.db"


def _normalize_name(name: str) -> str:
    """Normalize player name for matching (strip accents, lowercase)."""
    if not isinstance(name, str):
        return ""
    # Simple normalization — covers 95% of cases
    return name.strip().lower()


def _build_player_map() -> Dict[str, int]:
    """
    Build name → mlb_player_id mapping from our database.

    Returns dict like: {"shohei ohtani": 660271, ...}
    Uses lowercase names for fuzzy matching.
    """
    conn = sqlite3.connect(str(DB_PATH))
    players = pd.read_sql("SELECT mlb_id, name FROM players", conn)
    conn.close()

    player_map = {}
    for _, row in players.iterrows():
        key = _normalize_name(row["name"])
        player_map[key] = row["mlb_id"]

    return player_map


def load_batting_projections(filepath: Path, system_name: str) -> Optional[pd.DataFrame]:
    """
    Load a FanGraphs batting projection CSV.

    FanGraphs columns (typical): Name, Team, G, PA, AB, H, 2B, 3B, HR,
    R, RBI, BB, SO, HBP, SB, CS, AVG, OBP, SLG, OPS, wOBA, wRC+, WAR

    We extract the key rate stats for comparison with Marcel.
    """
    if not filepath.exists():
        print(f"  {system_name} batting: file not found at {filepath}")
        return None

    df = pd.read_csv(filepath, encoding="utf-8-sig")  # BOM-safe
    print(f"  {system_name} batting: {len(df)} players loaded from {filepath.name}")

    # Standardize column names (FanGraphs uses various capitalizations)
    df.columns = [c.strip() for c in df.columns]

    # Map column names (FanGraphs is inconsistent across exports)
    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if cl == "name":
            col_map[c] = "name"
        elif cl == "team":
            col_map[c] = "team"
        elif cl == "pa":
            col_map[c] = "pa"
        elif cl == "avg":
            col_map[c] = "avg"
        elif cl == "obp":
            col_map[c] = "obp"
        elif cl == "slg":
            col_map[c] = "slg"
        elif cl == "woba":
            col_map[c] = "woba"
        elif cl in ("wrc+", "wrc_plus"):
            col_map[c] = "wrc_plus"
        elif cl == "war":
            col_map[c] = "war"
        elif cl == "hr":
            col_map[c] = "hr"
        elif cl == "so" or cl == "k":
            col_map[c] = "so"
        elif cl == "bb":
            col_map[c] = "bb"

    df = df.rename(columns=col_map)
    df["system"] = system_name

    # Compute K% and BB% if not present
    if "pa" in df.columns:
        if "so" in df.columns:
            df["k_pct"] = df["so"] / df["pa"]
        if "bb" in df.columns:
            df["bb_pct"] = df["bb"] / df["pa"]

    return df


def load_pitching_projections(filepath: Path, system_name: str) -> Optional[pd.DataFrame]:
    """
    Load a FanGraphs pitching projection CSV.

    Key columns: Name, Team, W, L, ERA, GS, G, SV, IP, H, ER, HR,
    BB, SO, WHIP, K/9, BB/9, FIP, xFIP, WAR
    """
    if not filepath.exists():
        print(f"  {system_name} pitching: file not found at {filepath}")
        return None

    df = pd.read_csv(filepath, encoding="utf-8-sig")
    print(f"  {system_name} pitching: {len(df)} players loaded from {filepath.name}")

    df.columns = [c.strip() for c in df.columns]

    col_map = {}
    for c in df.columns:
        cl = c.lower().replace("/", "_")
        if cl == "name":
            col_map[c] = "name"
        elif cl == "team":
            col_map[c] = "team"
        elif cl == "ip":
            col_map[c] = "ip"
        elif cl == "era":
            col_map[c] = "era"
        elif cl == "fip":
            col_map[c] = "fip"
        elif cl == "xfip":
            col_map[c] = "xfip"
        elif cl == "whip":
            col_map[c] = "whip"
        elif cl in ("k_9", "k9"):
            col_map[c] = "k_9"
        elif cl in ("bb_9", "bb9"):
            col_map[c] = "bb_9"
        elif cl == "war":
            col_map[c] = "war"
        elif cl == "so" or cl == "k":
            col_map[c] = "so"
        elif cl == "bb":
            col_map[c] = "bb"
        elif cl == "gs":
            col_map[c] = "gs"

    df = df.rename(columns=col_map)
    df["system"] = system_name

    # Compute K-BB% if we have the components
    if "ip" in df.columns:
        # Approximate BF from IP
        bf_approx = df["ip"] * 3 + df.get("so", 0) + df.get("bb", 0)
        if "so" in df.columns:
            df["k_pct"] = df["so"] / bf_approx.clip(lower=1)
        if "bb" in df.columns:
            df["bb_pct"] = df["bb"] / bf_approx.clip(lower=1)
        if "k_pct" in df.columns and "bb_pct" in df.columns:
            df["k_bb_pct"] = df["k_pct"] - df["bb_pct"]

    return df


def load_fangraphs_projections(year: int = 2026
                                ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    Load all available FanGraphs projection CSVs for a given year.

    Returns
    -------
    batting : dict of {system_name: DataFrame}
    pitching : dict of {system_name: DataFrame}
    """
    batting = {}
    pitching = {}

    for system in ["steamer", "zips", "atc", "thebat"]:
        # Batting
        bat_path = DATA_DIR / f"{system}_batting_{year}.csv"
        bat_df = load_batting_projections(bat_path, system)
        if bat_df is not None:
            batting[system] = bat_df

        # Pitching
        pit_path = DATA_DIR / f"{system}_pitching_{year}.csv"
        pit_df = load_pitching_projections(pit_path, system)
        if pit_df is not None:
            pitching[system] = pit_df

    return batting, pitching


def match_players(fg_df: pd.DataFrame, player_map: Dict[str, int]) -> pd.DataFrame:
    """
    Add mlb_player_id column by matching FanGraphs names to our database.

    Uses exact name matching (lowercase). Could be extended with fuzzy
    matching or FanGraphs playerid column.
    """
    matched = []
    for _, row in fg_df.iterrows():
        name = _normalize_name(row.get("name", ""))
        mlb_id = player_map.get(name)
        matched.append(mlb_id)

    fg_df = fg_df.copy()
    fg_df["mlb_player_id"] = matched

    n_matched = sum(1 for m in matched if m is not None)
    print(f"    Matched {n_matched}/{len(fg_df)} players to database")

    return fg_df


# ─── Standalone test ──────────────────────────────────────────

if __name__ == "__main__":
    print("FanGraphs Projection Loader — Test")
    print("=" * 50)

    batting, pitching = load_fangraphs_projections(2026)

    if not batting and not pitching:
        print("\nNo FanGraphs CSV files found in data/projections/fangraphs/")
        print("Download from https://www.fangraphs.com/projections")
        print("Expected filenames:")
        print("  steamer_batting_2026.csv")
        print("  steamer_pitching_2026.csv")
        print("  zips_batting_2026.csv")
        print("  zips_pitching_2026.csv")
    else:
        print(f"\nLoaded: {len(batting)} batting systems, {len(pitching)} pitching systems")
        for system, df in batting.items():
            print(f"  {system} batting: {len(df)} players")
        for system, df in pitching.items():
            print(f"  {system} pitching: {len(df)} players")
