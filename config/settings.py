"""
Central configuration for MLB Analytics Platform.
All settings in one place. No magic numbers in code.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─── Paths ────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
EXTERNAL_DIR = DATA_DIR / "external"

# ─── Database ─────────────────────────────────────────────
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    f"sqlite:///{DATA_DIR / 'mlb_analytics.db'}"
)

# ─── MLB API ──────────────────────────────────────────────
MLB_API_BASE = os.getenv("MLB_API_BASE", "https://statsapi.mlb.com/api/v1")
MLB_CURRENT_SEASON = 2026
MLB_HISTORICAL_SEASONS = [2023, 2024, 2025]

# ─── Model Parameters ────────────────────────────────────
# How much each source contributes to final model
SOURCE_WEIGHTS = {
    "fangraphs": 0.50,
    "stats_based": 0.30,
    "user_criteria": 0.20,
}

# Rolling window sizes (in days)
ROLLING_WINDOWS = [7, 14, 30]

# ─── Feature Engineering ─────────────────────────────────
# Minimum sample sizes (NEVER use stats below these thresholds)
MIN_PA_HITTER = 50          # Plate appearances
MIN_IP_PITCHER = 20         # Innings pitched  
MIN_PA_SPLIT = 100          # For platoon/situational splits
MIN_IP_SPLIT = 50           # For pitcher splits

# Regression targets (league averages for mean reversion)
LEAGUE_AVG_BABIP = 0.300
LEAGUE_AVG_HR_FB = 0.120
LEAGUE_AVG_LOB_PCT = 0.720
LEAGUE_AVG_K_PCT = 0.224
LEAGUE_AVG_BB_PCT = 0.085

# ─── Betting (Phase 3) ───────────────────────────────────
MIN_EV_THRESHOLD = 0.03     # Minimum 3% EV to flag a bet
KELLY_FRACTION = 0.25       # Quarter Kelly for conservative sizing
BANKROLL = 100_000          # In ARS or your currency. Adjust.