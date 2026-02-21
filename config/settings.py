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

# ─── WAR / fWAR Calculation ──────────────────────────────
# Position adjustments per 162 games (runs above/below average).
# Based on FanGraphs positional adjustments.
# Positive = harder position (more valuable), negative = easier.
POSITION_ADJUSTMENT_RUNS = {
    "C":   +12.5,   # Catcher — hardest position, biggest bonus
    "SS":  +7.5,    # Shortstop
    "2B":  +2.5,    # Second base
    "3B":  +2.5,    # Third base
    "CF":  +2.5,    # Center field
    "LF":  -7.5,    # Left field
    "RF":  -7.5,    # Right field
    "1B":  -12.5,   # First base — easiest position
    "DH":  -17.5,   # Designated hitter — no defensive value at all
}

# League averages for wRC+ calculation
LEAGUE_AVG_WOBA = 0.310        # 2023-2025 average
LEAGUE_AVG_WOBA_SCALE = 1.15   # wOBA to runs conversion factor
LEAGUE_AVG_R_PA = 0.120        # League runs per plate appearance

# WAR constants
RUNS_PER_WIN = 10              # ~10 runs = 1 win (standard)
REPLACEMENT_LEVEL_WINS = 48    # Replacement-level team wins ~48 games

# ─── Statcast League Averages (Marcel Regression Targets) ──
# Used when regressing Marcel-weighted Statcast metrics toward the mean.
# Values represent ~2023-2025 MLB averages from Baseball Savant.

LEAGUE_AVG_STATCAST_HITTER = {
    'avg_exit_velocity': 88.5,   # mph
    'barrel_rate': 8.5,          # % of BBE
    'hard_hit_rate': 37.0,       # balls 95+ mph %
    'xwoba': 0.310,
    'xslg': 0.390,
    'xba': 0.248,
    'launch_angle_sweet_spot_pct': 33.0,  # balls 8-32 degrees %
    'chase_rate': 28.0,          # O-Swing %
    'whiff_rate': 25.0,          # swing & miss %
    'z_contact_rate': 82.0,      # contact on strikes %
    'pull_air_rate': 4.0,        # pulled fly ball % (key HR predictor)
    'gb_rate': 43.0,             # ground ball %
    'fb_rate': 35.0,             # fly ball %
    'ld_rate': 22.0,             # line drive %
    'babip': 0.300,
    'hr_per_fb': 12.0,           # HR / fly ball %
}

LEAGUE_AVG_STATCAST_PITCHER = {
    'avg_exit_velocity_against': 88.5,
    'barrel_rate_against': 8.5,
    'hard_hit_rate_against': 37.0,
    'xwoba_against': 0.310,
    'xera': 4.20,
    'whiff_rate': 25.0,
    'swstr_rate': 11.0,          # swinging strike %
    'chase_rate_induced': 28.0,  # O-Swing % induced
    'z_contact_rate_against': 82.0,
    'pull_air_rate_against': 4.0,
    'gb_rate': 43.0,
    'fb_rate': 35.0,
    'ld_rate': 22.0,
    'babip_against': 0.300,
    'hr_per_fb': 12.0,
    'k_minus_bb': 10.0,          # K-BB% league average
    'arsenal_depth_score': 1.5,   # Avg pitches with usage>10% AND RV<0
    'fastball_velocity': 93.9,    # Avg fastball velocity (FF/SI)
    'best_secondary_rv': -0.63,   # Avg best secondary run value per 100
}

# ─── Betting (Phase 3) ───────────────────────────────────
MIN_EV_THRESHOLD = 0.03     # Minimum 3% EV to flag a bet
KELLY_FRACTION = 0.25       # Quarter Kelly for conservative sizing
BANKROLL = 100_000          # In ARS or your currency. Adjust.