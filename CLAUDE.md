# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MLB analytics platform with game-level win probability, run total predictions, player projections, and season win total forecasting. Python 3.12+, SQLAlchemy 2.0+, SQLite. Models beat Vegas on both moneyline and O/U.

## Key Commands

```bash
# Setup
python -m venv venv && venv\Scripts\activate  # Windows
pip install -r requirements.txt
python scripts/initialize_db.py
python scripts/seed_teams.py

# Data ingestion
python scripts/backfill_games.py                # 2024-2025 seasons
python scripts/backfill_games.py --season 2026  # Single season
python scripts/backfill_games.py --summary      # Check DB counts

# Feature engineering
python scripts/build_features.py                # Full feature matrix → game_features.csv
python scripts/build_features.py --season 2025

# Model training
python scripts/train_win_model.py --pruned --save          # Win probability (48 features)
python scripts/train_run_model.py --save                   # Run totals (52 features)

# Ablation testing
python scripts/ablation_run_model.py           # Drop-one feature ablation (run model)
python scripts/ablation_bp_fatigue.py          # Bullpen fatigue test (win prob)

# Player projections (Marcel + Statcast)
python -m src.features.player_projections      # Hitter + pitcher projections
python -m src.features.depth_chart             # PA/IP allocation + team WAR

# Season win projections
python scripts/project_season_wins.py          # Full pipeline (A+B+C)
python scripts/project_season_wins.py --phase A  # Calibrated WAR only
python scripts/project_season_wins.py --sims 50000

# Daily predictions
python scripts/predict_today.py                    # Today's games
python scripts/predict_today.py --date 2026-04-15  # Specific date
python scripts/predict_today.py --backtest 2025-09-28

# Test connections
python -m src.ingestion.mlb_api
python -c "from src.storage.database import test_connection; test_connection()"
```

## Rules
- Always test features layer-by-layer, measuring log loss / MAE impact
- Never change working code without explicit permission
- Use max_depth=2 for LightGBM (num_leaves=3)
- Explain every code section as if I'm learning Python
- Chronological train/test splits only (no random splits)
- Train on 2021-2024, test on 2025 (skip COVID 2020)
- 5-seed averaging for ablation studies (seeds: 42, 123, 456, 789, 2025)

## Architecture

### Data Flow
```
MLB Stats API → src/ingestion/mlb_api.py → SQLite DB
    ↓
scripts/build_features.py → data/features/game_features.csv
    ↓
scripts/train_win_model.py  → models/win_probability_lgbm.txt  (48 features, 0.6667 log loss)
scripts/train_run_model.py  → models/run_home_lgbm.txt         (52 features, 3.531 MAE)
                              models/run_away_lgbm.txt
    ↓
scripts/predict_today.py → daily win probability predictions
```

### Player Projection Flow
```
DB (2023-2025 stats) → src/features/player_projections.py → Marcel projections
    ↓
src/features/depth_chart.py → PA/IP allocation → team WAR
    ↓
scripts/project_season_wins.py → calibrated wins + Monte Carlo → season projections
```

### Core Modules
- **config/settings.py** — Central config: DB URL, API base, rolling windows [14, 30], league averages, MLB_CURRENT_SEASON=2026
- **src/ingestion/mlb_api.py** — MLB Stats API client with 0.5s rate limiting (v1.1 for game feeds)
- **src/storage/models.py** — SQLAlchemy ORM: teams, players, games, pitching/batting/bullpen logs, lineups, ballparks, umpires
- **src/storage/database.py** — DB session management via `get_session()` context manager
- **scripts/build_features.py** — Full feature pipeline: team rolling, SP rolling, bullpen, lineup, projections, Elo, handedness, venue splits, weather, park factors → game_features.csv
- **scripts/train_win_model.py** — LightGBM classifier, 48 pruned features, binary log loss
- **scripts/train_run_model.py** — Two LightGBM regressors (home/away), 52 features, MAE objective, +0.1 bias correction
- **scripts/predict_today.py** — Daily prediction pipeline: fetches schedule, computes features, runs model
- **src/features/player_projections.py** — Marcel method (5:4:3 weighted), Statcast adjustments, aging curves
- **src/features/depth_chart.py** — Position-based PA/IP allocation, team WAR aggregation
- **scripts/project_season_wins.py** — WAR→wins calibration, Monte Carlo simulation, Vegas benchmark

## PROJECT STRUCTURE

```
sports-analytics/
├── config/                    # Settings, weights
├── src/
│   ├── ingestion/             # MLB API client
│   ├── storage/               # SQLAlchemy models, DB session
│   ├── features/              # Player projections, depth chart, team features
│   ├── models/                # Power rankings (legacy)
│   ├── betting/               # (stub — planned)
│   └── evaluation/            # (stub — planned)
├── scripts/
│   ├── backfill_games.py      # Game data ingestion
│   ├── build_features.py      # Feature matrix pipeline
│   ├── train_win_model.py     # Win probability model
│   ├── train_run_model.py     # Run total model
│   ├── predict_today.py       # Daily predictions
│   ├── project_season_wins.py # Season projections
│   ├── generate_marcel_snapshots.py
│   └── ablation_*.py          # Feature ablation scripts
├── models/                    # Saved LightGBM models + metadata
├── data/
│   ├── mlb_analytics.db       # SQLite database (~59MB)
│   ├── features/              # CSVs: game_features, projections, depth charts
│   ├── external/              # Schedule cache, Vegas lines
│   └── odds/                  # Historical odds (2010-2021)
└── CLAUDE.md
```

Database: `data/mlb_analytics.db`

## DATA ARCHITECTURE CONTRACT

Never read directly from raw_* tables in model code.
Floor 1 models read ONLY from game_features.csv (via build_features.py).
Projections come ONLY from batter/pitcher_projections CSVs.
Odds data lives in data/odds/ (raw Excel) and odds table (processed).

When adding new features:
1. Add raw metric to raw_* table first
2. Add Marcel-weighted projection to batter/pitcher_projections
3. Add aggregated game-level feature to game_features
4. Only then update model feature list

Layer order: raw → projections → game_features → model → output

### Database
SQLite at `data/mlb_analytics.db`. Schema uses foreign keys with cascading, unique constraints to prevent duplicates, indexes on (date, season, team_id). Batch commits every 50 records.

## Model Status

### Floor 1: Win Probability — DONE
- LightGBM classifier, 48 pruned features, **0.6667 log loss** (5-seed avg)
- Beats Vegas by **118 basis points** on 2025 test set
- Optimal ensemble: 85% model / 15% Vegas
- Top features: home_platoon_adv (imp=125), home_elo (67), diff_bp_bb_pct (37)
- Elo: K=4, home_adv=24, mean=1500, season_revert=0.33

### Floor 1+: Run Total Model — DONE
- Two LightGBM regressors (home/away), 52 features, **MAE=3.531** (bias-corrected)
- Beats Vegas MAE of 3.552 by **21 points**
- Bias correction: +0.1 per side from training residuals
- Negative binomial distribution for O/U probability
- Top features: away_elo (129), away_k_pct_sp10 (85), game_temperature (66)

### Floor 2: Player Projections — DONE
- Marcel method (5:4:3 year weighting) + Statcast adjustments
- ~950 hitters, ~450 pitchers projected for 2026
- Depth chart handles PA/IP allocation by position hierarchy
- Bounce-back, regression risk, sustainability, breakout scores

### Floor 3: Season Win Projections — DONE
- WAR→wins calibration via z-score mapping (no ML — 30 teams/year would overfit)
- Monte Carlo simulation (10K+ sims) with Log5 per-game probabilities
- MAE vs Vegas: **3.8 wins** (updated March 2026)
- Division title and playoff probabilities per team

## Git Workflow

**Auto-commit after every major change.** After completing a significant piece of work:
1. Create a feature branch from main (e.g., `feature/run-model-improvements`)
2. Stage and commit the relevant files with a descriptive message
3. Push the branch and create a PR to main

Branch naming: `feature/`, `fix/`, `refactor/`, or `data/` prefix + short kebab-case description.

**Important**: LightGBM model binaries (.txt) corrupt on git checkout across branches. After any merge/checkout, retrain models with `--save` flag.

## Conventions

- SQLAlchemy ORM for all queries (no raw SQL)
- DataFrames for data processing; merge on `team` column (standardized name)
- Feature columns use descriptive prefixes (e.g., `home_era_sp10`, `diff_bp_whip_bp35`)
- Modules are runnable via `python -m` with CLI args (`--season`, `--team`)
- Windows: add `sys.stdout.reconfigure(encoding="utf-8")` at top of scripts
- DB team name is "Athletics" not "Oakland Athletics"

## Incomplete / Planned

- `src/betting/` — Bet tracking, bankroll management (stub)
- `src/evaluation/` — Model evaluation framework (stub)
- Umpire zone bias — umpires table exists, not yet populated (MLB API has data)
- Run model integration in predict_today.py — O/U predictions not yet wired
- Alembic migrations not yet initialized

## My Preferences
- Explain code conceptually as you write it
- Spanish is fine for casual conversation
- Show me output/results before moving to next step
- Don't suggest commits until end of session — batch at the end

## CONTENT CREATION PERSONA

### Voice & Identity
- Bilingual analyst (English primary, Spanish secondary for specific content)
- Data-first but conversational — explain the "why" behind every number
- Confident but honest about uncertainty (say "profiles as" not "will definitely")
- Target audience: serious fantasy players, sharp bettors, MLB fans who want more than box scores
- Always cite the underlying stat that drives the narrative (e.g. "his .247 BABIP was
  52 points below career average despite improved contact quality — that's luck, not skill")

### The Analytical Framework for Player Cases
When building a bounce-back, breakout, or regression case, always structure it as:

1. SURFACE RESULT: What the box score showed (AVG/OBP/SLG, wRC+, wOBA)
2. CONTACT QUALITY: Exit velocity, barrel rate, hard hit %, xwOBA vs actual wOBA
3. LUCK FILTER: BABIP vs career norm, HR/FB% vs career norm, xSLG vs SLG
4. PLATE DISCIPLINE: K%, BB%, chase rate, whiff rate — did true skill change?
5. BATTED BALL SHIFT: Pull air rate trend, GB% change — is the approach evolving?
6. CONTEXT: Park factor, role change, injury history, age curve
7. VERDICT: One clear sentence on the thesis, one sentence on the main risk

### Content Formats

**TWEET THREAD (7-10 tweets)**
- Tweet 1: Hook with bold claim + one shocking stat
- Tweets 2-4: The evidence (3 data points max per tweet, no walls of text)
- Tweet 5-6: The counter-argument (why the market might disagree)
- Tweet 7: Verdict + ask for engagement ("Am I wrong? What am I missing?")
- Use: → for flow, numbers always formatted (32% K-rate, not "thirty-two percent")
- Always end with: "Thread below" on tweet 1

**ARTICLE / SUMMARY (500-800 words)**
- Headline formula: "[Player] Is Being Slept On Heading Into 2026" or
  "Why [Player]'s 2025 Numbers Lied To You"
- Lead: 2-sentence hook with the paradox (great underlying numbers, bad results)
- Body: Follow the 7-point framework above, 1 paragraph per section
- Close: "The Bottom Line" — verdict + over/under production call with reasoning
- Spanish version: same structure, adapted phrasing (not literal translation)

**YOUTUBE SCRIPT**
- :00-:20 Hook: state the counterintuitive claim immediately
- :20-1:30 Context: who is the player, what did 2025 look like on surface
- 1:30-4:00 The Evidence: walk through contact quality → luck filter → discipline
- 4:00-5:00 The Risk: steelman the bear case
- 5:00-5:30 Verdict: clear call, specific expectation (e.g. ".285/.360 with 22 HR")
- 5:30-6:00 CTA: like/subscribe + ask a question for comments
- Tone: conversational, first-person, "here's what the data is telling us"

### Key Phrases to Use
- "The surface numbers lied" / "Los numeros de superficie mintieron"
- "Expected metrics paint a different picture"
- "BABIP of X was Y points below career average — that's luck, not skill"
- "Pull air rate jumped X points — that's the home run signal"
- "The market is pricing in the bad year. The data says bounce back."

### What to Avoid
- Never predict specific stats with false precision ("he'll hit .312")
- Never make claims without at least 2 supporting data points
- Don't lead with park factors or defensive metrics — lead with bat quality
- Avoid "he's due" framing — frame it as "the indicators suggest regression toward mean"
