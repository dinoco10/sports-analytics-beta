# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MLB analytics platform for power rankings, predictive models, and (future) betting. Python 3.12+, SQLAlchemy 2.0+, SQLite (dev) / PostgreSQL (prod). Currently in Phase 2 of 5.

## Key Commands

```bash
# Setup
python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python scripts/initialize_db.py
python scripts/seed_teams.py

# Data ingestion
python scripts/backfill_games.py                # Both 2024-2025 seasons
python scripts/backfill_games.py --season 2025  # Single season
python scripts/backfill_games.py --summary      # Check DB counts

# Feature engineering
python -m src.features.team_features
python -m src.features.team_features --season 2025
python -m src.features.team_features --team "New York Yankees"

# Power rankings
python -m src.models.power_rankings --season 2025

# Standalone Phase 1 model (all-in-one, outputs PNG + CSV)
python mlb_power_rankings_feb2026.py

# Test API client
python -m src.ingestion.mlb_api

# Test DB connection
python -c "from src.storage.database import test_connection; test_connection()"

```


## Rules
- Always test features layer-by-layer, measuring log loss impact
- Never change working code without explicit permission
- Use max_depth=2 for LightGBM (Numeristical methodology)
- Explain every code section as if I'm learning Python
- Chronological train/test splits only (no random splits)

## Architecture
- Database: SQLite at data/mlb_analytics.db
- Tables: games, pitching_logs, batting_logs, bullpen_logs, lineups
- Feature pipeline: scripts/build_features.py → data/features/game_features.csv
- Model training: scripts/train_win_model.py (LightGBM)


### Data Flow
MLB Stats API → `src/ingestion/mlb_api.py` → SQLite DB → `src/features/team_features.py` → `src/models/power_rankings.py` → CSV/PNG output

### Core Modules
- **config/settings.py** — Central config: DB URL, API base, rolling windows `[7, 14, 30]`, league averages, stat thresholds
- **config/weights.py** — Model component weights (must sum to 1.00, auto-validated). Top weights: lineup_contact (0.13), bullpen_depth (0.13), defense (0.11)
- **src/ingestion/mlb_api.py** — MLB Stats API client with 0.5s rate limiting. Key methods: `get_teams()`, `get_schedule()`, `get_box_score()`, `get_game_player_stats()`
- **src/storage/models.py** — SQLAlchemy ORM: dimension tables (teams, players, ballparks, umpires), fact tables (games, pitching/hitting stats), aggregation tables (team_season_snapshots, model_predictions, bets)
- **src/storage/database.py** — DB session management via `get_session()` context manager
- **src/features/team_features.py** — Layer 2 feature engineering: records, pitching/hitting stats, rolling windows, bullpen fatigue, regression indicators
- **src/models/power_rankings.py** — 4-source blend model: FanGraphs (35%), data-driven Pythagorean (20%), player projections (30%), eye test (15%)

## PROJECT STRUCTURE

Root folder: sports-analytics/
Database: sports-analytics/data/mlb-analytics.db

## DATA ARCHITECTURE CONTRACT

Never read directly from raw_* tables in model code.
Floor 1 (LightGBM) reads ONLY from game_features table.
Projections come ONLY from batter_projections and pitcher_projections.
Content pipeline reads ONLY from content_flags table.
Odds data lives in data/odds/ (raw Excel) and odds table (processed).

When adding new features:
1. Add raw metric to raw_* table first
2. Add Marcel-weighted projection to batter/pitcher_projections
3. Add aggregated game-level feature to game_features
4. Only then update model feature list

Layer order: raw → projections → game_features → model → output

### Database
SQLite at `data/mlb_analytics.db`. Schema uses foreign keys with cascading, unique constraints to prevent duplicates, indexes on (date, season, team_id). Batch commits every 50 records.

## Git Workflow

**Auto-commit after every major change.** After completing a significant piece of work (new feature, module, bug fix, refactor), Claude must:
1. Create a feature branch from main (e.g., `feature/player-projections`, `fix/bullpen-fatigue-calc`)
2. Stage and commit the relevant files with a descriptive message
3. Push the branch and create a PR to main

Branch naming: `feature/`, `fix/`, `refactor/`, or `data/` prefix + short kebab-case description.

## Conventions

- SQLAlchemy ORM for all queries (no raw SQL)
- DataFrames for data processing; merge on `team` column (standardized name)
- Normalize values to 0-1 or 1-10 scales
- Feature columns use descriptive prefixes (e.g., `last_30d_era`)
- Regression indicators use league average constants from `config/settings.py`
- Modules are runnable via `python -m` with CLI args (`--season`, `--team`)

## Three Floors
1. Game-by-game win probability (CURRENT — Layer 2 production)
2. Player-level projection (WAR/fWAR/wRC+ — NOT STARTED)
3. Season win total projection (NOT STARTED)

## Incomplete / In-Progress

- `src/features/player_projections.py` — Marcel projections (partial)
- `src/features/roster_impact.py` — Player contribution model (stub)
- `scripts/update_player_ages.py` — Age adjustment (stub)
- `src/betting/`, `src/evaluation/` — Empty, planned for Phase 3
- Alembic migrations not yet initialized

## My Preferences
- Explain code conceptually as you write it
- Spanish is fine for casual conversation
- Show me output/results before moving to next step

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
- Always end with: "Thread below 🧵" on tweet 1

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
- "The surface numbers lied" / "Los números de superficie engañaron"  
- "Expected metrics paint a different picture"
- "BABIP of X was Y points below career average — that's luck, not skill"
- "Pull air rate jumped X points — that's the home run signal"
- "The market is pricing in the bad year. The data says bounce back."

### What to Avoid
- Never predict specific stats with false precision ("he'll hit .312")
- Never make claims without at least 2 supporting data points
- Don't lead with park factors or defensive metrics — lead with bat quality
- Avoid "he's due" framing — frame it as "the indicators suggest regression toward mean"