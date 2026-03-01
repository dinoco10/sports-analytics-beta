# MLB Analytics Platform -- Model Status

## Quick Summary

This is an MLB analytics platform that predicts game-level win probabilities, projects player performance using Marcel+Statcast methodology, simulates full seasons via Monte Carlo, and identifies value betting opportunities against sportsbook lines. The platform is built in Python 3.12+ with SQLAlchemy/SQLite, powered by a LightGBM classifier with 45 curated features. Currently in Phase 2 of 5, with Floors 1-3 complete: game win probability (0.6677 log loss, beats Vegas by 118bp), Marcel+Statcast player projections for both hitters and pitchers, and season win total projections (MAE 3.9 wins vs Vegas with depth chart allocation). The betting infrastructure (daily recommendations, bankroll management, systematic backtesting) is built and ready for validation.

---

## Architecture

### Data Flow

```
MLB Stats API
    |
    v
src/ingestion/mlb_api.py  (0.5s rate limiting)
    |
    v
SQLite DB  (data/mlb_analytics.db)
    |
    +---> scripts/build_features.py  --->  data/features/game_features.csv
    |         (Layers 1-6: team, SP, BP, lineup, projections, Elo/handedness/splits)
    |
    +---> src/features/player_projections.py  --->  data/features/*_projections_2026.csv
    |         (Marcel 5/4/3 + Statcast overlay, hitters + pitchers)
    |
    +---> src/features/depth_chart.py  --->  data/features/depth_chart_*_2026.csv
              (Layer 2: PA/IP allocation by position hierarchy)

game_features.csv
    |
    v
scripts/train_win_model.py  --->  models/win_probability_lgbm.txt
    |                              models/win_probability_meta.json
    |                              models/feature_medians.json
    v
scripts/predict_today.py  --->  Win probabilities for today's games
    |
    v
scripts/daily_bets.py  --->  Value bet recommendations (EV, Kelly sizing)

depth_chart_*_2026.csv
    |
    v
scripts/project_season_wins.py  --->  Season win totals (Monte Carlo)
    |                                  Playoff probabilities
    v
Phase C: Vegas benchmark (MAE, value flags)
```

### Three Floors

| Floor | Description | Status |
|-------|-------------|--------|
| **Floor 1** | Game-by-game win probability (LightGBM, 45 features) | DONE |
| **Floor 2** | Player-level projections (Marcel+Statcast, hitters+pitchers) | DONE |
| **Floor 3** | Season win total projections (Monte Carlo, 10K sims) | DONE |

### Layer Architecture

```
Layer 1: player_projections.py  (Marcel rate stats: wOBA, FIP, aging curves)
    |
    v
Layer 2: depth_chart.py  (PA/IP volume allocation by position tier)
    |
    v
Layer 3: project_season_wins.py  (Monte Carlo season simulation)
```

---

## Floor 1: Game Win Probability Model

### Model Type & Hyperparameters

- **Algorithm**: LightGBM (LGBMClassifier)
- **Objective**: binary (binary cross-entropy / log loss)
- **max_depth**: 2 (Numeristical methodology -- prevents overfitting)
- **num_leaves**: 3 (must equal 2^max_depth - 1; using 4 gives false signal)
- **learning_rate**: 0.05
- **n_estimators**: 500 (with early stopping at 50 rounds patience)
- **min_child_samples**: 50
- **subsample**: 0.8
- **colsample_bytree**: 0.8
- **reg_alpha**: 0.1
- **reg_lambda**: 1.0

### Training Data

- **Training seasons**: 2021-2024 (skips COVID 2020 -- adds noise)
- **Test season**: 2025
- **Split strategy**: Chronological only (NEVER random splits)
- **Training games**: ~4,936
- **Test games**: ~2,471
- **NaN handling**: Median imputation from training set (medians saved to JSON for live pipeline)

### Complete Feature List (45 Features)

The model uses a pruned set of 45 features (43 curated + 2 Elo ratings), selected by LightGBM split importance.

#### Projections (11 features -- strongest signal)

| # | Feature | Description |
|---|---------|-------------|
| 1 | `away_proj_lineup_woba` | Away lineup's avg Marcel+Statcast projected wOBA |
| 2 | `diff_proj_sp_k_bb` | Diff in SP projected K-BB% (home minus away) |
| 3 | `diff_proj_sp_sc_era` | Diff in SP Statcast-adjusted ERA (away minus home; lower=better) |
| 4 | `diff_proj_lineup_woba` | Diff in lineup projected wOBA (home minus away) |
| 5 | `home_proj_lineup_woba` | Home lineup's avg Marcel+Statcast projected wOBA |
| 6 | `home_proj_lineup_bb_score` | Home lineup's avg bounce-back score |
| 7 | `away_proj_sp_k_bb` | Away SP's projected K-BB% |
| 8 | `home_proj_sp_sc_era` | Home SP's Statcast-adjusted ERA |
| 9 | `home_proj_sp_fip` | Home SP's projected FIP |
| 10 | `diff_proj_sp_sust` | Diff in SP sustainability score |
| 11 | `home_proj_sp_war` | Home SP's projected fWAR |

#### Elo Ratings (2 features -- trajectory signal, -17.8bp improvement)

| # | Feature | Description |
|---|---------|-------------|
| 12 | `home_elo` | Home team's pre-game Elo rating |
| 13 | `away_elo` | Away team's pre-game Elo rating |

#### Team Rolling Stats (6 features)

| # | Feature | Description |
|---|---------|-------------|
| 14 | `diff_pyth_t30` | Diff in 30-game Pythagorean win% |
| 15 | `diff_pyth_t14` | Diff in 14-game Pythagorean win% |
| 16 | `away_team_whip_t14` | Away team 14-game rolling WHIP |
| 17 | `home_team_whip_t30` | Home team 30-game rolling WHIP |
| 18 | `away_team_k_pct_t14` | Away team 14-game rolling K% |
| 19 | `home_team_k_pct_t14` | Home team 14-game rolling K% |

#### Starting Pitcher Rolling Stats (12 features)

| # | Feature | Description |
|---|---------|-------------|
| 20 | `home_era_sp10` | Home SP ERA over last 10 starts |
| 21 | `away_k_pct_sp10` | Away SP K% over last 10 starts |
| 22 | `away_bb_pct_sp10` | Away SP BB% over last 10 starts |
| 23 | `away_ip_per_start_sp10` | Away SP IP/start over last 10 starts |
| 24 | `home_fip_sp10` | Home SP FIP over last 10 starts |
| 25 | `home_bb_pct_sp10` | Home SP BB% over last 10 starts |
| 26 | `away_era_sp5` | Away SP ERA over last 5 starts |
| 27 | `away_fip_sp10` | Away SP FIP over last 10 starts |
| 28 | `home_ip_per_start_sp5` | Home SP IP/start over last 5 starts |
| 29 | `away_whip_sp10` | Away SP WHIP over last 10 starts |
| 30 | `away_era_sp10` | Away SP ERA over last 10 starts |
| 31 | `away_k_pct_sp5` | Away SP K% over last 5 starts |
| 32 | `home_ip_per_start_sp10` | Home SP IP/start over last 10 starts |

#### Bullpen Diffs (3 features -- bp35 window only)

| # | Feature | Description |
|---|---------|-------------|
| 33 | `diff_bp_bb_pct_bp35` | Diff in bullpen BB% over 35 team games |
| 34 | `diff_bp_whip_bp35` | Diff in bullpen WHIP over 35 team games |
| 35 | `diff_bp_k_pct_bp35` | Diff in bullpen K% over 35 team games |

#### Lineup Stats (2 features)

| # | Feature | Description |
|---|---------|-------------|
| 36 | `home_lineup_slg` | Home lineup avg SLG (rolling 30 games) |
| 37 | `away_lineup_slg` | Away lineup avg SLG (rolling 30 games) |

#### Handedness Matchups (2 features -- ~80bp improvement)

| # | Feature | Description |
|---|---------|-------------|
| 38 | `home_platoon_adv` | Fraction of home lineup with platoon advantage vs opposing SP |
| 39 | `away_platoon_adv` | Fraction of away lineup with platoon advantage vs opposing SP |

#### Arsenal x Handedness Interactions (3 features)

| # | Feature | Description |
|---|---------|-------------|
| 40 | `away_velo_x_same_hand` | Away SP velocity * fraction of same-hand batters faced |
| 41 | `home_velo_x_same_hand` | Home SP velocity * fraction of same-hand batters faced |
| 42 | `home_ivb_x_same_hand` | Home SP induced vertical break * same-hand fraction |

#### Venue-Specific Splits (3 features)

| # | Feature | Description |
|---|---------|-------------|
| 43 | `home_venue_wpct` | Home team's rolling win% in home games only (last 30) |
| 44 | `away_venue_wpct` | Away team's rolling win% in away games only (last 30) |
| 45 | `diff_venue_wpct` | Diff: home_venue_wpct minus away_venue_wpct |

### Performance Metrics

| Metric | Value |
|--------|-------|
| **Log Loss** | 0.6677 (honest, no lookahead, per-season Marcel snapshots) |
| **Previous log loss** | 0.6694 (before Elo, improvement = -17.8bp) |
| **Inflated log loss** | 0.6635 (old number with lookahead bias -- NOT real) |
| **Accuracy** | ~57% |
| **Bias** | -0.023 (slightly underconfident -- safe for betting) |
| **Top feature** | `home_platoon_adv` (importance=113) |

### Vegas Benchmark

| Metric | Value |
|--------|-------|
| **Model beats Vegas by** | 118 basis points on 2025 test set |
| **Optimal ensemble** | 85% model / 15% Vegas (model is dominant) |
| **Value betting ROI (3% edge)** | +13% (flat $100 bets) |
| **Value betting ROI (10% edge)** | +23% (flat $100 bets) |

### Elo Rating Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| K | 4 | Update magnitude per game (low for MLB's high variance) |
| Home advantage | 24 | ~54% implied home win rate |
| Mean | 1500 | League average rating |
| Season revert | 0.33 | Regress 1/3 toward mean between seasons |
| MOV multiplier | Yes | FiveThirtyEight margin-of-victory formula |

---

## Floor 2: Player Projections (Marcel + Statcast)

### Marcel Methodology

The Marcel method ("so simple a monkey could do it") is the baseline:

1. **Weight 3 years of data**: 2025 x5, 2024 x4, 2023 x3
2. **Weight by volume within each year**: min(PA/550, 1.0) for hitters, min(IP/150, 1.0) for pitchers
3. **Regress toward league average**: regression_factor = min(total_sample / full_confidence_sample, 1.0)
   - Hitters: 1,500 PA = full confidence
   - Pitchers: 500 IP = full confidence
4. **Apply aging adjustment**

### Aging Curves (Tool-Specific)

**Hitter wOBA aging**:
- Age 25 and under: +0.008 (still improving)
- Age 26-29: 0.000 (peak)
- Age 30-31: -0.005
- Age 32-33: -0.012
- Age 34-35: -0.020
- Age 36-37: -0.030
- Age 38+: -0.045 (cliff)

**Pitcher ERA aging**:
- Age 25 and under: -0.10 (still improving)
- Age 26-28: 0.00 (peak)
- Age 29-30: +0.05
- Age 31-32: +0.12
- Age 33-34: +0.20
- Age 35-36: +0.30
- Age 37-38: +0.40
- Age 39+: +0.55 (cliff)

**Playing time multiplier**: Separate for SP, RP, and hitters. Reduces projected PA/IP for older players (injury risk).

### Statcast Metrics (Marcel-Weighted)

All Statcast metrics are Marcel-weighted across 3 years (5/4/3) with regression toward league average.

**Hitter Statcast metrics**: avg_exit_velocity, barrel_rate, hard_hit_rate, xwoba, xslg, xba, launch_angle_sweet_spot_pct, chase_rate, whiff_rate, z_contact_rate, pull_air_rate, gb_rate, fb_rate, ld_rate, babip, hr_per_fb

**Pitcher Statcast metrics**: avg_exit_velocity_against, barrel_rate_against, hard_hit_rate_against, xwoba_against, xera, whiff_rate, swstr_rate, chase_rate_induced, z_contact_rate_against, pull_air_rate_against, gb_rate, fb_rate, ld_rate, babip_against, hr_per_fb, k_minus_bb

### Hitter Projection Details

- **Output**: statcast_adjusted_woba, wrc_plus, proj_war (fWAR with position adjustment), bounce_back_score (0-100), regression_risk_score (0-100)
- **Statcast adjustment**: Capped at +/- 0.030 wOBA. Rules: xwOBA/BABIP luck correction, Marcel-weighted barrel rate, pull air rate trend, Marcel-weighted chase rate + trend
- **fWAR formula**: batting_runs = (wOBA - lgWOBA) / wOBA_scale * PA; pos_adj = position_adjustment * (PA/650); fWAR = (batting_runs + pos_adj) / 10

### Pitcher Projection Details

- **Output**: statcast_adjusted_era, proj_fip, proj_war, sustainability_score (0-100), regression_risk_score (0-100), breakout_score (0-100), key_pitch, primary_risk_flag, primary_upside_flag
- **Statcast ERA adjustment**: Capped at +/- 0.60 ERA. Rules: ERA/xERA + BABIP luck, contact suppression (barrel rate against), pull air rate trend, K-BB%/whiff%/SwStr% sustainability, fastball velocity trend
- **Pitcher WAR formula**: WAR = (lgFIP - projFIP) / 10 * (IP/9)
- **Arsenal analysis**: Pitch-level metrics (run_value_per_100, usage, whiff_rate, velocity) identify key pitch, arsenal diversity, velocity trends

### Data Sources

- **Traditional stats**: SQLite DB (pitching_game_stats, hitting_game_stats) for 2023-2025
- **Statcast**: player_statcast_metrics and pitcher_statcast_metrics tables (season-level from Baseball Savant)
- **Pitch-level**: pitcher_pitch_metrics table (per pitch type per season)
- **Rosters**: Live MLB API 40-man roster (fetched fresh each run, not stale DB data)

---

## Floor 3: Season Win Totals

### Three-Phase Pipeline

**Phase A -- Calibrated WAR Baseline**:
- Load depth chart WAR projections (not raw Marcel)
- Query historical W-L records from 2024-2025 (regular season only, exclude October)
- Z-score calibration: calibrated_wins = hist_mean + (proj_war - war_mean) * scale_factor + regression_adj + sos_adj
- Dampening: preseason projection std = actual std * 0.75

**Phase B -- Monte Carlo Simulation**:
- Fetch real 2026 schedule from MLB API (cached to CSV, 2,430 games)
- Fallback: synthetic balanced schedule if API unavailable
- Log5 formula for per-game probabilities: P(A beats B) = pA*(1-pB) / (pA*(1-pB) + pB*(1-pA))
- 10,000 simulations (configurable via --sims)
- Home-field advantage from historical data (~53.5%)
- Division titles and wild card tracking per simulation
- Output: mean_wins, median_wins, std_wins, 80%/90% confidence intervals, division_pct, playoff_pct

**Phase C -- Vegas Benchmark**:
- Compare to sportsbook O/U lines (bet365, stored in data/external/vegas_win_totals_2026.csv)
- Flag value bets where |diff| > 2 wins

### Z-Score Calibration Parameters

| Parameter | Value |
|-----------|-------|
| Historical wins mean | 81.0 |
| Historical wins std | 12.0 |
| Preseason dampening | 0.75 |
| Regression adjustment cap | +/- 3 wins |
| SOS adjustment | -(div_avg_war - league_avg_war) * 0.15 |

### Vegas Benchmark Results (with Depth Chart)

| Metric | Value |
|--------|-------|
| MAE vs Vegas | 3.9 wins (down from 5.8 without depth chart) |
| Correlation | 0.854 |
| Bias | -0.1 |
| Teams within 3 wins | 15/30 |
| Teams within 5 wins | 20/30 |
| Remaining big misses | SD -10.5, MIL -11.4 (transaction-driven), COL +9.0, WSH +8.1 |

---

## Layer 2: Depth Chart

### Purpose

Sits between Marcel (Layer 1, rate stats) and Monte Carlo (Layer 3, simulation). Marcel says how good a player is per PA/IP. The depth chart says how many PA/IP they will accumulate based on roster position.

### Position Hierarchy and PA Tiers (Hitters)

| Tier | PA | Description |
|------|-----|-------------|
| Starter | 560 | Full-time regular |
| Platoon | 280 | Platoon partner or catcher backup |
| Bench | 150 | Utility / pinch hitter |
| Emergency | 0 | 40-man filler |

**Catcher special case**: Forced 65/35 split (390 PA starter / 210 PA backup). No team runs a full-time catcher.

### Pitching IP Tiers

| Role | IP |
|------|-----|
| SP1 | 175 |
| SP2 | 160 |
| SP3 | 160 |
| SP4 | 150 |
| SP5 | 145 |
| SP6 (swingman) | 75 |
| Closer | 65 |
| Setup | 65 |
| Middle relief (x4) | 50 each |
| Low leverage (x3) | 35 each |

### Team Caps

| Cap | Value |
|-----|-------|
| Team PA (hitters) | 5,700 |
| Team IP (pitchers) | 1,450 |
| Individual hitter PA cap | 600 |
| Individual SP IP cap | 185 |
| Per-player WAR floor | -1.5 |

### Ranking Logic

- **Hitters**: Ranked by wRAA/PA = (wOBA - lgWOBA) / wOBA_scale within each position slot
- **Pitchers**: SP ranked by FIP (lower = better), RP ranked by FIP
- Players with wRAA/PA > -0.030 (above replacement) get utility/bench PA even if unassigned
- If team total exceeds cap, all players scaled proportionally

### WAR Calculation

- **Hitter fWAR**: (wRAA_per_pa * depth_pa + position_adj * depth_pa/600) / 10
- **Pitcher WAR**: (4.20 - proj_fip) / 10 * (depth_ip / 9)
- Position adjustments: C +12.5, SS +7.5, 2B/3B/CF +2.5, LF/RF -7.5, 1B -12.5, DH -17.5

---

## Betting Infrastructure

### daily_bets.py

The main betting recommendation script. Capabilities:
- Combines model win probabilities with sportsbook odds to find value bets
- Supports: manual odds entry, CSV odds files, historical odds from DB
- Minimum edge threshold: 3% (configurable via --min-edge)
- EV calculation: model_prob * (decimal_odds - 1) - (1 - model_prob)
- Vig removal from moneyline pairs to get fair probabilities
- Three confidence tiers: STRONG (10%+ edge), LEAN (5-10%), MILD (3-5%)
- Bet tracking: saves to CSV with pending/settled status
- Settlement: looks up actual results from DB, calculates flat and Kelly P/L
- Backtest mode: uses historical odds from DB + actual results
- P/L summary across all tracked dates

### bankroll.py (Kelly Sizing)

`KellyCalculator` class:
- `full_kelly(prob, ml)`: Mathematically optimal but volatile
- `fractional_kelly(prob, ml, fraction=0.25)`: Quarter-Kelly default (75% of growth, 50% less variance)
- `kelly_with_cap(prob, ml, max_bet_pct=0.05)`: Hard cap at 5% of bankroll
- `recommended_stake(prob, ml, bankroll)`: Main method -- fractional Kelly with cap

`BankrollManager` class:
- Tracks all bets from placement to settlement
- Maintains equity curve for visualization
- Computes: win_rate, ROI, max_drawdown, units_won
- Peak/drawdown tracking for risk management
- `print_summary()` for formatted performance report

### backtest.py (Systematic Backtesting)

`Backtester` class:
- Runs full pipeline across a date range: predict -> find edges -> size bets -> settle -> track P/L
- Strategies: "flat", "kelly", "fractional_kelly"
- Daily results DataFrame with cumulative P/L
- Best/worst day reporting
- `plot_equity_curve()`: Saves PNG with bankroll chart + daily P/L bars
- CLI: `python -m src.evaluation.backtest --start 2025-08-01 --end 2025-08-15`

### Value Betting Results (from Vegas benchmark)

| Edge Threshold | ROI (flat $100) |
|---------------|-----------------|
| 3% minimum edge | +13% |
| 10% minimum edge | +23% |

---

## Data Sources

### MLB Stats API

- **Base URL**: https://statsapi.mlb.com/api/v1
- **Rate limiting**: 0.5s between requests
- **Key endpoints**: teams, schedule, boxscore, game player stats
- **Client**: `src/ingestion/mlb_api.py` (MLBApiClient class)

### Database Schema (SQLite)

**Dimension tables**: teams, players, ballparks, umpires
**Fact tables**: games, pitching_game_stats, hitting_game_stats
**Statcast tables**: player_statcast_metrics, pitcher_statcast_metrics, pitcher_pitch_metrics
**Roster tracking**: roster_snapshots
**Aggregation**: team_season_snapshots, model_predictions, bets

Key constraints:
- Unique constraints prevent duplicate records
- Indexes on (date, season, team_id) for performance
- Batch commits every 50 records
- Foreign keys with cascading

### Odds Data

- **Excel files**: data/odds/ (historical, 2021 overlaps with JSON -- skip Excel for overlapping years)
- **JSON**: Historical closing lines
- **DB table**: odds (processed, used for backtesting)
- **Daily CSV**: data/odds/daily/ (manually entered or scraped)
- **Vegas win totals**: data/external/vegas_win_totals_2026.csv (bet365 lines)

### Statcast / Baseball Savant

- Season-level metrics loaded via pybaseball
- Stored in player_statcast_metrics and pitcher_statcast_metrics tables
- Pitch-level data in pitcher_pitch_metrics (per pitch type per season)

---

## What's Next (Roadmap)

### Immediate (Track A-E from last session)

1. **Multi-day backtest validation**: Run daily_bets.py across 2025-08-01 to 2025-08-15 to verify P/L tracking, settlement, and edge calculations
2. **Daily workflow automation**: Create scripts/daily_workflow.py chaining pull odds -> predict -> recommend bets -> settle yesterday -> log P/L
3. **Spring training research**: What data is available from MLB Stats API (velocity trends, health reports, roster moves)
4. **Merge PR**: Merge feature/live-predictions to main (has Elo + bullpen avail + predict_today + daily_bets)

### Lower Priority

- **Calibration monitoring**: Model bias is -0.023 (not stable across years). Raw model is best. Monitor in 2026.
- **Marcel projection fixes**: PA/IP weighting refinement, playing time multiplier, roster filtering
- **Alembic migrations**: Not yet initialized. Currently using ALTER TABLE with try/except.

---

## Key Decisions & Lessons Learned

### Why max_depth=2

Numeristical methodology. At max_depth=2 with num_leaves=3, the model learns simple interaction rules that generalize well. Deeper trees overfit to training data (especially with only ~5,000 training games). This is the single most important hyperparameter choice.

### Why 85/15 Model/Vegas Ensemble

Grid search over ensemble weights showed the model now dominates Vegas (previously was 20% model / 80% Vegas). At 85/15, the model contributes the vast majority of signal while Vegas provides a small stabilizing anchor. This flipped from the early days when the model was weaker.

### Why COVID 2020 Is Excluded

The shortened 60-game 2020 season adds noise to training. Player performance in 2020 was anomalous (no fans, shortened ramp-up, small sample), and including it degraded log loss. Training starts from 2021 by default (--train-start 2021).

### Features Tested and Rejected (12+ candidates)

None of the following improved the curated 43-feature set by more than 1bp:

- **Rest days**: No predictive signal for MLB win probability (importance=0)
- **Team-level WAR features** (diff_team_proj_war etc.): Add less than 1bp, not worth complexity
- **bp10 bullpen window**: Too noisy compared to bp35
- **Raw home/away bullpen columns**: Redundant with team-level stats (r=0.75+)
- **Venue RS/RA**: Too noisy with 8 features; only venue wpct helps
- **Arsenal features standalone** (velo, IVB, depth): Don't help alone, but velo_x_same_hand interaction DOES help
- **SP same-hand percentage** (raw): Captured better through interaction terms

### The Bullpen Availability Trap

Bullpen fatigue features (bp_ip/pitches last 1-3 games, fatigue diffs) look like strong signals when tested with minimal regularization. They capture whether a team's bullpen is fresh or gassed. However, with proper model parameters (max_depth=2, num_leaves=3, min_child_samples=50), these features do not survive regularization. This is a classic overfitting trap -- the signal is real but too noisy to be useful at the constrained model complexity needed for generalization. The features are still computed in build_features.py but excluded from the pruned feature list.

### Other Important Patterns

- **LightGBM model corrupts on git checkout** across branches. Use `git show branch:file > file` instead.
- **DB team name**: "Athletics" not "Oakland Athletics"; JSON has both variants.
- **Player table column**: `name` not `full_name` (for raw SQL queries).
- **Windows encoding**: Add `sys.stdout.reconfigure(encoding="utf-8")` at top of every script.
- **Marcel PA/IP**: Uses 5/4/3 weighted average (fixed from single-year bug that caused bad projections).
- **Pitcher statcast query**: Needs `pa_against as statcast_pa` alias -- without it Marcel returns all defaults.
- **~6.6% of games**: Have unknown SP throwing hand -- default to 0.50 platoon_adv.
- **Lookahead bias prevention**: Per-season Marcel snapshots in data/features/snapshots/. 2023 games use only 2021-2022 data, etc.
- **num_leaves must match max_depth**: num_leaves = 2^max_depth - 1 = 3. Setting num_leaves=4 with max_depth=2 gives false signal improvement.
