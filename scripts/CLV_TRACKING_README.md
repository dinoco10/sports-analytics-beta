# CLV Tracking — README

## What is CLV?

**Closing Line Value (CLV)** is the gold standard metric for evaluating sports betting models. It measures whether your predictions are "sharper" than the market's final consensus (the closing line).

The closing line is the most efficient price because:
- Sharp bettors move lines toward true probability
- By game time, all information is priced in
- Consistently beating the closing line = long-term profitability

**CLV Formula:**
```
CLV = Model Probability - Closing Line Implied Probability (devigged)
```

**Example:**
- Model says Yankees have 58% win probability
- Closing line: Yankees -150 (60% raw, 58.5% devigged)
- CLV = 58% - 58.5% = **-0.5%** (market was sharper)

**Why CLV matters:**
- Positive CLV → Model is beating the market (profitable long-term)
- Negative CLV → Model is weaker than the market (losing long-term)
- Research shows CLV predicts profitability better than win rate

---

## Workflow

### **1. Morning Run (10am ET)**
Log model predictions and opening lines for today's games.

```bash
python scripts/track_clv.py --morning
```

**What it does:**
1. Loads predictions from `predict_today.py` (or runs it fresh)
2. Fetches opening lines from available sources:
   - Live API (if configured)
   - Historical DB
   - Excel files (`data/odds/mlb-odds-YYYY.xlsx`)
   - JSON dataset (`data/odds/mlb_odds_dataset.json`)
3. Logs to `clv_tracking` table with:
   - Model probabilities
   - Opening odds (devigged)
   - Timestamp

**Output:**
```
============================================================
  MORNING RUN — 2026-04-15
============================================================
  Loaded 15 predictions from predictions_2026-04-15.csv
  Loaded 15 odds lines from DB for 2026-04-15

  Logged 15 games to CLV tracking table
```

---

### **2. Evening Run (5pm ET, ~30min before first pitch)**
Fetch closing lines and calculate CLV.

```bash
python scripts/track_clv.py --closing
```

**What it does:**
1. Fetches closing lines from odds sources
2. Devigs using multiplicative method (equal margin assumption)
3. Calculates CLV:
   - `clv_home = model_home_prob - close_home_implied`
   - `clv_away = model_away_prob - close_away_implied`
4. Updates `clv_tracking` table

**Output:**
```
============================================================
  CLOSING RUN — 2026-04-15
============================================================
  Loaded 15 odds lines from DB for 2026-04-15
  Found 15 tracked games

  Updated 15 games with closing lines + CLV
```

---

### **3. Settlement (Next Morning)**
Settle yesterday's games and track P/L.

```bash
python scripts/track_clv.py --settle
# Or specify date:
python scripts/track_clv.py --settle --date 2026-04-15
```

**What it does:**
1. Loads game results from `games` table
2. Marks actual winner (home/away/tie)
3. If bets were logged, calculates P/L
4. Updates `clv_tracking` table with results

**Output:**
```
============================================================
  SETTLEMENT — 2026-04-15
============================================================
  Found 15 game results

  Settled 15 games
```

---

### **4. Reporting**
Show CLV statistics and performance metrics.

```bash
# All-time report
python scripts/track_clv.py --report

# Last 30 days
python scripts/track_clv.py --report --last 30

# Last 7 days
python scripts/track_clv.py --report --last 7
```

**Output:**
```
======================================================================
  CLV REPORT — All time
======================================================================

  CLOSING LINE VALUE
  --------------------------------------------------
  Games tracked:          150
  Positive CLV rate:      52.3% (157/300 sides)
  Avg CLV (home):         +1.2 bp
  Avg CLV (away):         +0.8 bp
  Avg CLV (overall):      +1.0 bp

  TOP 5 POSITIVE CLV:
    2026-04-15  Boston Red Sox @ New York Yankees  (HOME)  CLV=+5.2bp
    2026-04-14  Chicago Cubs @ Milwaukee Brewers   (AWAY)  CLV=+4.8bp
    ...

  PERFORMANCE BY CLV TIER (settled games)
  --------------------------------------------------

    Strong CLV (>5%)
      Games:     25
      Record:    16-9 (64.0%)
      Avg CLV:   +7.2 bp

    Positive CLV (0-5%)
      Games:     132
      Record:    72-60 (54.5%)
      Avg CLV:   +2.1 bp

    Negative CLV (<0%)
      Games:     143
      Record:    68-75 (47.6%)
      Avg CLV:   -2.8 bp

  ROLLING TRENDS
  --------------------------------------------------
  7-day avg CLV:   +1.5 bp
  30-day avg CLV:  +0.9 bp
```

**Key metrics explained:**
- **Positive CLV rate:** % of sides where model beats closing line
- **Avg CLV:** Average edge vs closing line (in basis points)
- **Performance by tier:** Win rate for games where model had positive CLV
- **Calibration check:** Are CLV+ bets actually more profitable?

---

## Database Schema

The script creates a `clv_tracking` table:

```sql
CREATE TABLE clv_tracking (
    id INTEGER PRIMARY KEY,
    date DATE NOT NULL,
    mlb_game_id TEXT,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,

    -- Model predictions
    model_home_prob REAL,
    model_away_prob REAL,

    -- Opening lines
    open_home_odds REAL,
    open_away_odds REAL,
    open_home_implied REAL,
    open_away_implied REAL,

    -- Closing lines
    close_home_odds REAL,
    close_away_odds REAL,
    close_home_implied REAL,
    close_away_implied REAL,

    -- CLV
    clv_home REAL,
    clv_away REAL,

    -- Bet tracking
    bet_side TEXT,
    bet_amount REAL,
    bet_odds REAL,

    -- Settlement
    actual_winner TEXT,
    bet_result TEXT,
    pnl REAL,

    -- Metadata
    sportsbook TEXT,
    created_at TIMESTAMP,
    settled_at TIMESTAMP,

    UNIQUE(date, mlb_game_id)
);
```

---

## Devigging Methods

The script uses **multiplicative devig** by default (most common):

```python
def remove_vig_multiplicative(home_ml, away_ml):
    """Equal margin assumption — simple and robust."""
    home_raw = american_to_implied_prob(home_ml)
    away_raw = american_to_implied_prob(away_ml)
    total = home_raw + away_raw
    return home_raw / total, away_raw / total
```

**Alternative: Power method** (more accurate for asymmetric markets):

```python
def remove_vig_power(home_ml, away_ml, k=1.0):
    """Shin/Pinnacle method — better for favorites vs dogs."""
    home_raw = american_to_implied_prob(home_ml) ** k
    away_raw = american_to_implied_prob(away_ml) ** k
    total = home_raw + away_raw
    return home_raw / total, away_raw / total
```

To use power method, edit the script and replace calls to `remove_vig_multiplicative()` with `remove_vig_power()`.

---

## Odds Data Sources

### **Priority order:**

1. **Live API** (The Odds API) — Real-time odds, requires API key
2. **Database** (`odds` table) — Historical closing lines
3. **Excel files** (`data/odds/mlb-odds-YYYY.xlsx`) — Historical data
4. **JSON dataset** (`data/odds/mlb_odds_dataset.json`) — Backup

### **Configuring The Odds API (for live 2026 season):**

1. Sign up at https://the-odds-api.com/
2. Get API key (500 free requests/month)
3. Edit `track_clv.py`:

```python
USE_LIVE_ODDS_API = True
ODDS_API_KEY = "your_key_here"
```

4. Implement `fetch_odds_live()` function:

```python
import requests

def fetch_odds_live(target_date, line_type="opening"):
    url = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": "h2h",
        "oddsFormat": "american",
    }

    response = requests.get(url, params=params)
    data = response.json()

    # Parse response and return DataFrame
    # (structure depends on API response format)
    ...
```

### **Historical backtesting:**

For backtesting, the script automatically uses DB/Excel/JSON sources. No API needed.

---

## Expected Behavior

### **What "good" CLV looks like:**

- **Overall CLV:** +0.5 to +2.0 bp is excellent
  - Market is very efficient
  - Even +0.5bp compounds to significant profit over 1000s of bets
- **Positive CLV rate:** 50-55% is very good
  - 50% = perfect calibration (model is as sharp as market)
  - >55% = model has persistent edge
  - <50% = model needs improvement
- **Win rate on CLV+ bets:** Should be higher than CLV- bets
  - E.g., CLV+ bets: 53%, CLV- bets: 48%
  - If no difference → model is not well-calibrated

### **Red flags:**

- **Negative overall CLV** → Model is weaker than market
- **Positive CLV but lower win rate** → Model is poorly calibrated (overconfident)
- **Large CLV swings day-to-day** → Unstable model or data issues

---

## Integration with Daily Workflow

### **Automated daily workflow:**

```bash
# Morning (10am ET)
python scripts/track_clv.py --morning

# Evening (5pm ET, before first pitch)
python scripts/track_clv.py --closing

# Next morning (8am ET)
python scripts/track_clv.py --settle --date $(date -d "yesterday" +%Y-%m-%d)
```

### **Weekly review:**

```bash
# Every Monday, review last 7 days
python scripts/track_clv.py --report --last 7
```

### **Monthly review:**

```bash
# First of month, review last 30 days
python scripts/track_clv.py --report --last 30
```

---

## Example: Full Day Workflow

**April 15, 2026 — Yankees vs Red Sox**

### **Morning (10am):**

```bash
$ python scripts/track_clv.py --morning
```

**Logged:**
- Model: Yankees 58% (Red Sox 42%)
- Opening line: Yankees -145, Red Sox +125
- Opening devigged: Yankees 58.8%, Red Sox 41.2%

### **Evening (5pm):**

```bash
$ python scripts/track_clv.py --closing
```

**Updated:**
- Closing line: Yankees -150, Red Sox +130
- Closing devigged: Yankees 60.0%, Red Sox 40.0%
- CLV (Yankees): 58% - 60% = **-2.0%** (market moved away from us)
- CLV (Red Sox): 42% - 40% = **+2.0%** (positive CLV on underdog!)

**Interpretation:**
- Sharp money came in on Yankees, pushed line from -145 to -150
- Our model was on the wrong side of the closing line
- If we bet Yankees at opening, we got -145 instead of -150 (still -1.2% CLV vs close)

### **Next Morning (8am):**

```bash
$ python scripts/track_clv.py --settle --date 2026-04-15
```

**Settled:**
- Yankees won 5-3
- Our model predicted Yankees 58%, market closed 60%
- Outcome: Model was right, but market was more confident

---

## Advanced Features

### **Custom devig parameters:**

Edit the script to tune devigging:

```python
# Use power method with k=2.0 (more weight to underdogs)
close_home_impl, close_away_impl = remove_vig_power(
    close_home_odds, close_away_odds, k=2.0
)
```

### **Bet logging:**

To track actual bets placed (for P/L calculation):

```python
# After finding value bets in daily_bets.py,
# update clv_tracking with bet details:

conn.execute("""
    UPDATE clv_tracking
    SET bet_side = ?, bet_amount = ?, bet_odds = ?
    WHERE date = ? AND home_team = ? AND away_team = ?
""", (bet_side, bet_amount, bet_odds, date, home, away))
```

### **Export to CSV:**

```python
import pandas as pd
import sqlite3

conn = sqlite3.connect("data/mlb_analytics.db")
df = pd.read_sql("SELECT * FROM clv_tracking", conn)
df.to_csv("clv_export.csv", index=False)
```

---

## Troubleshooting

### **"No odds data found"**

- Check that `data/odds/` has historical files
- For live odds, verify `USE_LIVE_ODDS_API=True` and API key is set
- For backtest dates, ensure `odds` table is populated

### **"No predictions found"**

- Script will auto-run `predict_today.py` if CSV missing
- Ensure DB has historical game/stats data for the date
- Check that `models/win_probability_lgbm.txt` exists

### **"No matching games for settlement"**

- Verify games finished and are in `games` table
- Check team name matching (script handles common variations)
- Run with `--date` flag to specify exact date

### **CLV seems wrong (too high/low)**

- Verify odds are closing lines, not opening
- Check devig method (multiplicative vs power)
- Ensure sportsbook is sharp (Pinnacle > DraftKings > low-limit books)

---

## Next Steps

1. **Run morning workflow** for today's games
2. **Collect 30 days of CLV data** before making conclusions
3. **Review weekly reports** to spot trends
4. **Compare CLV to actual P/L** — they should correlate
5. **Use CLV to filter bets** — only bet games with positive CLV

**Key insight:** Even if model has 52% win rate, negative CLV means you're getting -EV prices. CLV is the true profitability indicator.

---

## References

- [Pinnacle: Closing Line Value Explained](https://www.pinnacle.com/en/betting-articles/Betting-Strategy/closing-line-value-bet/5P22XWPM9WDPJM6G)
- [The Hidden Signal of Sports Betting](https://www.espn.com/chalk/story/_/id/25225947/the-hidden-signal-sports-betting-clv)
- [How to Calculate CLV](https://www.sportsbookreview.com/picks/best-bets/how-to-calculate-closing-line-value/)
