"""
backtest.py — Systematic Backtesting Framework
===============================================
Runs the full betting pipeline over a historical date range:
predict → find edges → size bets → settle → track P/L.

This automates what daily_bets.py --backtest does for a single day,
but runs it across many days with proper bankroll tracking.

Usage:
    from src.evaluation.backtest import Backtester
    from src.betting.bankroll import BankrollManager

    mgr = BankrollManager(starting_bankroll=10000)
    bt = Backtester(bankroll_manager=mgr,
                    start_date="2025-08-01",
                    end_date="2025-08-15")
    bt.run(min_edge=0.03, strategy="fractional_kelly")
    bt.summary()
    bt.plot_equity_curve()

    # Or from CLI:
    python -m src.evaluation.backtest --start 2025-08-01 --end 2025-08-15
"""

import argparse
import sqlite3
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

# Fix Windows encoding
if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving PNGs
import matplotlib.pyplot as plt
import pandas as pd

from src.betting.bankroll import (
    BankrollManager,
    KellyCalculator,
    american_to_decimal,
    american_to_implied_prob,
    ml_payout,
)

ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = ROOT / "data" / "mlb_analytics.db"
MODEL_PATH = ROOT / "models" / "lgb_win_model.txt"

# ═══════════════════════════════════════════════════════════════
# BACKTESTER
# ═══════════════════════════════════════════════════════════════

class Backtester:
    """Runs a full backtest over a date range.

    For each day:
    1. Load historical odds from the DB
    2. Run predict_today.predict_games() in backtest mode
    3. Find value bets (model_prob - vegas_prob >= min_edge)
    4. Size bets using chosen strategy (flat, kelly, fractional_kelly)
    5. Settle immediately using actual game results
    6. Track P/L in the BankrollManager

    The result is a complete equity curve and performance report.
    """

    def __init__(self, bankroll_manager: BankrollManager = None,
                 start_date: str = "2025-08-01",
                 end_date: str = "2025-08-15",
                 model_path: str = None):
        """
        Args:
            bankroll_manager: BankrollManager instance (creates one if None)
            start_date: First day to backtest (YYYY-MM-DD)
            end_date: Last day to backtest (YYYY-MM-DD)
            model_path: Path to LightGBM model (uses default if None)
        """
        self.mgr = bankroll_manager or BankrollManager(starting_bankroll=10_000.0)
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
        self.model_path = model_path or str(MODEL_PATH)
        self.kelly = KellyCalculator()

        # Results storage
        self.daily_results: list[dict] = []
        self._ran = False

    def run(self, min_edge: float = 0.03,
            strategy: str = "fractional_kelly",
            kelly_fraction: float = 0.25,
            flat_amount: float = 100.0,
            max_bet_pct: float = 0.05,
            verbose: bool = True):
        """Execute the backtest across all dates.

        Args:
            min_edge: Minimum model-vs-vegas edge to place a bet (default 3%)
            strategy: "flat", "kelly", or "fractional_kelly"
            kelly_fraction: Fraction for fractional Kelly (default 0.25)
            flat_amount: Fixed bet amount for flat strategy
            max_bet_pct: Maximum bet as % of bankroll (safety cap)
            verbose: Print day-by-day results
        """
        # Import predict_games here to avoid circular imports.
        # predict_today lives in scripts/, so we add it to path.
        sys.path.insert(0, str(ROOT / "scripts"))
        from predict_today import predict_games

        if verbose:
            print(f"\n{'=' * 65}")
            print(f"  BACKTEST: {self.start_date} → {self.end_date}")
            print(f"  Strategy: {strategy} | Min edge: {min_edge:.0%} "
                  f"| Bankroll: ${self.mgr.current_bankroll:,.0f}")
            print(f"{'=' * 65}")

        current = self.start_date
        while current <= self.end_date:
            date_str = current.strftime("%Y-%m-%d")
            day_result = self._run_single_day(
                date_str, predict_games,
                min_edge=min_edge,
                strategy=strategy,
                kelly_fraction=kelly_fraction,
                flat_amount=flat_amount,
                max_bet_pct=max_bet_pct,
            )

            self.daily_results.append(day_result)

            if verbose and day_result["bets"] > 0:
                print(f"  {date_str}  |  {day_result['wins']}W-{day_result['losses']}L  "
                      f"|  P/L: ${day_result['daily_pl']:>+8,.0f}  "
                      f"|  Bankroll: ${day_result['bankroll']:>10,.0f}")
            elif verbose:
                print(f"  {date_str}  |  no bets (no odds or no edge)")

            current += timedelta(days=1)

        self._ran = True

        if verbose:
            print(f"\n{'=' * 65}")
            self.summary()

    def _run_single_day(self, date_str: str, predict_games_fn,
                        min_edge: float, strategy: str,
                        kelly_fraction: float, flat_amount: float,
                        max_bet_pct: float) -> dict:
        """Backtest a single day — returns dict with day's results."""
        result = {
            "date": date_str,
            "bets": 0, "wins": 0, "losses": 0,
            "daily_pl": 0.0,
            "bankroll": self.mgr.current_bankroll,
            "games": 0,
        }

        # 1. Load historical odds from DB
        odds_df = self._load_odds(date_str)
        if odds_df is None or len(odds_df) == 0:
            return result

        # 2. Generate predictions (backtest mode uses actual results)
        try:
            d = datetime.strptime(date_str, "%Y-%m-%d").date()
            preds_df = predict_games_fn(d, save=False, backtest=True)
        except Exception as e:
            print(f"    Error predicting {date_str}: {e}")
            return result

        if preds_df is None or len(preds_df) == 0:
            return result

        result["games"] = len(preds_df)

        # 3. Find value bets and settle them
        daily_pl = 0.0
        wins = 0
        losses = 0
        bets_placed = 0

        for _, pred in preds_df.iterrows():
            # Match prediction to odds
            odds_match = odds_df[
                odds_df["home_team"].str.contains(
                    pred["home_team"], case=False, na=False
                )
            ]
            if len(odds_match) == 0:
                continue

            odds_row = odds_match.iloc[0]
            home_ml = odds_row["home_ml"]
            away_ml = odds_row["away_ml"]

            # Remove vig to get fair vegas probs
            home_raw = american_to_implied_prob(home_ml)
            away_raw = american_to_implied_prob(away_ml)
            total = home_raw + away_raw
            vegas_home = home_raw / total
            vegas_away = away_raw / total

            model_home = pred["home_win_prob"]
            model_away = 1 - model_home

            # Check both sides for value
            for side, model_p, vegas_p, ml_odds in [
                ("HOME", model_home, vegas_home, home_ml),
                ("AWAY", model_away, vegas_away, away_ml),
            ]:
                edge = model_p - vegas_p
                if edge < min_edge:
                    continue

                # Size the bet based on strategy
                if strategy == "flat":
                    amount = flat_amount
                elif strategy == "kelly":
                    amount = self.kelly.recommended_stake(
                        model_p, ml_odds, self.mgr.current_bankroll,
                        fraction=1.0, max_bet_pct=max_bet_pct
                    )
                else:  # fractional_kelly
                    amount = self.kelly.recommended_stake(
                        model_p, ml_odds, self.mgr.current_bankroll,
                        fraction=kelly_fraction, max_bet_pct=max_bet_pct
                    )

                if amount <= 0:
                    continue

                # Determine if the bet won using actual results
                if pd.isna(pred.get("home_score")) or pd.isna(pred.get("away_score")):
                    continue

                home_won = pred["home_score"] > pred["away_score"]
                won = home_won if side == "HOME" else not home_won

                # Place and immediately settle
                team = pred["home_team"] if side == "HOME" else pred["away_team"]
                opponent = pred["away_team"] if side == "HOME" else pred["home_team"]

                bet_id = self.mgr.place_bet(
                    amount=amount, ml=ml_odds, team=team,
                    opponent=opponent, side=side,
                    prob=model_p, date=date_str, strategy=strategy,
                )
                pl = self.mgr.settle_bet(bet_id, won)

                daily_pl += pl
                bets_placed += 1
                if won:
                    wins += 1
                else:
                    losses += 1

        result["bets"] = bets_placed
        result["wins"] = wins
        result["losses"] = losses
        result["daily_pl"] = daily_pl
        result["bankroll"] = self.mgr.current_bankroll

        return result

    def _load_odds(self, date_str: str) -> pd.DataFrame:
        """Load historical odds from SQLite DB."""
        try:
            conn = sqlite3.connect(str(DB_PATH))
            df = pd.read_sql_query(
                """SELECT date, home_team, away_team, home_ml, away_ml,
                          sportsbook
                   FROM odds WHERE date = ?""",
                conn, params=(date_str,)
            )
            conn.close()
            return df if len(df) > 0 else None
        except Exception:
            return None

    # ═══════════════════════════════════════════════════════════
    # RESULTS
    # ═══════════════════════════════════════════════════════════

    def results(self) -> pd.DataFrame:
        """Return daily results as a DataFrame.

        Columns: date, bets, wins, losses, daily_pl, cumulative_pl, bankroll
        """
        if not self.daily_results:
            return pd.DataFrame()

        df = pd.DataFrame(self.daily_results)
        df["cumulative_pl"] = df["daily_pl"].cumsum()
        return df

    def summary(self):
        """Print a comprehensive backtest report."""
        stats = self.mgr.get_stats()
        df = self.results()

        if len(df) == 0 or stats["settled"] == 0:
            print("  No bets were placed during the backtest period.")
            return

        days_with_bets = (df["bets"] > 0).sum()
        total_days = len(df)

        print(f"\n  BACKTEST RESULTS")
        print(f"  {'─' * 45}")
        print(f"  Period:     {self.start_date} → {self.end_date} "
              f"({total_days} days, {days_with_bets} with bets)")
        print(f"  Bankroll:   ${stats['starting_bankroll']:,.0f} → "
              f"${stats['current_bankroll']:,.0f}")
        print(f"  P/L:        ${stats['total_profit_loss']:>+,.0f}")
        print(f"  ROI:        {stats['roi']:+.1f}%")
        print(f"  Record:     {stats['wins']}W-{stats['losses']}L "
              f"({stats['win_rate']:.1%})")
        print(f"  Max DD:     {stats['max_drawdown']:.1f}%")
        print(f"  Avg bets/day: {stats['settled'] / days_with_bets:.1f}" if days_with_bets else "")

        # Best and worst days
        if len(df[df["bets"] > 0]) > 0:
            active = df[df["bets"] > 0]
            best = active.loc[active["daily_pl"].idxmax()]
            worst = active.loc[active["daily_pl"].idxmin()]
            print(f"\n  Best day:   {best['date']}  ${best['daily_pl']:>+,.0f} "
                  f"({best['wins']:.0f}W-{best['losses']:.0f}L)")
            print(f"  Worst day:  {worst['date']}  ${worst['daily_pl']:>+,.0f} "
                  f"({worst['wins']:.0f}W-{worst['losses']:.0f}L)")

    def plot_equity_curve(self, save_path: str = None):
        """Plot bankroll over time — the equity curve.

        This is THE chart for evaluating a betting strategy.
        Ideally you want a smooth upward slope with small drawdowns.
        """
        df = self.results()
        if len(df) == 0:
            print("  No results to plot.")
            return

        active = df[df["bets"] > 0].copy()
        if len(active) == 0:
            print("  No active betting days to plot.")
            return

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1])

        # Top: Equity curve
        ax1 = axes[0]
        ax1.plot(range(len(active)), active["bankroll"].values,
                 "b-", linewidth=2, label="Bankroll")
        ax1.axhline(y=self.mgr.starting_bankroll, color="gray",
                     linestyle="--", alpha=0.5, label="Starting")
        ax1.fill_between(
            range(len(active)),
            self.mgr.starting_bankroll,
            active["bankroll"].values,
            alpha=0.1,
            color="green" if active["bankroll"].iloc[-1] > self.mgr.starting_bankroll else "red",
        )
        ax1.set_ylabel("Bankroll ($)")
        ax1.set_title(f"Backtest Equity Curve: {self.start_date} → {self.end_date}")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Bottom: Daily P/L bars
        ax2 = axes[1]
        colors = ["green" if pl >= 0 else "red" for pl in active["daily_pl"]]
        ax2.bar(range(len(active)), active["daily_pl"].values, color=colors, alpha=0.7)
        ax2.axhline(y=0, color="gray", linestyle="-", alpha=0.5)
        ax2.set_ylabel("Daily P/L ($)")
        ax2.set_xlabel("Trading Day")
        ax2.grid(True, alpha=0.3)

        # X-axis labels (show dates)
        tick_positions = range(0, len(active), max(1, len(active) // 10))
        ax2.set_xticks(tick_positions)
        ax2.set_xticklabels(
            [active.iloc[i]["date"][5:] for i in tick_positions],
            rotation=45, ha="right"
        )

        plt.tight_layout()

        if save_path is None:
            save_path = str(ROOT / "data" / "backtest_equity_curve.png")
        plt.savefig(save_path, dpi=150)
        print(f"\n  Equity curve saved to: {save_path}")
        plt.close()


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Run systematic backtest over date range"
    )
    parser.add_argument("--start", type=str, required=True,
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, required=True,
                        help="End date (YYYY-MM-DD)")
    parser.add_argument("--bankroll", type=float, default=10000,
                        help="Starting bankroll (default: $10,000)")
    parser.add_argument("--strategy", type=str, default="fractional_kelly",
                        choices=["flat", "kelly", "fractional_kelly"],
                        help="Bet sizing strategy")
    parser.add_argument("--min-edge", type=float, default=0.03,
                        help="Minimum edge to bet (default: 3%%)")
    parser.add_argument("--kelly-fraction", type=float, default=0.25,
                        help="Kelly fraction (default: 0.25 = quarter Kelly)")
    parser.add_argument("--flat-amount", type=float, default=100,
                        help="Flat bet amount (default: $100)")
    parser.add_argument("--plot", action="store_true",
                        help="Save equity curve plot")
    args = parser.parse_args()

    mgr = BankrollManager(starting_bankroll=args.bankroll)
    bt = Backtester(
        bankroll_manager=mgr,
        start_date=args.start,
        end_date=args.end,
    )

    bt.run(
        min_edge=args.min_edge,
        strategy=args.strategy,
        kelly_fraction=args.kelly_fraction,
        flat_amount=args.flat_amount,
    )

    if args.plot:
        bt.plot_equity_curve()


if __name__ == "__main__":
    main()
