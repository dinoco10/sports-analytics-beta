"""
bankroll.py — Bankroll Management & Kelly Sizing
=================================================
Manages bankroll for MLB betting: tracks bets, calculates optimal
stake sizes using Kelly criterion, and reports performance metrics.

Key concepts:
- Kelly criterion: mathematically optimal bet sizing based on edge
- Quarter-Kelly: betting 25% of full Kelly — much safer, smoother equity curve
- Max drawdown: largest peak-to-trough decline — measures worst-case pain

Usage:
    from src.betting.bankroll import KellyCalculator, BankrollManager

    kelly = KellyCalculator()
    stake = kelly.recommended_stake(prob=0.55, ml=-110, bankroll=10000)

    mgr = BankrollManager(starting_bankroll=10000)
    bet_id = mgr.place_bet(amount=250, ml=-110, team="Yankees", date="2025-08-01")
    mgr.settle_bet(bet_id, won=True)
    print(mgr.get_stats())
"""

import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import pandas as pd

# Fix Windows encoding
if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ═══════════════════════════════════════════════════════════════
# ODDS HELPERS
# ═══════════════════════════════════════════════════════════════
# These mirror daily_bets.py but live here so the module is self-contained.

def american_to_decimal(ml: float) -> float:
    """Convert American moneyline to decimal odds.

    Examples:
      -150 → 1.667  (bet $150 to win $100, get back $166.67)
      +130 → 2.300  (bet $100 to win $130, get back $230)
    """
    if ml < 0:
        return 1 + 100 / abs(ml)
    else:
        return 1 + ml / 100


def american_to_implied_prob(ml: float) -> float:
    """Convert American moneyline to raw implied probability.

    This includes the vig (sportsbook margin), so probabilities
    for both sides will sum to more than 100%.
    """
    if ml < 0:
        return abs(ml) / (abs(ml) + 100)
    else:
        return 100 / (ml + 100)


def ml_payout(stake: float, ml: float, won: bool) -> float:
    """Calculate profit/loss from a moneyline bet.

    Returns positive profit if won, negative stake if lost.
    """
    if not won:
        return -stake
    if ml < 0:
        return stake * (100 / abs(ml))
    else:
        return stake * (ml / 100)


# ═══════════════════════════════════════════════════════════════
# KELLY CALCULATOR
# ═══════════════════════════════════════════════════════════════

class KellyCalculator:
    """Calculates optimal bet sizes using Kelly criterion.

    The Kelly criterion tells you what fraction of your bankroll
    to bet to maximize long-term growth. In practice, full Kelly
    is too aggressive (huge swings), so we use fractional Kelly.

    The formula:
        f* = (p * (d - 1) - (1 - p)) / (d - 1)

    Where:
        p = your estimated probability of winning
        d = decimal odds
        f* = fraction of bankroll to bet
    """

    def full_kelly(self, prob: float, ml: float) -> float:
        """Full Kelly fraction — mathematically optimal but volatile.

        Args:
            prob: Model's estimated win probability (0-1)
            ml: American moneyline odds

        Returns:
            Fraction of bankroll to bet (0.0 to 1.0)
        """
        dec_odds = american_to_decimal(ml)
        denom = dec_odds - 1
        if denom <= 0:
            return 0.0

        # Kelly formula: f* = (p * b - q) / b
        # where b = decimal_odds - 1, q = 1 - p
        kelly_f = (prob * denom - (1 - prob)) / denom
        return max(kelly_f, 0.0)

    def fractional_kelly(self, prob: float, ml: float,
                         fraction: float = 0.25) -> float:
        """Fractional Kelly — safer, smoother equity curve.

        Quarter-Kelly (fraction=0.25) is the sweet spot:
        - Captures ~75% of full Kelly's growth rate
        - But with ~50% less variance (drawdowns)

        Args:
            prob: Model's estimated win probability
            ml: American moneyline odds
            fraction: Kelly fraction (0.25 = quarter Kelly)

        Returns:
            Fraction of bankroll to bet
        """
        return self.full_kelly(prob, ml) * fraction

    def kelly_with_cap(self, prob: float, ml: float,
                       max_bet_pct: float = 0.05) -> float:
        """Kelly with a hard cap on maximum bet size.

        Even quarter-Kelly can suggest large bets on high-edge spots.
        This caps the max at 5% of bankroll regardless of edge.

        Args:
            prob: Model's estimated win probability
            ml: American moneyline odds
            max_bet_pct: Maximum fraction of bankroll per bet

        Returns:
            Fraction of bankroll to bet (capped)
        """
        kelly_f = self.fractional_kelly(prob, ml)
        return min(kelly_f, max_bet_pct)

    def recommended_stake(self, prob: float, ml: float,
                          bankroll: float,
                          fraction: float = 0.25,
                          max_bet_pct: float = 0.05) -> float:
        """Get dollar amount to bet — the main method you'll use.

        Combines fractional Kelly with a hard cap for safety.

        Args:
            prob: Model's estimated win probability
            ml: American moneyline odds
            bankroll: Current bankroll in dollars
            fraction: Kelly fraction (default quarter)
            max_bet_pct: Max bet as fraction of bankroll

        Returns:
            Dollar amount to bet
        """
        frac = min(self.fractional_kelly(prob, ml, fraction), max_bet_pct)
        return round(bankroll * frac, 2)


# ═══════════════════════════════════════════════════════════════
# BET RECORD
# ═══════════════════════════════════════════════════════════════

@dataclass
class BetRecord:
    """One bet — tracks everything from placement to settlement.

    Fields:
        bet_id: Unique identifier (auto-incremented)
        date: Game date (YYYY-MM-DD)
        team: Team we're betting on
        opponent: The other team
        side: "HOME" or "AWAY"
        ml: American moneyline at time of bet
        prob: Model's estimated win probability
        amount: Dollar amount wagered
        strategy: How we sized it ("kelly", "flat", etc.)
        result: "pending", "win", or "loss"
        profit_loss: Dollar P/L after settlement
        bankroll_after: Bankroll balance after this bet settles
    """
    bet_id: int
    date: str
    team: str
    opponent: str = ""
    side: str = ""
    ml: float = 0.0
    prob: float = 0.0
    amount: float = 0.0
    strategy: str = "kelly"
    result: str = "pending"
    profit_loss: float = 0.0
    bankroll_after: float = 0.0
    placed_at: str = field(default_factory=lambda: datetime.now().isoformat())


# ═══════════════════════════════════════════════════════════════
# BANKROLL MANAGER
# ═══════════════════════════════════════════════════════════════

class BankrollManager:
    """Tracks bankroll, places bets, settles results, reports stats.

    This is the central class for managing your betting bankroll.
    It maintains a history of all bets and computes performance metrics.

    Usage:
        mgr = BankrollManager(starting_bankroll=10000)

        # Place a bet
        bet_id = mgr.place_bet(
            amount=250, ml=-110,
            team="Yankees", opponent="Red Sox",
            side="HOME", prob=0.58, date="2025-08-01"
        )

        # Settle after the game
        mgr.settle_bet(bet_id, won=True)

        # Check performance
        print(mgr.get_stats())
        print(mgr.get_history())
    """

    def __init__(self, starting_bankroll: float = 10_000.0):
        self.starting_bankroll = starting_bankroll
        self.current_bankroll = starting_bankroll
        self.bets: list[BetRecord] = []
        self._next_id = 1

        # Track peak for drawdown calculation
        self._peak_bankroll = starting_bankroll
        self._max_drawdown = 0.0

        # Bankroll snapshots over time (for equity curve)
        self.equity_curve: list[tuple[str, float]] = [
            ("start", starting_bankroll)
        ]

    def place_bet(self, amount: float, ml: float, team: str,
                  opponent: str = "", side: str = "",
                  prob: float = 0.0, date: str = "",
                  strategy: str = "kelly") -> int:
        """Place a bet — deducts from bankroll and logs it.

        Args:
            amount: Dollar amount to wager
            ml: American moneyline odds
            team: Team we're betting on
            opponent: Other team
            side: "HOME" or "AWAY"
            prob: Model probability
            date: Game date
            strategy: Sizing method used

        Returns:
            bet_id for later settlement
        """
        if amount > self.current_bankroll:
            amount = self.current_bankroll  # Can't bet more than you have

        if amount <= 0:
            return -1

        bet = BetRecord(
            bet_id=self._next_id,
            date=date,
            team=team,
            opponent=opponent,
            side=side,
            ml=ml,
            prob=prob,
            amount=amount,
            strategy=strategy,
            bankroll_after=self.current_bankroll,  # Updated on settle
        )

        self.bets.append(bet)
        self._next_id += 1
        return bet.bet_id

    def settle_bet(self, bet_id: int, won: bool) -> float:
        """Settle a bet — updates bankroll based on result.

        Args:
            bet_id: ID from place_bet()
            won: True if our team won

        Returns:
            Profit/loss amount
        """
        bet = self._find_bet(bet_id)
        if bet is None or bet.result != "pending":
            return 0.0

        # Calculate P/L
        pl = ml_payout(bet.amount, bet.ml, won)
        bet.result = "win" if won else "loss"
        bet.profit_loss = pl
        self.current_bankroll += pl
        bet.bankroll_after = self.current_bankroll

        # Update peak and drawdown tracking
        if self.current_bankroll > self._peak_bankroll:
            self._peak_bankroll = self.current_bankroll
        else:
            drawdown = (self._peak_bankroll - self.current_bankroll) / self._peak_bankroll
            if drawdown > self._max_drawdown:
                self._max_drawdown = drawdown

        # Record equity curve point
        self.equity_curve.append((bet.date, self.current_bankroll))

        return pl

    def _find_bet(self, bet_id: int) -> Optional[BetRecord]:
        """Look up a bet by ID."""
        for bet in self.bets:
            if bet.bet_id == bet_id:
                return bet
        return None

    def get_history(self) -> pd.DataFrame:
        """Return all bets as a DataFrame — easy to filter and analyze."""
        if not self.bets:
            return pd.DataFrame()

        records = []
        for b in self.bets:
            records.append({
                "bet_id": b.bet_id,
                "date": b.date,
                "team": b.team,
                "opponent": b.opponent,
                "side": b.side,
                "ml": b.ml,
                "prob": b.prob,
                "amount": b.amount,
                "strategy": b.strategy,
                "result": b.result,
                "profit_loss": b.profit_loss,
                "bankroll_after": b.bankroll_after,
            })
        return pd.DataFrame(records)

    def get_stats(self) -> dict:
        """Compute overall performance metrics.

        Returns dict with:
            - total_bets, wins, losses, pending
            - win_rate: percentage of settled bets won
            - total_profit_loss: cumulative P/L in dollars
            - roi: return on investment (P/L / total staked)
            - max_drawdown: worst peak-to-trough decline (%)
            - current_bankroll: what you have now
            - units_won: P/L divided by average stake
        """
        settled = [b for b in self.bets if b.result != "pending"]
        pending = [b for b in self.bets if b.result == "pending"]
        wins = [b for b in settled if b.result == "win"]
        losses = [b for b in settled if b.result == "loss"]

        total_staked = sum(b.amount for b in settled)
        total_pl = sum(b.profit_loss for b in settled)
        avg_stake = total_staked / len(settled) if settled else 0

        return {
            "total_bets": len(self.bets),
            "settled": len(settled),
            "pending": len(pending),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(settled) if settled else 0.0,
            "total_staked": round(total_staked, 2),
            "total_profit_loss": round(total_pl, 2),
            "roi": round(total_pl / total_staked * 100, 2) if total_staked else 0.0,
            "max_drawdown": round(self._max_drawdown * 100, 2),
            "current_bankroll": round(self.current_bankroll, 2),
            "starting_bankroll": self.starting_bankroll,
            "units_won": round(total_pl / avg_stake, 2) if avg_stake else 0.0,
        }

    def reset(self):
        """Reset bankroll to starting state — useful for backtesting."""
        self.current_bankroll = self.starting_bankroll
        self.bets = []
        self._next_id = 1
        self._peak_bankroll = self.starting_bankroll
        self._max_drawdown = 0.0
        self.equity_curve = [("start", self.starting_bankroll)]

    def print_summary(self):
        """Pretty-print performance summary."""
        stats = self.get_stats()

        print(f"\n{'=' * 55}")
        print(f"  BANKROLL SUMMARY")
        print(f"{'=' * 55}")
        print(f"  Starting:   ${stats['starting_bankroll']:>10,.2f}")
        print(f"  Current:    ${stats['current_bankroll']:>10,.2f}")
        print(f"  P/L:        ${stats['total_profit_loss']:>+10,.2f}")
        print(f"")
        print(f"  Bets:       {stats['settled']} settled, {stats['pending']} pending")
        print(f"  Record:     {stats['wins']}W-{stats['losses']}L "
              f"({stats['win_rate']:.1%})")
        print(f"  ROI:        {stats['roi']:+.1f}%")
        print(f"  Max DD:     {stats['max_drawdown']:.1f}%")
        print(f"  Units won:  {stats['units_won']:+.1f}")
