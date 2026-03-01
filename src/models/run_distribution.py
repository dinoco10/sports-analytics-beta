"""
run_distribution.py — Negative Binomial Run Distribution Model
===============================================================
Converts point predictions (expected home/away runs) into full
probability distributions using negative binomial parameterization.

Architecture:
  1. LightGBM predicts mu_home, mu_away (conditional means)
  2. League-wide phi (overdispersion) estimated from training residuals
  3. P(home=k) = NB(k | mu_home, phi_home) for k=0..MAX_RUNS
  4. P(away=k) = NB(k | mu_away, phi_away) for k=0..MAX_RUNS
  5. P(total=t) = convolution of home and away distributions
  6. P(over line) = sum P(total > line) + 0.5 * P(total = line)

Why Negative Binomial over Poisson:
  - MLB runs are OVERDISPERSED (variance > mean)
  - Poisson assumes Var(Y) = mu; real data shows Var(Y) ≈ 1.3 * mu
  - NB adds one parameter (phi) to capture this extra variance
  - Better calibrated tail probabilities = better O/U predictions

Usage:
  from src.models.run_distribution import RunDistribution
  rd = RunDistribution(phi_home=5.0, phi_away=5.0)
  p_over, p_under = rd.over_under_probability(mu_home=4.5, mu_away=3.8, line=8.5)
"""

import numpy as np
from scipy.stats import nbinom
from typing import Tuple, Dict, Optional


MAX_RUNS = 20  # Maximum runs per team to model (covers 99.9%+ of games)


class RunDistribution:
    """
    Full probability distribution for game run totals.

    Parameterized by two negative binomial distributions (home/away)
    that are convolved to produce total runs probabilities.
    """

    def __init__(self, phi_home: float = 5.0, phi_away: float = 5.0,
                 shutout_boost: float = 0.02):
        """
        Parameters
        ----------
        phi_home : float
            Overdispersion parameter for home runs.
            Higher phi = closer to Poisson. phi=infinity is exact Poisson.
            Typical MLB: phi ≈ 4-7.
        phi_away : float
            Overdispersion for away runs.
        shutout_boost : float
            Zero-inflation correction — boosts P(0 runs) by this amount
            and renormalizes. Empirically, shutouts happen ~3-4% more than
            NB predicts, likely due to ace-vs-weak-lineup matchups.
        """
        self.phi_home = phi_home
        self.phi_away = phi_away
        self.shutout_boost = shutout_boost

    def _nb_pmf(self, k: np.ndarray, mu: float, phi: float) -> np.ndarray:
        """
        Negative binomial PMF parameterized by mean (mu) and overdispersion (phi).

        scipy's nbinom uses (n, p) parameterization where:
          n = phi (number of successes)
          p = phi / (phi + mu)

        This gives: E[Y] = mu, Var(Y) = mu + mu^2/phi
        """
        if mu <= 0:
            # Degenerate case: predict 0 runs with certainty
            result = np.zeros_like(k, dtype=float)
            result[0] = 1.0
            return result

        n = phi
        p = phi / (phi + mu)

        pmf = nbinom.pmf(k, n, p)

        # Zero-inflation correction
        if self.shutout_boost > 0:
            pmf[0] += self.shutout_boost
            pmf = pmf / pmf.sum()  # Renormalize

        return pmf

    def predict_run_distribution(self, mu_home: float, mu_away: float
                                  ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate P(home=k) and P(away=k) for k=0..MAX_RUNS.

        Parameters
        ----------
        mu_home : float
            Expected home runs (from LightGBM regressor)
        mu_away : float
            Expected away runs

        Returns
        -------
        p_home : ndarray of shape (MAX_RUNS+1,)
            P(home_runs = k) for k = 0, 1, ..., MAX_RUNS
        p_away : ndarray of shape (MAX_RUNS+1,)
            P(away_runs = k) for k = 0, 1, ..., MAX_RUNS
        """
        k = np.arange(MAX_RUNS + 1)
        p_home = self._nb_pmf(k, mu_home, self.phi_home)
        p_away = self._nb_pmf(k, mu_away, self.phi_away)

        return p_home, p_away

    def game_total_distribution(self, mu_home: float, mu_away: float
                                 ) -> np.ndarray:
        """
        Compute P(total_runs = t) for t = 0, 1, ..., 2*MAX_RUNS.

        Uses convolution of home and away distributions:
        P(total = t) = sum_{k=0}^{t} P(home=k) * P(away=t-k)

        This is exact (not approximate) because we enumerate all
        combinations of home and away scores that sum to t.

        Returns
        -------
        p_total : ndarray of shape (2*MAX_RUNS+1,)
            P(total_runs = t) for t = 0, 1, ..., 2*MAX_RUNS
        """
        p_home, p_away = self.predict_run_distribution(mu_home, mu_away)
        p_total = np.convolve(p_home, p_away)
        return p_total

    def over_under_probability(self, mu_home: float, mu_away: float,
                                line: float) -> Tuple[float, float]:
        """
        Compute P(over) and P(under) for a given total line.

        For half-integer lines (8.5, 9.5, etc.):
          P(over) = P(total > line)
          P(under) = P(total < line)
          P(push) = 0

        For integer lines (8, 9, etc.):
          P(over) = P(total > line)
          P(under) = P(total < line)
          P(push) = P(total = line)
          We split push: P(over) += 0.5 * P(push), P(under) += 0.5 * P(push)

        Returns
        -------
        p_over : float
        p_under : float
        """
        p_total = self.game_total_distribution(mu_home, mu_away)

        # Is line a half-integer?
        is_half = (line % 1) != 0

        if is_half:
            threshold = int(np.ceil(line))
            p_over = p_total[threshold:].sum()
            p_under = 1.0 - p_over
        else:
            line_int = int(line)
            p_push = p_total[line_int] if line_int < len(p_total) else 0.0
            p_over = p_total[line_int + 1:].sum() + 0.5 * p_push
            p_under = p_total[:line_int].sum() + 0.5 * p_push

        return float(p_over), float(p_under)

    def run_line_probability(self, mu_home: float, mu_away: float,
                              spread: float = -1.5) -> Tuple[float, float]:
        """
        Compute probability of covering a run line (spread).

        Standard MLB run line: home -1.5 (favorite must win by 2+)

        Parameters
        ----------
        spread : float
            Run line spread (negative = home favored).
            -1.5: home must win by 2+
            +1.5: away must win or lose by 1

        Returns
        -------
        p_cover : float — P(home covers spread)
        p_not_cover : float — P(home does NOT cover)
        """
        p_home, p_away = self.predict_run_distribution(mu_home, mu_away)

        # Compute P(home_score - away_score > -spread)
        # For spread=-1.5: need home_score - away_score >= 2
        p_cover = 0.0

        for h in range(MAX_RUNS + 1):
            for a in range(MAX_RUNS + 1):
                margin = h - a
                if margin > -spread:
                    p_cover += p_home[h] * p_away[a]
                elif margin == -spread and (spread % 1) == 0:
                    # Push on integer spread — split
                    p_cover += 0.5 * p_home[h] * p_away[a]

        return float(p_cover), float(1.0 - p_cover)

    def full_game_analysis(self, mu_home: float, mu_away: float,
                            total_line: float = 8.5,
                            run_line: float = -1.5) -> Dict:
        """
        Complete probabilistic analysis for one game.

        Returns dict with all betting-relevant probabilities.
        """
        p_home, p_away = self.predict_run_distribution(mu_home, mu_away)
        p_total = self.game_total_distribution(mu_home, mu_away)

        p_over, p_under = self.over_under_probability(mu_home, mu_away, total_line)
        p_cover_home, p_cover_away = self.run_line_probability(
            mu_home, mu_away, run_line
        )

        # Most likely scores (mode)
        home_mode = int(np.argmax(p_home))
        away_mode = int(np.argmax(p_away))
        total_mode = int(np.argmax(p_total))

        # Win probability from run distribution (secondary check vs main model)
        p_home_win = 0.0
        for h in range(MAX_RUNS + 1):
            for a in range(MAX_RUNS + 1):
                if h > a:
                    p_home_win += p_home[h] * p_away[a]
                elif h == a:
                    p_home_win += 0.5 * p_home[h] * p_away[a]  # Tie → coin flip

        return {
            "mu_home": round(mu_home, 3),
            "mu_away": round(mu_away, 3),
            "mu_total": round(mu_home + mu_away, 3),
            "total_line": total_line,
            "p_over": round(p_over, 4),
            "p_under": round(p_under, 4),
            "run_line": run_line,
            "p_home_cover": round(p_cover_home, 4),
            "p_away_cover": round(p_cover_away, 4),
            "p_home_win_dist": round(p_home_win, 4),
            "home_mode": home_mode,
            "away_mode": away_mode,
            "total_mode": total_mode,
        }


# ─── Convenience functions ──────────────────────────────────

def create_distribution(phi_home: float = 5.0, phi_away: float = 5.0) -> RunDistribution:
    """Create a RunDistribution with estimated phi parameters."""
    return RunDistribution(phi_home=phi_home, phi_away=phi_away)


# ─── Standalone test ─────────────────────────────────────────

if __name__ == "__main__":
    print("Run Distribution Model — Test")
    print("=" * 50)

    rd = RunDistribution(phi_home=5.0, phi_away=5.0)

    # Test: Average MLB game (home=4.5, away=4.0)
    result = rd.full_game_analysis(mu_home=4.5, mu_away=4.0, total_line=8.5)

    print(f"\nTest game: home={result['mu_home']}, away={result['mu_away']}")
    print(f"  Total line: {result['total_line']}")
    print(f"  P(over):  {result['p_over']:.4f}")
    print(f"  P(under): {result['p_under']:.4f}")
    print(f"  Run line: {result['run_line']}")
    print(f"  P(home covers -1.5): {result['p_home_cover']:.4f}")
    print(f"  P(away covers +1.5): {result['p_away_cover']:.4f}")
    print(f"  P(home wins):        {result['p_home_win_dist']:.4f}")
    print(f"  Most likely score:   {result['home_mode']}-{result['away_mode']}")
    print(f"  Most likely total:   {result['total_mode']}")

    # Test: Coors Field blowout (home=6.5, away=5.5)
    result2 = rd.full_game_analysis(mu_home=6.5, mu_away=5.5, total_line=11.5)
    print(f"\nCoors game: home={result2['mu_home']}, away={result2['mu_away']}")
    print(f"  P(over 11.5): {result2['p_over']:.4f}")
    print(f"  P(under 11.5): {result2['p_under']:.4f}")

    # Test: Pitchers' duel (home=2.5, away=2.0)
    result3 = rd.full_game_analysis(mu_home=2.5, mu_away=2.0, total_line=7.5)
    print(f"\nPitchers duel: home={result3['mu_home']}, away={result3['mu_away']}")
    print(f"  P(over 7.5): {result3['p_over']:.4f}")
    print(f"  P(under 7.5): {result3['p_under']:.4f}")
