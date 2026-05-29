"""Monte Carlo robustness check on a backtest's trade sequence.

A single backtest is one ordering of trades — you might just have got lucky (or
unlucky). We bootstrap-resample the per-trade returns thousands of times to see
the DISTRIBUTION of outcomes: where the median lands, how bad a plausible bad run
is, and how often the account ends underwater.

Assumption: each trade is resampled as a return on capital and compounded full-
size. It ignores position sizing, so read it as the shape of risk, not a forecast.

  python montecarlo.py                          # uses logs/backtest_trades.csv
  python montecarlo.py --sims 10000 --ruin 0.5
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

LOG_DIR = Path(__file__).parent / "logs"


def simulate(pnl_pct: np.ndarray, sims: int = 5000, ruin_level: float = 0.5, seed: int = 0):
    """Bootstrap-resample the trade returns. Returns a dict of outcome stats."""
    rng = np.random.default_rng(seed)
    n = len(pnl_pct)
    rmult = 1.0 + pnl_pct / 100.0

    final, max_dd, ruined = np.empty(sims), np.empty(sims), 0
    for i in range(sims):
        path = rng.choice(rmult, size=n, replace=True)
        equity = np.cumprod(path)
        final[i] = equity[-1]
        peak = np.maximum.accumulate(equity)
        max_dd[i] = float((1.0 - equity / peak).max())
        if equity.min() <= ruin_level:
            ruined += 1

    pct = lambda a, q: float(np.percentile(a, q))
    return {
        "n_trades": n,
        "median_return_pct": (pct(final, 50) - 1.0) * 100.0,
        "p5_return_pct": (pct(final, 5) - 1.0) * 100.0,
        "p95_return_pct": (pct(final, 95) - 1.0) * 100.0,
        "prob_loss_pct": float((final < 1.0).mean() * 100.0),
        "median_maxdd_pct": pct(max_dd, 50) * 100.0,
        "p95_maxdd_pct": pct(max_dd, 95) * 100.0,
        "risk_of_ruin_pct": ruined / sims * 100.0,
        "ruin_level": ruin_level,
    }


def format_report(r: dict) -> str:
    return "\n".join([
        "", "=" * 52, "  MONTE CARLO (bootstrapped trade sequence)", "=" * 52,
        f"  Trades resampled : {r['n_trades']}",
        f"  Return  median   : {r['median_return_pct']:+.2f}%",
        f"          5–95%    : {r['p5_return_pct']:+.2f}%  …  {r['p95_return_pct']:+.2f}%",
        f"  P(ending in loss): {r['prob_loss_pct']:.1f}%",
        f"  Max drawdown med : {r['median_maxdd_pct']:.1f}%   (95th pct {r['p95_maxdd_pct']:.1f}%)",
        f"  Risk of ruin     : {r['risk_of_ruin_pct']:.1f}%  (equity ≤ {r['ruin_level']:.0%} of start)",
        "=" * 52, "",
    ])


def main():
    ap = argparse.ArgumentParser(description="Monte Carlo on backtest trades")
    ap.add_argument("--trades", default=str(LOG_DIR / "backtest_trades.csv"))
    ap.add_argument("--sims", type=int, default=5000)
    ap.add_argument("--ruin", type=float, default=0.5, help="ruin threshold as fraction of start")
    args = ap.parse_args()

    path = Path(args.trades)
    if not path.exists():
        print(f"No trades file at {path}. Run:  python backtest.py --save")
        return
    trades = pd.read_csv(path)
    if "pnl_pct" not in trades or trades.empty:
        print("Trades file has no pnl_pct rows — nothing to simulate.")
        return
    if len(trades) < 10:
        print(f"Only {len(trades)} trades — Monte Carlo needs ~30+ to be meaningful.")

    r = simulate(trades["pnl_pct"].to_numpy(), args.sims, args.ruin)
    print(format_report(r))


if __name__ == "__main__":
    main()
