"""Performance metrics computed from an equity curve and a round-trip trade ledger.

Pure functions, no I/O. Annualisation assumes evenly-spaced bars of `granularity`
seconds. Per-minute Sharpe/Sortino are noisy by nature — read them alongside the
trade-level stats (win rate, profit factor, expectancy), not in isolation.
"""
import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

SECONDS_PER_YEAR = 365 * 24 * 3600


@dataclass
class Metrics:
    start_equity: float
    end_equity: float
    total_return_pct: float
    cagr_pct: float
    sharpe: float
    sortino: float
    max_drawdown_pct: float
    exposure_pct: float          # fraction of bars holding a position
    num_trades: int
    win_rate_pct: float
    profit_factor: float
    avg_win_pct: float
    avg_loss_pct: float
    expectancy_pct: float        # average P&L per trade, in %
    best_trade_pct: float
    worst_trade_pct: float
    buy_hold_return_pct: float


def max_drawdown(equity: pd.Series) -> float:
    """Worst peak-to-trough decline as a positive fraction (0.20 = -20%)."""
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    dd = (equity - running_max) / running_max
    return float(-dd.min())


def _annualisation_factor(granularity: int) -> float:
    return math.sqrt(SECONDS_PER_YEAR / granularity)


def compute(equity: pd.Series, trades: pd.DataFrame, prices: pd.Series,
            exposure_bars: int, granularity: int = 60) -> Metrics:
    start_eq = float(equity.iloc[0])
    end_eq = float(equity.iloc[-1])
    total_ret = (end_eq / start_eq - 1.0) * 100.0

    seconds = (equity.index[-1] - equity.index[0]).total_seconds() or granularity
    years = seconds / SECONDS_PER_YEAR
    cagr = ((end_eq / start_eq) ** (1.0 / years) - 1.0) * 100.0 if years > 0 and end_eq > 0 else 0.0

    bar_rets = equity.pct_change().dropna()
    ann = _annualisation_factor(granularity)
    std = float(bar_rets.std())
    sharpe = float(bar_rets.mean()) / std * ann if std > 0 else 0.0
    downside = bar_rets[bar_rets < 0]
    dstd = float(downside.std()) if len(downside) > 1 else 0.0
    sortino = float(bar_rets.mean()) / dstd * ann if dstd > 0 else 0.0

    mdd = max_drawdown(equity) * 100.0
    exposure = exposure_bars / len(equity) * 100.0 if len(equity) else 0.0
    bh = (float(prices.iloc[-1]) / float(prices.iloc[0]) - 1.0) * 100.0

    if not trades.empty and "pnl_pct" in trades:
        p = trades["pnl_pct"]
        wins, losses = p[p > 0], p[p <= 0]
        win_rate = len(wins) / len(p) * 100.0
        gross_win = float(wins.sum())
        gross_loss = float(-losses.sum())
        profit_factor = gross_win / gross_loss if gross_loss > 0 else float("inf")
        avg_win = float(wins.mean()) if len(wins) else 0.0
        avg_loss = float(losses.mean()) if len(losses) else 0.0
        expectancy = float(p.mean())
        best, worst = float(p.max()), float(p.min())
        n = len(p)
    else:
        win_rate = profit_factor = avg_win = avg_loss = expectancy = best = worst = 0.0
        n = 0

    return Metrics(
        start_equity=round(start_eq, 2), end_equity=round(end_eq, 2),
        total_return_pct=round(total_ret, 2), cagr_pct=round(cagr, 2),
        sharpe=round(sharpe, 2), sortino=round(sortino, 2),
        max_drawdown_pct=round(mdd, 2), exposure_pct=round(exposure, 1),
        num_trades=n, win_rate_pct=round(win_rate, 1),
        profit_factor=round(profit_factor, 2) if math.isfinite(profit_factor) else profit_factor,
        avg_win_pct=round(avg_win, 2), avg_loss_pct=round(avg_loss, 2),
        expectancy_pct=round(expectancy, 3),
        best_trade_pct=round(best, 2), worst_trade_pct=round(worst, 2),
        buy_hold_return_pct=round(bh, 2),
    )


def format_report(m: Metrics, title: str = "BACKTEST RESULTS") -> str:
    d = asdict(m)
    verdict = _verdict(m)
    lines = [
        "", "=" * 52, f"  {title}", "=" * 52,
        f"  Equity        ${m.start_equity:,.2f}  →  ${m.end_equity:,.2f}",
        f"  Total return  {m.total_return_pct:+.2f}%   (buy & hold {m.buy_hold_return_pct:+.2f}%)",
        f"  CAGR          {m.cagr_pct:+.2f}%",
        f"  Sharpe        {m.sharpe:.2f}        Sortino {m.sortino:.2f}",
        f"  Max drawdown  {m.max_drawdown_pct:.2f}%      Exposure {m.exposure_pct:.1f}%",
        "  " + "-" * 48,
        f"  Trades        {m.num_trades}        Win rate {m.win_rate_pct:.1f}%",
        f"  Profit factor {m.profit_factor}        Expectancy {m.expectancy_pct:+.3f}%/trade",
        f"  Avg win       {m.avg_win_pct:+.2f}%      Avg loss {m.avg_loss_pct:+.2f}%",
        f"  Best/worst    {m.best_trade_pct:+.2f}% / {m.worst_trade_pct:+.2f}%",
        "  " + "-" * 48,
        f"  Verdict: {verdict}",
        "=" * 52, "",
    ]
    return "\n".join(lines)


def _verdict(m: Metrics) -> str:
    if m.num_trades < 10:
        return "INCONCLUSIVE — too few trades to judge (need ~30+)."
    if m.expectancy_pct <= 0 or m.total_return_pct <= 0:
        return "NEGATIVE EDGE — loses money after costs. Do not deploy."
    if m.total_return_pct < m.buy_hold_return_pct:
        return "WEAK — profitable but underperforms buy & hold."
    if m.sharpe < 1.0:
        return "MARGINAL — positive but low risk-adjusted return."
    return "PROMISING — beats buy & hold with a positive Sharpe. Validate out-of-sample."
