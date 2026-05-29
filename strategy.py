"""EMA crossover strategy. Pure function — no I/O.

EMA (exponential moving average) weights recent prices more heavily than the
older simple moving average. Every paper in knowledge/applied/ that tests
crossover strategies on BTC uses EMA, not SMA: faster reaction to real moves,
less lag at the cost of slightly noisier signals.
"""
from dataclasses import dataclass
from enum import Enum

import pandas as pd


class Signal(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class StrategyResult:
    signal: Signal
    short_ma: float
    long_ma: float
    last_price: float
    reason: str


def ma_crossover(
    bars: pd.DataFrame,
    short_window: int,
    long_window: int,
    has_position: bool,
) -> StrategyResult:
    if bars.empty or len(bars) < long_window + 1:
        return StrategyResult(
            Signal.HOLD, 0.0, 0.0, 0.0,
            f"not enough bars yet ({len(bars)}/{long_window + 1})",
        )

    closes = bars["close"]
    short_ma = closes.ewm(span=short_window, adjust=False).mean()
    long_ma = closes.ewm(span=long_window, adjust=False).mean()

    s_now, s_prev = float(short_ma.iloc[-1]), float(short_ma.iloc[-2])
    l_now, l_prev = float(long_ma.iloc[-1]), float(long_ma.iloc[-2])
    last_price = float(closes.iloc[-1])

    crossed_up = s_prev <= l_prev and s_now > l_now
    crossed_down = s_prev >= l_prev and s_now < l_now

    if crossed_up and not has_position:
        return StrategyResult(
            Signal.BUY, s_now, l_now, last_price,
            "short EMA crossed above long EMA",
        )
    if crossed_down and has_position:
        return StrategyResult(
            Signal.SELL, s_now, l_now, last_price,
            "short EMA crossed below long EMA",
        )
    return StrategyResult(Signal.HOLD, s_now, l_now, last_price, "no crossover")
