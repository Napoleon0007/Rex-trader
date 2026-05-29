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


def position_confidence(
    short_ma: float,
    long_ma: float,
    last_price: float,
    recent_vol: float,
    vol_mult: float,
) -> float:
    """Confidence in [0, 1] that the BUY 'view' is strong.

    Black-Litterman blends a neutral prior with a view weighted by conviction.
    Here conviction is the EMA gap normalised by recent volatility: a gap of
    `vol_mult` volatility-units counts as full confidence. At a fresh crossover
    the gap is near zero, so confidence (and size) starts modest and grows when
    we enter into an already-separating trend.
    """
    if last_price <= 0 or recent_vol <= 0 or vol_mult <= 0:
        return 0.0
    gap_norm = abs(short_ma - long_ma) / last_price
    return max(0.0, min(1.0, gap_norm / (vol_mult * recent_vol)))


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
