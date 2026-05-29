"""Shared decision engine.

The live bot (bot.py) and the backtester (backtest.py) both drive `decide()`,
so the strategy they run is identical — no reimplementation drift.

Indicators are built once by `build_features_frame()` (vectorised over a whole
price series). The live bot calls it on its short trailing window each tick; the
backtester calls it once over all history and indexes per bar — same numbers,
~50x faster than recomputing inside the loop. `decide()` is pure logic over a
single `Features` snapshot.
"""
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from strategy import Signal, position_confidence


class Action(Enum):
    ENTER = "enter"
    EXIT = "exit"
    HOLD = "hold"


@dataclass
class PositionState:
    """Mutable per-position bookkeeping owned by the caller (bot or backtest)."""
    qty: float = 0.0
    entry_price: float = 0.0
    peak_price: float = 0.0          # highest close seen since entry (trailing stop)
    bars_held: int = 0
    bars_since_exit: int = 10 ** 9   # large => no cooldown before the first trade

    @property
    def has_position(self) -> bool:
        return self.qty > 0

    def on_bar(self, price: float) -> None:
        if self.has_position:
            self.peak_price = max(self.peak_price, price)
            self.bars_held += 1
        else:
            self.bars_since_exit += 1

    def on_enter(self, qty: float, price: float) -> None:
        self.qty = qty
        self.entry_price = price
        self.peak_price = price
        self.bars_held = 0

    def on_exit(self) -> None:
        self.qty = 0.0
        self.entry_price = 0.0
        self.peak_price = 0.0
        self.bars_held = 0
        self.bars_since_exit = 0


@dataclass
class Features:
    """One bar's indicator snapshot — the only thing decide() reads from the market."""
    price: float
    short_ma: float
    long_ma: float
    short_prev: float
    long_prev: float
    recent_vol: float
    trend_up: bool
    atr: float
    rsi: float
    volume_ok: bool

    @property
    def valid(self) -> bool:
        return (
            self.price > 0
            and not np.isnan(self.short_prev)
            and not np.isnan(self.long_prev)
        )


@dataclass
class Decision:
    action: Action
    fraction: float = 0.0     # ENTER: fraction of available cash to deploy
    reason: str = ""
    signal: str = "hold"
    short_ma: float = 0.0
    long_ma: float = 0.0
    last_price: float = 0.0
    confidence: float = 0.0
    recent_vol: float = 0.0


def slippage_fraction(recent_vol: float, cfg) -> float:
    """Base slippage (bps) plus a volatility-scaled component. Shared by bot + backtest."""
    return cfg.paper_slippage_bps / 10_000.0 + cfg.slippage_vol_mult * recent_vol


def build_features_frame(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Vectorised indicators over an OHLCV frame. Columns map 1:1 to Features."""
    close, high, low = df["close"], df["high"], df["low"]
    short = close.ewm(span=cfg.short_window, adjust=False).mean()
    long = close.ewm(span=cfg.long_window, adjust=False).mean()
    recent_vol = close.pct_change().rolling(cfg.long_window).std()
    trend_ema = close.ewm(span=cfg.trend_window, adjust=False).mean()

    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(cfg.atr_window).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / cfg.rsi_window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / cfg.rsi_window, adjust=False).mean()
    rsi = (100.0 - 100.0 / (1.0 + gain / loss.replace(0.0, np.nan))).fillna(100.0)

    if cfg.volume_mult > 0 and "volume" in df:
        vavg = df["volume"].rolling(cfg.volume_window).mean()
        volume_ok = (df["volume"] >= cfg.volume_mult * vavg).fillna(True)
    else:
        volume_ok = pd.Series(True, index=df.index)

    return pd.DataFrame({
        "price": close, "short_ma": short, "long_ma": long,
        "short_prev": short.shift(1), "long_prev": long.shift(1),
        "recent_vol": recent_vol.fillna(0.0), "trend_up": close > trend_ema,
        "atr": atr.fillna(0.0), "rsi": rsi, "volume_ok": volume_ok,
    })


def features_from_window(bars: pd.DataFrame, cfg) -> Features:
    """Latest Features from a trailing window — used live (once per tick)."""
    if bars.empty:
        return Features(0, 0, 0, np.nan, np.nan, 0, True, 0, 50, True)
    row = build_features_frame(bars, cfg).iloc[-1]
    return _row_to_features(row)


def _row_to_features(row) -> Features:
    return Features(
        price=float(row.price), short_ma=float(row.short_ma), long_ma=float(row.long_ma),
        short_prev=float(row.short_prev), long_prev=float(row.long_prev),
        recent_vol=float(row.recent_vol), trend_up=bool(row.trend_up),
        atr=float(row.atr), rsi=float(row.rsi), volume_ok=bool(row.volume_ok),
    )


def _entry_fraction(feat: Features, cfg) -> tuple:
    conf = position_confidence(
        feat.short_ma, feat.long_ma, feat.price, feat.recent_vol, cfg.confidence_vol_mult,
    )
    fraction = cfg.position_fraction + (cfg.max_position_fraction - cfg.position_fraction) * conf
    if cfg.vol_target > 0 and feat.recent_vol > 0:
        fraction *= max(cfg.vol_scale_min, min(1.0, cfg.vol_target / feat.recent_vol))
    return fraction, conf


def decide(feat: Features, pos: PositionState, cfg, funding_rate=None) -> Decision:
    """Decide what to do on one bar. Pure: reads state, mutates nothing."""
    crossed_up = feat.short_prev <= feat.long_prev and feat.short_ma > feat.long_ma
    crossed_down = feat.short_prev >= feat.long_prev and feat.short_ma < feat.long_ma
    signal = "buy" if crossed_up else "sell" if crossed_down else "hold"

    d = Decision(
        action=Action.HOLD, signal=signal, short_ma=feat.short_ma, long_ma=feat.long_ma,
        last_price=feat.price, recent_vol=feat.recent_vol,
    )
    if not feat.valid:
        d.reason = "warming up"
        return d

    if pos.has_position:
        entry = pos.entry_price
        if cfg.stop_loss_pct > 0 and feat.price <= entry * (1 - cfg.stop_loss_pct):
            d.action, d.reason = Action.EXIT, "stop-loss"
        elif cfg.atr_stop_mult > 0 and feat.atr > 0 and feat.price <= entry - cfg.atr_stop_mult * feat.atr:
            d.action, d.reason = Action.EXIT, "atr-stop"
        elif cfg.trail_stop_pct > 0 and feat.price <= pos.peak_price * (1 - cfg.trail_stop_pct):
            d.action, d.reason = Action.EXIT, "trailing-stop"
        elif cfg.take_profit_pct > 0 and feat.price >= entry * (1 + cfg.take_profit_pct):
            d.action, d.reason = Action.EXIT, "take-profit"
        elif crossed_down:
            d.action, d.reason = Action.EXIT, "ema-cross-down"
        elif cfg.trend_exit and cfg.trend_filter_enabled and not feat.trend_up:
            d.action, d.reason = Action.EXIT, "trend-flip"
        else:
            d.reason = "hold position"
        return d

    # Flat: look for an entry.
    if pos.bars_since_exit < cfg.cooldown_bars:
        d.reason = "cooldown"
        return d
    if not crossed_up:
        d.reason = "no crossover"
        return d
    if cfg.trend_filter_enabled and not feat.trend_up:
        d.reason = "skip buy: trend down"
        return d
    if not feat.volume_ok:
        d.reason = "skip buy: low volume"
        return d
    if cfg.rsi_max and 0 < cfg.rsi_max < 100 and feat.rsi > cfg.rsi_max:
        d.reason = "skip buy: overbought (RSI)"
        return d
    if funding_rate is not None and funding_rate > cfg.funding_max:
        d.reason = "skip buy: funding overheated"
        return d

    fraction, conf = _entry_fraction(feat, cfg)
    d.action, d.reason, d.fraction, d.confidence = Action.ENTER, "ema-cross-up", fraction, conf
    return d
