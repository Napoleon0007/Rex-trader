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
    zscore: float
    er: float          # Kaufman efficiency ratio (ranging vs trending)
    mom_return: float  # trailing return over tsmom_lookback bars
    mom_frac: float    # fraction of ensemble horizons trending up (0..1)

    @property
    def valid(self) -> bool:
        return (
            self.price > 0
            and not np.isnan(self.short_prev)
            and not np.isnan(self.long_prev)
            and not np.isnan(self.zscore)
            and not np.isnan(self.mom_return)
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

    mr_mean = close.rolling(cfg.mr_window).mean()
    mr_std = close.rolling(cfg.mr_window).std()
    zscore = (close - mr_mean) / mr_std.replace(0.0, np.nan)

    net_move = (close - close.shift(cfg.regime_window)).abs()
    path = close.diff().abs().rolling(cfg.regime_window).sum()
    er = net_move / path.replace(0.0, np.nan)

    mom_return = close.pct_change(cfg.tsmom_lookback)
    # Fraction of ensemble horizons whose trailing return is positive (0..1).
    positives = [(close.pct_change(lb) > 0).astype(float) for lb in cfg.tsmom_lookbacks]
    mom_frac = sum(positives) / len(positives)

    return pd.DataFrame({
        "price": close, "short_ma": short, "long_ma": long,
        "short_prev": short.shift(1), "long_prev": long.shift(1),
        "recent_vol": recent_vol.fillna(0.0), "trend_up": close > trend_ema,
        "atr": atr.fillna(0.0), "rsi": rsi, "volume_ok": volume_ok,
        "zscore": zscore, "er": er, "mom_return": mom_return, "mom_frac": mom_frac,
    })


def features_from_window(bars: pd.DataFrame, cfg) -> Features:
    """Latest Features from a trailing window — used live (once per tick)."""
    if bars.empty:
        return Features(0, 0, 0, np.nan, np.nan, 0, True, 0, 50, True, np.nan, np.nan, np.nan, 0.0)
    row = build_features_frame(bars, cfg).iloc[-1]
    return _row_to_features(row)


def _row_to_features(row) -> Features:
    return Features(
        price=float(row.price), short_ma=float(row.short_ma), long_ma=float(row.long_ma),
        short_prev=float(row.short_prev), long_prev=float(row.long_prev),
        recent_vol=float(row.recent_vol), trend_up=bool(row.trend_up),
        atr=float(row.atr), rsi=float(row.rsi), volume_ok=bool(row.volume_ok),
        zscore=float(row.zscore), er=float(row.er), mom_return=float(row.mom_return),
        mom_frac=float(row.mom_frac),
    )


def _signals(feat: Features, cfg) -> tuple:
    """(raw_buy, raw_sell, confidence, entry_reason, exit_reason) for the active strategy.

    confidence in [0,1] scales position size; entry/exit reasons are for logging.
    """
    if cfg.strategy == "tsmom":
        if cfg.tsmom_ensemble:
            buy = feat.mom_frac >= cfg.tsmom_enter_frac
            sell = feat.mom_frac < cfg.tsmom_exit_frac
            conf = feat.mom_frac if buy else 0.0
            return buy, sell, conf, "ensemble momentum up", "ensemble momentum down"
        buy = feat.mom_return > 0
        sell = feat.mom_return < 0
        conf = max(0.0, min(1.0, feat.mom_return / cfg.tsmom_conf_scale)) if buy else 0.0
        return buy, sell, conf, "momentum up", "momentum down"
    if cfg.strategy == "meanrev":
        buy = feat.zscore <= -cfg.mr_entry_z
        sell = feat.zscore >= cfg.mr_exit_z
        # Deeper than the entry threshold = more conviction in the bounce.
        conf = max(0.0, min(1.0, (abs(feat.zscore) - cfg.mr_entry_z) / cfg.mr_entry_z)) if buy else 0.0
        return buy, sell, conf, "oversold (z<=-%.1f)" % cfg.mr_entry_z, "reverted to mean"
    # crossover (default trend-following)
    buy = feat.short_prev <= feat.long_prev and feat.short_ma > feat.long_ma
    sell = feat.short_prev >= feat.long_prev and feat.short_ma < feat.long_ma
    conf = position_confidence(
        feat.short_ma, feat.long_ma, feat.price, feat.recent_vol, cfg.confidence_vol_mult,
    )
    return buy, sell, conf, "ema-cross-up", "ema-cross-down"


def _blend_fraction(conf: float, feat: Features, cfg) -> float:
    fraction = cfg.position_fraction + (cfg.max_position_fraction - cfg.position_fraction) * conf
    if cfg.vol_target > 0 and feat.recent_vol > 0:
        fraction *= max(cfg.vol_scale_min, min(1.0, cfg.vol_target / feat.recent_vol))
    return fraction


def decide(feat: Features, pos: PositionState, cfg, funding_rate=None) -> Decision:
    """Decide what to do on one bar. Pure: reads state, mutates nothing."""
    raw_buy, raw_sell, conf, entry_reason, exit_reason = _signals(feat, cfg)
    signal = "buy" if raw_buy else "sell" if raw_sell else "hold"

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
        elif cfg.take_profit_atr > 0 and feat.atr > 0 and feat.price >= entry + cfg.take_profit_atr * feat.atr:
            d.action, d.reason = Action.EXIT, "atr-target"
        elif cfg.max_hold_bars > 0 and pos.bars_held >= cfg.max_hold_bars:
            d.action, d.reason = Action.EXIT, "time-stop"
        elif raw_sell:
            d.action, d.reason = Action.EXIT, exit_reason
        elif cfg.trend_exit and cfg.trend_filter_enabled and not feat.trend_up:
            d.action, d.reason = Action.EXIT, "trend-flip"
        else:
            d.reason = "hold position"
        return d

    # Flat: look for an entry.
    if pos.bars_since_exit < cfg.cooldown_bars:
        d.reason = "cooldown"
        return d
    if not raw_buy:
        d.reason = "no entry signal"
        return d
    if cfg.trend_filter_enabled and not feat.trend_up:
        d.reason = "skip buy: trend down"
        return d
    if cfg.regime_filter_enabled and (np.isnan(feat.er) or feat.er > cfg.regime_er_max):
        d.reason = "skip buy: trending (not ranging)"
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

    d.action, d.reason = Action.ENTER, entry_reason
    d.fraction, d.confidence = _blend_fraction(conf, feat, cfg), conf
    return d
