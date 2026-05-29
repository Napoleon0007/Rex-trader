"""Rex Trader configuration. Tune these to change behavior."""
from dataclasses import dataclass


@dataclass
class Config:
    # Symbol on Coinbase Exchange public data feed (used for paper-mode prices).
    # BTC-USD = Bitcoin priced in US dollars. Other ideas: ETH-USD, SOL-USD.
    symbol: str = "BTC-USD"

    # Which signal generates entries/exits:
    #   "crossover" — EMA trend-following (tested: no edge, dead)
    #   "meanrev"   — buy statistical dips (tested: marginal)
    #   "tsmom"     — time-series momentum: long when the trailing lookback return
    #                 is positive, flat when negative. The most robust edge in the
    #                 literature (Moskowitz; Two Centuries of Trend Following).
    #                 Designed for daily bars — trades rarely, so costs barely matter.
    strategy: str = "tsmom"

    # EMA crossover parameters (computed on 1-minute bars).
    # EMA spans behave like SMA periods of the same length but weight recent
    # prices more — see knowledge/applied/ for the supporting papers.
    short_window: int = 9
    long_window: int = 21

    # Mean-reversion parameters. z = (price - rolling mean) / rolling std over
    # mr_window bars. Enter long when z <= -mr_entry_z (statistically cheap),
    # exit when z >= mr_exit_z (reverted to / above the mean).
    mr_window: int = 20
    mr_entry_z: float = 2.0
    mr_exit_z: float = 0.0
    # MR-appropriate exits (fix the "falling mean books a loss" problem):
    # take profit at entry + k*ATR, and a time-stop after N bars if it hasn't worked.
    take_profit_atr: float = 0.0   # 0 = off
    max_hold_bars: int = 0         # 0 = off

    # Regime filter (mean-reversion only works in ranging markets). Kaufman
    # efficiency ratio = |net move| / |total path| over regime_window bars:
    # ~0 = choppy/ranging (good for MR), ~1 = strong trend (bad). Only enter
    # when ER <= regime_er_max.
    regime_filter_enabled: bool = False
    regime_window: int = 50
    regime_er_max: float = 0.30

    # Time-series momentum. mom = trailing return over tsmom_lookback bars. Long
    # while mom > 0, flat while mom < 0. tsmom_conf_scale is the lookback return
    # that counts as full-size conviction (0.20 = a +20% trailing move).
    tsmom_lookback: int = 30
    tsmom_conf_scale: float = 0.20
    # Ensemble: instead of betting on one "magic" lookback (which overfits and
    # broke out-of-sample), average the trend signal across several horizons and
    # size by the fraction that are positive. Far more robust (Moskowitz uses
    # multiple horizons). Long when >= enter_frac of horizons are up; exit below
    # exit_frac.
    tsmom_ensemble: bool = True
    tsmom_lookbacks: tuple = (40, 80, 120, 160, 200)
    tsmom_enter_frac: float = 0.6
    tsmom_exit_frac: float = 0.4

    # How often the bot wakes up and re-evaluates the market (seconds)
    poll_seconds: int = 30

    # Paper-portfolio starting cash (US dollars). Real money never touched in paper mode.
    paper_starting_cash: float = 100.00

    # Position sizing. position_fraction is the BASELINE (neutral prior) spent on
    # a BUY. In the Black-Litterman spirit we blend that prior with the strength
    # of the "view" — how far the short EMA has separated from the long EMA,
    # measured in units of recent volatility — scaling up toward max_position_fraction
    # as confidence rises. confidence_vol_mult sets how many vol-units of EMA gap
    # count as full confidence.
    position_fraction: float = 0.50
    max_position_fraction: float = 0.90
    confidence_vol_mult: float = 3.0

    # Simulated trading fee per fill (0.001 = 0.10%, similar to most exchanges)
    paper_fee_rate: float = 0.001

    # Realistic paper slippage. The MEV / front-running literature's core lesson
    # for a taker is adverse selection: you never fill at the mid. We model a base
    # slippage (basis points) plus a component that scales with recent volatility.
    # Buys fill slightly higher, sells slightly lower — keeps the equity curve honest.
    paper_slippage_bps: float = 2.0
    slippage_vol_mult: float = 0.5

    # Funding-rate filter. Perp funding is the cheapest read on crowd positioning:
    # extreme positive funding = crowded longs = mean-reversion risk. When enabled,
    # a BUY signal is suppressed if BTC perp funding is above funding_max (per 8h).
    # If funding can't be fetched we fail open and trade normally.
    funding_filter_enabled: bool = True
    funding_max: float = 0.0005  # 0.05% / 8h ≈ overheated longs
    funding_refresh_seconds: int = 900  # re-fetch funding at most every 15 min

    # ---- Risk management (Phase B) ----------------------------------------
    # All exits are checked every bar while in a position; the first one that
    # trips wins. Set a value to 0 to disable that exit.

    # Hard stop-loss: exit if price falls this fraction below the entry price.
    # Off by default — the validated TSMOM strategy exits on the trend turning,
    # not a fixed stop (backtests showed fixed stops barely triggered and the
    # ensemble's own exit handles risk).
    stop_loss_pct: float = 0.0
    # ATR stop: exit if price falls this many ATRs below entry (0 = off).
    atr_stop_mult: float = 0.0
    atr_window: int = 14
    # Trailing stop: exit if price falls this fraction below the peak since entry.
    trail_stop_pct: float = 0.0
    # Take-profit: exit if price rises this fraction above entry (0 = off, let it run).
    take_profit_pct: float = 0.0

    # Trend filter: only take BUYs when price is above a slow EMA (regime = up).
    # Useful for crossover/meanrev; off for TSMOM (its lookbacks ARE the trend).
    trend_filter_enabled: bool = False
    trend_window: int = 200
    # Also exit an open position if the trend flips down.
    trend_exit: bool = False

    # Cooldown: minimum bars to wait after an exit before opening a new position.
    cooldown_bars: int = 0

    # Volatility targeting: scale the position DOWN when recent volatility exceeds
    # this per-bar target (never scales up). 0 = off. Caps risk in turbulent tape.
    vol_target: float = 0.0
    vol_scale_min: float = 0.25            # never shrink below 25% of intended size

    # ---- Signal confirmation (Phase C) ------------------------------------
    # Require above-average volume on the entry bar (1.0 = off).
    volume_mult: float = 0.0
    volume_window: int = 20
    # RSI gate: skip BUYs when RSI is above this (overbought). 0/100 = off.
    rsi_window: int = 14
    rsi_max: float = 0.0                   # e.g. 70 to avoid buying overbought

    # Mode. Only "paper" is implemented today. "live" comes after VALR signup.
    mode: str = "paper"


CONFIG = Config()
