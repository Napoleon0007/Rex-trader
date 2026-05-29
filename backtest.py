"""Backtester. Replays historical candles through the SAME engine the live bot
uses, with realistic slippage + fees, and reports performance.

  python backtest.py                      # 14d BTC-USD, current config
  python backtest.py --days 30            # longer history
  python backtest.py --short 9 --long 21 --stop 0.02 --trend 200
  python backtest.py --no-trend           # disable the trend filter
"""
import argparse
import dataclasses
from pathlib import Path

import pandas as pd

from config import CONFIG
from data import fetch_history
from engine import (
    Action, PositionState, _row_to_features, build_features_frame, decide, slippage_fraction,
)
import metrics as metrics_mod

LOG_DIR = Path(__file__).parent / "logs"


def window_size(cfg) -> int:
    """Trailing window the engine sees — identical to bot.py's broker.get_bars limit."""
    return max(cfg.long_window * 4, cfg.trend_window + 2, cfg.atr_window + 2)


def run_backtest(df: pd.DataFrame, cfg, granularity: int = 60) -> dict:
    """Simulate the strategy over df (ascending OHLCV). Funding filter is inert
    here (no aligned historical funding); everything else matches the live bot."""
    need = window_size(cfg)
    if len(df) <= need + 2:
        raise ValueError(f"Not enough bars ({len(df)}) for window {need}.")

    cash = cfg.paper_starting_cash
    qty = 0.0
    pos = PositionState()

    # Precompute every indicator once, then iterate via itertuples — orders of
    # magnitude faster than rebuilding them from a window each bar, identical decisions.
    feats = build_features_frame(df, cfg)

    equity_idx, equity_val, exposure_bars = [], [], 0
    trades = []
    open_trade = None

    for row in feats.iloc[need - 1:].itertuples():
        ts = row.Index
        price = float(row.price)

        pos.on_bar(price)
        feat = _row_to_features(row)
        d = decide(feat, pos, cfg, funding_rate=None)
        slip = slippage_fraction(d.recent_vol, cfg)

        if d.action == Action.ENTER and qty == 0:
            dollars = cash * d.fraction
            if dollars >= 1.0:
                fill = price * (1 + slip)
                fee = dollars * cfg.paper_fee_rate
                q = (dollars - fee) / fill
                cash -= dollars
                qty += q
                pos.on_enter(q, fill)
                open_trade = {"entry_ts": ts, "entry_price": fill, "qty": q,
                              "entry_reason": d.reason}
        elif d.action == Action.EXIT and qty > 0:
            fill = price * (1 - slip)
            gross = qty * fill
            fee = gross * cfg.paper_fee_rate
            proceeds = gross - fee
            cost = open_trade["entry_price"] * qty
            cash += proceeds
            trades.append({
                **open_trade,
                "exit_ts": ts, "exit_price": fill, "exit_reason": d.reason,
                "bars_held": pos.bars_held,
                "pnl_usd": proceeds - cost,
                "pnl_pct": (proceeds / cost - 1.0) * 100.0 if cost > 0 else 0.0,
            })
            qty = 0.0
            pos.on_exit()
            open_trade = None

        equity_idx.append(ts)
        equity_val.append(cash + qty * price)
        if qty > 0:
            exposure_bars += 1

    equity = pd.Series(equity_val, index=pd.DatetimeIndex(equity_idx))
    trades_df = pd.DataFrame(trades)
    sim_prices = df["close"].iloc[need - 1:]
    m = metrics_mod.compute(equity, trades_df, sim_prices, exposure_bars, granularity)
    return {"equity": equity, "trades": trades_df, "metrics": m,
            "open_position": qty > 0}


def _config_with_overrides(args) -> object:
    cfg = dataclasses.replace(CONFIG)
    if args.short is not None: cfg.short_window = args.short
    if args.long is not None: cfg.long_window = args.long
    if args.stop is not None: cfg.stop_loss_pct = args.stop
    if args.trail is not None: cfg.trail_stop_pct = args.trail
    if args.take is not None: cfg.take_profit_pct = args.take
    if args.trend is not None:
        cfg.trend_filter_enabled = True
        cfg.trend_window = args.trend
    if args.no_trend:
        cfg.trend_filter_enabled = False
    return cfg


def main():
    ap = argparse.ArgumentParser(description="Rex Trader backtester")
    ap.add_argument("--symbol", default=CONFIG.symbol)
    ap.add_argument("--days", type=float, default=14.0)
    ap.add_argument("--granularity", type=int, default=60)
    ap.add_argument("--short", type=int)
    ap.add_argument("--long", type=int)
    ap.add_argument("--stop", type=float, help="stop-loss fraction, e.g. 0.02")
    ap.add_argument("--trail", type=float, help="trailing-stop fraction")
    ap.add_argument("--take", type=float, help="take-profit fraction")
    ap.add_argument("--trend", type=int, help="enable trend filter with this EMA window")
    ap.add_argument("--no-trend", action="store_true", help="disable the trend filter")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--save", action="store_true", help="write equity + trades to logs/")
    args = ap.parse_args()

    cfg = _config_with_overrides(args)
    print(f"Fetching {args.days}d of {args.symbol} ({args.granularity}s bars)…")
    df = fetch_history(args.symbol, args.granularity, args.days, use_cache=not args.no_cache)

    print(f"Backtesting {cfg.short_window}/{cfg.long_window} EMA"
          f"{' + trend ' + str(cfg.trend_window) if cfg.trend_filter_enabled else ' (no trend filter)'}"
          f", stop {cfg.stop_loss_pct:.1%}, fee {cfg.paper_fee_rate:.2%}…")
    res = run_backtest(df, cfg, args.granularity)
    print(metrics_mod.format_report(res["metrics"]))
    if res["open_position"]:
        print("  (note: still holding an open position at the end — equity marks it to market)\n")

    if args.save:
        LOG_DIR.mkdir(exist_ok=True)
        res["equity"].to_csv(LOG_DIR / "backtest_equity.csv", header=["equity"])
        res["trades"].to_csv(LOG_DIR / "backtest_trades.csv", index=False)
        print(f"  saved equity + {len(res['trades'])} trades to logs/")


if __name__ == "__main__":
    main()
