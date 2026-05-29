"""Walk-forward analysis — the honesty check against curve-fitting.

Split history into contiguous blocks. For each fold, find the best parameters
on the IN-SAMPLE block, then score them on the NEXT (OUT-OF-SAMPLE) block the
optimiser never saw. Stitch the OOS results together: that combined OOS curve is
the only one you should believe. A big gap between in-sample and out-of-sample
performance is the signature of overfitting (López de Prado / Bailey).

  python walkforward.py --days 30 --folds 5
"""
import argparse
import dataclasses
import itertools

import pandas as pd

from config import CONFIG
from data import fetch_history
from backtest import run_backtest, window_size
import metrics as metrics_mod

# Slow, fee-aware grids — a 1-min crossover must trade rarely to beat ~0.3%
# round-trip costs, so the candidates skew toward longer windows.
GRIDS = {
    "quick": {"short": [9, 21], "long": [50, 100], "stop": [0.02]},
    "wide": {"short": [9, 21, 50], "long": [50, 100, 200, 400], "stop": [0.01, 0.02, 0.05]},
}


def _candidates(grid):
    for s, l, stop in itertools.product(grid["short"], grid["long"], grid["stop"]):
        if s < l:
            yield {"short_window": s, "long_window": l, "stop_loss_pct": stop}


def _cfg_with(cfg, params):
    c = dataclasses.replace(cfg)
    for k, v in params.items():
        setattr(c, k, v)
    return c


def find_best(df_is, cfg, grid, objective="total_return_pct"):
    """Return (best_params, best_metrics) on the in-sample slice."""
    best, best_m, best_score = None, None, float("-inf")
    need = window_size(_cfg_with(cfg, {"short_window": max(grid["short"]),
                                       "long_window": max(grid["long"])}))
    if len(df_is) <= need + 2:
        return None, None
    for params in _candidates(grid):
        c = _cfg_with(cfg, params)
        try:
            res = run_backtest(df_is, c)
        except ValueError:
            continue
        m = res["metrics"]
        if m.num_trades < 3:
            continue
        score = getattr(m, objective)
        if score > best_score:
            best, best_m, best_score = params, m, score
    return best, best_m


def walk_forward(df, cfg, folds=5, grid_name="wide", objective="total_return_pct"):
    grid = GRIDS[grid_name]
    blocks = folds + 1
    edges = [int(len(df) * i / blocks) for i in range(blocks + 1)]

    rows = []
    oos_equity_pieces = []
    oos_trades = []
    for k in range(folds):
        df_is = df.iloc[edges[k]:edges[k + 1]]
        df_oos = df.iloc[edges[k + 1]:edges[k + 2]]
        best, is_m = find_best(df_is, cfg, grid, objective)
        if best is None:
            continue
        try:
            oos = run_backtest(df_oos, _cfg_with(cfg, best))
        except ValueError:
            continue
        oos_m = oos["metrics"]
        rows.append({
            "fold": k + 1,
            "params": f"{best['short_window']}/{best['long_window']} stop {best['stop_loss_pct']:.0%}",
            "is_return": is_m.total_return_pct,
            "oos_return": oos_m.total_return_pct,
            "oos_trades": oos_m.num_trades,
            "oos_winrate": oos_m.win_rate_pct,
        })
        # Re-base each OOS equity piece so they compound into one continuous curve.
        piece = oos["equity"]
        if oos_equity_pieces:
            piece = piece / piece.iloc[0] * oos_equity_pieces[-1].iloc[-1]
        oos_equity_pieces.append(piece)
        if not oos["trades"].empty:
            oos_trades.append(oos["trades"])

    report = pd.DataFrame(rows)
    combined_equity = pd.concat(oos_equity_pieces) if oos_equity_pieces else pd.Series(dtype=float)
    combined_trades = pd.concat(oos_trades, ignore_index=True) if oos_trades else pd.DataFrame()
    return report, combined_equity, combined_trades


def main():
    ap = argparse.ArgumentParser(description="Walk-forward analysis")
    ap.add_argument("--symbol", default=CONFIG.symbol)
    ap.add_argument("--days", type=float, default=30.0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--grid", choices=list(GRIDS), default="wide")
    ap.add_argument("--objective", default="total_return_pct")
    args = ap.parse_args()

    print(f"Fetching {args.days}d of {args.symbol}…")
    df = fetch_history(args.symbol, 60, args.days)
    print(f"Walk-forward: {args.folds} folds, '{args.grid}' grid, optimise {args.objective}\n")

    report, equity, trades = walk_forward(df, CONFIG, args.folds, args.grid, args.objective)
    if report.empty:
        print("No usable folds (try more --days or a smaller grid).")
        return

    print(report.to_string(index=False))
    is_avg = report["is_return"].mean()
    oos_avg = report["oos_return"].mean()
    print(f"\n  Avg in-sample return : {is_avg:+.2f}%")
    print(f"  Avg out-of-sample    : {oos_avg:+.2f}%   <-- the one that matters")
    print(f"  Overfit gap          : {is_avg - oos_avg:+.2f}%  (large = curve-fit)")

    if not equity.empty:
        comb = (equity.iloc[-1] / equity.iloc[0] - 1.0) * 100.0
        print(f"\n  Stitched OOS equity  : ${equity.iloc[0]:,.2f} → ${equity.iloc[-1]:,.2f} "
              f"({comb:+.2f}%), max DD {metrics_mod.max_drawdown(equity) * 100:.1f}%")
    if not trades.empty:
        wr = (trades["pnl_pct"] > 0).mean() * 100
        print(f"  OOS trades           : {len(trades)}, win rate {wr:.1f}%, "
              f"expectancy {trades['pnl_pct'].mean():+.3f}%/trade")


if __name__ == "__main__":
    main()
