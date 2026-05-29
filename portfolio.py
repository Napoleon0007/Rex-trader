"""Cross-sectional momentum portfolio.

The single-asset story is fragile (TSMOM works on BTC, fails on alts). The
historically robust form of momentum is cross-sectional + diversified (Asness,
"Value and Momentum Everywhere"): every rebalance, rank the whole universe by
trailing momentum, hold an equal-weight basket of the strongest trenders, and —
crucially — only hold names whose momentum is positive (absolute filter), so the
book moves to cash in a broad bear instead of riding everything down.

  python portfolio.py
  python portfolio.py --top 3 --rebalance 7 --days 1500
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from data import fetch_history
import metrics as metrics_mod

LOG_DIR = Path(__file__).parent / "logs"
# Long-history Coinbase symbols (all ~2021-05). XRP excluded — relisted 2023, so
# it would collapse the common window and cut out the 2022 bear.
DEFAULT_UNIVERSE = ["BTC-USD", "ETH-USD", "SOL-USD", "LTC-USD", "DOGE-USD",
                    "ADA-USD", "LINK-USD", "BCH-USD", "ETC-USD", "XLM-USD"]
LOOKBACKS = (40, 80, 120, 160, 200)
COST_RATE = 0.002  # one-way: ~0.1% fee + ~0.1% slippage on rebalance turnover


def load_universe(symbols, days, granularity=86400) -> pd.DataFrame:
    """Aligned close-price frame (inner join on common dates)."""
    cols = {}
    for s in symbols:
        try:
            df = fetch_history(s, granularity, days, verbose=False)
            if len(df) > max(LOOKBACKS) + 10:
                cols[s] = df["close"]
        except Exception as e:
            print(f"  skip {s}: {e}")
    closes = pd.DataFrame(cols).dropna()
    return closes


def momentum_scores(closes: pd.DataFrame) -> pd.DataFrame:
    """Blended momentum = mean trailing return across LOOKBACKS, per asset per day."""
    parts = [closes.pct_change(lb) for lb in LOOKBACKS]
    return sum(parts) / len(parts)


def run_portfolio(closes: pd.DataFrame, top_k=3, rebalance=7, cost_rate=COST_RATE) -> dict:
    scores = momentum_scores(closes)
    rets = closes.pct_change().fillna(0.0)
    assets = list(closes.columns)
    warmup = max(LOOKBACKS) + 1

    cash = 100.0
    holdings = {a: 0.0 for a in assets}
    eq_idx, eq_val, exposure_days, n_rebalances = [], [], 0, 0

    for i in range(warmup, len(closes)):
        ts = closes.index[i]
        for a in assets:                       # mark holdings to market
            holdings[a] *= (1.0 + rets[a].iloc[i])

        if (i - warmup) % rebalance == 0:      # rebalance day
            total = cash + sum(holdings.values())
            row = scores.iloc[i]
            ranked = row[row > 0].sort_values(ascending=False)  # positive momentum only
            selected = list(ranked.index[:top_k])
            w = 1.0 / len(selected) if selected else 0.0
            targets = {a: (total * w if a in selected else 0.0) for a in assets}
            turnover = sum(abs(targets[a] - holdings[a]) for a in assets) / total if total > 0 else 0
            total -= cost_rate * turnover * total
            holdings = {a: (total * w if a in selected else 0.0) for a in assets}
            cash = total - sum(holdings.values())
            n_rebalances += 1

        equity = cash + sum(holdings.values())
        eq_idx.append(ts)
        eq_val.append(equity)
        if sum(holdings.values()) > 0:
            exposure_days += 1

    equity = pd.Series(eq_val, index=pd.DatetimeIndex(eq_idx))
    basket = (closes.iloc[warmup:] / closes.iloc[warmup]).mean(axis=1)  # equal-weight B&H
    m = metrics_mod.compute(equity, pd.DataFrame(), basket, exposure_days, 86400)
    return {"equity": equity, "metrics": m, "rebalances": n_rebalances, "basket": basket}


def _print(label, m, extra=""):
    print(f"  {label:22s} ret {m.total_return_pct:+8.1f}%  CAGR {m.cagr_pct:+6.1f}%  "
          f"maxDD {m.max_drawdown_pct:5.1f}%  Sharpe {m.sharpe:5.2f}  Sortino {m.sortino:5.2f}{extra}")


def main():
    ap = argparse.ArgumentParser(description="Cross-sectional momentum portfolio")
    ap.add_argument("--days", type=float, default=1825)
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--rebalance", type=int, default=7)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    print(f"Loading universe ({len(DEFAULT_UNIVERSE)} coins, {args.days}d daily)…")
    closes = load_universe(DEFAULT_UNIVERSE, args.days)
    print(f"  {len(closes)} common days, {closes.shape[1]} assets: {list(closes.columns)}")
    print(f"  window {closes.index.min().date()} → {closes.index.max().date()}\n")

    res = run_portfolio(closes, args.top, args.rebalance)
    m = res["metrics"]
    print(f"PORTFOLIO  (top {args.top}, rebalance every {args.rebalance}d, {res['rebalances']} rebalances):")
    _print("strategy", m)
    print(f"  {'equal-weight B&H':22s} ret {m.buy_hold_return_pct:+8.1f}%  "
          f"(basket maxDD {metrics_mod.max_drawdown(res['basket'])*100:.1f}%)")

    # Out-of-sample: same params, unseen second half.
    n = len(closes); cut = int(n * 0.6)
    is_m = run_portfolio(closes.iloc[:cut], args.top, args.rebalance)["metrics"]
    oos_m = run_portfolio(closes.iloc[cut:], args.top, args.rebalance)["metrics"]
    print("\nOUT-OF-SAMPLE (same params):")
    _print("in-sample (~60%)", is_m)
    _print("out-sample (~40%)", oos_m, "  <== honest number")

    if args.save:
        LOG_DIR.mkdir(exist_ok=True)
        res["equity"].to_csv(LOG_DIR / "portfolio_equity.csv", header=["equity"])
        print("\n  saved portfolio equity to logs/")


if __name__ == "__main__":
    main()
