# Rex Trader — Research Log

Evidence-based strategy notes. **Rule: nothing goes live until it passes
walk-forward with positive out-of-sample expectancy.** Nothing has yet.

Run the tools yourself:
```
python backtest.py --days 14 --save        # single backtest + metrics
python walkforward.py --days 30 --folds 5  # in-sample vs out-of-sample
python montecarlo.py --sims 5000           # outcome range + risk of ruin
```

---

## Findings so far

### 1. EMA crossover — DEAD. Do not revisit.
No edge at any timeframe or parameter set, in-sample or out.

| Timeframe | Best result | Buy & hold |
|-----------|-------------|------------|
| 1-min     | −26%        | −8%        |
| 15-min    | −2% to −15% | +4% to +9% |
| 1-hour    | −7% to −22% | −19%       |
| Walk-fwd OOS | −9%, 12.7% win rate | — |

Every config: negative Sharpe, profit factor < 1. Monte Carlo: 100% of 3000
resamples ended in a loss. The overfit gap was *small* — it's not curve-fit, the
signal simply has no edge. Simple MA crossovers get chopped to death on liquid
crypto.

### 2. Mean-reversion (z-score dip buying) — REAL SIGNAL, but marginal.
Dips genuinely bounce: win rates 35–75% (vs crossover's 7–25%), small drawdowns
(2–7%). But still slightly negative after costs.

Best observed (15-min, 60d): trend filter ON + reversion exit (z≥+0.5):
- `z≤−2.5`: −1.65%, 14 trades, 64% win, PF 0.49
- `z≤−3.0`: −0.03% (flat), 4 trades, 75% win, PF 1.82 — but too few trades to trust

**Why it doesn't quite work:**
- The rolling mean chases price in a trend, so "reversion" can book a loss at a
  lower price than entry (even with a high win rate, losers are ~2–4× the winners).
- ~0.3% round-trip costs eat the small bounces.
- The high-quality deep-z signals are rare (trade-starved).

**What was tried and did NOT help** (made it worse or inert):
- ATR-based profit target + time-stop → exits before the bounce completes.
- Dropping the trend filter → buys dips in downtrends (falling knives).
- Kaufman efficiency-ratio regime filter → inert at deep-z (those trades are
  already in low-ER regimes).
- Stop-loss level (2/4/5/8%) → barely matters; dips revert before stops hit.

What consistently *helped*: the **trend filter** (buy dips only in uptrends) and
**deeper z entries** (fewer, higher-quality signals).

---

## Leads for next time
- **Anchor value to something that doesn't chase price** (e.g. VWAP, a fixed
  longer-horizon mean) so the exit isn't dragged down in trends.
- **Funding-rate carry** — a structural edge, not a price pattern. Needs aligned
  historical funding wired into the backtester.
- **Higher timeframe / daily time-series momentum** (Moskowitz) — the opposite end
  from the 1-min noise trap; proven over years on daily bars.
- **Multi-coin cross-sectional** mean-reversion or momentum (rank a basket).
- Consider whether the realistic cost assumptions are too harsh or too soft
  (currently 0.10% fee + vol-scaled slippage per side).

## Infrastructure notes
- `engine.decide()` is the single source of truth — live bot and backtester both
  drive it, so backtest results predict live behaviour (modulo tiny EMA-seeding
  differences from the trailing-window vs full-series view).
- Backtest is ~22k bars/s (vectorised features + itertuples).
- Strategy is selected by `config.strategy` ("crossover" | "meanrev").
