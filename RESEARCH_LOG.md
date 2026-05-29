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

### 3. Time-series momentum (TSMOM), daily bars — BEST SO FAR. Risk-managed, not a printer.
Long while the trailing trend is up, flat otherwise. Tested on 5y daily BTC
(2021–2026, a full cycle incl. the 2022 −65% bear). This is the current default
(`config.strategy = "tsmom"`, ensemble on).

**Single lookback** (e.g. 120d): +58% over 5y, maxDD 22% vs B&H's 65%. BUT it
**failed the strict out-of-sample test** (tuned on 2021–23 → −29% on 2024–26).
Betting on one "magic" lookback overfits.

**Ensemble** (long when ≥60% of the [40,80,120,160,200]-day horizons are up,
exit below 40%) — the fix for that instability:
- Full cycle: **+66% to +111%**, max drawdown **~29–31%** (vs B&H 65%), Sharpe 0.5–0.7.
- **Out-of-sample (2024–26): −0.5%** — fixed the −29% single-lookback failure
  (now ~break-even OOS, not bleeding).
- 2022: **−17% vs −65%** (dodged the bear). 2025: **+9% vs B&H −7%** (positive in a down year).

**Honest verdict:** not a clear OOS edge (≈break-even out-of-sample), but a
genuine *risk-managed BTC participation overlay* — roughly half the drawdown,
dodges bear markets, positive over the cycle. Good fit for "win some, don't blow
up." Caveats: low frequency (7–23 trades in 5y → small sample), and the high-
threshold configs (0.8/0.4 → +158%) are overfit (only 4 trades) — don't chase them.

**DEPLOYMENT GAP:** TSMOM is a *daily* strategy. The live bot polls 1-min bars
every 30s. To run it live it must operate on daily closes (resample or fetch
daily candles, act once/day). Not yet wired — do this before any live run.

---

## Leads for next time
- **Wire TSMOM for live daily execution** (the deployment gap above) — biggest
  unblocker if we want to actually run the best strategy we have.
- **Volatility-target the TSMOM position size** (classic TSMOM scales to constant
  vol) — usually lifts Sharpe meaningfully without changing the signal.
- **Validate TSMOM on more assets / more history** to grow the sample (ETH, SOL;
  or daily data from another source going back to 2015) — 22 trades is thin.
- **Funding-rate carry** — a structural edge, not a price pattern. Needs aligned
  historical funding wired into the backtester.
- **Multi-coin cross-sectional** momentum (rank a basket, long the strongest).
- Consider whether the realistic cost assumptions are too harsh or too soft
  (currently 0.10% fee + vol-scaled slippage per side).

## Infrastructure notes
- `engine.decide()` is the single source of truth — live bot and backtester both
  drive it, so backtest results predict live behaviour (modulo tiny EMA-seeding
  differences from the trailing-window vs full-series view).
- Backtest is ~22k bars/s (vectorised features + itertuples).
- Strategy is selected by `config.strategy` ("crossover" | "meanrev").
