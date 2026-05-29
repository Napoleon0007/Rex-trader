# Rex Trader — Handoff / Resume Here

_Last updated: 2026-05-29_

## TL;DR
Rex is now an honest **daily time-series-momentum (TSMOM) BTC bot** — a
risk-managed trend overlay, not a money-printer. All code is committed and pushed
(`Napoleon0007/Rex-trader`, latest commit `ca5d2a1`). **It is NOT deployed yet** —
deployment is blocked only on Railway login.

## ▶ To finish the deploy (the one open task)
The Railway CLI session is unauthorized. In a terminal:
```
railway login                       # opens browser OAuth — only Luke can do this
cd ~/Desktop/Github\ projects/Rextrader
RAILWAY_CALLER="skill:use-railway@1.2.1" railway up --service rex-trader --detach -m "Daily TSMOM strategy"
```
Then verify https://rex-trader-production.up.railway.app is live on the new
strategy. (Or just run `railway login`, tell Claude "done", and it'll deploy.)
NOTE: the Railway volume still has the old crossover-era paper portfolio + 1-min
ticks; harmless, but optionally clear logs/ on the volume for a clean slate.

## What the bot does now
- `config.strategy = "tsmom"` (ensemble). Long when ≥60% of BTC's momentum
  horizons (40/80/120/160/200-day) are trending up; cash otherwise.
- Runs on **daily bars** (`config.candle_granularity = 86400`), polls hourly.
- Honest performance (5y daily backtest): does NOT beat buy & hold on raw return,
  but cuts max drawdown ~in half (29% vs 77%) and sat out the 2022 −65% crash.
- Current signal as of handoff: `mom_frac ≈ 0.2` → bot sits in CASH (BTC weak).

## What we learned (full detail in RESEARCH_LOG.md)
Tested 5 approaches rigorously with real costs + out-of-sample validation:
1. EMA crossover — dead (no edge anywhere).
2. Mean-reversion — real signal (high win rate) but marginal after costs.
3. Single-asset TSMOM — works on BTC, FAILS on ETH/SOL/DOGE/LTC (BTC was luck).
4. Cross-sectional momentum portfolio — fails across the full cycle, all OOS negative.
5. Funding rate as predictor — ~zero correlation; kept only as a buy filter.
**Meta-conclusion:** no robust OOS price-based edge on crypto in 2021–2026. The
only robust property is trend-following's drawdown reduction → hence the TSMOM overlay.

## The toolbox (this is the durable asset)
`backtest.py` · `walkforward.py` · `montecarlo.py` · `portfolio.py` · `metrics.py`
· `data.py`. Engine in `engine.py` is the single source of truth (live bot AND
backtester both drive `decide()`). Workflow: change rule → backtest → walk-forward
(OOS must hold) → Monte Carlo → then paper. ~22k bars/s.

## Best next leads (when we come back)
- Validate TSMOM on more history/assets (ETH, SOL, longer daily series) to grow
  the thin sample (currently ~7–22 trades).
- Add volatility-targeted position sizing (classic TSMOM upgrade; usually lifts Sharpe).
- Funding-carry done properly would need shorting/perps (bot is spot long-only).
- Honest stretch: accept this is a risk overlay and optimise it as such, rather
  than hunting a price-pattern alpha that the evidence says isn't there.
