"""Main bot loop. Polls market, runs strategy, places trades, logs everything."""
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from broker import make_broker
from config import CONFIG
from strategy import Signal, ma_crossover

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
STATE_FILE = LOG_DIR / "state.json"
TRADES_FILE = LOG_DIR / "trades.jsonl"
TICKS_FILE = LOG_DIR / "ticks.jsonl"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "bot.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("bot")


def write_state(state: dict) -> None:
    # Atomic: write to a temp file then rename, so the dashboard never reads a
    # half-written state.json while the bot is mid-write (both run in serve.py).
    tmp = STATE_FILE.with_name(STATE_FILE.name + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, default=str))
    tmp.replace(STATE_FILE)


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def main() -> None:
    broker = make_broker(LOG_DIR)
    log.info(
        "Rex Trader starting | mode=%s | symbol=%s | %d/%d MA | poll=%ds",
        CONFIG.mode.upper(), CONFIG.symbol,
        CONFIG.short_window, CONFIG.long_window, CONFIG.poll_seconds,
    )

    while True:
        try:
            bars = broker.get_bars(CONFIG.symbol, limit=CONFIG.long_window * 4)
            position = broker.get_position(CONFIG.symbol)
            has_position = position is not None

            result = ma_crossover(
                bars, CONFIG.short_window, CONFIG.long_window, has_position,
            )

            # Recent 1-min return volatility — feeds slippage (and sizing below).
            rets = bars["close"].pct_change().dropna()
            recent_vol = (
                float(rets.tail(CONFIG.long_window).std()) if len(rets) >= 2 else 0.0
            )
            slippage_frac = (
                CONFIG.paper_slippage_bps / 10_000.0
                + CONFIG.slippage_vol_mult * recent_vol
            )

            account = broker.get_account_snapshot(result.last_price)
            now = datetime.now(timezone.utc).isoformat()
            tick = {
                "ts": now,
                "symbol": CONFIG.symbol,
                "mode": CONFIG.mode,
                "price": result.last_price,
                "short_ma": result.short_ma,
                "long_ma": result.long_ma,
                "signal": result.signal.value,
                "reason": result.reason,
                "has_position": has_position,
                **account,
            }
            append_jsonl(TICKS_FILE, tick)
            write_state(tick)

            log.info(
                "price=%.2f short=%.2f long=%.2f signal=%s pos=%s equity=$%.2f",
                result.last_price, result.short_ma, result.long_ma,
                result.signal.value, has_position, account["equity"],
            )

            if result.signal == Signal.BUY:
                dollars = account["cash"] * CONFIG.position_fraction
                if dollars >= 1.0:
                    trade = broker.buy_notional(dollars, result.last_price, slippage_frac)
                    log.warning(
                        "BUY $%.2f @ $%.2f (qty %.6f)",
                        dollars, result.last_price, trade["qty"],
                    )
                    append_jsonl(TRADES_FILE, {**trade, "reason": result.reason})
                else:
                    log.warning("BUY signal but cash too low ($%.2f)", account["cash"])
            elif result.signal == Signal.SELL:
                trade = broker.sell_position(result.last_price, slippage_frac)
                if trade:
                    log.warning(
                        "SELL %.6f @ $%.2f (proceeds $%.2f)",
                        trade["qty"], result.last_price, trade["notional"] - trade["fee"],
                    )
                    append_jsonl(TRADES_FILE, {**trade, "reason": result.reason})
        except KeyboardInterrupt:
            log.info("Bot stopped by user.")
            break
        except Exception as e:
            log.exception("loop error: %s", e)

        time.sleep(CONFIG.poll_seconds)


if __name__ == "__main__":
    main()
