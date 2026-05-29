"""Main bot loop. Polls market, runs the shared engine, places trades, logs."""
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from broker import fetch_funding_rate, make_broker
from config import CONFIG
from engine import Action, PositionState, decide, features_from_window, slippage_fraction

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
STATE_FILE = LOG_DIR / "state.json"
POSITION_FILE = LOG_DIR / "position.json"
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


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def write_state(state: dict) -> None:
    _atomic_write(STATE_FILE, json.dumps(state, indent=2, default=str))


def write_position(pos: PositionState) -> None:
    _atomic_write(POSITION_FILE, json.dumps(pos.__dict__, indent=2, default=str))


def load_position() -> PositionState:
    if POSITION_FILE.exists():
        try:
            return PositionState(**json.loads(POSITION_FILE.read_text()))
        except (json.JSONDecodeError, TypeError):
            pass
    return PositionState()


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(record, default=str) + "\n")


class FundingCache:
    """Caches the perp funding rate; funding only changes every 8h, so we refresh
    at most every funding_refresh_seconds instead of hammering the API each tick."""

    def __init__(self):
        self.value = None
        self.fetched_at = 0.0

    def get(self) -> float | None:
        if not CONFIG.funding_filter_enabled:
            return None
        now = time.time()
        if now - self.fetched_at >= CONFIG.funding_refresh_seconds:
            fetched = fetch_funding_rate()
            if fetched is not None:
                self.value = fetched
            self.fetched_at = now
        return self.value


def reconcile(pos: PositionState, broker, symbol: str, price: float) -> None:
    """Keep engine position state in sync with the broker (source of truth for qty)."""
    bpos = broker.get_position(symbol)
    if bpos:
        pos.qty = bpos["qty"]
        if pos.entry_price <= 0:
            pos.entry_price = bpos["avg_price"]
        if pos.peak_price <= 0:
            pos.peak_price = max(pos.entry_price, price)
    elif pos.qty > 0:
        pos.on_exit()


def main() -> None:
    broker = make_broker(LOG_DIR)
    pos = load_position()
    funding = FundingCache()
    log.info(
        "Rex Trader starting | mode=%s | symbol=%s | %d/%d EMA | trend=%s | poll=%ds",
        CONFIG.mode.upper(), CONFIG.symbol, CONFIG.short_window, CONFIG.long_window,
        CONFIG.trend_window if CONFIG.trend_filter_enabled else "off", CONFIG.poll_seconds,
    )

    while True:
        try:
            need = max(CONFIG.long_window * 4, CONFIG.trend_window + 2, CONFIG.atr_window + 2,
                       CONFIG.mr_window + 2, CONFIG.regime_window + 2, CONFIG.tsmom_lookback + 2,
                       max(CONFIG.tsmom_lookbacks) + 2)
            bars = broker.get_bars(CONFIG.symbol, limit=need)
            price = float(bars["close"].iloc[-1]) if not bars.empty else 0.0

            reconcile(pos, broker, CONFIG.symbol, price)
            pos.on_bar(price)
            feat = features_from_window(bars, CONFIG)
            d = decide(feat, pos, CONFIG, funding_rate=funding.get())
            slip = slippage_fraction(d.recent_vol, CONFIG)

            account = broker.get_account_snapshot(d.last_price)
            tick = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "symbol": CONFIG.symbol, "mode": CONFIG.mode, "strategy": CONFIG.strategy,
                "price": d.last_price, "short_ma": d.short_ma, "long_ma": d.long_ma,
                "signal": d.signal, "reason": d.reason, "has_position": pos.has_position,
                "mom_frac": feat.mom_frac, "poll_seconds": CONFIG.poll_seconds,
                **account,
            }
            append_jsonl(TICKS_FILE, tick)
            write_state(tick)
            log.info(
                "price=%.2f short=%.2f long=%.2f signal=%s action=%s pos=%s equity=$%.2f",
                d.last_price, d.short_ma, d.long_ma, d.signal, d.action.value,
                pos.has_position, account["equity"],
            )

            if d.action == Action.ENTER:
                dollars = account["cash"] * d.fraction
                if dollars >= 1.0:
                    trade = broker.buy_notional(dollars, d.last_price, slip)
                    pos.on_enter(trade["qty"], trade["price"])
                    log.warning(
                        "BUY $%.2f @ $%.2f (qty %.6f, conf %.2f, frac %.2f) — %s",
                        dollars, trade["price"], trade["qty"], d.confidence, d.fraction, d.reason,
                    )
                    append_jsonl(TRADES_FILE, {**trade, "reason": d.reason})
                else:
                    log.warning("ENTER signal but cash too low ($%.2f)", account["cash"])
            elif d.action == Action.EXIT:
                entry = pos.entry_price
                trade = broker.sell_position(d.last_price, slip)
                if trade:
                    proceeds = trade["notional"] - trade["fee"]
                    pnl = proceeds - entry * trade["qty"]
                    pos.on_exit()
                    log.warning(
                        "SELL %.6f @ $%.2f (proceeds $%.2f, P&L $%.2f) — %s",
                        trade["qty"], trade["price"], proceeds, pnl, d.reason,
                    )
                    append_jsonl(TRADES_FILE, {**trade, "reason": d.reason, "pnl": pnl})

            write_position(pos)
        except KeyboardInterrupt:
            log.info("Bot stopped by user.")
            break
        except Exception as e:
            log.exception("loop error: %s", e)

        time.sleep(CONFIG.poll_seconds)


if __name__ == "__main__":
    main()
