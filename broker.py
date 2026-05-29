"""Paper broker.

Reads real BTC prices from Coinbase Exchange's free public API (no auth, no keys).
Simulates a portfolio on disk so we can taste the strategy without risking money.
The interface (get_bars, get_position, buy_notional, sell_position,
get_account_snapshot) is designed to match a real broker, so bot.py stays
identical when we later swap in a live VALR broker.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from config import CONFIG


COINBASE_CANDLES = "https://api.exchange.coinbase.com/products/{product}/candles"

# Bybit public linear-perp ticker carries the current funding rate. No auth, and
# (unlike Binance) it is reachable from Railway's US region.
BYBIT_TICKERS = "https://api.bybit.com/v5/market/tickers"


def fetch_funding_rate(symbol: str = "BTCUSDT", timeout: int = 10):
    """Current BTC perpetual funding rate as a float (e.g. 0.0001 = 0.01% / 8h).

    Returns None on any failure so callers can fail open — a funding outage must
    never wedge the trading loop.
    """
    try:
        r = requests.get(
            BYBIT_TICKERS,
            params={"category": "linear", "symbol": symbol},
            timeout=timeout,
        )
        r.raise_for_status()
        rows = r.json().get("result", {}).get("list", [])
        if not rows:
            return None
        rate = rows[0].get("fundingRate")
        return float(rate) if rate not in (None, "") else None
    except (requests.RequestException, ValueError, KeyError):
        return None


class PaperBroker:
    """Reads real prices, simulates orders against a local cash + position state."""

    def __init__(self, state_path: Path, starting_cash: float, fee_rate: float):
        self.state_path = state_path
        self.fee_rate = fee_rate
        if not self.state_path.exists():
            self._write({
                "cash": starting_cash,
                "starting_cash": starting_cash,
                "position_qty": 0.0,
                "position_avg_price": 0.0,
            })

    def _read(self) -> dict:
        return json.loads(self.state_path.read_text())

    def _write(self, state: dict) -> None:
        # Atomic: temp file + rename, so the dashboard never reads a half-written
        # portfolio.json while we're mid-write (bot + dashboard share the process).
        tmp = self.state_path.with_name(self.state_path.name + ".tmp")
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(self.state_path)

    def get_bars(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        # Coinbase Exchange public candles (geo-open, no auth). Granularity is
        # configurable — the live TSMOM strategy runs on daily bars (86400s). One
        # request returns up to 300 candles; the last one is the still-forming
        # current bar, whose close ≈ live price.
        url = COINBASE_CANDLES.format(product=symbol)
        r = requests.get(url, params={"granularity": CONFIG.candle_granularity}, timeout=10)
        r.raise_for_status()
        rows = r.json()
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows, columns=["time", "low", "high", "open", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.set_index("timestamp").sort_index()  # Coinbase returns newest-first
        df = df.tail(limit)
        return df[["open", "high", "low", "close", "volume"]]

    def get_position(self, symbol: str):
        s = self._read()
        if s["position_qty"] > 0:
            return {
                "qty": s["position_qty"],
                "avg_price": s["position_avg_price"],
            }
        return None

    def get_account_snapshot(self, last_price: float) -> dict:
        s = self._read()
        position_value = s["position_qty"] * last_price
        return {
            "cash": s["cash"],
            "equity": s["cash"] + position_value,
            "starting_cash": s["starting_cash"],
            "position_qty": s["position_qty"],
            "position_value": position_value,
        }

    def buy_notional(self, dollars: float, price: float, slippage_frac: float = 0.0) -> dict:
        s = self._read()
        if dollars > s["cash"]:
            dollars = s["cash"]
        # Adverse selection: a market buy fills above the quoted price.
        fill_price = price * (1 + slippage_frac)
        fee = dollars * self.fee_rate
        spend = dollars - fee
        qty = spend / fill_price
        price = fill_price
        new_qty = s["position_qty"] + qty
        new_avg = (
            (s["position_qty"] * s["position_avg_price"] + qty * price) / new_qty
            if new_qty > 0 else 0.0
        )
        s["cash"] -= dollars
        s["position_qty"] = new_qty
        s["position_avg_price"] = new_avg
        self._write(s)
        return {
            "ts": datetime.now(timezone.utc).isoformat(),
            "side": "buy",
            "qty": qty,
            "price": price,
            "notional": dollars,
            "fee": fee,
        }

    def sell_position(self, price: float, slippage_frac: float = 0.0) -> dict:
        s = self._read()
        qty = s["position_qty"]
        if qty <= 0:
            return {}
        # Adverse selection: a market sell fills below the quoted price.
        price = price * (1 - slippage_frac)
        gross = qty * price
        fee = gross * self.fee_rate
        proceeds = gross - fee
        s["cash"] += proceeds
        s["position_qty"] = 0.0
        s["position_avg_price"] = 0.0
        self._write(s)
        return {
            "ts": datetime.now(timezone.utc).isoformat(),
            "side": "sell",
            "qty": qty,
            "price": price,
            "notional": gross,
            "fee": fee,
        }


def make_broker(logs_dir: Path):
    if CONFIG.mode == "paper":
        return PaperBroker(
            state_path=logs_dir / "portfolio.json",
            starting_cash=CONFIG.paper_starting_cash,
            fee_rate=CONFIG.paper_fee_rate,
        )
    raise NotImplementedError(
        "Live mode not built yet — finish the paper taste-test first, "
        "then we'll wire up VALR."
    )
