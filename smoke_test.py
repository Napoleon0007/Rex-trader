"""Quick check: can we reach Binance and read live BTC prices?"""
from pathlib import Path

from broker import make_broker
from config import CONFIG

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)


def main():
    print(f"Rex Trader smoke test (mode={CONFIG.mode})")
    broker = make_broker(LOG_DIR)

    print(f"Pulling {CONFIG.symbol} bars from Binance…")
    bars = broker.get_bars(CONFIG.symbol, limit=30)
    if bars.empty:
        print("  WARNING: no bars returned")
        return
    last = float(bars["close"].iloc[-1])
    print(f"  {len(bars)} bars, last close = ${last:,.2f}")

    acct = broker.get_account_snapshot(last)
    print(f"  paper portfolio: cash=${acct['cash']:.2f}  equity=${acct['equity']:.2f}")
    print("\nAll good. Run `python bot.py` to start trading on paper.")


if __name__ == "__main__":
    main()
