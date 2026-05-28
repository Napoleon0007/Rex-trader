"""Rex Trader configuration. Tune these to change behavior."""
from dataclasses import dataclass


@dataclass
class Config:
    # Symbol on Coinbase Exchange public data feed (used for paper-mode prices).
    # BTC-USD = Bitcoin priced in US dollars. Other ideas: ETH-USD, SOL-USD.
    symbol: str = "BTC-USD"

    # Moving-average crossover parameters (computed on 1-minute bars)
    short_window: int = 9
    long_window: int = 21

    # How often the bot wakes up and re-evaluates the market (seconds)
    poll_seconds: int = 30

    # Paper-portfolio starting cash (US dollars). Real money never touched in paper mode.
    paper_starting_cash: float = 100.00

    # Position sizing: fraction of available cash to spend on a BUY signal
    position_fraction: float = 0.50

    # Simulated trading fee per fill (0.001 = 0.10%, similar to most exchanges)
    paper_fee_rate: float = 0.001

    # Mode. Only "paper" is implemented today. "live" comes after VALR signup.
    mode: str = "paper"


CONFIG = Config()
