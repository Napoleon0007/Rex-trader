"""Rex Trader configuration. Tune these to change behavior."""
from dataclasses import dataclass


@dataclass
class Config:
    # Symbol on Coinbase Exchange public data feed (used for paper-mode prices).
    # BTC-USD = Bitcoin priced in US dollars. Other ideas: ETH-USD, SOL-USD.
    symbol: str = "BTC-USD"

    # EMA crossover parameters (computed on 1-minute bars).
    # EMA spans behave like SMA periods of the same length but weight recent
    # prices more — see knowledge/applied/ for the supporting papers.
    short_window: int = 9
    long_window: int = 21

    # How often the bot wakes up and re-evaluates the market (seconds)
    poll_seconds: int = 30

    # Paper-portfolio starting cash (US dollars). Real money never touched in paper mode.
    paper_starting_cash: float = 100.00

    # Position sizing. position_fraction is the BASELINE (neutral prior) spent on
    # a BUY. In the Black-Litterman spirit we blend that prior with the strength
    # of the "view" — how far the short EMA has separated from the long EMA,
    # measured in units of recent volatility — scaling up toward max_position_fraction
    # as confidence rises. confidence_vol_mult sets how many vol-units of EMA gap
    # count as full confidence.
    position_fraction: float = 0.50
    max_position_fraction: float = 0.90
    confidence_vol_mult: float = 3.0

    # Simulated trading fee per fill (0.001 = 0.10%, similar to most exchanges)
    paper_fee_rate: float = 0.001

    # Realistic paper slippage. The MEV / front-running literature's core lesson
    # for a taker is adverse selection: you never fill at the mid. We model a base
    # slippage (basis points) plus a component that scales with recent volatility.
    # Buys fill slightly higher, sells slightly lower — keeps the equity curve honest.
    paper_slippage_bps: float = 2.0
    slippage_vol_mult: float = 0.5

    # Mode. Only "paper" is implemented today. "live" comes after VALR signup.
    mode: str = "paper"


CONFIG = Config()
