"""Historical candle fetcher with on-disk caching.

Coinbase Exchange's public candles endpoint returns at most 300 bars per request,
newest-first, no auth. To backtest we page backwards in 300-bar chunks and cache
the result to data/ so repeat runs are instant and don't re-hit the API.
"""
import time
from pathlib import Path

import pandas as pd
import requests

COINBASE_CANDLES = "https://api.exchange.coinbase.com/products/{product}/candles"
DATA_DIR = Path(__file__).parent / "data"
MAX_PER_REQUEST = 300


def _cache_path(symbol: str, granularity: int) -> Path:
    return DATA_DIR / f"{symbol}_{granularity}s.csv"


def _fetch_chunk(symbol: str, granularity: int, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    r = requests.get(
        COINBASE_CANDLES.format(product=symbol),
        params={"granularity": granularity, "start": start.isoformat(), "end": end.isoformat()},
        timeout=15,
    )
    r.raise_for_status()
    rows = r.json()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["time", "low", "high", "open", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df.set_index("timestamp")[["open", "high", "low", "close", "volume"]]


def _fetch_range(symbol: str, granularity: int, start: pd.Timestamp, end: pd.Timestamp,
                 verbose: bool) -> tuple:
    """Page backwards over [start, end], returning (frame, n_requests)."""
    chunks, made = [], 0
    cursor_end = end
    span = pd.Timedelta(seconds=granularity * MAX_PER_REQUEST)
    while cursor_end > start:
        cursor_start = max(start, cursor_end - span)
        chunk = _fetch_chunk(symbol, granularity, cursor_start, cursor_end)
        made += 1
        if not chunk.empty:
            chunks.append(chunk)
        cursor_end = cursor_start
        time.sleep(0.25)  # polite: well under Coinbase's public rate limit
        if verbose and made % 10 == 0:
            print(f"  …fetched {made} chunks")
    return (pd.concat(chunks) if chunks else pd.DataFrame()), made


def fetch_history(symbol: str, granularity: int = 60, days: float = 14.0,
                  use_cache: bool = True, verbose: bool = True) -> pd.DataFrame:
    """Return a clean, ascending OHLCV frame for the last `days` of `symbol`.

    Cached to data/<symbol>_<g>s.csv. Fills whatever the cache is missing on
    BOTH ends — older history before the cache and newer bars up to now.
    """
    DATA_DIR.mkdir(exist_ok=True)
    cache = _cache_path(symbol, granularity)
    now = pd.Timestamp.now(tz="UTC").floor("min")
    want_start = now - pd.Timedelta(days=days)
    step = pd.Timedelta(seconds=granularity)

    existing = pd.DataFrame()
    if use_cache and cache.exists():
        existing = pd.read_csv(cache, index_col=0, parse_dates=True)
        existing.index = pd.to_datetime(existing.index, utc=True)

    pieces, total_requests = [existing] if not existing.empty else [], 0
    if existing.empty:
        fresh, n = _fetch_range(symbol, granularity, want_start, now, verbose)
        pieces.append(fresh); total_requests += n
    else:
        if want_start < existing.index.min():  # backfill older history
            older, n = _fetch_range(symbol, granularity, want_start, existing.index.min(), verbose)
            pieces.append(older); total_requests += n
        if existing.index.max() + step < now:  # top up recent bars
            newer, n = _fetch_range(symbol, granularity, existing.index.max() + step, now, verbose)
            pieces.append(newer); total_requests += n

    combined = pd.concat([p for p in pieces if not p.empty])
    if combined.empty:
        raise RuntimeError(f"No candle data returned for {symbol}.")
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()

    if use_cache:
        combined.to_csv(cache)
    out = combined[combined.index >= want_start]
    if verbose:
        print(f"  {len(out)} bars  {out.index.min()} → {out.index.max()}  "
              f"({total_requests} new request(s))")
    return out
