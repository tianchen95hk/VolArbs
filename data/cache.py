"""
Pickle-based caching layer for Deribit market data.

fetch_with_cache() returns a single dict with all data needed by the signal
and backtest layers.  The cache is invalidated after CACHE_TTL_MIN minutes.
Use --no-fetch (force_refresh=False) to reuse stale cache for faster iteration.
"""

from __future__ import annotations

import pickle
from datetime import datetime, timezone
from pathlib import Path

from config import CACHE_FILE, CACHE_TTL_MIN, CURRENCY, LOOKBACK_DAYS
from data.deribit import (
    get_options_summary,
    get_dvol_history,
    get_mark_price_history,
    get_futures_summary,
    get_ticker,
)


def _cache_path() -> Path:
    return Path(CACHE_FILE)


def _is_fresh(data: dict) -> bool:
    fetched_at = data.get("fetched_at")
    if not fetched_at:
        return False
    age_min = (datetime.now(timezone.utc) - fetched_at).total_seconds() / 60
    return age_min < CACHE_TTL_MIN


def _load() -> dict | None:
    path = _cache_path()
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _save(data: dict) -> None:
    with open(_cache_path(), "wb") as f:
        pickle.dump(data, f)


def fetch_with_cache(
    currency: str = CURRENCY,
    force_refresh: bool = True,
) -> dict:
    """
    Return a dict with all market data needed for scanning and backtesting:

        options_df   — pd.DataFrame  full options chain (get_options_summary)
        dvol_df      — pd.DataFrame  hourly DVOL index  (get_dvol_history)
        spot_history — pd.DataFrame  hourly mark price  (get_mark_price_history)
        futures_df   — pd.DataFrame  dated futures      (get_futures_summary)
        spot_price   — float         current index price (get_ticker)
        fetched_at   — datetime      UTC timestamp of fetch
        currency     — str

    If force_refresh=False and a fresh cache exists, returns cached data.
    """
    if not force_refresh:
        cached = _load()
        if cached and _is_fresh(cached) and cached.get("currency") == currency:
            print(f"[cache] Using cached data ({CACHE_TTL_MIN}-min TTL). "
                  "Pass force_refresh=True to re-fetch.")
            return cached

    print(f"[data] Fetching Deribit data for {currency} ...")

    perp = f"{currency}-PERPETUAL"
    ticker = get_ticker(perp)
    spot_price = float(ticker.get("index_price") or ticker.get("mark_price") or 0)

    print(f"  spot price : ${spot_price:,.0f}")

    options_df   = get_options_summary(currency)
    print(f"  options    : {len(options_df)} instruments")

    dvol_df      = get_dvol_history(currency, days=LOOKBACK_DAYS)
    print(f"  DVOL       : {len(dvol_df)} hourly bars")

    spot_history = get_mark_price_history(perp, days=LOOKBACK_DAYS)
    print(f"  spot hist  : {len(spot_history)} hourly bars")

    futures_df   = get_futures_summary(currency)
    print(f"  futures    : {len(futures_df)} instruments")

    data = {
        "options_df":   options_df,
        "dvol_df":      dvol_df,
        "spot_history": spot_history,
        "futures_df":   futures_df,
        "spot_price":   spot_price,
        "fetched_at":   datetime.now(timezone.utc),
        "currency":     currency,
    }

    _save(data)
    return data
