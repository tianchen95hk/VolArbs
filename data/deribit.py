"""
Deribit public REST API client.

All endpoints used here are unauthenticated — no API key required.

Key data fetched
----------------
- Perpetual ticker          → current mark price, funding rate, index price
- Funding rate history      → historical 8h funding rates (for carry signal)
- Volatility index (DVOL)   → Deribit's own IV benchmark (for vol signal)
- Futures summary           → all active futures with mark price & expiry
- Options summary           → full options chain with IV and greeks
- Mark price history        → hourly mark prices for realised-vol computation
"""

import time
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone

from config import DERIBIT_BASE, LOOKBACK_DAYS

# ── HTTP helper ────────────────────────────────────────────────────────────────

def _get(endpoint: str, params: dict | None = None, retries: int = 3) -> dict:
    url = f"{DERIBIT_BASE}/{endpoint}"
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params or {}, timeout=15)
            r.raise_for_status()
            body = r.json()
            if "error" in body:
                raise ValueError(f"Deribit API error: {body['error']}")
            return body.get("result", body)
        except requests.RequestException as exc:
            if attempt == retries - 1:
                raise
            time.sleep(1.5 ** attempt)
    return {}


def _now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def _start_ms(days: int) -> int:
    return int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)


def _to_ms(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


# ── Ticker ─────────────────────────────────────────────────────────────────────

def get_ticker(instrument: str) -> dict:
    """
    Current market snapshot for a single instrument.

    Returns dict with keys including:
        mark_price, index_price, current_funding, funding_8h,
        best_bid_price, best_ask_price, last_price, open_interest
    """
    return _get("ticker", {"instrument_name": instrument})


# ── Funding rate history ───────────────────────────────────────────────────────

def get_funding_rate_history(
    instrument: str = "BTC-PERPETUAL",
    days: int = LOOKBACK_DAYS,
    start_dt: datetime | None = None,
    end_dt:   datetime | None = None,
) -> pd.DataFrame:
    """
    Historical 8-hourly funding rates for a Deribit perpetual.

    Pass start_dt / end_dt (UTC datetimes) for a specific date range,
    or use days for a rolling lookback from now.

    Returns DataFrame indexed by UTC timestamp with column:
        interest_8h   — funding rate per 8h period (decimal, e.g. 0.0001 = 0.01%)
    """
    start_ms = _to_ms(start_dt) if start_dt else _start_ms(days)
    end_ms   = _to_ms(end_dt)   if end_dt   else _now_ms()

    # Paginate: Deribit returns max 720 rows per request
    all_rows: list[dict] = []
    chunk_end = end_ms
    while True:
        result = _get("get_funding_rate_history", {
            "instrument_name": instrument,
            "start_timestamp": start_ms,
            "end_timestamp":   chunk_end,
        })
        batch = result if isinstance(result, list) else []
        if not batch:
            break
        all_rows.extend(batch)
        if len(batch) < 720:
            break
        chunk_end = min(r["timestamp"] for r in batch) - 1
        if chunk_end <= start_ms:
            break

    result = all_rows

    if not result:
        return pd.DataFrame(columns=["interest_8h"])

    df = pd.DataFrame(result)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()
    df["interest_8h"] = df["interest_8h"].astype(float)
    return df[["interest_8h"]]


# ── DVOL — Deribit Volatility Index ───────────────────────────────────────────

def get_dvol_history(
    currency: str = "BTC",
    days: int = LOOKBACK_DAYS,
    resolution: int = 3600,
    start_dt: datetime | None = None,
    end_dt:   datetime | None = None,
) -> pd.DataFrame:
    """
    Historical DVOL (Deribit 30-day implied vol index) at hourly resolution.

    Pass start_dt / end_dt for a specific date range, or days for rolling lookback.

    Returns DataFrame indexed by UTC timestamp with columns:
        open, high, low, close   — DVOL values in % (e.g. 55.4 means 55.4% annualised IV)
    """
    start_ms = _to_ms(start_dt) if start_dt else _start_ms(days)
    end_ms   = _to_ms(end_dt)   if end_dt   else _now_ms()

    result = _get("get_volatility_index_data", {
        "currency":        currency,
        "start_timestamp": start_ms,
        "end_timestamp":   end_ms,
        "resolution":      str(resolution),
    })

    if not result or "data" not in result:
        return pd.DataFrame(columns=["open", "high", "low", "close"])

    df = pd.DataFrame(result["data"], columns=["timestamp", "open", "high", "low", "close"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()
    df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype(float)
    return df


# ── Futures summary ────────────────────────────────────────────────────────────

def get_futures_summary(currency: str = "BTC") -> pd.DataFrame:
    """
    All active futures (excluding perpetuals) with mark price.

    Returns DataFrame with columns:
        instrument_name, mark_price, underlying_price, creation_timestamp,
        estimated_delivery_price, open_interest, volume
    """
    result = _get("get_book_summary_by_currency", {
        "currency": currency,
        "kind":     "future",
    })

    if not result:
        return pd.DataFrame()

    df = pd.DataFrame(result)
    df = df[~df["instrument_name"].str.contains("PERPETUAL", na=False)].copy()
    numeric = ["mark_price", "underlying_price", "open_interest", "volume"]
    for col in numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.reset_index(drop=True)


# ── Options chain ──────────────────────────────────────────────────────────────

def get_options_summary(currency: str = "BTC") -> pd.DataFrame:
    """
    Full options chain summary with IV and greeks (where available).

    Returns DataFrame with columns:
        instrument_name, mark_price, mark_iv, bid_iv, ask_iv,
        underlying_price, open_interest, volume,
        delta, gamma, vega, theta   (may be NaN if not returned by API)
    """
    result = _get("get_book_summary_by_currency", {
        "currency": currency,
        "kind":     "option",
    })

    if not result:
        return pd.DataFrame()

    rows = []
    for item in result:
        greeks = item.get("greeks") or {}
        rows.append({
            "instrument_name":  item.get("instrument_name"),
            "mark_price":       item.get("mark_price"),
            "mark_iv":          item.get("mark_iv"),
            "bid_iv":           item.get("bid_iv"),
            "ask_iv":           item.get("ask_iv"),
            "underlying_price": item.get("underlying_price"),
            "open_interest":    item.get("open_interest"),
            "volume":           item.get("volume"),
            "delta":            greeks.get("delta"),
            "gamma":            greeks.get("gamma"),
            "vega":             greeks.get("vega"),
            "theta":            greeks.get("theta"),
        })

    df = pd.DataFrame(rows)
    numeric = ["mark_price", "mark_iv", "bid_iv", "ask_iv",
               "underlying_price", "open_interest", "volume",
               "delta", "gamma", "vega", "theta"]
    for col in numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.reset_index(drop=True)


# ── Mark price history ─────────────────────────────────────────────────────────

def get_mark_price_history(
    instrument: str = "BTC-PERPETUAL",
    days: int = LOOKBACK_DAYS,
    resolution: int = 60,
) -> pd.DataFrame:
    """
    Hourly OHLCV history for any Deribit instrument via TradingView chart data.

    Returns DataFrame indexed by UTC timestamp with columns:
        mark_price (close), open, high, low, volume

    Paginates automatically for the full requested lookback window.
    resolution=60 → 1-hour bars.
    """
    all_ticks:  list[int]   = []
    all_closes: list[float] = []
    all_opens:  list[float] = []
    all_highs:  list[float] = []
    all_lows:   list[float] = []

    end_ms   = _now_ms()
    start_ms = _start_ms(days)

    while True:
        result = _get("get_tradingview_chart_data", {
            "instrument_name": instrument,
            "start_timestamp": start_ms,
            "end_timestamp":   end_ms,
            "resolution":      str(resolution),
        })

        if not result or not isinstance(result, dict):
            break

        ticks  = result.get("ticks",  [])
        closes = result.get("close",  [])
        opens  = result.get("open",   [])
        highs  = result.get("high",   [])
        lows   = result.get("low",    [])

        if not ticks:
            break

        all_ticks.extend(ticks)
        all_closes.extend(closes)
        all_opens.extend(opens or [float("nan")] * len(ticks))
        all_highs.extend(highs or [float("nan")] * len(ticks))
        all_lows.extend(lows  or [float("nan")] * len(ticks))

        # Deribit returns up to ~1000 bars; paginate backwards if full
        if len(ticks) < 500:
            break

        end_ms = min(ticks) - 1
        if end_ms <= start_ms:
            break

    if not all_ticks:
        return pd.DataFrame(columns=["mark_price", "open", "high", "low"])

    df = pd.DataFrame({
        "timestamp":  all_ticks,
        "mark_price": all_closes,
        "open":       all_opens,
        "high":       all_highs,
        "low":        all_lows,
    })
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    for col in ["mark_price", "open", "high", "low"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.set_index("timestamp").sort_index().drop_duplicates()
    return df
