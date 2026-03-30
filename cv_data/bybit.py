"""
cross_vol/data/bybit.py — Bybit options + spot data for the extended universe.

Bybit supports options for: BTC, ETH, SOL, XRP, DOGE (and more).
Unlike Deribit, Bybit does NOT publish a rolling-30d IV index, so we
compute ATM IV directly from the live options chain.

Key functions:
    get_atm_iv(coin, target_dte)  → current ATM implied vol %
    get_spot_klines(coin, days)   → OHLCV DataFrame for RV computation
    get_spot_price(coin)          → current mid price
"""
from __future__ import annotations

import warnings
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import requests

# ── Constants ────────────────────────────────────────────────────────────────

BASE = "https://api.bybit.com/v5"
TIMEOUT = 10
ATM_BAND = 0.08          # moneyness tolerance: |K/S - 1| < 8%
MIN_IV   = 0.01          # minimum plausible mark IV (decimal, not %)
MAX_IV   = 20.0          # maximum plausible mark IV (decimal)


# ── Internal helpers ─────────────────────────────────────────────────────────

def _get(endpoint: str, params: dict) -> list | dict:
    url = f"{BASE}/{endpoint}"
    r = requests.get(url, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    if data.get("retCode", 0) != 0:
        raise RuntimeError(f"Bybit error [{data.get('retCode')}]: {data.get('retMsg')}")
    return data.get("result", {})


def _parse_bybit_symbol(symbol: str) -> dict | None:
    """
    Parse e.g. 'SOL-31MAR26-84-C-USDT' → {expiry, dte, strike, otype}
    """
    parts = symbol.split("-")
    if len(parts) < 4:
        return None
    try:
        expiry = datetime.strptime(parts[1], "%d%b%y").replace(tzinfo=timezone.utc)
        strike = float(parts[2])
        otype  = parts[3].upper()  # "C" or "P"
        dte    = max(1, (expiry - datetime.now(timezone.utc)).days)
        return {"expiry": expiry, "dte": dte, "strike": strike, "otype": otype}
    except (ValueError, IndexError):
        return None


# ── Public functions ─────────────────────────────────────────────────────────

def get_spot_price(coin: str) -> float:
    """Current mid-market spot price for coin/USDT perpetual."""
    coin = coin.upper()
    result = _get("market/tickers", {"category": "linear", "symbol": f"{coin}USDT"})
    items = result.get("list", [])
    if not items:
        raise ValueError(f"No spot ticker for {coin} on Bybit")
    return float(items[0]["lastPrice"])


def get_atm_iv(coin: str, target_dte: int = 21) -> tuple[float, int]:
    """
    Compute ATM implied vol (%) from Bybit's live options chain.

    Finds the expiry closest to target_dte, then averages mark_iv across
    all near-ATM strikes (moneyness within ATM_BAND). Returns both the
    IV in % and the actual DTE used.

    Returns (atm_iv_pct, dte_used) or raises ValueError if no usable data.
    """
    coin = coin.upper()
    result = _get("market/tickers", {"category": "option", "baseCoin": coin})
    items = result.get("list", [])
    if not items:
        raise ValueError(f"No Bybit options found for {coin}")

    records = []
    for item in items:
        meta = _parse_bybit_symbol(item["symbol"])
        if meta is None:
            continue
        try:
            mark_iv = float(item.get("markIv") or 0)
            spot    = float(item.get("underlyingPrice") or 0)
            if spot <= 0 or mark_iv <= MIN_IV or mark_iv >= MAX_IV:
                continue
            moneyness = abs(meta["strike"] / spot - 1)
            records.append({
                "dte":       meta["dte"],
                "mark_iv":   mark_iv,
                "moneyness": moneyness,
                "spot":      spot,
            })
        except (ValueError, TypeError):
            continue

    if not records:
        raise ValueError(f"No parseable options for {coin}")

    # Pick expiry nearest to target_dte
    dtes = sorted(set(r["dte"] for r in records))
    best_dte = min(dtes, key=lambda d: abs(d - target_dte))

    # ATM filter
    expiry_recs = [r for r in records if r["dte"] == best_dte]
    atm_recs    = [r for r in expiry_recs if r["moneyness"] < ATM_BAND]
    if not atm_recs:
        # Fallback: 6 nearest-to-money
        atm_recs = sorted(expiry_recs, key=lambda r: r["moneyness"])[:6]
    if not atm_recs:
        raise ValueError(f"No ATM options for {coin} at {best_dte}d DTE")

    avg_iv_pct = float(np.mean([r["mark_iv"] for r in atm_recs])) * 100.0
    return avg_iv_pct, best_dte


def get_spot_klines(coin: str, days: int = 500, interval: str = "D") -> pd.DataFrame:
    """
    Fetch daily OHLCV klines for coin/USDT from Bybit.

    Returns DataFrame indexed by UTC date with columns:
        open, high, low, close, volume
    """
    coin    = coin.upper()
    symbol  = f"{coin}USDT"
    end_ms  = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = end_ms - days * 86400 * 1000

    result = _get("market/kline", {
        "category": "linear",
        "symbol":   symbol,
        "interval": interval,
        "start":    start_ms,
        "end":      end_ms,
        "limit":    min(days, 1000),
    })

    raw = result.get("list", [])
    if not raw:
        return pd.DataFrame()

    # Bybit returns [startTime, open, high, low, close, volume, turnover]
    rows = []
    for bar in raw:
        try:
            ts = pd.Timestamp(int(bar[0]), unit="ms", tz="UTC")
            rows.append({
                "date":   ts,
                "open":   float(bar[1]),
                "high":   float(bar[2]),
                "low":    float(bar[3]),
                "close":  float(bar[4]),
                "volume": float(bar[5]),
            })
        except (ValueError, IndexError):
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("date").sort_index()
    return df


def get_rv30(klines: pd.DataFrame) -> float:
    """
    Compute the 30-day realized volatility (annualized %) from daily klines.
    Uses close-to-close log returns.
    """
    if len(klines) < 5:
        return float("nan")
    log_ret = np.log(klines["close"] / klines["close"].shift(1)).dropna()
    rv = float(log_ret.tail(30).std() * np.sqrt(252) * 100)
    return rv


def get_rv30_series(klines: pd.DataFrame, window: int = 30) -> pd.Series:
    """
    Rolling 30-day realized vol (annualized %) at each daily bar.
    Returns a pd.Series aligned to klines index.
    """
    log_ret = np.log(klines["close"] / klines["close"].shift(1)).dropna()
    rv = log_ret.rolling(window, min_periods=max(5, window // 2)).std() * np.sqrt(252) * 100
    return rv.rename("rv30")
