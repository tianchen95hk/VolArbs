"""
cross_vol/data/binance.py — Binance spot history via CCXT.

Used for:
  - Historical daily OHLCV for any large-cap crypto
  - Realized volatility (RV30) computation across the extended universe
  - More data history and more symbols than Bybit's kline API

Reads BINANCE_API_KEY / BINANCE_API_SECRET from .env (falls back to
the factor_quant .env if option_vol/.env doesn't exist).
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# ── Load .env ──────────────────────────────────────────────────────────────
def _load_env():
    try:
        from dotenv import load_dotenv
        # Try option_vol root first, then factor_quant
        root = Path(__file__).resolve().parents[3]  # …/Claude Code/
        for candidate in [
            root / "option_vol" / ".env",
            root / "factor_quant" / ".env",
        ]:
            if candidate.exists():
                load_dotenv(candidate)
                return
        load_dotenv()
    except ImportError:
        pass

_load_env()

# ── CCXT exchange ──────────────────────────────────────────────────────────
def _make_exchange(authenticated: bool = False):
    """
    Return a Binance CCXT instance.
    OHLCV / ticker data is public — no auth needed.
    Pass authenticated=True only for private (trading) endpoints.
    """
    import ccxt
    opts: dict = {"enableRateLimit": True, "options": {"defaultType": "spot"}}
    if authenticated:
        opts["apiKey"] = os.getenv("BINANCE_API_KEY", "")
        opts["secret"] = os.getenv("BINANCE_API_SECRET", "")
    return ccxt.binance(opts)


# ── Public functions ───────────────────────────────────────────────────────

def get_spot_klines(coin: str, days: int = 500) -> pd.DataFrame:
    """
    Fetch daily OHLCV for coin/USDT from Binance via CCXT.

    Returns DataFrame indexed by UTC date with columns:
        open, high, low, close, volume
    """
    ex = _make_exchange()
    symbol = f"{coin.upper()}/USDT"
    since_ms = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
    raw = ex.fetch_ohlcv(symbol, timeframe="1d", since=since_ms, limit=min(days + 5, 1000))
    if not raw:
        return pd.DataFrame()
    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").astype(float).sort_index()
    return df


def get_rv30_series(klines: pd.DataFrame, window: int = 30) -> pd.Series:
    """
    Rolling annualized RV (%) using close-to-close log returns.
    Returns a pd.Series aligned to klines index.
    """
    log_ret = np.log(klines["close"] / klines["close"].shift(1)).dropna()
    rv = log_ret.rolling(window, min_periods=max(5, window // 2)).std() * np.sqrt(252) * 100
    return rv.rename("rv30")


def get_rv30(klines: pd.DataFrame, window: int = 30) -> float:
    """Current 30-day realized vol (annualized %) — scalar."""
    s = get_rv30_series(klines, window)
    s = s.dropna()
    return float(s.iloc[-1]) if len(s) > 0 else float("nan")


def get_spot_price(coin: str) -> float:
    """Current spot price for coin/USDT."""
    ex = _make_exchange()
    ticker = ex.fetch_ticker(f"{coin.upper()}/USDT")
    return float(ticker["last"])
