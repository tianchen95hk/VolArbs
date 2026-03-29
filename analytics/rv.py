"""
Realized volatility estimators.

All functions return annualised vol in % (e.g. 55.0 = 55%).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import EWMA_LAMBDA, RV_WINDOW_DAYS


def realised_vol_cc(prices: pd.Series, window: int = RV_WINDOW_DAYS) -> float:
    """
    Close-to-close annualised realized volatility (%).

    Uses log returns over a rolling window.
    Returns NaN if there are fewer than `window` observations.
    """
    log_ret = np.log(prices / prices.shift(1)).dropna()
    if len(log_ret) < window:
        return float("nan")
    return float(log_ret.tail(window).std() * np.sqrt(365) * 100)


def realised_vol_ewma(prices: pd.Series, lam: float = EWMA_LAMBDA) -> float:
    """
    EWMA (RiskMetrics) annualised realized volatility (%).

    More responsive to recent moves than the rolling-window estimator.
    sigma²_t = lam × sigma²_{t-1} + (1 − lam) × r_t²
    """
    log_ret = np.log(prices / prices.shift(1)).dropna()
    if len(log_ret) < 2:
        return float("nan")

    var = float(log_ret.iloc[0] ** 2)
    for r in log_ret.iloc[1:]:
        var = lam * var + (1 - lam) * r ** 2

    return float(np.sqrt(var * 365) * 100)


def realised_vol_parkinson(ohlc: pd.DataFrame, window: int = RV_WINDOW_DAYS) -> float:
    """
    Parkinson high-low estimator of annualised vol (%).

    More efficient than close-to-close (captures intraday range).
    Requires columns: high, low.
    Returns NaN if columns missing or insufficient data.
    """
    if "high" not in ohlc.columns or "low" not in ohlc.columns:
        return float("nan")

    hl = ohlc[["high", "low"]].dropna()
    if len(hl) < window:
        return float("nan")

    hl = hl.tail(window)
    term = (np.log(hl["high"] / hl["low"]) ** 2) / (4 * np.log(2))
    var_daily = float(term.mean())
    return float(np.sqrt(var_daily * 365) * 100)


def rv_term_structure(prices: pd.Series) -> dict[int, float]:
    """
    Realized vol at standard lookback windows (7, 14, 30, 60, 90 days).

    Returns dict {window_days: rv_%}. Windows with insufficient data return NaN.
    Useful for comparing with the IV term structure.
    """
    return {w: realised_vol_cc(prices, window=w) for w in [7, 14, 30, 60, 90]}
