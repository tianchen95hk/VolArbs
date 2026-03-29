"""
Black-Scholes-Merton option pricing, greeks, and IV solver.

All functions assume:
  - European-style options (Deribit BTC/ETH options are European)
  - No dividends (crypto assets pay no dividends)
  - Continuous risk-free rate r

Vol convention: sigma is a decimal (0.60 = 60% annualised vol).
IV convention: returned as decimal; callers multiply by 100 for display.
"""

from __future__ import annotations

import math
import numpy as np
from scipy.stats import norm


# ── Core BSM formula ──────────────────────────────────────────────────────────

def _d1d2(S: float, K: float, T: float, r: float, sigma: float) -> tuple[float, float]:
    """Compute d1 and d2 for BSM formula. T must be > 0."""
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2


def bsm_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str,
) -> float:
    """
    Black-Scholes-Merton option price.

    Parameters
    ----------
    S           : spot price
    K           : strike price
    T           : time to expiry in years (must be > 0)
    r           : risk-free rate (annualised, decimal)
    sigma       : volatility (annualised, decimal, e.g. 0.60 for 60%)
    option_type : "C" for call, "P" for put

    Returns price in the same currency unit as S and K (USD for BTC options).
    Returns NaN if T <= 0 or sigma <= 0.
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return float("nan")

    d1, d2 = _d1d2(S, K, T, r, sigma)
    disc = math.exp(-r * T)

    if option_type.upper() == "C":
        return S * norm.cdf(d1) - K * disc * norm.cdf(d2)
    else:
        return K * disc * norm.cdf(-d2) - S * norm.cdf(-d1)


def bsm_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str,
) -> dict[str, float]:
    """
    BSM greeks.

    Returns
    -------
    delta : dV/dS
    gamma : d²V/dS²
    vega  : dV/d(sigma) per 1% move in vol (i.e. divided by 100)
    theta : dV/dt per calendar day (negative for long options)
    rho   : dV/dr (for completeness)
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return {k: float("nan") for k in ["delta", "gamma", "vega", "theta", "rho"]}

    d1, d2 = _d1d2(S, K, T, r, sigma)
    sqrt_T = math.sqrt(T)
    disc   = math.exp(-r * T)
    pdf_d1 = norm.pdf(d1)

    gamma = pdf_d1 / (S * sigma * sqrt_T)
    # Vega per 1% vol move (divide by 100 to convert from per-unit to per-%)
    vega  = S * pdf_d1 * sqrt_T / 100

    if option_type.upper() == "C":
        delta = norm.cdf(d1)
        theta = (
            -(S * pdf_d1 * sigma) / (2 * sqrt_T)
            - r * K * disc * norm.cdf(d2)
        ) / 365
        rho = K * T * disc * norm.cdf(d2) / 100
    else:
        delta = norm.cdf(d1) - 1
        theta = (
            -(S * pdf_d1 * sigma) / (2 * sqrt_T)
            + r * K * disc * norm.cdf(-d2)
        ) / 365
        rho = -K * T * disc * norm.cdf(-d2) / 100

    return {
        "delta": delta,
        "gamma": gamma,
        "vega":  vega,
        "theta": theta,
        "rho":   rho,
    }


# ── Implied volatility solver ─────────────────────────────────────────────────

def _bs_approx(market_price: float, S: float, T: float) -> float:
    """Brenner-Subrahmanyam approximation for ATM IV as Newton starting point."""
    if S <= 0 or T <= 0:
        return 0.5
    return math.sqrt(2 * math.pi / T) * market_price / S


def implied_vol(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float:
    """
    Implied volatility via Newton-Raphson, with bisection fallback.

    Returns sigma as a decimal (e.g. 0.60 for 60% vol).
    Returns NaN if T <= 0, price is non-positive, or solver fails to converge.
    """
    if T <= 0 or market_price <= 0 or S <= 0 or K <= 0:
        return float("nan")

    # Intrinsic value check
    disc = math.exp(-r * T)
    intrinsic = max(0.0, S - K * disc) if option_type.upper() == "C" else max(0.0, K * disc - S)
    if market_price < intrinsic - 1e-4:
        return float("nan")

    # Newton-Raphson
    sigma = max(0.01, _bs_approx(market_price, S, T))
    for _ in range(max_iter):
        price = bsm_price(S, K, T, r, sigma, option_type)
        if math.isnan(price):
            break
        diff = price - market_price
        if abs(diff) < tol:
            return sigma
        # vega (per unit sigma, not per %)
        d1, _ = _d1d2(S, K, T, r, sigma)
        vega_raw = S * norm.pdf(d1) * math.sqrt(T)
        if vega_raw < 1e-10:
            break
        sigma -= diff / vega_raw
        if sigma <= 0:
            sigma = 1e-4

    # Bisection fallback
    lo, hi = 1e-4, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2
        price = bsm_price(S, K, T, r, mid, option_type)
        if math.isnan(price):
            return float("nan")
        if abs(price - market_price) < tol:
            return mid
        if price < market_price:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            return (lo + hi) / 2

    return float("nan")


# ── Put-call parity ───────────────────────────────────────────────────────────

def parity_check(
    call_price: float,
    put_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
) -> float:
    """
    Put-call parity deviation in USD.

    Theoretical: C − P = S − K·exp(−r·T)
    Returns:     (C − P) − (S − K·exp(−r·T))

    A positive value means calls are relatively expensive vs puts.
    A negative value means puts are relatively expensive vs calls.
    Near zero means no parity violation.
    """
    if T <= 0:
        return float("nan")
    forward_pv = S - K * math.exp(-r * T)
    market_diff = call_price - put_price
    return market_diff - forward_pv
