"""
Volatility surface construction from the Deribit options chain.

Migrated and extended from dex_deribit/signals/vol_surface.py (_parse_chain).
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from config import ATM_MONEYNESS_PCT, DELTA_25D_BAND

_MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


# ── Chain parsing ──────────────────────────────────────────────────────────────

def parse_chain(options_df: pd.DataFrame, spot: float) -> pd.DataFrame:
    """
    Enrich the raw options summary with derived columns:
        expiry, dte, dte_years, strike, option_type, moneyness_pct

    Instruments that fail to parse are silently dropped.
    """
    if options_df.empty or spot <= 0:
        return pd.DataFrame()

    now     = datetime.now(timezone.utc)
    records = []

    for _, row in options_df.iterrows():
        name = str(row.get("instrument_name", ""))
        try:
            parts    = name.split("-")
            # e.g. BTC-27DEC24-50000-C
            exp_str  = parts[1]
            strike   = float(parts[2])
            opt_type = parts[3]   # "C" or "P"
            day      = int(exp_str[:2])
            mon      = _MONTH_MAP[exp_str[2:5]]
            year     = 2000 + int(exp_str[5:])
            expiry   = datetime(year, mon, day, 8, 0, tzinfo=timezone.utc)
            dte      = max(1, (expiry - now).days)
        except (IndexError, KeyError, ValueError):
            continue

        records.append({
            **row.to_dict(),
            "expiry":        expiry,
            "dte":           dte,
            "dte_years":     dte / 365.0,
            "strike":        strike,
            "option_type":   opt_type,
            "moneyness_pct": (strike / spot - 1.0) * 100,
        })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    numeric = ["mark_iv", "bid_iv", "ask_iv", "mark_price",
               "open_interest", "volume", "delta", "gamma", "vega", "theta"]
    for col in numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ── Surface grid ──────────────────────────────────────────────────────────────

def build_surface_grid(chain: pd.DataFrame) -> pd.DataFrame:
    """
    Build a vol surface grid indexed by (expiry, strike) → mark_iv.

    Filters to instruments with non-zero open interest or volume to
    focus on liquid strikes.
    """
    if chain.empty:
        return pd.DataFrame()

    liquid = chain[(chain["open_interest"] > 0) | (chain["volume"] > 0)].copy()
    if liquid.empty:
        liquid = chain.copy()

    grid = (
        liquid.groupby(["expiry", "strike"])["mark_iv"]
        .mean()
        .reset_index()
        .set_index(["expiry", "strike"])
    )
    return grid


# ── ATM IV per expiry ─────────────────────────────────────────────────────────

def atm_iv_by_expiry(chain: pd.DataFrame, spot: float) -> pd.Series:
    """
    For each expiry, return the average ATM IV (%).

    "ATM" = |moneyness_pct| < ATM_MONEYNESS_PCT (default 5%).
    Falls back to the 5 nearest-to-money strikes if nothing qualifies.

    Returns Series indexed by expiry (datetime), values in %.
    """
    if chain.empty:
        return pd.Series(dtype=float)

    result = {}
    for expiry, grp in chain.groupby("expiry"):
        atm = grp[grp["moneyness_pct"].abs() < ATM_MONEYNESS_PCT]
        if atm.empty:
            atm = grp.nsmallest(5, grp["moneyness_pct"].abs().name
                                if "moneyness_pct" in grp else "dte")
        iv = atm["mark_iv"].dropna()
        if not iv.empty:
            result[expiry] = float(iv.mean())

    return pd.Series(result).sort_index()


# ── 25-delta skew ─────────────────────────────────────────────────────────────

def skew_25d(chain: pd.DataFrame) -> pd.Series:
    """
    25-delta skew per expiry: put_iv_25d − call_iv_25d (vol points).

    Uses delta column when available; falls back to moneyness bands (−15% to −5%
    for puts, +5% to +15% for calls).

    Returns Series indexed by expiry, values in vol points.
    Positive = puts rich relative to calls (typical in crypto).
    """
    if chain.empty:
        return pd.Series(dtype=float)

    has_delta = "delta" in chain.columns and chain["delta"].notna().any()
    lo, hi = DELTA_25D_BAND

    result = {}
    for expiry, grp in chain.groupby("expiry"):
        if has_delta:
            puts_25  = grp[(grp["option_type"] == "P") &
                           (grp["delta"].between(-hi, -lo))]
            calls_25 = grp[(grp["option_type"] == "C") &
                           (grp["delta"].between(lo, hi))]
        else:
            puts_25  = grp[(grp["option_type"] == "P") &
                           (grp["moneyness_pct"].between(-15, -5))]
            calls_25 = grp[(grp["option_type"] == "C") &
                           (grp["moneyness_pct"].between(5, 15))]

        put_iv  = float(puts_25["mark_iv"].mean())  if not puts_25.empty  else float("nan")
        call_iv = float(calls_25["mark_iv"].mean()) if not calls_25.empty else float("nan")

        if not (np.isnan(put_iv) or np.isnan(call_iv)):
            result[expiry] = put_iv - call_iv

    return pd.Series(result).sort_index()


# ── Term structure ────────────────────────────────────────────────────────────

def term_structure_ivs(chain: pd.DataFrame, spot: float) -> pd.DataFrame:
    """
    ATM IV at each available expiry, sorted by DTE, with term slope.

    Returns DataFrame with columns:
        expiry, dte, atm_iv, term_slope_vs_front

    term_slope_vs_front = atm_iv − front_atm_iv (positive = contango).
    """
    atm_series = atm_iv_by_expiry(chain, spot)
    if atm_series.empty:
        return pd.DataFrame(columns=["expiry", "dte", "atm_iv", "term_slope_vs_front"])

    now = datetime.now(timezone.utc)
    rows = []
    for expiry, iv in atm_series.items():
        dte = max(1, (expiry - now).days)
        rows.append({"expiry": expiry, "dte": dte, "atm_iv": iv})

    df = pd.DataFrame(rows).sort_values("dte").reset_index(drop=True)
    if df.empty:
        return df

    front_iv = df["atm_iv"].iloc[0]
    df["term_slope_vs_front"] = df["atm_iv"] - front_iv
    return df
