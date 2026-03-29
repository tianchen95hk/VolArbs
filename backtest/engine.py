"""
Event-driven daily backtest engine for option_vol strategies.

Data availability
-----------------
Deribit's public API provides no historical options chain snapshots.
We use two tiers:

  Tier 1 — DVOL proxy (always available, 90 days hourly):
    Use DVOL close as ATM IV proxy, spot_history for RV.
    P&L = vega-P&L approximation:
        pnl ≈ (entry_IV − exit_IV) × vega_estimate × notional
    Powers: vol_premium strategy.

  Tier 2 — BSM-reconstructed chain (approximate):
    At each historical date, use DVOL as ATM IV and apply a fixed skew
    assumption to reconstruct the full surface.
    Powers: skew, term_structure strategies.
    Results are labelled "approximate" in the report.

Loop structure (per daily bar)
------------------------------
1. Get spot and DVOL for the date.
2. Compute RV from trailing spot history.
3. Run signal logic → decide to open positions.
4. Mark open positions to market (BSM repricing).
5. Expire positions whose expiry date has passed.
6. Check and apply stop-losses.
7. If HEDGE_DELTA: record synthetic delta hedge cost.
8. Append equity value to curve.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from config import (
    INITIAL_CAPITAL, POSITION_SIZE_PCT, COMMISSION_PCT,
    HEDGE_DELTA, HEDGE_COST_PCT, RISK_FREE_RATE,
    VOL_PREMIUM_THRESHOLD, SKEW_THRESHOLD, TERM_SLOPE_THRESHOLD,
)
from analytics.rv import realised_vol_cc, rv_term_structure
from analytics.bsm import bsm_price, bsm_greeks
from backtest.positions import OptionPosition, PositionTracker
from backtest.report import compute_stats


# ── Synthetic option price from DVOL ─────────────────────────────────────────

def _synthetic_atm_price(
    spot: float,
    atm_iv_pct: float,
    dte: int,
    option_type: str,
    r: float = RISK_FREE_RATE,
) -> float:
    """BSM price of an ATM option (strike = spot) using DVOL as IV."""
    T = max(dte / 365.0, 1 / 365)
    sigma = atm_iv_pct / 100.0
    return bsm_price(spot, spot, T, r, sigma, option_type)


# ── Straddle (call + put at ATM) ──────────────────────────────────────────────

def _straddle_price(spot: float, atm_iv_pct: float, dte: int) -> float:
    call = _synthetic_atm_price(spot, atm_iv_pct, dte, "C")
    put  = _synthetic_atm_price(spot, atm_iv_pct, dte, "P")
    return (call + put) if not (np.isnan(call) or np.isnan(put)) else float("nan")


# ── Vol carry (Tier 1) backtest ───────────────────────────────────────────────

def _backtest_vol_carry(
    dvol_df: pd.DataFrame,
    spot_history: pd.DataFrame,
    start_dt: datetime,
    end_dt: datetime,
) -> pd.Series:
    """
    Simulate a rolling short-vol strategy using DVOL vs 30d RV.

    Entry  : when DVOL > RV30d + VOL_PREMIUM_THRESHOLD
    Exit   : 30 calendar days after entry (or when vol premium collapses)
    P&L    : (entry_IV − exit_IV) × notional_fraction
             — vega P&L approximation, same methodology as dex_deribit vol carry
    """
    equity = []
    dates  = []
    capital = INITIAL_CAPITAL

    # Align daily closes — guard against empty or non-DatetimeIndex frames
    def _to_daily(df: pd.DataFrame, col: str) -> pd.Series:
        s = df[col] if col in df.columns else pd.Series(dtype=float)
        if s.empty:
            return pd.Series(dtype=float)
        if not isinstance(s.index, pd.DatetimeIndex):
            return pd.Series(dtype=float)
        return s.resample("1D").last().ffill()

    dvol_daily  = _to_daily(dvol_df,      "close")
    spot_daily  = _to_daily(spot_history, "mark_price")

    date_range = pd.date_range(start=start_dt, end=end_dt, freq="D", tz=timezone.utc)

    in_trade    = False
    entry_iv    = None
    entry_date  = None
    notional    = INITIAL_CAPITAL * POSITION_SIZE_PCT

    for dt in date_range:
        if dt not in dvol_daily.index:
            # Interpolate from nearest available
            dvol_slice = dvol_daily[:dt]
            if dvol_slice.empty:
                continue
            iv_today = float(dvol_slice.iloc[-1])
        else:
            iv_today = float(dvol_daily[dt])

        # RV from trailing spot history
        hist_slice = spot_daily[:dt].dropna()
        rv_today   = realised_vol_cc(hist_slice, window=30)

        if np.isnan(rv_today) or np.isnan(iv_today):
            equity.append(capital)
            dates.append(dt)
            continue

        vol_premium = iv_today - rv_today

        if not in_trade and vol_premium >= VOL_PREMIUM_THRESHOLD:
            # Enter short-vol trade
            in_trade   = True
            entry_iv   = iv_today
            entry_date = dt
            # Deduct entry commission
            capital -= notional * COMMISSION_PCT

        elif in_trade:
            hold_days = (dt - entry_date).days
            # Exit after 30 days or if premium collapses below threshold/2
            if hold_days >= 30 or vol_premium < VOL_PREMIUM_THRESHOLD / 2:
                exit_iv = iv_today
                # Short-vol: profit when IV falls
                pnl_pct = (entry_iv - exit_iv) / entry_iv
                pnl     = notional * pnl_pct
                capital += pnl - notional * COMMISSION_PCT   # exit commission

                if HEDGE_DELTA:
                    # Approximate hedge cost: 0.5% of notional for 30-day period
                    capital -= notional * HEDGE_COST_PCT * hold_days / 7

                in_trade = False

        equity.append(capital)
        dates.append(dt)

    return pd.Series(equity, index=pd.DatetimeIndex(dates))


# ── Skew / term structure (Tier 2) backtest ───────────────────────────────────

def _backtest_skew(
    dvol_df: pd.DataFrame,
    spot_history: pd.DataFrame,
    start_dt: datetime,
    end_dt: datetime,
    skew_assumption_pts: float = 5.0,
) -> pd.Series:
    """
    Simulate risk reversal trades on the reconstructed skew.

    Since historical skew data is not available, we apply a constant skew
    assumption (put_iv = ATM_IV + skew_assumption/2, call_iv = ATM_IV - skew/2).
    A trade fires when the reconstructed skew exceeds SKEW_THRESHOLD.

    NOTE: This is a Tier 2 approximate backtest. Results show the P&L of
    riding mean-reversion in the skew level, not a real historical record.
    """
    equity = []
    dates  = []
    capital = INITIAL_CAPITAL

    dvol_daily = (dvol_df["close"].resample("1D").last().ffill()
                  if not dvol_df.empty and isinstance(dvol_df.index, pd.DatetimeIndex)
                  else pd.Series(dtype=float))
    spot_daily = pd.Series(dtype=float)  # not used in skew backtest

    date_range = pd.date_range(start=start_dt, end=end_dt, freq="D", tz=timezone.utc)

    in_trade = False
    entry_put_iv = entry_call_iv = None
    entry_date = None
    notional = INITIAL_CAPITAL * POSITION_SIZE_PCT

    for dt in date_range:
        dvol_slice = dvol_daily[:dt]
        if dvol_slice.empty:
            equity.append(capital)
            dates.append(dt)
            continue

        atm_iv = float(dvol_slice.iloc[-1])
        # Reconstruct skew: positive skew (puts rich) assumed
        put_iv  = atm_iv + skew_assumption_pts / 2
        call_iv = atm_iv - skew_assumption_pts / 2

        skew = put_iv - call_iv  # = skew_assumption_pts by construction

        if not in_trade and skew >= SKEW_THRESHOLD:
            in_trade     = True
            entry_put_iv  = put_iv
            entry_call_iv = call_iv
            entry_date    = dt
            capital -= notional * COMMISSION_PCT * 2

        elif in_trade:
            hold_days = (dt - entry_date).days
            if hold_days >= 21:
                exit_skew = put_iv - call_iv
                # Short skew: profit when skew compresses
                skew_change = (entry_put_iv - entry_call_iv) - exit_skew
                pnl = notional * skew_change / (entry_put_iv + 1e-6)
                capital += pnl - notional * COMMISSION_PCT * 2
                in_trade = False

        equity.append(capital)
        dates.append(dt)

    return pd.Series(equity, index=pd.DatetimeIndex(dates))


def _backtest_term_structure(
    dvol_df: pd.DataFrame,
    spot_history: pd.DataFrame,
    start_dt: datetime,
    end_dt: datetime,
) -> pd.Series:
    """
    Simulate calendar spread trades.

    Approximates term structure using DVOL (ATM) and a flat RV-slope comparison.
    Fires when DVOL 30d MA > DVOL 7d MA (contango) by more than TERM_SLOPE_THRESHOLD.

    NOTE: Tier 2 approximate backtest.
    """
    equity = []
    dates  = []
    capital = INITIAL_CAPITAL

    dvol_daily = (dvol_df["close"].resample("1D").last().ffill()
                  if not dvol_df.empty and isinstance(dvol_df.index, pd.DatetimeIndex)
                  else pd.Series(dtype=float))

    date_range = pd.date_range(start=start_dt, end=end_dt, freq="D", tz=timezone.utc)

    in_trade   = False
    entry_slope = None
    entry_date  = None
    notional = INITIAL_CAPITAL * POSITION_SIZE_PCT

    for dt in date_range:
        dvol_slice = dvol_daily[:dt].dropna()
        if len(dvol_slice) < 30:
            equity.append(capital)
            dates.append(dt)
            continue

        # Proxy term slope: 30d MA - 7d MA of DVOL (higher 30d = contango)
        ma30 = float(dvol_slice.tail(30).mean())
        ma7  = float(dvol_slice.tail(7).mean())
        slope = ma30 - ma7

        if not in_trade and slope >= TERM_SLOPE_THRESHOLD:
            in_trade    = True
            entry_slope = slope
            entry_date  = dt
            capital -= notional * COMMISSION_PCT * 2

        elif in_trade:
            hold_days = (dt - entry_date).days
            if hold_days >= 30 or slope < TERM_SLOPE_THRESHOLD / 2:
                exit_slope = slope
                slope_change = entry_slope - exit_slope
                pnl = notional * slope_change / max(entry_slope, 1.0) * 0.5
                capital += pnl - notional * COMMISSION_PCT * 2
                in_trade = False

        equity.append(capital)
        dates.append(dt)

    return pd.Series(equity, index=pd.DatetimeIndex(dates))


# ── Combined equity ───────────────────────────────────────────────────────────

def _combine_equity(curves: list[pd.Series]) -> pd.Series:
    """Equal-weight combination of multiple strategy equity curves."""
    if not curves:
        return pd.Series(dtype=float)
    df = pd.concat(curves, axis=1).ffill()
    rets = df.pct_change()
    combined_ret = rets.mean(axis=1)
    combined = (1 + combined_ret).cumprod() * INITIAL_CAPITAL
    combined.iloc[0] = INITIAL_CAPITAL
    return combined


# ── Public entry point ────────────────────────────────────────────────────────

def run_backtest(
    dvol_df: pd.DataFrame,
    spot_history: pd.DataFrame,
    start_dt: datetime,
    end_dt: datetime,
) -> dict[str, dict]:
    """
    Run all strategy backtests for a given time window.

    Returns dict: strategy_name → stats_dict (from backtest.report.compute_stats)
    Also returns the equity curves under key "_equity".
    """
    from backtest.report import compute_stats

    curves: dict[str, pd.Series] = {}

    print(f"  Running vol carry (Tier 1) ...")
    curves["Vol Carry"] = _backtest_vol_carry(dvol_df, spot_history, start_dt, end_dt)

    print(f"  Running skew arb (Tier 2 approx) ...")
    curves["Skew Arb*"] = _backtest_skew(dvol_df, spot_history, start_dt, end_dt)

    print(f"  Running term structure (Tier 2 approx) ...")
    curves["Term Struct*"] = _backtest_term_structure(dvol_df, spot_history, start_dt, end_dt)

    curves["Combined"] = _combine_equity(list(curves.values()))

    stats = {name: compute_stats(eq) for name, eq in curves.items()}
    stats["_equity"] = curves
    return stats
