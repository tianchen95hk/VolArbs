"""
cross_vol/signal.py — Cross-asset vol spread signal computation.

Computes the normalised spread between two assets' vol premiums (DVOL − RV30)
and generates entry/exit signals for the vega-neutral spread strategy.

Signal definition:
    spread_t = vol_premium_A(t) − vol_premium_B(t)
             where vol_premium_X = DVOL_X − RV30_X

Entry: |z_t| >= ZSCORE_ENTRY  AND  |spread_t| >= SPREAD_MIN_PTS
Exit:  |z_t| <= ZSCORE_EXIT   OR   holding_days >= HOLDING_DAYS_MAX

The dual entry condition (z-score + absolute floor) prevents firing during
low-variance regimes where 1.5σ would correspond to only 2 vol pts of spread.
"""
from __future__ import annotations

import sys
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd
import numpy as np

_PER_ASSET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "per_asset")
if _PER_ASSET not in sys.path:
    sys.path.insert(0, _PER_ASSET)

from cv_config import (
    CROSS_VOL_SPREAD_LOOKBACK,
    CROSS_VOL_ZSCORE_ENTRY,
    CROSS_VOL_ZSCORE_EXIT,
    CROSS_VOL_SPREAD_MIN_PTS,
    CROSS_VOL_HOLDING_DAYS_MAX,
)
from universe import AssetVolData


# ── Data containers ───────────────────────────────────────────────────────────

@dataclass
class CrossVolSpread:
    """One row of the spread time series at a specific date."""
    date:          datetime
    asset_long:    str       # currency to go LONG vol (cheaper vol premium)
    asset_short:   str       # currency to go SHORT vol (richer vol premium)
    vp_long:       float     # vol_premium of long leg (vol pts)
    vp_short:      float     # vol_premium of short leg (vol pts)
    raw_spread:    float     # vp_short − vp_long (positive = A is richer)
    spread_zscore: float     # rolling z-score
    spread_mean:   float
    spread_std:    float


@dataclass
class CrossVolSignal:
    """Actionable signal for one point in time."""
    direction:       str           # "enter_spread" | "exit_spread" | "neutral"
    asset_long:      str           # buy vol in this asset
    asset_short:     str           # sell vol in this asset
    raw_spread:      float         # vol pts
    spread_zscore:   float
    confidence:      float         # min(1.0, |zscore| / ZSCORE_ENTRY)
    entry_triggered: bool
    exit_triggered:  bool
    description:     str
    action_items:    list[str] = field(default_factory=list)


# ── Core functions ────────────────────────────────────────────────────────────

def compute_spread_series(
    aligned_df: pd.DataFrame,
    asset_a: str,
    asset_b: str,
    lookback: int = CROSS_VOL_SPREAD_LOOKBACK,
) -> pd.DataFrame:
    """
    Compute the full cross-asset vol spread time series.

    Spread = vol_premium[asset_a] − vol_premium[asset_b]
    Positive spread → asset_a has the richer vol premium → short A, long B.

    Z-score uses an expanding window for the first `lookback` dates, then
    switches to a rolling `lookback`-day window for stationarity.

    Returns a DataFrame with columns:
        date, raw_spread, spread_mean, spread_std, spread_zscore,
        asset_long, asset_short, vp_a, vp_b
    """
    asset_a = asset_a.upper()
    asset_b = asset_b.upper()

    if asset_a not in aligned_df.columns or asset_b not in aligned_df.columns:
        raise ValueError(
            f"Assets {asset_a}, {asset_b} not found in aligned_df "
            f"(available: {list(aligned_df.columns)})"
        )

    vp_a = aligned_df[asset_a]
    vp_b = aligned_df[asset_b]
    raw  = vp_a - vp_b  # positive → A is richer

    # Rolling mean and std: expanding for burn-in, rolling thereafter
    roll_mean = raw.expanding(min_periods=2).mean()
    roll_std  = raw.expanding(min_periods=2).std()

    # Switch to fixed rolling window once we have enough history
    fixed_mean = raw.rolling(lookback).mean()
    fixed_std  = raw.rolling(lookback).std()

    mask = raw.index.get_loc(raw.index[lookback]) if len(raw) > lookback else len(raw)
    roll_mean.iloc[lookback:] = fixed_mean.iloc[lookback:]
    roll_std.iloc[lookback:]  = fixed_std.iloc[lookback:]

    zscore = (raw - roll_mean) / roll_std.replace(0, float("nan"))

    rows = []
    for ts in aligned_df.index:
        r   = float(raw.loc[ts])
        z   = float(zscore.loc[ts]) if not np.isnan(zscore.loc[ts]) else 0.0
        vpa = float(vp_a.loc[ts])
        vpb = float(vp_b.loc[ts])

        # Directional assignment via z-score sign (mean-reversion logic):
        #   spread = VP_A − VP_B
        #   z > 0  → A is RELATIVELY expensive vs history → short A, long B
        #   z < 0  → B is RELATIVELY expensive vs history → short B, long A
        #
        # Using instantaneous VP (which asset is higher right now) gives the
        # WRONG direction when the spread is below its historical mean, because
        # "cheap" in absolute VP can still be "expensive" relative to history.
        if z >= 0:
            a_long, a_short = asset_b, asset_a   # long B (historically cheap), short A
            vp_long, vp_short = vpb, vpa
        else:
            a_long, a_short = asset_a, asset_b   # long A (historically cheap), short B
            vp_long, vp_short = vpa, vpb

        rows.append({
            "date":          ts,
            "raw_spread":    abs(r),      # magnitude for threshold comparison
            "signed_spread": r,           # positive if VP_A > VP_B
            "spread_mean":   float(roll_mean.loc[ts]),
            "spread_std":    float(roll_std.loc[ts]) if not roll_std.isna().loc[ts] else 0.0,
            "spread_zscore": abs(z),      # magnitude
            "signed_zscore": z,           # signed — determines direction
            "asset_long":    a_long,
            "asset_short":   a_short,
            "vp_long":       vp_long,
            "vp_short":      vp_short,
        })

    return pd.DataFrame(rows).set_index("date")


def generate_signal(
    spread_series: pd.DataFrame,
    current_date: datetime,
    in_trade: bool,
    holding_days: int = 0,
    zscore_entry: float = CROSS_VOL_ZSCORE_ENTRY,
    zscore_exit:  float = CROSS_VOL_ZSCORE_EXIT,
    min_spread_pts: float = CROSS_VOL_SPREAD_MIN_PTS,
    holding_days_max: int = CROSS_VOL_HOLDING_DAYS_MAX,
) -> CrossVolSignal:
    """
    At a given date, decide entry / exit for the spread trade.

    Entry: not in_trade AND |zscore| >= zscore_entry AND spread >= min_spread_pts
    Exit:  in_trade AND (|zscore| <= zscore_exit OR holding_days >= MAX)
    """
    # Find the row at or before current_date
    ts = pd.Timestamp(current_date, tz="UTC") if current_date.tzinfo is None else pd.Timestamp(current_date)
    available = spread_series[spread_series.index <= ts]
    if available.empty:
        return CrossVolSignal(
            direction="neutral", asset_long="", asset_short="",
            raw_spread=0.0, spread_zscore=0.0, confidence=0.0,
            entry_triggered=False, exit_triggered=False,
            description="No data available",
        )

    row = available.iloc[-1]
    z     = float(row["spread_zscore"])
    raw   = float(row["raw_spread"])
    a_l   = str(row["asset_long"])
    a_s   = str(row["asset_short"])

    confidence = min(1.0, z / zscore_entry) if zscore_entry > 0 else 0.0

    entry_triggered = (not in_trade) and (z >= zscore_entry) and (raw >= min_spread_pts)
    exit_triggered  = in_trade and (
        (z <= zscore_exit) or (holding_days >= holding_days_max)
    )

    if entry_triggered:
        direction = "enter_spread"
        description = (
            f"ENTER | Long {a_l} vol ({row['vp_long']:.1f} VP)  "
            f"Short {a_s} vol ({row['vp_short']:.1f} VP)  "
            f"Spread {raw:.1f} vol pts  z={z:.2f}"
        )
        action_items = [
            f"BUY ATM straddle on {a_l} (~{holding_days_max}d DTE)",
            f"SELL ATM straddle on {a_s} (~{holding_days_max}d DTE)",
            "Size legs vega-neutral (see sizing.py)",
            "Delta-hedge each leg daily via perpetual",
        ]
    elif exit_triggered:
        reason = "z-score mean reversion" if z <= zscore_exit else "max holding period"
        direction = "exit_spread"
        description = (
            f"EXIT | Spread {raw:.1f} vol pts  z={z:.2f}  ({reason})"
        )
        action_items = [
            f"CLOSE {a_l} straddle position",
            f"CLOSE {a_s} straddle position",
            "Unwind any delta hedges",
        ]
    else:
        direction = "neutral"
        description = (
            f"NEUTRAL | {a_l}/{a_s} spread {raw:.1f} vol pts  "
            f"z={z:.2f}  (need {zscore_entry:.1f}σ AND {min_spread_pts:.0f}pt)"
        )
        action_items = []

    return CrossVolSignal(
        direction       = direction,
        asset_long      = a_l,
        asset_short     = a_s,
        raw_spread      = raw,
        spread_zscore   = z,
        confidence      = confidence,
        entry_triggered = entry_triggered,
        exit_triggered  = exit_triggered,
        description     = description,
        action_items    = action_items,
    )


def compute_live_signal(
    universe: dict[str, AssetVolData],
    asset_a: str = "BTC",
    asset_b: str = "ETH",
) -> CrossVolSignal:
    """
    Convenience wrapper for live/scan mode.
    Builds the spread series from the latest universe data and returns
    the current-date signal.
    """
    from universe import align_assets  # local import to avoid circular

    aligned = align_assets(universe)
    spread  = compute_spread_series(aligned, asset_a, asset_b)
    now     = datetime.now(timezone.utc)
    return generate_signal(spread, now, in_trade=False)
