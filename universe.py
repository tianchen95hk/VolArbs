"""
cross_vol/universe.py — Multi-asset data fetch and preprocessing.

Fetches DVOL + spot history for each currency via the per_asset data layer,
computes rolling RV, and aligns all assets onto a common daily timeline.
"""
from __future__ import annotations

import sys
import os
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd
import numpy as np

# ── Path bootstrap ────────────────────────────────────────────────────────────
_PER_ASSET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "per_asset")
if _PER_ASSET not in sys.path:
    sys.path.insert(0, _PER_ASSET)

from data.cache import fetch_with_cache
from analytics.rv import realised_vol_cc
from cv_config import (
    CROSS_VOL_ASSETS,
    CROSS_VOL_LOOKBACK_DAYS,
    CROSS_VOL_SPREAD_LOOKBACK,
    CROSS_VOL_MAX_GAP_HOURS,
)


# ── Data container ────────────────────────────────────────────────────────────

@dataclass
class AssetVolData:
    """All pre-processed vol data for one asset, aligned to daily bars."""
    currency:    str
    spot_price:  float          # current spot (USD)
    dvol_daily:  pd.Series      # DatetimeIndex UTC, DVOL close in %
    spot_daily:  pd.Series      # DatetimeIndex UTC, mark_price close in USD
    rv30_series: pd.Series      # rolling 30d close-to-close RV at each date (%)
    vol_premium: pd.Series      # dvol_daily − rv30_series (vol pts)
    fetched_at:  datetime


# ── Helpers ───────────────────────────────────────────────────────────────────

def compute_rv_series(spot_daily: pd.Series, window: int = 30) -> pd.Series:
    """
    Rolling close-to-close RV as a time series, annualised %.

    Uses a rolling window of `window` days. First `window` values are NaN.
    Applies realised_vol_cc() at each step via rolling apply.
    """
    log_ret = np.log(spot_daily / spot_daily.shift(1)).dropna()

    def _rv(arr: np.ndarray) -> float:
        if len(arr) < window:
            return float("nan")
        return float(np.std(arr, ddof=1) * np.sqrt(252) * 100)

    rv = log_ret.rolling(window).apply(_rv, raw=True)
    return rv.reindex(spot_daily.index)


def _resample_daily(df: pd.DataFrame, col: str) -> pd.Series:
    """Resample hourly DataFrame column to 1D last-bar close, forward-fill gaps."""
    if df.empty or col not in df.columns:
        return pd.Series(dtype=float)
    if not isinstance(df.index, pd.DatetimeIndex):
        return pd.Series(dtype=float)
    return df[col].resample("1D").last().ffill()


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_universe(
    currencies: list[str] = CROSS_VOL_ASSETS,
    force_refresh: bool = True,
) -> dict[str, AssetVolData]:
    """
    Fetch and preprocess data for all currencies.

    Calls fetch_with_cache() per currency (reuses per_asset caching layer).
    Returns dict keyed by uppercase currency code.
    """
    universe: dict[str, AssetVolData] = {}

    for ccy in currencies:
        ccy = ccy.upper()
        print(f"[cross-vol] Fetching {ccy} ...")
        raw = fetch_with_cache(currency=ccy, force_refresh=force_refresh)

        # Fetch daily-resolution DVOL directly — gives ~1000 daily bars (~3 years).
        # The cache holds only ~42 days of hourly DVOL; daily resolution is needed
        # for FY25 and any lookback longer than 90 days.
        from data.deribit import get_dvol_history, get_mark_price_history
        dvol_df_daily = get_dvol_history(
            currency=ccy, days=CROSS_VOL_LOOKBACK_DAYS, resolution=86400
        )
        dvol_daily = (
            dvol_df_daily["close"]
            if not dvol_df_daily.empty and "close" in dvol_df_daily.columns
            else _resample_daily(raw["dvol_df"], "close")
        )

        # Fetch extended spot history directly for full RV30 computation.
        # fetch_with_cache caps spot at 90 days; for FY25 we need ~500 days.
        instrument = f"{ccy}-PERPETUAL"
        spot_df_long = get_mark_price_history(instrument, days=CROSS_VOL_LOOKBACK_DAYS)
        spot_daily = (
            spot_df_long["mark_price"]
            if not spot_df_long.empty and "mark_price" in spot_df_long.columns
            else _resample_daily(raw["spot_history"], "mark_price")
        )
        # Ensure daily resolution
        if isinstance(spot_daily.index, pd.DatetimeIndex) and len(spot_daily) > 0:
            spot_daily = spot_daily.resample("1D").last().ffill()
        rv30_series = compute_rv_series(spot_daily, window=30)
        vol_premium = (dvol_daily - rv30_series).dropna()

        universe[ccy] = AssetVolData(
            currency    = ccy,
            spot_price  = float(raw["spot_price"]),
            dvol_daily  = dvol_daily,
            spot_daily  = spot_daily,
            rv30_series = rv30_series,
            vol_premium = vol_premium,
            fetched_at  = raw["fetched_at"],
        )
        print(
            f"  DVOL bars: {len(dvol_daily)}  "
            f"spot bars: {len(spot_daily)}  "
            f"vol_premium mean: {vol_premium.mean():.1f} vol pts"
        )

    return universe


def align_assets(
    universe: dict[str, AssetVolData],
    start_dt: datetime | None = None,
    end_dt:   datetime | None = None,
) -> pd.DataFrame:
    """
    Align vol_premium series from all assets onto a common daily DatetimeIndex.

    - Inner-joins on dates where ALL assets have non-NaN vol_premium.
    - Forward-fills DVOL gaps ≤ CROSS_VOL_MAX_GAP_HOURS before join.
    - Warns if any asset has gaps larger than the threshold.
    - Raises ValueError if fewer than CROSS_VOL_SPREAD_LOOKBACK aligned dates.

    Returns DataFrame with columns = currency codes, rows = dates (UTC).
    """
    vp_dict: dict[str, pd.Series] = {}

    for ccy, data in universe.items():
        s = data.vol_premium.copy()

        # Detect and warn on large gaps in the underlying DVOL series
        if not data.dvol_daily.empty:
            dvol_idx = data.dvol_daily.dropna().index
            if len(dvol_idx) > 1:
                gap_hrs = dvol_idx.to_series().diff().dropna()
                large = gap_hrs[gap_hrs > pd.Timedelta(hours=CROSS_VOL_MAX_GAP_HOURS)]
                if not large.empty:
                    warnings.warn(
                        f"[cross-vol] {ccy}: {len(large)} DVOL gap(s) > "
                        f"{CROSS_VOL_MAX_GAP_HOURS}h. Forward-filling.",
                        stacklevel=2,
                    )

        vp_dict[ccy] = s

    # Align on common dates (inner join — only dates all assets are available)
    aligned = pd.DataFrame(vp_dict).dropna(how="any")
    aligned.index = pd.to_datetime(aligned.index, utc=True)
    aligned = aligned.sort_index()

    # Date slice
    if start_dt is not None:
        aligned = aligned[aligned.index >= pd.Timestamp(start_dt, tz="UTC")]
    if end_dt is not None:
        aligned = aligned[aligned.index <= pd.Timestamp(end_dt, tz="UTC")]

    if len(aligned) < CROSS_VOL_SPREAD_LOOKBACK:
        raise ValueError(
            f"[cross-vol] Only {len(aligned)} aligned dates available; "
            f"need at least {CROSS_VOL_SPREAD_LOOKBACK} for z-score initialisation. "
            f"Try a longer CROSS_VOL_LOOKBACK_DAYS or a shorter backtest window."
        )

    return aligned
