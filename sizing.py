"""
cross_vol/sizing.py — Vega-neutral sizing for cross-asset vol spread.

Core formula (derived from two constraints):
    Let v_A, v_B = straddle vega per USD notional for each asset
        N_total  = total capital allocated to this pair

    Constraint 1 (vega neutral): N_A × v_A = N_B × v_B
    Constraint 2 (budget):       N_A + N_B = N_total

    Solution:
        N_long  = N_total × v_short / (v_long + v_short)
        N_short = N_total × v_long  / (v_long + v_short)

The asset with the higher vega per notional receives proportionally less
capital, so both legs contribute equally in dollar-vega terms.
"""
from __future__ import annotations

import sys
import os
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

_PER_ASSET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "per_asset")
if _PER_ASSET not in sys.path:
    sys.path.insert(0, _PER_ASSET)

from analytics.bsm import bsm_price, bsm_greeks
from cv_config import (
    INITIAL_CAPITAL,
    RISK_FREE_RATE,
    CROSS_VOL_CAPITAL_FRACTION,
    CROSS_VOL_DTE_TARGET,
    CROSS_VOL_MAX_GROSS_VEGA,
    CROSS_VOL_MAX_CONCENTRATION,
)
from spread_signal import CrossVolSignal
from universe import AssetVolData


# ── Data containers ───────────────────────────────────────────────────────────

@dataclass
class LegSizing:
    """Sizing for one leg of the cross-asset vol spread."""
    currency:           str
    direction:          int      # +1 = long vol, -1 = short vol
    spot:               float
    atm_iv_pct:         float    # ATM IV in % (from DVOL proxy)
    dte:                int
    straddle_price_usd: float    # BSM ATM straddle price per unit of spot
    straddle_vega:      float    # dollar vega per 1% vol move, per unit notional
    notional_usd:       float    # capital allocated to this leg


@dataclass
class PairSizing:
    """Vega-neutral sizing for the full cross-asset pair."""
    long_leg:            LegSizing
    short_leg:           LegSizing
    total_notional:      float
    gross_vega_usd:      float   # |long vega| + |short vega| in USD
    net_vega_usd:        float   # target ≈ 0 after sizing
    vega_neutrality_pct: float   # |net_vega| / gross_vega × 100 (quality metric)
    approved:            bool
    rejection_reason:    str | None


# ── BSM helpers ───────────────────────────────────────────────────────────────

def _atm_straddle(
    spot: float,
    iv_pct: float,
    dte: int,
    r: float = RISK_FREE_RATE,
) -> tuple[float, float]:
    """
    Price and dollar-vega of an ATM straddle (call + put at K = spot).

    Returns (straddle_price_usd, straddle_vega_per_notional)
      straddle_price_usd         — total premium for the straddle in USD
      straddle_vega_per_notional — vega per USD of notional (dimensionless ratio)
    """
    if dte <= 0 or iv_pct <= 0 or spot <= 0:
        return 0.0, 0.0

    T      = dte / 365.0
    sigma  = iv_pct / 100.0
    K      = spot   # ATM: strike = spot

    call_price = bsm_price(spot, K, T, r, sigma, "C")
    put_price  = bsm_price(spot, K, T, r, sigma, "P")
    straddle   = call_price + put_price

    # Vega of ATM call == ATM put by BSM symmetry; straddle vega = 2×
    greeks     = bsm_greeks(spot, K, T, r, sigma, "C")
    # bsm_greeks returns vega per 1% vol move in USD (for one unit of underlying)
    straddle_vega_usd = 2.0 * greeks.get("vega", 0.0)

    # Per-notional vega: how much dollar-vega per $1 of straddle premium
    straddle_vega_per_notional = straddle_vega_usd / straddle if straddle > 0 else 0.0

    return straddle, straddle_vega_per_notional


# ── Public API ────────────────────────────────────────────────────────────────

def compute_vega_neutral_sizing(
    signal: CrossVolSignal,
    universe: dict[str, AssetVolData],
    total_capital_fraction: float = CROSS_VOL_CAPITAL_FRACTION,
    initial_capital: float = INITIAL_CAPITAL,
    dte: int = CROSS_VOL_DTE_TARGET,
) -> PairSizing:
    """
    Size each leg so that dollar-vega is equal across legs (vega-neutral).

    Uses the current DVOL value as ATM IV proxy for straddle pricing.
    """
    N_total = initial_capital * total_capital_fraction

    long_data  = universe[signal.asset_long]
    short_data = universe[signal.asset_short]

    # Use latest available DVOL close as ATM IV proxy
    iv_long  = float(long_data.dvol_daily.dropna().iloc[-1])
    iv_short = float(short_data.dvol_daily.dropna().iloc[-1])

    straddle_long,  v_long  = _atm_straddle(long_data.spot_price,  iv_long,  dte)
    straddle_short, v_short = _atm_straddle(short_data.spot_price, iv_short, dte)

    # Vega-neutral allocation:
    # N_long  = N_total × v_short / (v_long + v_short)
    # N_short = N_total × v_long  / (v_long + v_short)
    v_sum = v_long + v_short
    if v_sum <= 0:
        n_long = n_short = N_total / 2
    else:
        n_long  = N_total * v_short / v_sum
        n_short = N_total * v_long  / v_sum

    dollar_vega_long  = n_long  * v_long
    dollar_vega_short = n_short * v_short
    gross_vega        = dollar_vega_long + dollar_vega_short
    net_vega          = dollar_vega_long - dollar_vega_short
    neutrality_pct    = abs(net_vega) / gross_vega * 100 if gross_vega > 0 else 0.0

    long_leg = LegSizing(
        currency           = signal.asset_long,
        direction          = +1,
        spot               = long_data.spot_price,
        atm_iv_pct         = iv_long,
        dte                = dte,
        straddle_price_usd = straddle_long,
        straddle_vega      = v_long,
        notional_usd       = n_long,
    )
    short_leg = LegSizing(
        currency           = signal.asset_short,
        direction          = -1,
        spot               = short_data.spot_price,
        atm_iv_pct         = iv_short,
        dte                = dte,
        straddle_price_usd = straddle_short,
        straddle_vega      = v_short,
        notional_usd       = n_short,
    )

    sizing = PairSizing(
        long_leg            = long_leg,
        short_leg           = short_leg,
        total_notional      = N_total,
        gross_vega_usd      = gross_vega,
        net_vega_usd        = net_vega,
        vega_neutrality_pct = neutrality_pct,
        approved            = True,
        rejection_reason    = None,
    )

    approved, reason = check_risk_limits(sizing, initial_capital)
    sizing.approved         = approved
    sizing.rejection_reason = reason
    return sizing


def check_risk_limits(
    sizing: PairSizing,
    current_capital: float,
) -> tuple[bool, str | None]:
    """
    Validate PairSizing against gross vega and concentration limits.

    Returns (approved, rejection_reason).
    """
    gross_vega_ratio = sizing.gross_vega_usd / current_capital if current_capital > 0 else 0.0
    if gross_vega_ratio > CROSS_VOL_MAX_GROSS_VEGA:
        return False, (
            f"Gross vega {gross_vega_ratio:.1%} exceeds limit "
            f"{CROSS_VOL_MAX_GROSS_VEGA:.0%}"
        )

    if sizing.gross_vega_usd > 0:
        long_share  = (sizing.long_leg.notional_usd  * sizing.long_leg.straddle_vega)  / sizing.gross_vega_usd
        short_share = (sizing.short_leg.notional_usd * sizing.short_leg.straddle_vega) / sizing.gross_vega_usd
        max_share   = max(long_share, short_share)
        if max_share > CROSS_VOL_MAX_CONCENTRATION:
            return False, (
                f"One leg concentration {max_share:.1%} exceeds limit "
                f"{CROSS_VOL_MAX_CONCENTRATION:.0%}"
            )

    return True, None
