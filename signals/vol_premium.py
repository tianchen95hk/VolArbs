"""
Signal 1 — Vol Premium (IV vs Realised Vol / Vol Carry)

Compares near-term ATM implied volatility against realized volatility
computed from Deribit's BTC-PERPETUAL mark price history.

  IV >> RV  →  sell options (overpriced relative to actual moves)
  IV << RV  →  buy options  (cheap relative to actual moves)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from config import (
    VOL_PREMIUM_THRESHOLD, ATM_MONEYNESS_PCT,
    NEAR_DTE_MIN, NEAR_DTE_MAX,
)
from analytics.rv import realised_vol_cc, realised_vol_ewma
from analytics.surface import parse_chain


@dataclass
class VolPremiumSignal:
    direction:       str              # "short_vol" | "long_vol" | "neutral"
    atm_iv:          float            # near-term ATM IV (%)
    rv_30d:          float            # 30-day close-to-close RV (%)
    rv_ewma:         float            # EWMA RV as confirmation (%)
    vol_premium:     float            # atm_iv - rv_30d (vol points)
    dvol:            float            # DVOL index close (%) for context
    confidence:      float
    top_instruments: list[str] = field(default_factory=list)
    action_items:    list[str] = field(default_factory=list)
    description:     str = ""


def analyse_vol_premium(
    chain: pd.DataFrame,
    spot_history: pd.DataFrame,
    dvol_df: pd.DataFrame,
    threshold: float = VOL_PREMIUM_THRESHOLD,
) -> Optional[VolPremiumSignal]:
    """
    Compare near-term ATM IV against realized vol.

    Parameters
    ----------
    chain        : enriched options chain from analytics.surface.parse_chain()
    spot_history : hourly BTC-PERPETUAL mark price history (column: mark_price)
    dvol_df      : hourly DVOL index (column: close)
    threshold    : vol-point difference to trigger a signal

    Returns None if inputs are insufficient.
    """
    if chain.empty or spot_history.empty:
        return None

    # DVOL context
    dvol = float(dvol_df["close"].iloc[-1]) if not dvol_df.empty else float("nan")

    # Realized vol from spot price history
    prices = spot_history["mark_price"].dropna()
    rv_30d  = realised_vol_cc(prices, window=30)
    rv_ewma = realised_vol_ewma(prices)

    if np.isnan(rv_30d):
        return None

    # Near-term ATM IV
    near = chain[(chain["dte"] >= NEAR_DTE_MIN) & (chain["dte"] <= NEAR_DTE_MAX)]
    if near.empty:
        near = chain[chain["dte"] >= NEAR_DTE_MIN].nsmallest(30, "dte")
    if near.empty:
        return None

    atm = near[near["moneyness_pct"].abs() < ATM_MONEYNESS_PCT]
    if atm.empty:
        atm = near.nsmallest(5, "dte")
    atm_iv = float(atm["mark_iv"].dropna().mean()) if not atm.empty else dvol
    if np.isnan(atm_iv):
        atm_iv = dvol

    vol_premium = atm_iv - rv_30d

    # Top instruments by open interest
    top_instr = (
        near.sort_values("open_interest", ascending=False)["instrument_name"]
        .dropna().head(5).tolist()
    )

    # Direction and confidence
    if vol_premium >= threshold:
        direction  = "short_vol"
        confidence = min(1.0, vol_premium / (threshold * 2.5))
        action_items = [
            f"SELL near-term ATM straddle on Deribit (7–21 DTE)",
            f"IV {atm_iv:.1f}% vs RV30d {rv_30d:.1f}% → premium {vol_premium:+.1f} vol pts",
            f"EWMA RV {rv_ewma:.1f}% confirms: {'yes' if rv_ewma < atm_iv else 'no (divergence)'}",
            f"Delta-hedge daily to maintain vega exposure",
            f"Target: {', '.join(top_instr[:3])}",
            f"Exit when IV − RV falls below {threshold / 2:.0f} pts. DVOL: {dvol:.1f}%",
        ]
        desc = (
            f"SHORT VOL | ATM IV {atm_iv:.1f}% vs RV30d {rv_30d:.1f}% "
            f"(premium {vol_premium:+.1f} pts). DVOL: {dvol:.1f}%."
        )
    elif vol_premium <= -threshold:
        direction  = "long_vol"
        confidence = min(1.0, abs(vol_premium) / (threshold * 2.5))
        action_items = [
            f"BUY near-term ATM straddle on Deribit (14–30 DTE)",
            f"IV {atm_iv:.1f}% vs RV30d {rv_30d:.1f}% → discount {vol_premium:.1f} vol pts",
            f"EWMA RV {rv_ewma:.1f}% confirms: {'yes' if rv_ewma > atm_iv else 'no (divergence)'}",
            f"Delta-hedge to isolate gamma / vega",
            f"Target: {', '.join(top_instr[:3])}",
            f"Exit when RV − IV falls below {threshold / 2:.0f} pts. DVOL: {dvol:.1f}%",
        ]
        desc = (
            f"LONG VOL | ATM IV {atm_iv:.1f}% vs RV30d {rv_30d:.1f}% "
            f"(discount {vol_premium:.1f} pts). DVOL: {dvol:.1f}%."
        )
    else:
        direction  = "neutral"
        confidence = 0.0
        action_items = [
            f"No actionable vol edge. ATM IV {atm_iv:.1f}% vs RV30d {rv_30d:.1f}%.",
            f"Monitor for threshold breach (>{threshold:.0f} pts premium or discount).",
        ]
        desc = (
            f"VOL NEUTRAL | ATM IV {atm_iv:.1f}% vs RV30d {rv_30d:.1f}% "
            f"(premium {vol_premium:+.1f} pts). Below threshold."
        )

    return VolPremiumSignal(
        direction=direction,
        atm_iv=atm_iv,
        rv_30d=rv_30d,
        rv_ewma=rv_ewma,
        vol_premium=vol_premium,
        dvol=dvol,
        confidence=confidence,
        top_instruments=top_instr,
        action_items=action_items,
        description=desc,
    )
