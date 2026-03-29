"""
Signal 3 — Term Structure / Calendar Spread

Compares the IV term structure slope (back IV − front IV) against the
realized vol term structure slope (RV30d − RV7d).

  IV term slope >> RV term slope (steep contango):
      Sell front-month ATM straddle, buy back-month ATM straddle.
      Rationale: front vol is rich relative to its term-structure fair value.

  IV term slope << 0 (backwardation, front IV >> back IV):
      Buy front-month options (cheap relative to historical norm).
      Sell back-month as a hedge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from config import TERM_SLOPE_THRESHOLD, NEAR_DTE_MIN, FAR_DTE_MIN
from analytics.surface import term_structure_ivs
from analytics.rv import rv_term_structure


@dataclass
class TermStructureSignal:
    direction:       str       # "sell_front_buy_back" | "buy_front_sell_back" | "neutral"
    front_expiry:    datetime
    back_expiry:     datetime
    front_dte:       int
    back_dte:        int
    front_atm_iv:    float     # %
    back_atm_iv:     float     # %
    term_slope:      float     # back_atm_iv - front_atm_iv (vol points)
    rv_term_slope:   float     # rv_30d - rv_7d (RV term structure for comparison)
    confidence:      float
    action_items:    list[str] = field(default_factory=list)
    description:     str = ""


def analyse_term_structure(
    chain: pd.DataFrame,
    spot_history: pd.DataFrame,
    spot: float,
    threshold: float = TERM_SLOPE_THRESHOLD,
) -> Optional[TermStructureSignal]:
    """
    Compare IV term structure slope against RV term structure.

    Parameters
    ----------
    chain        : enriched options chain from analytics.surface.parse_chain()
    spot_history : hourly BTC-PERPETUAL mark price (column: mark_price)
    spot         : current spot price
    threshold    : excess slope (vol pts) required for a signal

    Returns None if there are fewer than two usable expiries.
    """
    if chain.empty or spot <= 0:
        return None

    ts = term_structure_ivs(chain, spot)
    if ts.empty or len(ts) < 2:
        return None

    # Front expiry: first with DTE >= NEAR_DTE_MIN
    front_rows = ts[ts["dte"] >= NEAR_DTE_MIN]
    if front_rows.empty:
        return None
    front = front_rows.iloc[0]

    # Back expiry: first with DTE >= FAR_DTE_MIN
    back_rows = ts[ts["dte"] >= FAR_DTE_MIN]
    if back_rows.empty:
        # Fall back to second available expiry
        if len(ts) < 2:
            return None
        back = ts.iloc[1]
    else:
        back = back_rows.iloc[0]

    # Avoid using same expiry for both legs
    if front["expiry"] == back["expiry"]:
        if len(ts) < 2:
            return None
        back = ts[ts["expiry"] != front["expiry"]].iloc[0]

    front_iv = float(front["atm_iv"])
    back_iv  = float(back["atm_iv"])
    term_slope = back_iv - front_iv

    # RV term structure
    prices = spot_history["mark_price"].dropna() if not spot_history.empty else pd.Series()
    rv_ts = rv_term_structure(prices)
    rv_7d  = rv_ts.get(7,  float("nan"))
    rv_30d = rv_ts.get(30, float("nan"))
    rv_term_slope = (rv_30d - rv_7d) if not (np.isnan(rv_7d) or np.isnan(rv_30d)) else 0.0

    # Excess slope: how much steeper is IV term structure vs RV term structure
    excess_slope = term_slope - rv_term_slope

    if excess_slope >= threshold:
        direction = "sell_front_buy_back"
        confidence = min(1.0, excess_slope / (threshold * 2.5))
        action_items = [
            f"CALENDAR SPREAD: Sell front-month ({int(front['dte'])}d), "
            f"Buy back-month ({int(back['dte'])}d) ATM straddles",
            f"Front IV {front_iv:.1f}% → back IV {back_iv:.1f}% (slope {term_slope:+.1f} pts)",
            f"RV slope ({rv_7d:.1f}% → {rv_30d:.1f}%) = {rv_term_slope:+.1f} pts → excess {excess_slope:+.1f} pts",
            f"Front vol overpriced relative to term structure fair value",
            f"Profit if front vol mean-reverts toward back vol",
        ]
        desc = (
            f"SELL FRONT BUY BACK | Front {int(front['dte'])}d IV {front_iv:.1f}% vs "
            f"Back {int(back['dte'])}d IV {back_iv:.1f}% "
            f"(slope {term_slope:+.1f} pts, excess vs RV {excess_slope:+.1f} pts)."
        )
    elif term_slope <= -threshold:
        direction = "buy_front_sell_back"
        confidence = min(1.0, abs(term_slope) / (threshold * 2.5))
        action_items = [
            f"REVERSE CALENDAR: Buy front-month ({int(front['dte'])}d), "
            f"Sell back-month ({int(back['dte'])}d) ATM straddles",
            f"Front IV {front_iv:.1f}% < Back IV {back_iv:.1f}% (backwardation {term_slope:.1f} pts)",
            f"RV slope {rv_term_slope:+.1f} pts → front vol unusually cheap",
            f"Profit if front vol mean-reverts higher",
        ]
        desc = (
            f"BUY FRONT SELL BACK | Backwardation: Front {int(front['dte'])}d IV {front_iv:.1f}% "
            f"vs Back {int(back['dte'])}d IV {back_iv:.1f}% "
            f"(slope {term_slope:.1f} pts)."
        )
    else:
        direction = "neutral"
        confidence = 0.0
        action_items = [
            f"No term structure edge. Slope {term_slope:+.1f} pts (threshold ±{threshold:.0f} pts).",
            f"Front {int(front['dte'])}d: {front_iv:.1f}%  |  Back {int(back['dte'])}d: {back_iv:.1f}%",
        ]
        desc = (
            f"TERM NEUTRAL | Front {int(front['dte'])}d {front_iv:.1f}% vs "
            f"Back {int(back['dte'])}d {back_iv:.1f}% (slope {term_slope:+.1f} pts). "
            f"Below threshold."
        )

    return TermStructureSignal(
        direction=direction,
        front_expiry=front["expiry"],
        back_expiry=back["expiry"],
        front_dte=int(front["dte"]),
        back_dte=int(back["dte"]),
        front_atm_iv=front_iv,
        back_atm_iv=back_iv,
        term_slope=term_slope,
        rv_term_slope=rv_term_slope,
        confidence=confidence,
        action_items=action_items,
        description=desc,
    )
