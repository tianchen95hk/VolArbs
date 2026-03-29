"""
Signal 2 — Skew Arbitrage (25-delta put/call IV imbalance)

Compares 25-delta put IV against 25-delta call IV for each near-term expiry.

  Rich puts  (skew > threshold) → risk reversal: sell OTM put, buy OTM call
  Rich calls (skew < -threshold) → reverse risk reversal: sell OTM call, buy OTM put

In crypto, a positive skew (puts > calls) is normal due to downside protection
demand. A signal fires when the skew is abnormally high or flips negative.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from config import SKEW_THRESHOLD, NEAR_DTE_MIN, NEAR_DTE_MAX, DELTA_25D_BAND


@dataclass
class SkewSignal:
    direction:       str       # "rich_puts" | "rich_calls" | "neutral"
    expiry:          datetime
    dte:             int
    put_iv_25d:      float     # 25-delta put IV (%)
    call_iv_25d:     float     # 25-delta call IV (%)
    skew_25d:        float     # put_iv_25d - call_iv_25d (vol points)
    atm_iv:          float     # ATM IV for context (%)
    confidence:      float
    put_instrument:  str = ""  # suggested put leg
    call_instrument: str = ""  # suggested call leg
    action_items:    list[str] = field(default_factory=list)
    description:     str = ""


def analyse_skew(
    chain: pd.DataFrame,
    spot: float,
    threshold: float = SKEW_THRESHOLD,
) -> list[SkewSignal]:
    """
    Compute 25-delta skew for each near-term expiry and return signals.

    Parameters
    ----------
    chain     : enriched options chain from analytics.surface.parse_chain()
    spot      : current spot price
    threshold : |skew| in vol points required to generate a signal

    Returns list sorted by |skew| descending (largest edge first).
    Returns empty list if no expiries breach threshold.
    """
    if chain.empty or spot <= 0:
        return []

    near = chain[(chain["dte"] >= NEAR_DTE_MIN) & (chain["dte"] <= NEAR_DTE_MAX)]
    if near.empty:
        return []

    has_delta = "delta" in near.columns and near["delta"].notna().any()
    lo, hi = DELTA_25D_BAND
    signals: list[SkewSignal] = []

    for expiry, grp in near.groupby("expiry"):
        # 25-delta options selection
        if has_delta:
            puts_25  = grp[(grp["option_type"] == "P") & grp["delta"].between(-hi, -lo)]
            calls_25 = grp[(grp["option_type"] == "C") & grp["delta"].between(lo, hi)]
        else:
            puts_25  = grp[(grp["option_type"] == "P") & grp["moneyness_pct"].between(-15, -5)]
            calls_25 = grp[(grp["option_type"] == "C") & grp["moneyness_pct"].between(5, 15)]

        if puts_25.empty or calls_25.empty:
            continue

        put_iv  = float(puts_25["mark_iv"].dropna().mean())
        call_iv = float(calls_25["mark_iv"].dropna().mean())

        if np.isnan(put_iv) or np.isnan(call_iv):
            continue

        skew = put_iv - call_iv

        if abs(skew) < threshold:
            continue

        # ATM IV
        atm = grp[grp["moneyness_pct"].abs() < 5.0]
        atm_iv = float(atm["mark_iv"].dropna().mean()) if not atm.empty else float("nan")

        dte = int(grp["dte"].iloc[0])

        # Best instruments by open interest
        put_instr  = (puts_25.sort_values("open_interest", ascending=False)
                      ["instrument_name"].dropna().iloc[0]
                      if not puts_25.empty else "")
        call_instr = (calls_25.sort_values("open_interest", ascending=False)
                      ["instrument_name"].dropna().iloc[0]
                      if not calls_25.empty else "")

        confidence = min(1.0, abs(skew) / (threshold * 2))

        if skew > threshold:
            direction = "rich_puts"
            action_items = [
                f"RISK REVERSAL: Sell {put_instr} (put), Buy {call_instr} (call)",
                f"Put 25d IV {put_iv:.1f}% vs Call 25d IV {call_iv:.1f}% → skew {skew:+.1f} pts",
                f"Put IV premium unusually high → expect mean reversion",
                f"Net position: short downside, long upside",
                f"Delta-hedge the combined position to isolate skew exposure",
            ]
            desc = (
                f"RICH PUTS | Expiry {expiry.strftime('%d%b%y')} ({dte}d) "
                f"Put25d {put_iv:.1f}% vs Call25d {call_iv:.1f}% "
                f"(skew {skew:+.1f} pts). ATM IV {atm_iv:.1f}%."
            )
        else:
            direction = "rich_calls"
            action_items = [
                f"REVERSE RISK REVERSAL: Sell {call_instr} (call), Buy {put_instr} (put)",
                f"Call 25d IV {call_iv:.1f}% vs Put 25d IV {put_iv:.1f}% → skew {skew:.1f} pts",
                f"Call IV premium unusually high → call overwriting opportunity",
                f"Net position: short upside, long downside",
                f"Delta-hedge the combined position to isolate skew exposure",
            ]
            desc = (
                f"RICH CALLS | Expiry {expiry.strftime('%d%b%y')} ({dte}d) "
                f"Call25d {call_iv:.1f}% vs Put25d {put_iv:.1f}% "
                f"(skew {skew:.1f} pts). ATM IV {atm_iv:.1f}%."
            )

        signals.append(SkewSignal(
            direction=direction,
            expiry=expiry,
            dte=dte,
            put_iv_25d=put_iv,
            call_iv_25d=call_iv,
            skew_25d=skew,
            atm_iv=atm_iv,
            confidence=confidence,
            put_instrument=put_instr,
            call_instrument=call_instr,
            action_items=action_items,
            description=desc,
        ))

    signals.sort(key=lambda s: abs(s.skew_25d), reverse=True)
    return signals
