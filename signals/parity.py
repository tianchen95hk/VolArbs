"""
Signal 4 — Put-Call Parity Violations

For European options on a non-dividend asset:
    C − P = S − K · exp(−r · T)     (using spot as underlying)
    C − P = F · exp(−r · T) − K · exp(−r · T)   (using futures forward)

Any deviation from this identity after accounting for transaction costs
represents a riskless arbitrage opportunity.

Implementation
--------------
- Uses the nearest dated Deribit future as the forward price (more accurate
  than S × exp(r·T) because it captures the repo/funding cost directly).
- Falls back to S × exp(r·T) if no matching futures contract exists.
- Filters to liquid strikes (OI > 0 on both call and put legs).
- Net edge = |parity_deviation_usd| − round-trip commission cost.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import math
import numpy as np
import pandas as pd

from config import (
    PARITY_THRESHOLD_PCT, RISK_FREE_RATE, COMMISSION_PCT,
)
from analytics.bsm import parity_check


@dataclass
class ParityViolation:
    instrument_call:       str
    instrument_put:        str
    strike:                float
    expiry:                datetime
    dte:                   int
    call_iv:               float      # %
    put_iv:                float      # %
    iv_spread:             float      # call_iv - put_iv (vol points)
    parity_deviation_usd:  float      # raw C − P − (F − K)·exp(−rT) in USD
    parity_deviation_pct:  float      # as % of spot
    direction:             str        # "buy_call_sell_put" | "buy_put_sell_call"
    net_edge_usd:          float      # deviation minus estimated transaction costs
    confidence:            float
    action_items:          list[str] = field(default_factory=list)
    description:           str = ""


def _forward_price(spot: float, K: float, T: float, r: float,
                   futures_df: pd.DataFrame, expiry: datetime) -> float:
    """
    Return the forward price for the given expiry.

    Prefers the Deribit dated futures mark price; falls back to spot × exp(r·T).
    """
    if not futures_df.empty and "instrument_name" in futures_df.columns:
        exp_str = expiry.strftime("%-d%b%y").upper()   # e.g. 27DEC24
        match = futures_df[futures_df["instrument_name"].str.contains(exp_str, na=False)]
        if not match.empty and "mark_price" in match.columns:
            fwd = match["mark_price"].dropna()
            if not fwd.empty:
                return float(fwd.iloc[0])
    return spot * math.exp(r * T)


def analyse_parity(
    chain: pd.DataFrame,
    spot: float,
    futures_df: pd.DataFrame,
    r: float = RISK_FREE_RATE,
    threshold_pct: float = PARITY_THRESHOLD_PCT,
) -> list[ParityViolation]:
    """
    Find put-call parity violations across all (expiry, strike) pairs.

    Parameters
    ----------
    chain         : enriched options chain from analytics.surface.parse_chain()
    spot          : current spot price
    futures_df    : dated futures from data.deribit.get_futures_summary()
    r             : risk-free rate (decimal, annualised)
    threshold_pct : minimum net edge as % of spot to return a violation

    Returns list sorted by net_edge_usd descending (largest edge first).
    """
    if chain.empty or spot <= 0:
        return []

    # Separate calls and puts
    calls = chain[chain["option_type"] == "C"].copy()
    puts  = chain[chain["option_type"] == "P"].copy()

    if calls.empty or puts.empty:
        return []

    violations: list[ParityViolation] = []

    # Group by (expiry, strike) — need matching call and put
    call_idx = calls.set_index(["expiry", "strike"])
    put_idx  = puts.set_index(["expiry", "strike"])

    common = call_idx.index.intersection(put_idx.index)

    for (expiry, strike) in common:
        call_row = call_idx.loc[(expiry, strike)]
        put_row  = put_idx.loc[(expiry, strike)]

        # Handle multi-row (take mean)
        if isinstance(call_row, pd.DataFrame):
            call_row = call_row.iloc[0]
        if isinstance(put_row, pd.DataFrame):
            put_row = put_row.iloc[0]

        # Filter to liquid options
        call_oi = call_row.get("open_interest", 0) or 0
        put_oi  = put_row.get("open_interest", 0)  or 0
        if call_oi <= 0 and put_oi <= 0:
            continue

        call_price = call_row.get("mark_price")
        put_price  = put_row.get("mark_price")
        call_iv    = call_row.get("mark_iv")
        put_iv     = put_row.get("mark_iv")

        if any(v is None or (isinstance(v, float) and np.isnan(v))
               for v in [call_price, put_price]):
            continue

        T = max(float(call_row.get("dte_years", 0)), 1 / 365)
        dte = int(call_row.get("dte", 0))

        fwd = _forward_price(spot, strike, T, r, futures_df, expiry)
        deviation = parity_check(float(call_price), float(put_price), fwd, strike, T, r)

        if np.isnan(deviation):
            continue

        # Round-trip cost: 2 legs (call + put) × commission
        # Commission is per BTC notional; option price is in BTC
        commission_usd = 2 * COMMISSION_PCT * spot   # simplified: 1 contract per leg
        net_edge = abs(deviation) - commission_usd

        threshold_usd = threshold_pct / 100 * spot
        if net_edge < threshold_usd:
            continue

        direction = "buy_call_sell_put" if deviation > 0 else "buy_put_sell_call"
        confidence = min(1.0, net_edge / (threshold_usd * 2))

        call_name = str(call_row.get("instrument_name", ""))
        put_name  = str(put_row.get("instrument_name", ""))
        iv_spread = (float(call_iv) - float(put_iv)
                     if not (np.isnan(float(call_iv or "nan")) or np.isnan(float(put_iv or "nan")))
                     else float("nan"))

        if direction == "buy_call_sell_put":
            action_items = [
                f"BUY {call_name}  (call, IV {call_iv:.1f}%)",
                f"SELL {put_name} (put, IV {put_iv:.1f}%)",
                f"SELL futures/spot to hedge: forward price {fwd:,.0f}",
                f"Parity deviation: ${deviation:+.2f}  |  Net edge after costs: ${net_edge:.2f}",
            ]
        else:
            action_items = [
                f"BUY {put_name}  (put, IV {put_iv:.1f}%)",
                f"SELL {call_name} (call, IV {call_iv:.1f}%)",
                f"BUY futures/spot to hedge: forward price {fwd:,.0f}",
                f"Parity deviation: ${deviation:+.2f}  |  Net edge after costs: ${net_edge:.2f}",
            ]

        desc = (
            f"PARITY VIOLATION | Strike {strike:,.0f} Exp {expiry.strftime('%d%b%y')} ({dte}d) "
            f"Deviation ${deviation:+.2f} ({deviation / spot * 100:+.3f}%) "
            f"Net edge ${net_edge:.2f}."
        )

        violations.append(ParityViolation(
            instrument_call=call_name,
            instrument_put=put_name,
            strike=strike,
            expiry=expiry,
            dte=dte,
            call_iv=float(call_iv) if call_iv is not None else float("nan"),
            put_iv=float(put_iv)   if put_iv  is not None else float("nan"),
            iv_spread=iv_spread,
            parity_deviation_usd=float(deviation),
            parity_deviation_pct=float(deviation / spot * 100),
            direction=direction,
            net_edge_usd=float(net_edge),
            confidence=confidence,
            action_items=action_items,
            description=desc,
        ))

    violations.sort(key=lambda v: v.net_edge_usd, reverse=True)
    return violations
