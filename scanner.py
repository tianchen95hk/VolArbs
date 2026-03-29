"""
Opportunity scanner — combines all four signals into a ranked output.

Scoring
-------
  composite_score = normalised_edge × confidence

  Normalisation per signal (heuristic, same approach as dex_deribit/scanner.py):
    vol_premium    : |vol_premium|   / VOL_PREMIUM_THRESHOLD   × confidence
    skew           : |skew_25d|      / SKEW_THRESHOLD          × confidence
    term_structure : |term_slope|    / TERM_SLOPE_THRESHOLD    × confidence
    parity         : net_edge_usd    / (spot × threshold_pct)  × confidence
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from config import (
    VOL_PREMIUM_THRESHOLD, SKEW_THRESHOLD,
    TERM_SLOPE_THRESHOLD, PARITY_THRESHOLD_PCT,
)
from signals.vol_premium    import VolPremiumSignal
from signals.skew           import SkewSignal
from signals.term_structure import TermStructureSignal
from signals.parity         import ParityViolation


@dataclass
class Opportunity:
    rank:            int
    signal_type:     str      # "vol_premium" | "skew" | "term_structure" | "parity"
    direction:       str
    composite_score: float
    confidence:      float
    description:     str
    action_items:    list[str] = field(default_factory=list)


def _score_vol_premium(sig: VolPremiumSignal) -> float:
    return abs(sig.vol_premium) / VOL_PREMIUM_THRESHOLD * sig.confidence

def _score_skew(sig: SkewSignal) -> float:
    return abs(sig.skew_25d) / SKEW_THRESHOLD * sig.confidence

def _score_term(sig: TermStructureSignal) -> float:
    return abs(sig.term_slope) / TERM_SLOPE_THRESHOLD * sig.confidence

def _score_parity(sig: ParityViolation, spot: float) -> float:
    baseline = spot * PARITY_THRESHOLD_PCT / 100
    if baseline <= 0:
        return 0.0
    return sig.net_edge_usd / baseline * sig.confidence


def rank_opportunities(
    vol_premium:    Optional[VolPremiumSignal]   = None,
    skew:           list[SkewSignal]             | None = None,
    term_structure: Optional[TermStructureSignal] = None,
    parity:         list[ParityViolation]        | None = None,
    spot:           float = 0.0,
) -> list[Opportunity]:
    """
    Combine all signals into a ranked list of Opportunity objects.

    Returns list sorted by composite_score descending.
    """
    opps: list[Opportunity] = []

    if vol_premium is not None and vol_premium.direction != "neutral":
        opps.append(Opportunity(
            rank=0,
            signal_type="vol_premium",
            direction=vol_premium.direction,
            composite_score=_score_vol_premium(vol_premium),
            confidence=vol_premium.confidence,
            description=vol_premium.description,
            action_items=vol_premium.action_items,
        ))

    for sig in (skew or [])[:3]:   # top 3 skew signals by expiry
        opps.append(Opportunity(
            rank=0,
            signal_type="skew",
            direction=sig.direction,
            composite_score=_score_skew(sig),
            confidence=sig.confidence,
            description=sig.description,
            action_items=sig.action_items,
        ))

    if term_structure is not None and term_structure.direction != "neutral":
        opps.append(Opportunity(
            rank=0,
            signal_type="term_structure",
            direction=term_structure.direction,
            composite_score=_score_term(term_structure),
            confidence=term_structure.confidence,
            description=term_structure.description,
            action_items=term_structure.action_items,
        ))

    for sig in (parity or [])[:3]:
        opps.append(Opportunity(
            rank=0,
            signal_type="parity",
            direction=sig.direction,
            composite_score=_score_parity(sig, spot),
            confidence=sig.confidence,
            description=sig.description,
            action_items=sig.action_items,
        ))

    opps.sort(key=lambda o: o.composite_score, reverse=True)
    for i, opp in enumerate(opps):
        opp.rank = i + 1

    return opps


# ── Pretty printer ─────────────────────────────────────────────────────────────

_SIGNAL_ICONS = {
    "vol_premium":    "σ",
    "skew":           "⟋",
    "term_structure": "⟿",
    "parity":         "=",
}


def _bar(value: float, width: int = 10) -> str:
    filled = min(width, max(0, int(value * width)))
    return "█" * filled + "░" * (width - filled)


def print_opportunity(opp: Opportunity) -> None:
    icon  = _SIGNAL_ICONS.get(opp.signal_type, "◆")
    label = opp.signal_type.replace("_", " ").upper()
    dir_  = opp.direction.replace("_", " ").upper()
    print(f"\n  #{opp.rank}  {icon}  {label}  ·  {dir_}")
    print(f"       Confidence  {_bar(opp.confidence)}  {opp.confidence:.0%}")
    print(f"       Score       {opp.composite_score:.4f}")
    print()
    print(f"  {opp.description}")
    print()
    print("  Action items:")
    for item in opp.action_items:
        print(f"    →  {item}")
    print("  " + "─" * 68)
