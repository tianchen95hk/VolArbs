"""
Option position tracker with Greek aggregation and risk controls.

Positions are tracked in USD terms.  Greeks are computed via BSM whenever
a position is marked to market.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from config import (
    INITIAL_CAPITAL, POSITION_SIZE_PCT, MAX_OPEN_POSITIONS,
    MAX_NET_DELTA, MAX_VEGA_NOTIONAL, MAX_GAMMA_NOTIONAL,
    STOP_LOSS_PCT, MAX_DRAWDOWN_HALT_PCT, RISK_FREE_RATE,
)
from analytics.bsm import bsm_price, bsm_greeks


@dataclass
class OptionPosition:
    instrument:      str
    option_type:     str        # "C" or "P"
    strike:          float
    expiry:          datetime
    dte_at_entry:    int
    direction:       int        # +1 = long, -1 = short
    notional_usd:    float      # USD capital allocated
    entry_iv:        float      # IV at entry (decimal, e.g. 0.60)
    entry_price:     float      # option price in USD at entry
    entry_spot:      float      # spot price at entry
    signal_type:     str        # which signal generated this

    # Filled in after open
    allocated_capital: float = 0.0   # same as notional_usd (set on open)
    peak_value:        float = 0.0   # for stop-loss tracking


@dataclass
class PortfolioGreeks:
    net_delta: float = 0.0
    net_gamma: float = 0.0
    net_vega:  float = 0.0
    net_theta: float = 0.0


class PositionTracker:
    """Tracks all open option positions and enforces risk limits."""

    def __init__(self, initial_capital: float = INITIAL_CAPITAL) -> None:
        self.capital = initial_capital
        self.peak_capital = initial_capital
        self.positions: dict[str, OptionPosition] = {}   # instrument → position
        self._trade_log: list[dict] = []

    # ── Greek helpers ──────────────────────────────────────────────────────────

    def _position_greeks(
        self, pos: OptionPosition, spot: float, current_iv: float | None = None
    ) -> dict[str, float]:
        T = max((pos.expiry - datetime.utcnow()).total_seconds() / (365 * 86400), 1 / 365)
        sigma = current_iv if current_iv is not None else pos.entry_iv
        g = bsm_greeks(spot, pos.strike, T, RISK_FREE_RATE, sigma, pos.option_type)
        return {k: v * pos.direction * pos.notional_usd / pos.entry_price
                if pos.entry_price > 0 else 0.0
                for k, v in g.items()}

    def portfolio_greeks(
        self, spot: float, current_ivs: dict[str, float] | None = None
    ) -> PortfolioGreeks:
        pg = PortfolioGreeks()
        for instr, pos in self.positions.items():
            iv = (current_ivs or {}).get(instr, pos.entry_iv)
            g  = self._position_greeks(pos, spot, iv)
            pg.net_delta += g.get("delta", 0.0)
            pg.net_gamma += g.get("gamma", 0.0)
            pg.net_vega  += g.get("vega",  0.0)
            pg.net_theta += g.get("theta", 0.0)
        return pg

    # ── Position value ─────────────────────────────────────────────────────────

    def _position_value(
        self, pos: OptionPosition, spot: float, current_iv: float
    ) -> float:
        """Current USD value of one position."""
        T = max((pos.expiry - datetime.utcnow()).total_seconds() / (365 * 86400), 1 / 365)
        price = bsm_price(spot, pos.strike, T, RISK_FREE_RATE, current_iv, pos.option_type)
        if math.isnan(price):
            return pos.notional_usd  # assume flat if BSM fails
        # contracts = notional / entry_price; current value = contracts × current_price
        if pos.entry_price <= 0:
            return pos.notional_usd
        contracts = pos.notional_usd / pos.entry_price
        return contracts * price

    # ── Risk checks ────────────────────────────────────────────────────────────

    def _drawdown_halted(self) -> bool:
        dd = (self.capital - self.peak_capital) / self.peak_capital
        return dd <= -MAX_DRAWDOWN_HALT_PCT

    def _would_breach_greeks(
        self, pos: OptionPosition, spot: float
    ) -> str | None:
        """Return a reason string if opening pos would breach a Greek limit, else None."""
        g = self._position_greeks(pos, spot, pos.entry_iv)

        pg = self.portfolio_greeks(spot)
        new_delta = pg.net_delta + g.get("delta", 0.0)
        new_vega  = pg.net_vega  + g.get("vega",  0.0)
        new_gamma = pg.net_gamma + g.get("gamma", 0.0)

        if abs(new_delta) > MAX_NET_DELTA * self.capital:
            return f"net delta {new_delta:.2f} would exceed limit {MAX_NET_DELTA * self.capital:.2f}"
        if abs(new_vega) > MAX_VEGA_NOTIONAL * self.capital:
            return f"total vega {new_vega:.2f} would exceed limit {MAX_VEGA_NOTIONAL * self.capital:.2f}"
        if abs(new_gamma) > MAX_GAMMA_NOTIONAL * self.capital:
            return f"total gamma {new_gamma:.4f} would exceed limit {MAX_GAMMA_NOTIONAL * self.capital:.4f}"
        return None

    # ── Open / close ──────────────────────────────────────────────────────────

    def open_position(self, pos: OptionPosition) -> bool:
        """
        Open a new position. Returns False (with printed reason) if rejected.
        """
        if self._drawdown_halted():
            print(f"  [risk] NEW TRADES HALTED — portfolio drawdown exceeds {MAX_DRAWDOWN_HALT_PCT:.0%}")
            return False

        if len(self.positions) >= MAX_OPEN_POSITIONS:
            print(f"  [risk] Max open positions ({MAX_OPEN_POSITIONS}) reached — skipping {pos.instrument}")
            return False

        if pos.instrument in self.positions:
            print(f"  [risk] Already have position in {pos.instrument} — skipping duplicate")
            return False

        reason = self._would_breach_greeks(pos, pos.entry_spot)
        if reason:
            print(f"  [risk] Greek limit breach — {reason} — skipping {pos.instrument}")
            return False

        notional = self.capital * POSITION_SIZE_PCT
        pos.notional_usd    = notional
        pos.allocated_capital = notional
        pos.peak_value      = notional

        self.positions[pos.instrument] = pos
        self._trade_log.append({
            "action":      "open",
            "instrument":  pos.instrument,
            "signal_type": pos.signal_type,
            "direction":   "long" if pos.direction == 1 else "short",
            "notional":    notional,
            "entry_iv":    pos.entry_iv,
            "entry_spot":  pos.entry_spot,
        })
        return True

    def close_position(
        self, instrument: str, current_iv: float, current_spot: float,
        reason: str = "signal"
    ) -> float:
        """
        Close a position. Returns realised P&L in USD.
        """
        pos = self.positions.pop(instrument, None)
        if pos is None:
            return 0.0

        exit_value = self._position_value(pos, current_spot, current_iv)
        pnl = (exit_value - pos.allocated_capital) * pos.direction
        self.capital += pnl

        self._trade_log.append({
            "action":     "close",
            "instrument": instrument,
            "reason":     reason,
            "pnl_usd":    pnl,
            "exit_spot":  current_spot,
            "exit_iv":    current_iv,
        })
        return pnl

    # ── Mark to market ─────────────────────────────────────────────────────────

    def mark_to_market(
        self, current_ivs: dict[str, float], current_spot: float
    ) -> float:
        """
        Revalue all open positions. Returns total portfolio USD value.
        Updates peak value for stop-loss tracking.
        """
        open_value = 0.0
        for instr, pos in self.positions.items():
            iv = current_ivs.get(instr, pos.entry_iv)
            val = self._position_value(pos, current_spot, iv)
            pos.peak_value = max(pos.peak_value, val)
            open_value += val

        # Cash not in positions
        cash = self.capital - sum(p.allocated_capital for p in self.positions.values())
        total = cash + open_value
        self.peak_capital = max(self.peak_capital, total)
        return total

    # ── Expiry settlement ──────────────────────────────────────────────────────

    def expire_positions(self, current_date: datetime, spot: float) -> float:
        """
        Settle expired positions at intrinsic value.
        Returns total expiry P&L.
        """
        total_pnl = 0.0
        expired = [instr for instr, pos in self.positions.items()
                   if pos.expiry.date() <= current_date.date()]

        for instr in expired:
            pos = self.positions[instr]
            K   = pos.strike
            if pos.option_type == "C":
                intrinsic = max(0.0, spot - K)
            else:
                intrinsic = max(0.0, K - spot)

            contracts = pos.notional_usd / pos.entry_price if pos.entry_price > 0 else 0
            exit_value = contracts * intrinsic
            pnl = (exit_value - pos.allocated_capital) * pos.direction
            self.capital += pnl
            total_pnl += pnl
            self.positions.pop(instr)

            self._trade_log.append({
                "action":    "expire",
                "instrument": instr,
                "intrinsic":  intrinsic,
                "pnl_usd":    pnl,
                "spot":       spot,
            })

        return total_pnl

    # ── Stop-loss ─────────────────────────────────────────────────────────────

    def check_stop_loss(
        self, current_ivs: dict[str, float], current_spot: float
    ) -> list[str]:
        """
        Return list of instruments that have breached STOP_LOSS_PCT.
        Caller should close these positions.
        """
        triggered = []
        for instr, pos in self.positions.items():
            iv  = current_ivs.get(instr, pos.entry_iv)
            val = self._position_value(pos, current_spot, iv)
            loss_pct = (pos.allocated_capital - val) / pos.allocated_capital
            if loss_pct > STOP_LOSS_PCT:
                triggered.append(instr)
        return triggered

    # ── Delta hedge size ──────────────────────────────────────────────────────

    def net_delta_hedge(
        self, spot: float, current_ivs: dict[str, float] | None = None
    ) -> float:
        """
        USD notional of BTC-PERPETUAL needed to flatten net delta.
        Positive = buy perp, negative = sell perp.
        """
        pg = self.portfolio_greeks(spot, current_ivs)
        # net_delta is in "USD per USD of spot move" → convert to perp notional
        return -pg.net_delta * spot

    # ── Trade log ─────────────────────────────────────────────────────────────

    @property
    def trade_log(self) -> list[dict]:
        return self._trade_log
