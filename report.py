"""
cross_vol/report.py — Performance attribution for the cross-asset vol spread.

Provides:
  - compute_attribution(): per-leg and trade-level stats
  - print_cross_vol_report(): formatted console output
"""
from __future__ import annotations

import sys
import os
from datetime import datetime

import pandas as pd
import numpy as np

_PER_ASSET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "per_asset")
if _PER_ASSET not in sys.path:
    sys.path.insert(0, _PER_ASSET)

from backtest.report import compute_stats
from engine import CrossVolTrade


# ── Attribution ───────────────────────────────────────────────────────────────

def compute_attribution(
    trades: list[CrossVolTrade],
    equity: pd.Series,
) -> dict:
    """
    Compute per-leg and trade-level attribution.

    Returns dict with keys:
        overall_stats    — from backtest.report.compute_stats(equity)
        long_leg_stats   — stats from long leg P&L series only
        short_leg_stats  — stats from short leg P&L series only
        trade_stats      — trade-level summary metrics
    """
    overall = compute_stats(equity)

    if not trades:
        return {
            "overall_stats":   overall,
            "long_leg_stats":  {},
            "short_leg_stats": {},
            "trade_stats":     {"n_trades": 0},
        }

    # Build per-leg P&L series indexed by exit_date
    long_pnls  = []
    short_pnls = []
    exit_dates = []
    for t in trades:
        if t.exit_date is not None:
            _ts = pd.Timestamp(t.exit_date)
            exit_dates.append(_ts if _ts.tzinfo else _ts.tz_localize("UTC"))
            long_pnls.append(t.pnl_long_usd)
            short_pnls.append(t.pnl_short_usd)

    from cv_config import INITIAL_CAPITAL
    n = len(exit_dates)
    if n > 0:
        long_series  = pd.Series(long_pnls,  index=exit_dates).sort_index()
        short_series = pd.Series(short_pnls, index=exit_dates).sort_index()
        long_equity  = INITIAL_CAPITAL + long_series.cumsum()
        short_equity = INITIAL_CAPITAL + short_series.cumsum()
        long_stats   = compute_stats(long_equity)
        short_stats  = compute_stats(short_equity)
    else:
        long_stats = short_stats = {}

    trade_stats = _compute_trade_stats(trades)

    return {
        "overall_stats":   overall,
        "long_leg_stats":  long_stats,
        "short_leg_stats": short_stats,
        "trade_stats":     trade_stats,
    }


def _compute_trade_stats(trades: list[CrossVolTrade]) -> dict:
    if not trades:
        return {"n_trades": 0}

    n          = len(trades)
    winners    = sum(1 for t in trades if t.total_pnl_usd > 0)
    win_rate   = winners / n * 100

    avg_pnl    = np.mean([t.total_pnl_usd for t in trades])
    avg_hold   = np.mean([t.holding_days  for t in trades])
    avg_z      = np.mean([t.entry_zscore  for t in trades])
    avg_spread = np.mean([t.entry_spread_pts for t in trades])

    exit_counts = {}
    for t in trades:
        exit_counts[t.exit_reason] = exit_counts.get(t.exit_reason, 0) + 1

    long_contrib  = sum(t.pnl_long_usd  for t in trades)
    short_contrib = sum(t.pnl_short_usd for t in trades)
    total_contrib = long_contrib + short_contrib

    long_pct  = long_contrib  / total_contrib * 100 if total_contrib != 0 else 0
    short_pct = short_contrib / total_contrib * 100 if total_contrib != 0 else 0

    return {
        "n_trades":         n,
        "win_rate":         win_rate,
        "avg_pnl_usd":      avg_pnl,
        "avg_holding_days": avg_hold,
        "avg_entry_zscore": avg_z,
        "avg_spread_pts":   avg_spread,
        "exit_reasons":     exit_counts,
        "long_contrib_pct": long_pct,
        "short_contrib_pct":short_pct,
    }


# ── Printing ──────────────────────────────────────────────────────────────────

def print_cross_vol_report(
    results: dict,
    attribution: dict,
    period_label: str = "",
    asset_a: str = "BTC",
    asset_b: str = "ETH",
) -> None:
    """Print formatted cross-vol performance report to stdout."""
    width = 72
    eq    = "═" * width
    dash  = "─" * width
    label = f"  Cross-Vol Spread — {period_label}  ({asset_a}/{asset_b})" if period_label else \
            f"  Cross-Vol Spread  ({asset_a}/{asset_b})"

    print(f"\n{eq}")
    print(label)
    print(eq)

    # ── Section 1: Overall stats ──────────────────────────────────────────────
    s = attribution.get("overall_stats", {})
    if s:
        hdr = f"  {'Strategy':<22} {'TotalRet':>9} {'AnnRet':>8} {'Sharpe':>8} {'MaxDD':>8} {'Win%':>7} {'Bars':>6}"
        print(hdr)
        print(f"  {dash[:66]}")
        tr = s.get("total_ret", 0) * 100
        ar = s.get("ann_ret",   0) * 100
        sh = s.get("sharpe",    0)
        dd = s.get("max_dd",    0) * 100
        wr = s.get("win_rate",  0) * 100
        nb = s.get("n_periods", 0)
        sh_str = f"{sh:>8.2f}" if np.isfinite(sh) else "     nan"
        print(
            f"  {'BTC/ETH Spread':<22} {tr:>+8.1f}%  {ar:>+7.1f}%  "
            f"{sh_str}  {dd:>+7.1f}%  {wr:>6.1f}%  {nb:>5}"
        )
        print(f"  {dash[:66]}")

    # ── Section 2: Leg attribution ────────────────────────────────────────────
    ts = attribution.get("trade_stats", {})
    if ts.get("n_trades", 0) > 0:
        print(f"\n  Leg Attribution")
        print(f"  {dash[:50]}")
        print(f"  {'Leg':<16} {'Contribution':>14}")
        print(f"  {'Long vol (' + asset_a + ')':<16} {ts['long_contrib_pct']:>+13.1f}%")
        print(f"  {'Short vol (' + asset_b + ')':<16} {ts['short_contrib_pct']:>+13.1f}%")

        # ── Section 3: Trade statistics ───────────────────────────────────────
        print(f"\n  Trade Statistics")
        print(f"  {dash[:50]}")
        print(f"  Trades: {ts['n_trades']}   Win rate: {ts['win_rate']:.0f}%   "
              f"Avg hold: {ts['avg_holding_days']:.0f}d   "
              f"Avg P&L: ${ts['avg_pnl_usd']:+,.0f}")
        print(f"  Entry: avg z={ts['avg_entry_zscore']:.2f}  "
              f"avg spread={ts['avg_spread_pts']:.1f} vol pts")

        exit_parts = [
            f"{reason}: {cnt}/{ts['n_trades']}"
            for reason, cnt in ts["exit_reasons"].items()
        ]
        print(f"  Exits:  {',  '.join(exit_parts)}")

    else:
        print("\n  No completed trades in this period.")

    print()


def print_trade_log(
    trades: list[CrossVolTrade],
    spread_series: pd.DataFrame | None = None,
) -> None:
    """
    Print a detailed chronological trade log with three phases per trade:
      1. SIGNAL  — when the z-score crossed the entry threshold
      2. OPEN    — position details at entry
      3. CLOSE   — P&L breakdown at exit
    """
    if not trades:
        print("  No trades to display.\n")
        return

    width = 72
    eq   = "═" * width
    dash = "─" * width

    print(f"\n{eq}")
    print(f"  Detailed Trade Log  ({len(trades)} trade{'s' if len(trades) != 1 else ''})")
    print(eq)

    for i, t in enumerate(trades, 1):
        win = "✓ WIN " if t.total_pnl_usd > 0 else "✗ LOSS"
        print(f"\n  Trade #{i}  {win}  {t.asset_long}/{t.asset_short}")
        print(f"  {dash[:50]}")

        # ── 1. SIGNAL ─────────────────────────────────────────────────────────
        print(f"  [SIGNAL]  {t.entry_date.strftime('%Y-%m-%d')}")
        from cv_config import CROSS_VOL_ZSCORE_ENTRY
        print(f"    Spread z-score : {t.entry_zscore:+.2f}σ  (threshold ±{CROSS_VOL_ZSCORE_ENTRY:.2f}σ)")
        print(f"    Raw spread     : {t.entry_spread_pts:.1f} vol pts")
        print(f"    Direction      : Long {t.asset_long} vol  ·  Short {t.asset_short} vol")

        # ── 2. OPEN ───────────────────────────────────────────────────────────
        print(f"\n  [OPEN]    {t.entry_date.strftime('%Y-%m-%d')}")
        total_notional = t.notional_long + t.notional_short
        print(f"    Long  {t.asset_long}  notional : ${t.notional_long:>9,.0f}")
        print(f"    Short {t.asset_short}  notional : ${t.notional_short:>9,.0f}")
        print(f"    Total notional  : ${total_notional:>9,.0f}")
        print(f"    Gross vega      : ${t.gross_vega_usd:>9,.2f}  (net ≈ $0 by construction)")

        # ── 3. CLOSE ──────────────────────────────────────────────────────────
        exit_date_str = t.exit_date.strftime('%Y-%m-%d') if t.exit_date else "open"
        print(f"\n  [CLOSE]   {exit_date_str}  ({t.holding_days}d held)  reason: {t.exit_reason}")
        if t.exit_zscore is not None:
            print(f"    Exit z-score   : {t.exit_zscore:+.2f}σ")
        print(f"    Long  leg P&L  : ${t.pnl_long_usd:>+10,.2f}")
        print(f"    Short leg P&L  : ${t.pnl_short_usd:>+10,.2f}")
        print(f"    ──────────────────────────────")
        print(f"    Net P&L        : ${t.total_pnl_usd:>+10,.2f}  "
              f"({t.total_pnl_usd / total_notional * 100:+.1f}% of notional)")

    # ── Summary footer ────────────────────────────────────────────────────────
    total_pnl = sum(t.total_pnl_usd for t in trades)
    winners   = sum(1 for t in trades if t.total_pnl_usd > 0)
    print(f"\n{dash}")
    print(f"  Total P&L: ${total_pnl:>+,.2f}   "
          f"Win rate: {winners}/{len(trades)} ({winners/len(trades)*100:.0f}%)")
    print()
