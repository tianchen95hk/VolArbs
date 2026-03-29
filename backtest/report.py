"""
Backtest performance reporting.

_stats() is adapted from dex_deribit/backtest.py (lines 66-85).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_stats(equity: pd.Series, periods_per_year: int = 365) -> dict:
    """
    Compute standard performance statistics from an equity curve.

    Parameters
    ----------
    equity           : pd.Series of portfolio values (daily or sub-daily)
    periods_per_year : 365 for daily bars

    Returns dict with keys:
        total_ret, ann_ret, ann_vol, sharpe, max_dd, win_rate, n_periods
    """
    if equity.empty or len(equity) < 2:
        return {}

    ret   = equity.pct_change().dropna()
    total = equity.iloc[-1] / equity.iloc[0] - 1
    n     = len(equity)
    ann   = (1 + total) ** (periods_per_year / n) - 1
    vol   = ret.std() * np.sqrt(periods_per_year)
    sr    = ann / vol if vol > 0 else float("nan")
    mdd   = float((equity / equity.cummax() - 1).min())
    win   = float((ret > 0).mean())

    return dict(
        total_ret=total,
        ann_ret=ann,
        ann_vol=vol,
        sharpe=sr,
        max_dd=mdd,
        win_rate=win,
        n_periods=n,
    )


def print_backtest_report(
    results: dict[str, dict],
    period_label: str = "",
) -> None:
    """
    Print a formatted performance table.

    Parameters
    ----------
    results      : dict mapping strategy_name → stats_dict (from compute_stats)
    period_label : optional string for the header line
    """
    header = f"  Backtest Results{f' — {period_label}' if period_label else ''}"
    print()
    print("=" * 72)
    print(header)
    print("=" * 72)
    print(f"  {'Strategy':<22}  {'TotalRet':>8}  {'AnnRet':>7}  "
          f"{'Sharpe':>7}  {'MaxDD':>7}  {'Win%':>6}  {'Bars':>6}")
    print("  " + "─" * 68)

    for name, s in results.items():
        if not s:
            print(f"  {name:<22}  {'N/A':>8}")
            continue
        print(
            f"  {name:<22}"
            f"  {s['total_ret']:>+8.1%}"
            f"  {s['ann_ret']:>+7.1%}"
            f"  {s['sharpe']:>7.2f}"
            f"  {s['max_dd']:>7.1%}"
            f"  {s['win_rate']:>5.1%}"
            f"  {s['n_periods']:>6d}"
        )

    print("  " + "─" * 68)
    print()
