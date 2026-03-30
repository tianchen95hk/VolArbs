"""
cross_vol/report_portfolio.py — Multi-pair portfolio backtest report.

Extends the single-pair report.py to handle:
  - Multiple pairs trading simultaneously
  - Per-pair attribution (which pair drove results)
  - Factor-agreement vs outcome analysis
  - Dynamic sizing analysis (did larger sizes work?)
  - Detailed trade log: [SIGNAL] → [OPEN] → [CLOSE]
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime

from portfolio_engine import PairTrade


# ── Equity stats ──────────────────────────────────────────────────────────────

def _compute_stats(equity: pd.Series) -> dict:
    if len(equity) < 2:
        return {}
    ret = equity.pct_change().dropna()
    total_ret = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
    n_days    = max((equity.index[-1] - equity.index[0]).days, 1)
    ann_ret   = ((1 + total_ret / 100) ** (365 / n_days) - 1) * 100
    sharpe    = float(ret.mean() / ret.std() * np.sqrt(252)) if ret.std() > 0 else 0.0
    max_dd    = float(((equity / equity.cummax()) - 1).min() * 100)
    return {
        "total_ret": total_ret, "ann_ret": ann_ret,
        "sharpe": sharpe, "max_dd": max_dd, "n_bars": len(equity),
    }


# ── Main report ───────────────────────────────────────────────────────────────

def print_portfolio_report(
    results: dict,
    period_label: str,
    coins: list[str],
) -> None:
    """
    Print a full multi-pair portfolio performance report.

    results dict keys: equity, trades, pair_spreads, summary
    """
    equity  = results.get("equity", pd.Series(dtype=float))
    trades: list[PairTrade] = results.get("trades", [])
    summary = results.get("summary", {})

    stats = _compute_stats(equity)

    print(f"\n{'═'*72}")
    print(f"  Multi-Pair Vol Portfolio — {period_label}  ({', '.join(coins)})")
    print(f"{'═'*72}")

    # ── Overall performance ───────────────────────────────────────────────────
    print(f"\n  {'Strategy':<26s}  {'TotalRet':>8s}  {'AnnRet':>7s}  "
          f"{'Sharpe':>7s}  {'MaxDD':>7s}  {'Win%':>5s}  {'Bars':>5s}")
    print(f"  {'─'*68}")
    tr   = stats.get("total_ret", 0)
    ar   = stats.get("ann_ret",   0)
    sh   = stats.get("sharpe",    0)
    dd   = stats.get("max_dd",    0)
    wr   = summary.get("win_rate_pct", 0)
    bars = stats.get("n_bars", 0)
    print(f"  {'Multi-Pair Portfolio':<26s}  {tr:>+7.1f}%  {ar:>+6.1f}%  "
          f"{sh:>7.2f}  {dd:>+7.1f}%  {wr:>4.1f}%  {bars:>5d}")
    print(f"  {'─'*68}")

    # ── Per-pair attribution ──────────────────────────────────────────────────
    pair_pnl: dict[str, float] = {}
    pair_trades: dict[str, list[PairTrade]] = {}
    for t in trades:
        pair_pnl[t.pair]    = pair_pnl.get(t.pair, 0.0) + t.total_pnl_usd
        pair_trades.setdefault(t.pair, []).append(t)

    if pair_pnl:
        total_abs = sum(abs(v) for v in pair_pnl.values()) or 1.0
        print(f"\n  {'─'*72}")
        print(f"  Per-Pair Attribution")
        print(f"  {'─'*72}")
        print(f"  {'Pair':<12s}  {'Trades':>6s}  {'Win%':>5s}  {'P&L':>10s}  {'Contribution':>12s}")
        print(f"  {'─'*52}")
        for pair in sorted(pair_pnl, key=lambda p: -abs(pair_pnl[p])):
            pts = pair_trades[pair]
            wins = sum(1 for t in pts if t.total_pnl_usd > 0)
            wr_p = wins / len(pts) * 100 if pts else 0.0
            pnl  = pair_pnl[pair]
            contrib_pct = pnl / (equity.iloc[0] if len(equity) else 100000) * 100
            print(f"  {pair:<12s}  {len(pts):>6d}  {wr_p:>4.0f}%  "
                  f"  ${pnl:>+9,.0f}  {contrib_pct:>+11.1f}%")

    # ── Trade statistics ──────────────────────────────────────────────────────
    if trades:
        avg_hold   = np.mean([t.holding_days for t in trades])
        avg_pnl    = np.mean([t.total_pnl_usd for t in trades])
        avg_z      = np.mean([t.entry_zscore for t in trades])
        avg_agree  = np.mean([t.factor_agreement for t in trades])
        avg_mult   = np.mean([t.size_mult for t in trades])
        exit_counts: dict[str, int] = {}
        for t in trades:
            exit_counts[t.exit_reason] = exit_counts.get(t.exit_reason, 0) + 1

        print(f"\n  {'─'*72}")
        print(f"  Trade Statistics")
        print(f"  {'─'*72}")
        print(f"  Trades: {len(trades)}   "
              f"Win rate: {sum(1 for t in trades if t.total_pnl_usd>0)}/{len(trades)} "
              f"({wr:.0f}%)   "
              f"Avg hold: {avg_hold:.0f}d   "
              f"Avg P&L: ${avg_pnl:,.0f}")
        print(f"  Entry: avg z={avg_z:.2f}  "
              f"avg factor-agreement={avg_agree:.0%}  "
              f"avg size-mult={avg_mult:.2f}×")
        exits_str = "  ".join(f"{r}: {c}/{len(trades)}" for r, c in exit_counts.items())
        print(f"  Exits:  {exits_str}")

        # Factor agreement vs outcome
        wins_high_agree  = [t for t in trades if t.factor_agreement >= 0.7 and t.total_pnl_usd > 0]
        wins_low_agree   = [t for t in trades if t.factor_agreement <  0.7 and t.total_pnl_usd > 0]
        total_high_agree = [t for t in trades if t.factor_agreement >= 0.7]
        total_low_agree  = [t for t in trades if t.factor_agreement <  0.7]
        if total_high_agree and total_low_agree:
            wr_high = len(wins_high_agree) / len(total_high_agree) * 100
            wr_low  = len(wins_low_agree)  / len(total_low_agree)  * 100
            print(f"\n  Factor Agreement vs Outcome:")
            print(f"    ≥70% agreement ({len(total_high_agree)} trades): "
                  f"win rate {wr_high:.0f}%")
            print(f"    <70% agreement ({len(total_low_agree)} trades):  "
                  f"win rate {wr_low:.0f}%")


def print_portfolio_trade_log(trades: list[PairTrade]) -> None:
    """Print detailed [SIGNAL] → [OPEN] → [CLOSE] log for every trade."""
    if not trades:
        print("\n  (no trades)\n")
        return

    print(f"\n{'═'*72}")
    print(f"  Detailed Trade Log  ({len(trades)} trades)")
    print(f"{'═'*72}")

    for i, t in enumerate(trades, 1):
        win   = t.total_pnl_usd > 0
        label = "✓ WIN " if win else "✗ LOSS"
        print(f"\n  Trade #{i}  {label}  {t.asset_long}/{t.asset_short}  [{t.pair}]")
        print(f"  {'─'*60}")

        # SIGNAL
        print(f"  [SIGNAL]  {t.entry_date.strftime('%Y-%m-%d')}")
        print(f"    Spread z-score   : {t.entry_zscore:+.2f}σ")
        print(f"    Factor agreement : {t.factor_agreement:.0%}  "
              f"({'✓ strong' if t.factor_agreement >= 0.7 else '~ medium' if t.factor_agreement >= 0.5 else '✗ weak'})")
        print(f"    Size multiplier  : {t.size_mult:.2f}×")
        print(f"    Direction        : Long {t.asset_long} vol  ·  Short {t.asset_short} vol")

        # OPEN
        total_n = t.notional_long + t.notional_short
        print(f"\n  [OPEN]    {t.entry_date.strftime('%Y-%m-%d')}")
        print(f"    Long  {t.asset_long:<5s} notional : ${t.notional_long:>10,.0f}")
        print(f"    Short {t.asset_short:<5s} notional : ${t.notional_short:>10,.0f}")
        print(f"    Total notional   : ${total_n:>10,.0f}")
        print(f"    Gross vega       : ${t.gross_vega_usd:>10,.2f}  (net ≈ $0 by construction)")

        # CLOSE
        exit_date_str = t.exit_date.strftime('%Y-%m-%d') if t.exit_date else "open"
        exit_z_str    = f"{t.exit_zscore:+.2f}σ" if t.exit_zscore is not None else "—"
        pnl_pct = t.total_pnl_usd / total_n * 100 if total_n > 0 else 0.0
        print(f"\n  [CLOSE]   {exit_date_str}  ({t.holding_days}d held)  "
              f"reason: {t.exit_reason}")
        print(f"    Exit z-score     : {exit_z_str}")
        print(f"    Long  leg P&L    : ${t.pnl_long_usd:>+11,.2f}")
        print(f"    Short leg P&L    : ${t.pnl_short_usd:>+11,.2f}")
        print(f"    {'─'*40}")
        print(f"    Net P&L          : ${t.total_pnl_usd:>+11,.2f}  "
              f"({pnl_pct:+.1f}% of notional)")

    total_pnl = sum(t.total_pnl_usd for t in trades)
    wins = sum(1 for t in trades if t.total_pnl_usd > 0)
    print(f"\n{'─'*72}")
    print(f"  Total P&L: ${total_pnl:>+,.2f}   "
          f"Win rate: {wins}/{len(trades)} ({wins/len(trades)*100:.0f}%)\n")
