"""
Backtest chart utilities.

Creates PNG charts for equity curves and drawdowns.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip()).strip("_").lower()
    return slug or "backtest"


def _drawdown(equity: pd.Series) -> pd.Series:
    if equity.empty:
        return pd.Series(dtype=float)
    return equity / equity.cummax() - 1.0


def _safe_total_return(equity: pd.Series) -> float:
    s = equity.dropna()
    if len(s) < 2:
        return float("nan")
    base = float(s.iloc[0])
    if base == 0:
        return float("nan")
    return float(s.iloc[-1] / base - 1.0)


def save_backtest_charts(
    equity_curves: dict[str, pd.Series],
    period_label: str,
    output_dir: str | Path = "backtest/charts",
) -> dict[str, str]:
    """
    Save backtest charts and return created file paths.

    Returns a dict with keys:
      - equity_drawdown
      - total_return_bar
    """
    if not equity_curves:
        return {}

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.concat(equity_curves, axis=1).sort_index().ffill().dropna(how="all")
    if df.empty:
        return {}

    slug = _slugify(period_label)
    combined_name = "Combined" if "Combined" in df.columns else str(df.columns[0])

    # Chart 1: Equity + combined drawdown
    drawdown = _drawdown(df[combined_name])

    fig, (ax_equity, ax_dd) = plt.subplots(
        2, 1, figsize=(13, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    for col in df.columns:
        is_combined = col == combined_name
        ax_equity.plot(
            df.index,
            df[col],
            label=col,
            linewidth=2.2 if is_combined else 1.3,
            alpha=1.0 if is_combined else 0.85,
        )

    ax_equity.set_title(f"Backtest Equity Curves - {period_label}")
    ax_equity.set_ylabel("Portfolio Value (USD)")
    ax_equity.grid(alpha=0.25)
    ax_equity.legend(loc="best")
    ax_equity.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:,.0f}"))

    ax_dd.fill_between(drawdown.index, drawdown.values * 100, 0, color="#C0392B", alpha=0.25)
    ax_dd.plot(drawdown.index, drawdown.values * 100, color="#C0392B", linewidth=1.8)
    ax_dd.set_title(f"Drawdown - {combined_name}")
    ax_dd.set_ylabel("Drawdown (%)")
    ax_dd.grid(alpha=0.25)
    ax_dd.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f}%"))

    fig.tight_layout()
    fig.autofmt_xdate()

    equity_path = out_dir / f"{slug}_equity_drawdown.png"
    fig.savefig(equity_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Chart 2: Total return per strategy
    totals = pd.Series({col: _safe_total_return(df[col]) for col in df.columns}).dropna()
    if totals.empty:
        return {"equity_drawdown": str(equity_path)}

    fig2, ax_bar = plt.subplots(figsize=(10, 5))
    colors = ["#1F78B4" if idx != combined_name else "#0B3C5D" for idx in totals.index]
    bars = ax_bar.bar(totals.index, totals.values * 100, color=colors, alpha=0.9)
    ax_bar.axhline(0, color="black", linewidth=1)
    ax_bar.set_title(f"Total Return by Strategy - {period_label}")
    ax_bar.set_ylabel("Total Return (%)")
    ax_bar.grid(axis="y", alpha=0.25)
    ax_bar.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax_bar.tick_params(axis="x", rotation=20)

    for bar, val in zip(bars, totals.values):
        y = bar.get_height()
        va = "bottom" if y >= 0 else "top"
        offset = 0.6 if y >= 0 else -0.6
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            y + offset,
            f"{val * 100:.1f}%",
            ha="center",
            va=va,
            fontsize=9,
        )

    fig2.tight_layout()
    return_path = out_dir / f"{slug}_total_return.png"
    fig2.savefig(return_path, dpi=160, bbox_inches="tight")
    plt.close(fig2)

    return {
        "equity_drawdown": str(equity_path),
        "total_return_bar": str(return_path),
    }
