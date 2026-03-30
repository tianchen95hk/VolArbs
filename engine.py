"""
cross_vol/engine.py — Daily event-driven backtest for the cross-asset vol spread.

Loop per daily bar:
    1. Read aligned vol_premium_A, vol_premium_B
    2. Compute z-score from pre-built spread_series
    3. If not in trade → check entry signal → size vega-neutral → open trade
    4. If in trade → mark-to-market → check stop-loss → check exit signal
    5. Charge delta-hedge cost daily (if CROSS_VOL_DELTA_HEDGE)

P&L methodology:
    CROSS_VOL_USE_FULL_BSM = True  (default):
        contracts = notional_entry / straddle_price_entry
        value_t   = contracts × bsm_straddle(S_t, DVOL_t, remaining_dte_t)
        leg_pnl_t = direction × (value_t − notional_entry)

    CROSS_VOL_USE_FULL_BSM = False (vega approximation):
        leg_pnl_t = direction × (exit_IV − entry_IV) × straddle_vega × notional

Note on PositionTracker: not used in the backtest loop. The cross-vol engine
tracks its own PnL via CrossVolTrade objects. PositionTracker is appropriate
for live trading (per-leg Greek limits, instrument registry) but adds overhead
for a paired strategy that enforces vega-neutrality through sizing.
"""
from __future__ import annotations

import sys
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import pandas as pd
import numpy as np

_PER_ASSET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "per_asset")
if _PER_ASSET not in sys.path:
    sys.path.insert(0, _PER_ASSET)

from analytics.bsm import bsm_price
from cv_config import (
    INITIAL_CAPITAL,
    COMMISSION_PCT,
    HEDGE_COST_PCT,
    STOP_LOSS_PCT,
    MAX_DRAWDOWN_HALT_PCT,
    CROSS_VOL_USE_FULL_BSM,
    CROSS_VOL_DELTA_HEDGE,
    BACKTEST_PERIODS,
    CROSS_VOL_REGIME_FILTER,
    CROSS_VOL_REGIME_LOOKBACK,
    CROSS_VOL_REGIME_ZSCORE_MAX,
    CROSS_VOL_SPREAD_LOOKBACK,
    CROSS_VOL_ZSCORE_ENTRY,
    CROSS_VOL_ZSCORE_EXIT,
    CROSS_VOL_SPREAD_MIN_PTS,
    CROSS_VOL_HOLDING_DAYS_MAX,
    CROSS_VOL_CAPITAL_FRACTION,
    CROSS_VOL_DTE_TARGET,
)
from universe import AssetVolData, align_assets
from spread_signal import compute_spread_series, generate_signal
from sizing import compute_vega_neutral_sizing, PairSizing


# ── Data containers ───────────────────────────────────────────────────────────

@dataclass
class CrossVolTrade:
    """One complete round-trip trade (entry → exit)."""
    entry_date:       datetime
    exit_date:        datetime | None
    asset_long:       str
    asset_short:      str
    entry_zscore:     float
    exit_zscore:      float | None
    entry_spread_pts: float
    notional_long:    float
    notional_short:   float
    gross_vega_usd:   float
    pnl_long_usd:     float
    pnl_short_usd:    float
    total_pnl_usd:    float
    exit_reason:      str   # "reversion" | "max_holding" | "stop_loss" | "end_of_period"
    holding_days:     int


# ── BSM straddle helpers ──────────────────────────────────────────────────────

def _bsm_straddle_price(spot: float, iv_pct: float, dte: int, r: float = 0.05) -> float:
    """BSM ATM straddle price: call(K=S) + put(K=S)."""
    if dte <= 0 or iv_pct <= 0 or spot <= 0:
        return 0.0
    T = dte / 365.0
    sigma = iv_pct / 100.0
    return (
        bsm_price(spot, spot, T, r, sigma, "C") +
        bsm_price(spot, spot, T, r, sigma, "P")
    )


def _leg_pnl(
    direction: int,
    entry_spot: float,
    current_spot: float,
    entry_iv: float,
    current_iv: float,
    entry_straddle_price: float,
    straddle_vega: float,
    notional: float,
    dte_at_entry: int,
    days_held: int,
    use_full_bsm: bool = CROSS_VOL_USE_FULL_BSM,
) -> float:
    """Mark-to-market P&L for one straddle leg."""
    if use_full_bsm:
        remaining_dte = max(0, dte_at_entry - days_held)
        current_price = _bsm_straddle_price(current_spot, current_iv, remaining_dte)
        if entry_straddle_price <= 0:
            return 0.0
        contracts = notional / entry_straddle_price
        return direction * (current_price - entry_straddle_price) * contracts
    else:
        # Vega approximation: P&L ≈ direction × ΔIV × vega × notional
        delta_iv = current_iv - entry_iv
        return direction * delta_iv * straddle_vega * notional


def _commission(notional_long: float, notional_short: float) -> float:
    """Round-trip commission on both legs (entry + exit = 2×)."""
    return 2.0 * COMMISSION_PCT * (notional_long + notional_short)


def _build_regime_series(
    spot_a: pd.Series,
    spot_b: pd.Series,
    lookback: int = CROSS_VOL_REGIME_LOOKBACK,
) -> pd.Series:
    """
    Compute rolling z-score of the A/B spot price ratio.

    A high |z| means the ratio is in a strong trend → vol spread is likely
    trending too, breaking the mean-reversion assumption.

    Returns a pd.Series indexed like spot_a with z-score values.
    """
    ratio = (spot_a / spot_b).dropna()
    log_ratio = np.log(ratio)
    roll_mean = log_ratio.rolling(lookback, min_periods=max(2, lookback // 2)).mean()
    roll_std  = log_ratio.rolling(lookback, min_periods=max(2, lookback // 2)).std()
    z = (log_ratio - roll_mean) / roll_std.replace(0, float("nan"))
    return z.fillna(0.0)


def _resolve_backtest_params(overrides: dict | None) -> dict:
    """Build the effective parameter set for one backtest run."""
    params = {
        "spread_lookback": CROSS_VOL_SPREAD_LOOKBACK,
        "zscore_entry": CROSS_VOL_ZSCORE_ENTRY,
        "zscore_exit": CROSS_VOL_ZSCORE_EXIT,
        "spread_min_pts": CROSS_VOL_SPREAD_MIN_PTS,
        "holding_days_max": CROSS_VOL_HOLDING_DAYS_MAX,
        "capital_fraction": CROSS_VOL_CAPITAL_FRACTION,
        "dte_target": CROSS_VOL_DTE_TARGET,
        "regime_lookback": CROSS_VOL_REGIME_LOOKBACK,
        "regime_zscore_max": CROSS_VOL_REGIME_ZSCORE_MAX,
    }
    if overrides:
        params.update(overrides)
    return params


# ── Main backtest function ────────────────────────────────────────────────────

def run_cross_vol_backtest(
    universe: dict[str, AssetVolData],
    asset_a: str,
    asset_b: str,
    start_dt: datetime,
    end_dt: datetime,
    initial_capital: float = INITIAL_CAPITAL,
    params: dict | None = None,
) -> dict:
    """
    Event-driven daily backtest for the cross-asset vol spread strategy.

    Returns:
        equity         — pd.Series (daily USD equity curve)
        trades         — list[CrossVolTrade]
        spread_series  — pd.DataFrame (daily spread/z-score history)
        _equity        — dict[str, pd.Series] for visualize.py compatibility
    """
    asset_a = asset_a.upper()
    asset_b = asset_b.upper()
    p = _resolve_backtest_params(params)

    # Build full aligned series (includes burn-in before start_dt)
    aligned_full = align_assets(universe)
    spread_full  = compute_spread_series(
        aligned_full, asset_a, asset_b, lookback=int(p["spread_lookback"])
    )

    # Regime filter: rolling z-score of spot A/B ratio
    regime_z: pd.Series | None = None
    if CROSS_VOL_REGIME_FILTER:
        spot_a_s = universe[asset_a].spot_daily.reindex(aligned_full.index).ffill()
        spot_b_s = universe[asset_b].spot_daily.reindex(aligned_full.index).ffill()
        regime_z = _build_regime_series(
            spot_a_s, spot_b_s, lookback=int(p["regime_lookback"])
        )

    # Restrict to backtest window
    def _to_utc(dt: datetime) -> pd.Timestamp:
        ts = pd.Timestamp(dt)
        return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")

    start_ts = _to_utc(start_dt)
    end_ts   = _to_utc(end_dt)
    date_range = spread_full[
        (spread_full.index >= start_ts) & (spread_full.index <= end_ts)
    ].index

    if len(date_range) == 0:
        print(f"  [!] No aligned dates in [{start_dt.date()} → {end_dt.date()}]")
        empty = pd.Series([initial_capital], index=[start_ts], name="equity")
        return {"equity": empty, "trades": [], "spread_series": spread_full,
                "_equity": {"BTC/ETH Spread": empty}}

    capital       = initial_capital
    peak_capital  = initial_capital
    equity_vals   = []
    equity_idx    = []
    trades: list[CrossVolTrade] = []

    # Live trade state
    in_trade        = False
    trade_entry: dict = {}

    for ts in date_range:
        row = spread_full.loc[ts]

        # ── If in trade: mark-to-market ──────────────────────────────────────
        if in_trade:
            days_held = (ts - trade_entry["entry_ts"]).days

            # Get current DVOL for each leg
            long_data  = universe[trade_entry["asset_long"]]
            short_data = universe[trade_entry["asset_short"]]

            cur_iv_long    = float(long_data.dvol_daily.get(ts, long_data.dvol_daily.dropna().iloc[-1]))
            cur_spot_long  = float(long_data.spot_daily.get(ts, long_data.spot_daily.dropna().iloc[-1]))
            cur_iv_short   = float(short_data.dvol_daily.get(ts, short_data.dvol_daily.dropna().iloc[-1]))
            cur_spot_short = float(short_data.spot_daily.get(ts, short_data.spot_daily.dropna().iloc[-1]))

            # Cumulative BSM P&L from entry (gross, no costs)
            pnl_long_gross = _leg_pnl(
                +1,
                trade_entry["spot_long"], cur_spot_long,
                trade_entry["iv_long"], cur_iv_long,
                trade_entry["straddle_price_long"], trade_entry["straddle_vega_long"],
                trade_entry["notional_long"],
                trade_entry["dte"], days_held,
            )
            pnl_short_gross = _leg_pnl(
                -1,
                trade_entry["spot_short"], cur_spot_short,
                trade_entry["iv_short"], cur_iv_short,
                trade_entry["straddle_price_short"], trade_entry["straddle_vega_short"],
                trade_entry["notional_short"],
                trade_entry["dte"], days_held,
            )

            # Accumulate daily delta-hedge cost (one day's worth added each bar)
            if CROSS_VOL_DELTA_HEDGE:
                trade_entry["cumulative_hedge_cost"] += HEDGE_COST_PCT * (
                    trade_entry["notional_long"] + trade_entry["notional_short"]
                )

            hedge_so_far = trade_entry["cumulative_hedge_cost"]
            cumulative_pnl = pnl_long_gross + pnl_short_gross - hedge_so_far

            # Stop-loss: use total drawdown from entry as fraction of notional
            total_notional = trade_entry["notional_long"] + trade_entry["notional_short"]
            loss_ratio = -cumulative_pnl / total_notional if total_notional > 0 else 0.0
            stop_triggered = loss_ratio >= STOP_LOSS_PCT

            # Exit signal
            sig = generate_signal(spread_full, ts.to_pydatetime(), in_trade=True,
                                   holding_days=days_held,
                                   zscore_entry=float(p["zscore_entry"]),
                                   zscore_exit=float(p["zscore_exit"]),
                                   min_spread_pts=float(p["spread_min_pts"]),
                                   holding_days_max=int(p["holding_days_max"]))

            if stop_triggered:
                exit_reason = "stop_loss"
            elif sig.exit_triggered:
                exit_reason = (
                    "reversion"
                    if float(row["spread_zscore"]) <= float(p["zscore_exit"])
                    else "max_holding"
                )
            else:
                exit_reason = None

            if exit_reason or days_held >= int(trade_entry["max_holding"]):
                exit_reason = exit_reason or "max_holding"
                exit_comm = _commission(trade_entry["notional_long"], trade_entry["notional_short"])
                net_pnl = cumulative_pnl - exit_comm / 2  # entry comm already deducted
                capital = trade_entry["capital_at_entry"] + net_pnl

                trade = CrossVolTrade(
                    entry_date       = trade_entry["entry_ts"].to_pydatetime(),
                    exit_date        = ts.to_pydatetime(),
                    asset_long       = trade_entry["asset_long"],
                    asset_short      = trade_entry["asset_short"],
                    entry_zscore     = trade_entry["entry_zscore"],
                    exit_zscore      = float(row["spread_zscore"]),
                    entry_spread_pts = trade_entry["entry_spread"],
                    notional_long    = trade_entry["notional_long"],
                    notional_short   = trade_entry["notional_short"],
                    gross_vega_usd   = trade_entry["gross_vega"],
                    pnl_long_usd     = pnl_long_gross - hedge_so_far / 2,
                    pnl_short_usd    = pnl_short_gross - hedge_so_far / 2,
                    total_pnl_usd    = net_pnl,
                    exit_reason      = exit_reason,
                    holding_days     = days_held,
                )
                trades.append(trade)
                in_trade = False
                trade_entry = {}
            else:
                # MTM: record mark-to-market equity without modifying capital
                mtm_equity = trade_entry["capital_at_entry"] + cumulative_pnl
                equity_vals.append(mtm_equity)
                equity_idx.append(ts)
                peak_capital = max(peak_capital, mtm_equity)
                continue  # skip the append at end of loop

        # ── If not in trade: check entry ─────────────────────────────────────
        if not in_trade:
            # Halt new trades if portfolio is in significant drawdown
            drawdown = (peak_capital - capital) / peak_capital if peak_capital > 0 else 0.0
            if drawdown >= MAX_DRAWDOWN_HALT_PCT:
                equity_vals.append(capital)
                equity_idx.append(ts)
                continue

            # Regime gate: skip entry if spot ratio is in a strong trend
            regime_blocked = False
            if CROSS_VOL_REGIME_FILTER and regime_z is not None and ts in regime_z.index:
                rz = abs(float(regime_z.loc[ts]))
                if rz > float(p["regime_zscore_max"]):
                    regime_blocked = True

            sig = generate_signal(
                spread_full,
                ts.to_pydatetime(),
                in_trade=False,
                zscore_entry=float(p["zscore_entry"]),
                zscore_exit=float(p["zscore_exit"]),
                min_spread_pts=float(p["spread_min_pts"]),
                holding_days_max=int(p["holding_days_max"]),
            )
            if sig.entry_triggered and not regime_blocked:
                sizing = compute_vega_neutral_sizing(
                    sig, universe,
                    total_capital_fraction=float(p["capital_fraction"]),
                    initial_capital=capital,
                    dte=int(p["dte_target"]),
                )
                if sizing.approved:
                    # Deduct entry commission
                    entry_comm = _commission(sizing.long_leg.notional_usd, sizing.short_leg.notional_usd)
                    capital -= entry_comm

                    trade_entry = {
                        "entry_ts":             ts,
                        "asset_long":           sig.asset_long,
                        "asset_short":          sig.asset_short,
                        "entry_zscore":         float(row["spread_zscore"]),
                        "entry_spread":         float(row["raw_spread"]),
                        "notional_long":        sizing.long_leg.notional_usd,
                        "notional_short":       sizing.short_leg.notional_usd,
                        "gross_vega":           sizing.gross_vega_usd,
                        "iv_long":              sizing.long_leg.atm_iv_pct,
                        "iv_short":             sizing.short_leg.atm_iv_pct,
                        "spot_long":            sizing.long_leg.spot,
                        "spot_short":           sizing.short_leg.spot,
                        "straddle_price_long":  sizing.long_leg.straddle_price_usd,
                        "straddle_price_short": sizing.short_leg.straddle_price_usd,
                        "straddle_vega_long":   sizing.long_leg.straddle_vega,
                        "straddle_vega_short":  sizing.short_leg.straddle_vega,
                        "dte":                  sizing.long_leg.dte,
                        "max_holding":          int(p["holding_days_max"]),
                        "capital_at_entry":     capital,      # snapshot for MTM equity
                        "cumulative_hedge_cost": 0.0,         # accumulated each bar
                    }
                    in_trade = True

        peak_capital = max(peak_capital, capital)
        equity_vals.append(capital)
        equity_idx.append(ts)

    # Close any open trade at end of period
    if in_trade and equity_idx:
        last_ts = equity_idx[-1]
        days_held = (last_ts - trade_entry["entry_ts"]).days
        long_data  = universe[trade_entry["asset_long"]]
        short_data = universe[trade_entry["asset_short"]]
        cur_iv_long    = float(long_data.dvol_daily.dropna().iloc[-1])
        cur_iv_short   = float(short_data.dvol_daily.dropna().iloc[-1])
        cur_spot_long  = float(long_data.spot_daily.dropna().iloc[-1])
        cur_spot_short = float(short_data.spot_daily.dropna().iloc[-1])

        pnl_long_gross = _leg_pnl(
            +1, trade_entry["spot_long"], cur_spot_long,
            trade_entry["iv_long"], cur_iv_long,
            trade_entry["straddle_price_long"], trade_entry["straddle_vega_long"],
            trade_entry["notional_long"], trade_entry["dte"], days_held,
        )
        pnl_short_gross = _leg_pnl(
            -1, trade_entry["spot_short"], cur_spot_short,
            trade_entry["iv_short"], cur_iv_short,
            trade_entry["straddle_price_short"], trade_entry["straddle_vega_short"],
            trade_entry["notional_short"], trade_entry["dte"], days_held,
        )
        hedge_so_far = trade_entry["cumulative_hedge_cost"]
        exit_comm = _commission(trade_entry["notional_long"], trade_entry["notional_short"])
        final_pnl = pnl_long_gross + pnl_short_gross - hedge_so_far - exit_comm / 2
        trades.append(CrossVolTrade(
            entry_date=trade_entry["entry_ts"].to_pydatetime(),
            exit_date=last_ts.to_pydatetime(),
            asset_long=trade_entry["asset_long"], asset_short=trade_entry["asset_short"],
            entry_zscore=trade_entry["entry_zscore"], exit_zscore=None,
            entry_spread_pts=trade_entry["entry_spread"],
            notional_long=trade_entry["notional_long"], notional_short=trade_entry["notional_short"],
            gross_vega_usd=trade_entry["gross_vega"],
            pnl_long_usd=pnl_long_gross - hedge_so_far / 2,
            pnl_short_usd=pnl_short_gross - hedge_so_far / 2,
            total_pnl_usd=final_pnl,
            exit_reason="end_of_period", holding_days=days_held,
        ))

    equity = pd.Series(equity_vals, index=equity_idx, name="equity")
    pair_label = f"{asset_a}/{asset_b} Spread"
    return {
        "equity":        equity,
        "trades":        trades,
        "spread_series": spread_full,
        "params":        p,
        "_equity":       {pair_label: equity},
    }
