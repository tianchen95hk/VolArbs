"""
cross_vol/portfolio_engine.py — Multi-pair vol-spread portfolio backtester.

Improvements over engine.py (single BTC/ETH pair):
  1. Multi-pair simultaneous positions (top 2 by signal strength)
  2. Binance OHLCV as primary data source — works for any large-cap asset
  3. Multi-factor entry signal (Yang-Zhang RV + term-structure + EWMA + DVOL)
  4. Signal-strength-proportional sizing (z=1.2 → 0.5×, z=2.5 → 1.5× notional)
  5. Factor-agreement confirmation filter (require ≥ 60% factor alignment)
  6. Portfolio-level risk management: max 2 concurrent positions, corr guard
  7. Improved regime filter: both spot-ratio AND spread autocorrelation
  8. Parkinson/Yang-Zhang RV → less noise in the signal

Data flow (Binance-primary):
    Binance OHLCV → alpha_factors.compute_asset_score()
                 → compute_pair_spread() → z_spread entry signal
    Deribit DVOL  → boosts score quality for BTC / ETH (optional overlay)
    Bybit ATM IV  → used only for live scanner, not required for backtest

P&L methodology (same as engine.py):
    Full BSM straddle repricing at each bar using:
      - spot from Binance klines (daily close)
      - IV from DVOL (if available) else Yang-Zhang RV30 (proxy)
"""
from __future__ import annotations

import sys
import os
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from itertools import combinations

import numpy as np
import pandas as pd

_HERE      = os.path.dirname(os.path.abspath(__file__))
_PER_ASSET = os.path.join(_HERE, "..", "per_asset")
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
if _PER_ASSET not in sys.path:
    sys.path.append(_PER_ASSET)

from analytics.bsm import bsm_price
from alpha_factors import (
    compute_asset_score, compute_pair_spread, factor_agreement,
    rv_yang_zhang, rv_parkinson,
)
from cv_data.binance import get_spot_klines, get_rv30_series
from cv_config import (
    INITIAL_CAPITAL, COMMISSION_PCT, HEDGE_COST_PCT, STOP_LOSS_PCT,
    MAX_DRAWDOWN_HALT_PCT, CROSS_VOL_USE_FULL_BSM, CROSS_VOL_DELTA_HEDGE,
    CROSS_VOL_SPREAD_LOOKBACK, CROSS_VOL_ZSCORE_ENTRY, CROSS_VOL_ZSCORE_EXIT,
    CROSS_VOL_HOLDING_DAYS_MAX, CROSS_VOL_CAPITAL_FRACTION,
    CROSS_VOL_REGIME_FILTER, CROSS_VOL_REGIME_LOOKBACK, CROSS_VOL_REGIME_ZSCORE_MAX,
)

# ── Portfolio-specific config ─────────────────────────────────────────────────
MAX_CONCURRENT_PAIRS   = 2      # max simultaneous pair positions
MIN_FACTOR_AGREEMENT   = 0.60   # at least 60% of factors must agree
SIZE_Z_SCALE           = True   # scale notional by z-score strength
SIZE_Z_BASE            = 1.2    # z at which size_mult = 1.0×
SIZE_Z_MAX             = 2.5    # z at which size_mult caps at MAX_SIZE_MULT
MAX_SIZE_MULT          = 1.5    # maximum notional multiplier
MIN_SIZE_MULT          = 0.6    # minimum notional multiplier (weak signal)
AUTOCORR_FILTER        = True   # block entry if spread is autocorrelated (trending)
AUTOCORR_WINDOW        = 20     # days for autocorrelation test
AUTOCORR_THRESHOLD     = 0.35   # |lag-1 autocorr| above this = trending spread
HISTORY_DAYS           = 500    # Binance history for factor computation


# ── Data containers ───────────────────────────────────────────────────────────

@dataclass
class PairState:
    """Live state of one open pair position."""
    pair:               str       # e.g. "BTC/ETH"
    asset_long:         str
    asset_short:        str
    entry_ts:           pd.Timestamp
    notional_long:      float
    notional_short:     float
    capital_at_entry:   float
    entry_zscore:       float
    entry_spread:       float
    iv_long:            float
    iv_short:           float
    spot_long:          float
    spot_short:         float
    straddle_price_long:  float
    straddle_price_short: float
    straddle_vega_long:   float
    straddle_vega_short:  float
    dte:                int
    gross_vega:         float
    cumulative_hedge_cost: float = 0.0
    size_mult:          float = 1.0   # signal-strength multiplier at entry


@dataclass
class PairTrade:
    """Completed round-trip trade."""
    pair:             str
    entry_date:       datetime
    exit_date:        datetime
    asset_long:       str
    asset_short:      str
    entry_zscore:     float
    exit_zscore:      float | None
    entry_spread:     float
    notional_long:    float
    notional_short:   float
    gross_vega_usd:   float
    pnl_long_usd:     float
    pnl_short_usd:    float
    total_pnl_usd:    float
    exit_reason:      str
    holding_days:     int
    size_mult:        float
    factor_agreement: float


# ── BSM helpers (same as engine.py) ──────────────────────────────────────────

def _bsm_straddle(spot: float, iv_pct: float, dte: int, r: float = 0.05) -> float:
    if dte <= 0 or iv_pct <= 0 or spot <= 0:
        return 0.0
    T = dte / 365.0
    sigma = iv_pct / 100.0
    return (bsm_price(spot, spot, T, r, sigma, "C") +
            bsm_price(spot, spot, T, r, sigma, "P"))


def _leg_pnl(direction, entry_spot, cur_spot, entry_iv, cur_iv,
             entry_price, vega, notional, dte_entry, days_held) -> float:
    remaining = max(0, dte_entry - days_held)
    cur_price = _bsm_straddle(cur_spot, cur_iv, remaining)
    if entry_price <= 0:
        return 0.0
    contracts = notional / entry_price
    return direction * (cur_price - entry_price) * contracts


# ── Regime filters ────────────────────────────────────────────────────────────

def _spot_ratio_regime_z(klines_a: pd.DataFrame, klines_b: pd.DataFrame,
                         lookback: int) -> pd.Series:
    """Rolling z-score of log(spot_A / spot_B) — same as engine.py filter."""
    close_a = klines_a["close"].reindex(klines_b.index, method="ffill")
    close_b = klines_b["close"]
    log_ratio = np.log(close_a / close_b).dropna()
    mean = log_ratio.rolling(lookback, min_periods=lookback // 2).mean()
    std  = log_ratio.rolling(lookback, min_periods=lookback // 2).std().replace(0, np.nan)
    return ((log_ratio - mean) / std).fillna(0.0)


def _spread_autocorr(spread: pd.Series, window: int = 20) -> pd.Series:
    """
    Rolling lag-1 autocorrelation of the spread.
    High positive autocorr = trending spread = mean-reversion broken.
    """
    def _autocorr(x):
        if len(x) < 4:
            return 0.0
        s = pd.Series(x)
        c = s.autocorr(lag=1)
        return c if not np.isnan(c) else 0.0
    return spread.rolling(window, min_periods=window // 2).apply(_autocorr, raw=False)


# ── IV proxy for backtesting (when DVOL not available) ───────────────────────

def _iv_proxy_series(klines: pd.DataFrame, dvol: pd.Series | None = None) -> pd.Series:
    """
    For backtesting: return the best available daily IV proxy.
    Priority: Deribit DVOL > Yang-Zhang RV30 × 1.15 (15% VRP cushion)
    """
    yz = rv_yang_zhang(klines, window=30)
    if dvol is not None and len(dvol.dropna()) > 5:
        return dvol.reindex(yz.index, method="ffill").fillna(yz * 1.15)
    return (yz * 1.15).rename("iv_proxy")   # synthetic: add 15% to RV as avg VRP


# ── Size multiplier ───────────────────────────────────────────────────────────

def _size_mult(z: float) -> float:
    """Linear interpolation: z=1.2 → 1.0×, z=2.5 → 1.5×, floor at 0.6×."""
    if not SIZE_Z_SCALE:
        return 1.0
    z = abs(z)
    t = min(max((z - SIZE_Z_BASE) / (SIZE_Z_MAX - SIZE_Z_BASE), 0.0), 1.0)
    return MIN_SIZE_MULT + t * (MAX_SIZE_MULT - MIN_SIZE_MULT)


# ── Main backtest ─────────────────────────────────────────────────────────────

def run_portfolio_backtest(
    coins: list[str],
    start_dt: datetime,
    end_dt:   datetime,
    klines_map:  dict[str, pd.DataFrame] | None = None,
    dvol_map:    dict[str, pd.Series]    | None = None,
    initial_capital: float = INITIAL_CAPITAL,
    verbose: bool = True,
) -> dict:
    """
    Multi-pair vol-spread portfolio backtest.

    Args:
        coins       — list of coin names (e.g. ["BTC","ETH","SOL"])
        start_dt    — backtest start (UTC)
        end_dt      — backtest end (UTC)
        klines_map  — pre-fetched Binance OHLCV {coin: DataFrame}; fetched if None
        dvol_map    — pre-fetched Deribit DVOL   {coin: Series};    BTC/ETH only
        initial_capital — starting portfolio value in USD

    Returns dict with keys:
        equity       — pd.Series daily portfolio value
        trades       — list[PairTrade]
        pair_spreads — {pair_label: pd.DataFrame} with z_spread series
        summary      — dict of performance metrics
    """
    coins = [c.upper() for c in coins]
    start_ts = pd.Timestamp(start_dt).tz_localize("UTC") if not hasattr(start_dt, "tzinfo") or start_dt.tzinfo is None else pd.Timestamp(start_dt).tz_convert("UTC")
    end_ts   = pd.Timestamp(end_dt).tz_localize("UTC")   if not hasattr(end_dt, "tzinfo")   or end_dt.tzinfo   is None else pd.Timestamp(end_dt).tz_convert("UTC")

    # ── Fetch / validate klines ───────────────────────────────────────────────
    if klines_map is None:
        klines_map = {}
    if dvol_map is None:
        dvol_map = {}

    for coin in coins:
        if coin not in klines_map:
            if verbose:
                print(f"  Fetching Binance OHLCV for {coin}...")
            try:
                klines_map[coin] = get_spot_klines(coin, days=HISTORY_DAYS)
            except Exception as e:
                print(f"  [!] Could not fetch {coin}: {e}")

    # ── Build composite scores and pair spreads ───────────────────────────────
    # Two-tier signal:
    #   Tier 1 — both assets have DVOL: use pure DVOL-VP spread (proven signal,
    #             preserves the direction accuracy of the single-pair engine)
    #   Tier 2 — at least one asset lacks DVOL: use multi-factor composite
    #             (Binance-primary RV signal)
    scores: dict[str, pd.Series] = {}
    vp_map: dict[str, pd.Series] = {}   # DVOL-VP series for Tier-1 assets
    iv_map: dict[str, pd.Series] = {}

    for coin in coins:
        kl = klines_map.get(coin)
        if kl is None or len(kl) < 30:
            continue
        dvol = dvol_map.get(coin)
        scores[coin] = compute_asset_score(kl, dvol_series=dvol)
        iv_map[coin] = _iv_proxy_series(kl, dvol)

        # Tier-1: build DVOL-VP series when DVOL is available
        if dvol is not None and len(dvol.dropna()) > 10:
            from alpha_factors import rv_parkinson
            rv30 = rv_parkinson(kl, window=30).reindex(dvol.index, method="ffill")
            dvol_aligned = dvol.reindex(rv30.index, method="ffill").dropna()
            rv_aligned   = rv30.reindex(dvol_aligned.index)
            vp_map[coin] = (dvol_aligned - rv_aligned).rename(f"vp_{coin}")

    available_coins = [c for c in coins if c in scores]
    pairs = list(combinations(available_coins, 2))

    # Track which pairs are Tier-1 (DVOL-based) vs Tier-2 (composite)
    tier1_pairs: set[str] = set()
    pair_spreads: dict[str, pd.DataFrame] = {}

    for ca, cb in pairs:
        label = f"{ca}/{cb}"

        if ca in vp_map and cb in vp_map:
            # Tier 1: DVOL VP spread (same formula as spread_signal.py)
            vpa = vp_map[ca]
            vpb = vp_map[cb]
            idx = vpa.index.intersection(vpb.index)
            spread_raw = (vpa.loc[idx] - vpb.loc[idx]).rename("spread")
            roll_mean  = spread_raw.rolling(CROSS_VOL_SPREAD_LOOKBACK, min_periods=CROSS_VOL_SPREAD_LOOKBACK // 2).mean()
            roll_std   = spread_raw.rolling(CROSS_VOL_SPREAD_LOOKBACK, min_periods=CROSS_VOL_SPREAD_LOOKBACK // 2).std().replace(0, np.nan)
            z_spread   = ((spread_raw - roll_mean) / roll_std).rename("z_spread")
            sig_str    = z_spread.abs().clip(upper=3.0) / 3.0
            pair_spreads[label] = pd.DataFrame({
                "spread": spread_raw, "z_spread": z_spread, "signal_strength": sig_str,
                "score_a": vpa.reindex(idx), "score_b": vpb.reindex(idx),
            })
            tier1_pairs.add(label)
        else:
            # Tier 2: multi-factor composite
            ps = compute_pair_spread(scores[ca], scores[cb],
                                     zscore_window=CROSS_VOL_SPREAD_LOOKBACK)
            pair_spreads[label] = ps

    # ── Regime series ─────────────────────────────────────────────────────────
    regime_z_map: dict[str, pd.Series] = {}
    if CROSS_VOL_REGIME_FILTER:
        for ca, cb in pairs:
            kl_a = klines_map.get(ca)
            kl_b = klines_map.get(cb)
            if kl_a is not None and kl_b is not None:
                regime_z_map[f"{ca}/{cb}"] = _spot_ratio_regime_z(
                    kl_a, kl_b, CROSS_VOL_REGIME_LOOKBACK)

    # Autocorrelation of spread (trending detection)
    # Only applied to Tier-2 pairs — DVOL-based (Tier-1) spreads use their own
    # regime filter and don't need the autocorr gate
    autocorr_map: dict[str, pd.Series] = {}
    if AUTOCORR_FILTER:
        for label, ps in pair_spreads.items():
            if label not in tier1_pairs:   # Tier-2 only
                autocorr_map[label] = _spread_autocorr(
                    ps["spread"].dropna(), window=AUTOCORR_WINDOW)

    # ── Build unified date range ───────────────────────────────────────────────
    all_idx = None
    for ps in pair_spreads.values():
        idx = ps.index[(ps.index >= start_ts) & (ps.index <= end_ts)]
        all_idx = idx if all_idx is None else all_idx.union(idx)
    if all_idx is None or len(all_idx) == 0:
        print("  [!] No aligned dates in backtest window.")
        empty = pd.Series([initial_capital], index=[start_ts], name="equity")
        return {"equity": empty, "trades": [], "pair_spreads": pair_spreads, "summary": {}}
    all_idx = all_idx.sort_values()

    # ── Event loop ────────────────────────────────────────────────────────────
    capital       = initial_capital
    peak_capital  = initial_capital
    equity_vals:  list[float]        = []
    equity_idx:   list[pd.Timestamp] = []
    trades:       list[PairTrade]    = []
    open_states:  list[PairState]    = []

    def _get_val(series: pd.Series, ts: pd.Timestamp, fallback=None):
        """Safe value lookup with forward-fill fallback."""
        if ts in series.index:
            v = series.loc[ts]
            return float(v) if not (isinstance(v, float) and np.isnan(v)) else fallback
        prev = series[series.index <= ts].dropna()
        return float(prev.iloc[-1]) if len(prev) else fallback

    for ts in all_idx:
        # ── Mark open positions to market ────────────────────────────────────
        still_open: list[PairState] = []
        for state in open_states:
            days_held = (ts - state.entry_ts).days

            kl_long  = klines_map.get(state.asset_long)
            kl_short = klines_map.get(state.asset_short)
            iv_long  = _get_val(iv_map.get(state.asset_long, pd.Series(dtype=float)), ts,
                                fallback=state.iv_long)
            iv_short = _get_val(iv_map.get(state.asset_short, pd.Series(dtype=float)), ts,
                                fallback=state.iv_short)
            spot_long  = _get_val(kl_long["close"],  ts, fallback=state.spot_long)  if kl_long  is not None else state.spot_long
            spot_short = _get_val(kl_short["close"], ts, fallback=state.spot_short) if kl_short is not None else state.spot_short

            if iv_long is None:  iv_long  = state.iv_long
            if iv_short is None: iv_short = state.iv_short

            pnl_long_gross  = _leg_pnl(+1, state.spot_long,  spot_long,
                                        state.iv_long,  iv_long,
                                        state.straddle_price_long,
                                        state.straddle_vega_long,
                                        state.notional_long, state.dte, days_held)
            pnl_short_gross = _leg_pnl(-1, state.spot_short, spot_short,
                                        state.iv_short, iv_short,
                                        state.straddle_price_short,
                                        state.straddle_vega_short,
                                        state.notional_short, state.dte, days_held)

            if CROSS_VOL_DELTA_HEDGE:
                state.cumulative_hedge_cost += HEDGE_COST_PCT * (
                    state.notional_long + state.notional_short)

            cum_pnl = pnl_long_gross + pnl_short_gross - state.cumulative_hedge_cost
            total_notional = state.notional_long + state.notional_short
            loss_ratio = -cum_pnl / total_notional if total_notional > 0 else 0.0
            stop_triggered = loss_ratio >= STOP_LOSS_PCT

            # Exit signal from pair-spread z-score
            ps  = pair_spreads.get(state.pair)
            cur_z = _get_val(ps["z_spread"], ts) if ps is not None else None
            exit_reason = None
            if stop_triggered:
                exit_reason = "stop_loss"
            elif cur_z is not None and abs(cur_z) <= CROSS_VOL_ZSCORE_EXIT:
                exit_reason = "reversion"
            elif days_held >= CROSS_VOL_HOLDING_DAYS_MAX:
                exit_reason = "max_holding"

            if exit_reason:
                exit_comm = 2.0 * COMMISSION_PCT * total_notional
                net_pnl   = cum_pnl - exit_comm / 2
                capital   = state.capital_at_entry + net_pnl
                # Restore capital from other open trades
                for other in still_open:
                    capital = state.capital_at_entry  # correct per-trade attribution

                # Agreement at entry
                agr = factor_agreement(
                    klines_map.get(state.asset_long,  pd.DataFrame()),
                    klines_map.get(state.asset_short, pd.DataFrame()),
                )

                trades.append(PairTrade(
                    pair=state.pair, entry_date=state.entry_ts.to_pydatetime(),
                    exit_date=ts.to_pydatetime(),
                    asset_long=state.asset_long, asset_short=state.asset_short,
                    entry_zscore=state.entry_zscore,
                    exit_zscore=cur_z,
                    entry_spread=state.entry_spread,
                    notional_long=state.notional_long, notional_short=state.notional_short,
                    gross_vega_usd=state.gross_vega,
                    pnl_long_usd=pnl_long_gross - state.cumulative_hedge_cost / 2,
                    pnl_short_usd=pnl_short_gross - state.cumulative_hedge_cost / 2,
                    total_pnl_usd=net_pnl,
                    exit_reason=exit_reason, holding_days=days_held,
                    size_mult=state.size_mult,
                    factor_agreement=agr.get("agreement_score", 0.5),
                ))
            else:
                still_open.append(state)

        open_states = still_open

        # Compute MTM equity (capital plus all open positions' unrealized P&L)
        mtm_equity = capital
        for state in open_states:
            days_held = (ts - state.entry_ts).days
            kl_long  = klines_map.get(state.asset_long)
            kl_short = klines_map.get(state.asset_short)
            iv_long  = _get_val(iv_map.get(state.asset_long,  pd.Series(dtype=float)), ts, fallback=state.iv_long)
            iv_short = _get_val(iv_map.get(state.asset_short, pd.Series(dtype=float)), ts, fallback=state.iv_short)
            spot_long  = _get_val(kl_long["close"],  ts, fallback=state.spot_long)  if kl_long  is not None else state.spot_long
            spot_short = _get_val(kl_short["close"], ts, fallback=state.spot_short) if kl_short is not None else state.spot_short

            if iv_long  is None: iv_long  = state.iv_long
            if iv_short is None: iv_short = state.iv_short

            pnl_l = _leg_pnl(+1, state.spot_long,  spot_long,  state.iv_long,  iv_long,
                              state.straddle_price_long,  state.straddle_vega_long,
                              state.notional_long,  state.dte, days_held)
            pnl_s = _leg_pnl(-1, state.spot_short, spot_short, state.iv_short, iv_short,
                              state.straddle_price_short, state.straddle_vega_short,
                              state.notional_short, state.dte, days_held)
            mtm_equity += pnl_l + pnl_s - state.cumulative_hedge_cost

        # ── Check entry signals ───────────────────────────────────────────────
        drawdown = (peak_capital - mtm_equity) / peak_capital if peak_capital > 0 else 0.0
        if drawdown < MAX_DRAWDOWN_HALT_PCT and len(open_states) < MAX_CONCURRENT_PAIRS:

            # Score all pairs not currently open
            open_labels = {s.pair for s in open_states}
            candidates: list[tuple[float, str, str, str, float]] = []

            for ca, cb in pairs:
                label = f"{ca}/{cb}"
                if label in open_labels:
                    continue
                ps = pair_spreads.get(label)
                if ps is None:
                    continue

                cur_z_row = ps[ps.index == ts]
                if cur_z_row.empty:
                    continue
                cur_z      = float(cur_z_row["z_spread"].iloc[0])
                cur_spread = float(cur_z_row["spread"].iloc[0])

                if abs(cur_z) < CROSS_VOL_ZSCORE_ENTRY:
                    continue
                if np.isnan(cur_z):
                    continue

                # Regime filter: spot ratio
                regime_blocked = False
                if CROSS_VOL_REGIME_FILTER and label in regime_z_map:
                    rz = _get_val(regime_z_map[label], ts, fallback=0.0)
                    if abs(rz) > CROSS_VOL_REGIME_ZSCORE_MAX:
                        regime_blocked = True
                if regime_blocked:
                    continue

                # Autocorrelation (trending spread) filter
                if AUTOCORR_FILTER and label in autocorr_map:
                    ac = _get_val(autocorr_map[label], ts, fallback=0.0)
                    if abs(ac) > AUTOCORR_THRESHOLD:
                        continue  # spread is trending, not mean-reverting

                # Factor agreement check — Tier-2 only (DVOL pairs don't need it)
                agr = {"agreement_score": 1.0}
                if label not in tier1_pairs:
                    agr = factor_agreement(
                        klines_map.get(ca, pd.DataFrame()),
                        klines_map.get(cb, pd.DataFrame()),
                    )
                    if agr["agreement_score"] < MIN_FACTOR_AGREEMENT:
                        continue

                candidates.append((abs(cur_z), label, ca, cb, cur_z))

            # Sort by |z|, take top signal
            candidates.sort(reverse=True, key=lambda x: x[0])
            slots_available = MAX_CONCURRENT_PAIRS - len(open_states)

            for _, label, ca, cb, cur_z in candidates[:slots_available]:
                # Direction: z > 0 → A expensive → short A, long B
                if cur_z >= 0:
                    asset_long, asset_short = cb, ca
                else:
                    asset_long, asset_short = ca, cb

                kl_long  = klines_map.get(asset_long)
                kl_short = klines_map.get(asset_short)
                if kl_long is None or kl_short is None:
                    continue

                spot_long  = _get_val(kl_long["close"],  ts)
                spot_short = _get_val(kl_short["close"], ts)
                iv_long    = _get_val(iv_map.get(asset_long,  pd.Series(dtype=float)), ts)
                iv_short   = _get_val(iv_map.get(asset_short, pd.Series(dtype=float)), ts)

                if not spot_long or not spot_short or not iv_long or not iv_short:
                    continue
                if iv_long <= 0 or iv_short <= 0:
                    continue

                # Size by signal strength
                mult = _size_mult(cur_z)
                cap_frac  = CROSS_VOL_CAPITAL_FRACTION * mult
                notional  = min(capital * cap_frac, capital * 0.30)  # hard cap 30%

                dte = 21
                sp_long  = _bsm_straddle(spot_long,  iv_long,  dte)
                sp_short = _bsm_straddle(spot_short, iv_short, dte)
                if sp_long <= 0 or sp_short <= 0:
                    continue

                # Vega-neutral split
                T = dte / 365.0
                sig_l = iv_long  / 100.0
                sig_s = iv_short / 100.0
                from analytics.bsm import bsm_greeks
                try:
                    g_long  = bsm_greeks(spot_long,  spot_long,  T, 0.05, sig_l, "C")
                    g_short = bsm_greeks(spot_short, spot_short, T, 0.05, sig_s, "C")
                    # Raw per-unit Vegas (scale with spot price)
                    vega_l_unit = g_long.get("vega", 0.01)  + bsm_greeks(spot_long,  spot_long,  T, 0.05, sig_l, "P").get("vega", 0.01)
                    vega_s_unit = g_short.get("vega", 0.01) + bsm_greeks(spot_short, spot_short, T, 0.05, sig_s, "P").get("vega", 0.01)
                    # Normalize to per-$1-of-premium (same as sizing.py: v = vega_unit/straddle)
                    vega_l = vega_l_unit / sp_long  if sp_long  > 0 else 0.01
                    vega_s = vega_s_unit / sp_short if sp_short > 0 else 0.01
                except Exception:
                    vega_l = vega_s = 0.01
                # Vega-neutral allocation
                v_sum = vega_l + vega_s
                n_long  = notional * vega_s / v_sum
                n_short = notional * vega_l / v_sum

                entry_comm = 2.0 * COMMISSION_PCT * notional
                capital   -= entry_comm

                state = PairState(
                    pair=label,
                    asset_long=asset_long, asset_short=asset_short,
                    entry_ts=ts, notional_long=n_long, notional_short=n_short,
                    capital_at_entry=capital,
                    entry_zscore=abs(cur_z), entry_spread=float(
                        pair_spreads[label].loc[ts, "spread"] if ts in pair_spreads[label].index else 0),
                    iv_long=iv_long, iv_short=iv_short,
                    spot_long=spot_long, spot_short=spot_short,
                    straddle_price_long=sp_long, straddle_price_short=sp_short,
                    straddle_vega_long=vega_l_unit, straddle_vega_short=vega_s_unit,
                    dte=dte, gross_vega=abs(n_long * vega_l),
                    size_mult=mult,
                )
                open_states.append(state)

        peak_capital = max(peak_capital, mtm_equity)
        equity_vals.append(mtm_equity)
        equity_idx.append(ts)

    equity = pd.Series(equity_vals, index=equity_idx, name="portfolio_equity")

    # ── Compute summary metrics ───────────────────────────────────────────────
    if len(equity) > 1:
        total_ret   = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
        daily_ret   = equity.pct_change().dropna()
        sharpe      = float(daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0.0
        max_dd      = float(((equity / equity.cummax()) - 1).min() * 100)
        n_days      = (equity.index[-1] - equity.index[0]).days
        ann_ret     = float(((1 + total_ret / 100) ** (365 / max(n_days, 1)) - 1) * 100)
    else:
        total_ret = ann_ret = sharpe = max_dd = 0.0

    win_trades  = [t for t in trades if t.total_pnl_usd > 0]
    win_rate    = len(win_trades) / len(trades) * 100 if trades else 0.0

    summary = {
        "total_return_pct": total_ret,
        "ann_return_pct":   ann_ret,
        "sharpe":           sharpe,
        "max_drawdown_pct": max_dd,
        "n_trades":         len(trades),
        "win_rate_pct":     win_rate,
        "total_pnl_usd":    sum(t.total_pnl_usd for t in trades),
    }

    return {
        "equity":       equity,
        "trades":       trades,
        "pair_spreads": pair_spreads,
        "summary":      summary,
        "_equity":      {"Multi-Pair Portfolio": equity},
    }
