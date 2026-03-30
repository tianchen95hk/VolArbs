"""
cross_vol/alpha_factors.py — Multi-factor realized-vol signal construction.

All factors are computed from daily OHLCV (Binance), so they work for
any large-cap crypto — no implied-vol data required.

Factors
-------
F1  Parkinson RV          — high/low range estimator (5× more efficient than C2C)
F2  Yang-Zhang RV         — handles overnight gaps, most efficient (8× C2C)
F3  EWMA vol              — exponentially-weighted, reacts quickly to spikes
F4  RV term-structure     — RV7 / RV30 ratio (vol spike vs baseline)
F5  Vol trend             — gradient of rolling RV (expanding vs contracting)

Composite score per asset
-------------------------
    score = Σ wᵢ × zᵢ   where z = rolling z-score of each factor

    When combined across two assets → cross-sectional spread signal:
        spread = score_A − score_B
        z_spread = rolling z-score of spread (the entry/exit signal)

This z_spread is a multi-factor, more robust version of the current
single-factor VP spread used in spread_signal.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ── Individual factor estimators ──────────────────────────────────────────────

def rv_parkinson(klines: pd.DataFrame, window: int = 30) -> pd.Series:
    """
    Parkinson (1980) realized volatility — uses High-Low range.

    σ² = 1/(4N·ln2) × Σ [ln(Hᵢ/Lᵢ)]²

    ~5× more efficient than close-to-close. Annualised %, rolling window.
    """
    hl = np.log(klines["high"] / klines["low"]) ** 2
    var = hl.rolling(window, min_periods=max(3, window // 3)).mean() / (4 * np.log(2))
    return (np.sqrt(var * 252) * 100).rename(f"rv_parkinson_{window}d")


def rv_yang_zhang(klines: pd.DataFrame, window: int = 30) -> pd.Series:
    """
    Yang-Zhang (2000) realized volatility — handles overnight gaps.

    Uses open-close, close-open (overnight), and Parkinson intraday components.
    ~8× more efficient than close-to-close. Annualised %, rolling window.
    """
    k = 0.34 / (1.34 + (window + 1) / (window - 1))

    # Overnight return: close-to-open
    log_oc = np.log(klines["open"] / klines["close"].shift(1))
    sig2_oc = log_oc.rolling(window, min_periods=max(3, window // 3)).var()

    # Intraday (Rogers-Satchell component)
    log_ho = np.log(klines["high"] / klines["open"])
    log_lo = np.log(klines["low"]  / klines["open"])
    log_hc = np.log(klines["high"] / klines["close"])
    log_lc = np.log(klines["low"]  / klines["close"])
    rs = (log_ho * log_hc + log_lo * log_lc)
    sig2_rs = rs.rolling(window, min_periods=max(3, window // 3)).mean()

    # Close-to-close
    log_cc = np.log(klines["close"] / klines["close"].shift(1))
    sig2_cc = log_cc.rolling(window, min_periods=max(3, window // 3)).var()

    yz_var = sig2_oc + k * sig2_cc + (1 - k) * sig2_rs
    yz_var = yz_var.clip(lower=0)
    return (np.sqrt(yz_var * 252) * 100).rename(f"rv_yz_{window}d")


def rv_ewma(klines: pd.DataFrame, span: int = 10) -> pd.Series:
    """
    Exponentially-weighted moving average volatility.

    Faster response to recent vol regime changes.
    RiskMetrics-style: λ = 1 - 2/(span+1). Annualised %, no window bias.
    """
    log_ret = np.log(klines["close"] / klines["close"].shift(1)).dropna()
    ewma_var = log_ret.ewm(span=span, adjust=False).var()
    rv = (np.sqrt(ewma_var * 252) * 100).reindex(klines.index)
    return rv.rename(f"rv_ewma_{span}d")


def rv_term_structure(klines: pd.DataFrame,
                      short_w: int = 7,
                      long_w: int = 30) -> pd.Series:
    """
    Vol term-structure ratio: RV(short) / RV(long).

    > 1  → vol spike (short-term vol above baseline) → IV typically elevated → sell
    < 1  → vol compression → IV typically cheap → buy
    """
    rv_s = rv_parkinson(klines, window=short_w)
    rv_l = rv_parkinson(klines, window=long_w)
    ratio = (rv_s / rv_l.replace(0, np.nan)).rename(f"rv_ts_{short_w}vs{long_w}")
    return ratio


def vol_trend(klines: pd.DataFrame, window: int = 14, diff_lag: int = 5) -> pd.Series:
    """
    First-difference of rolling RV — captures vol expansion (> 0) vs
    contraction (< 0). Useful as a confirmation signal.
    """
    rv = rv_parkinson(klines, window=window)
    trend = rv.diff(diff_lag).rename(f"vol_trend_{window}d")
    return trend


# ── Cross-sectional z-scoring ─────────────────────────────────────────────────

def rolling_zscore(series: pd.Series, window: int = 60) -> pd.Series:
    """
    Rolling z-score with burn-in:
        - First `window` bars: expanding window (no lookahead)
        - After `window` bars: fixed rolling window
    """
    roll_mean = series.rolling(window, min_periods=window // 2).mean()
    roll_std  = series.rolling(window, min_periods=window // 2).std()
    z = (series - roll_mean) / roll_std.replace(0, np.nan)
    return z.rename(f"{series.name}_z")


# ── Composite asset score ─────────────────────────────────────────────────────

# Factor weights — tuned for vol-spread mean-reversion alpha
_FACTOR_WEIGHTS = {
    "yz30":    0.40,   # Yang-Zhang RV30 — most efficient baseline
    "ts_7_30": 0.30,   # Term-structure ratio — catches IV response lag
    "ewma10":  0.20,   # EWMA — current regime speed
    "trend14": 0.10,   # Vol expansion/contraction direction
}

def compute_asset_score(
    klines: pd.DataFrame,
    dvol_series: pd.Series | None = None,
    zscore_window: int = 60,
    weights: dict | None = None,
) -> pd.Series:
    """
    Compute a composite vol-richness score for one asset.

    Higher score → IV is likely elevated vs fair value → vol is EXPENSIVE (sell)
    Lower score  → IV is likely cheap vs fair value     → vol is CHEAP    (buy)

    If dvol_series is provided (Deribit DVOL), the score also incorporates the
    true implied vol premium (IV − RV), which improves signal quality.

    Returns daily pd.Series aligned to klines.index.
    """
    w = weights or _FACTOR_WEIGHTS
    factors: dict[str, pd.Series] = {}

    # Factor 1: Yang-Zhang RV30 level (higher RV → IV likely elevated)
    factors["yz30"]    = rv_yang_zhang(klines, window=30)

    # Factor 2: Term-structure ratio RV7/RV30 (spike vs baseline)
    factors["ts_7_30"] = rv_term_structure(klines, short_w=7, long_w=30)

    # Factor 3: EWMA vol (current regime)
    factors["ewma10"]  = rv_ewma(klines, span=10)

    # Factor 4: Vol trend (expansion = +, contraction = −)
    factors["trend14"] = vol_trend(klines, window=14)

    # Optional Factor 5: DVOL − RV30 (true vol premium, highest quality)
    if dvol_series is not None and len(dvol_series.dropna()) > 10:
        rv30 = rv_parkinson(klines, window=30).reindex(dvol_series.index).ffill()
        dvol_aligned = dvol_series.reindex(rv30.index).ffill()
        vp = (dvol_aligned - rv30).rename("true_vp")
        factors["true_vp"] = vp
        # Boost DVOL-based VP weight, reduce others
        w = {"yz30": 0.15, "ts_7_30": 0.20, "ewma10": 0.15, "trend14": 0.10,
             "true_vp": 0.40}

    # Z-score each factor and compute weighted composite
    zscores = {name: rolling_zscore(f, window=zscore_window)
               for name, f in factors.items() if name in w}

    # Align all z-scores to common index
    idx = None
    for z in zscores.values():
        z_idx = z.dropna().index
        idx = z_idx if idx is None else idx.intersection(z_idx)

    if idx is None or len(idx) == 0:
        return pd.Series(dtype=float, name="composite_score")

    composite = sum(w[n] * zscores[n].reindex(idx) for n in zscores)
    if not isinstance(composite, pd.Series):
        composite = pd.Series(composite, index=idx)
    composite.name = "composite_score"
    return composite.sort_index()


# ── Pair spread signal ────────────────────────────────────────────────────────

def compute_pair_spread(
    score_a: pd.Series,
    score_b: pd.Series,
    zscore_window: int = 60,
) -> pd.DataFrame:
    """
    Given composite scores for two assets, compute the spread and its z-score.

    Returns DataFrame with columns:
        score_a, score_b, spread (a − b), z_spread, signal_strength
    """
    idx = score_a.index.intersection(score_b.index)
    sa  = score_a.reindex(idx)
    sb  = score_b.reindex(idx)

    spread     = (sa - sb).rename("spread")
    z_spread   = rolling_zscore(spread, window=zscore_window).rename("z_spread")

    # Signal strength: |z| normalised — useful for proportional sizing
    sig_strength = z_spread.abs().clip(upper=3.0) / 3.0  # 0-1 scale

    return pd.DataFrame({
        "score_a":        sa,
        "score_b":        sb,
        "spread":         spread,
        "z_spread":       z_spread,
        "signal_strength": sig_strength,
    })


# ── Factor agreement (confirmation filter) ───────────────────────────────────

def factor_agreement(
    klines_a: pd.DataFrame,
    klines_b: pd.DataFrame,
    dvol_a: pd.Series | None = None,
    dvol_b: pd.Series | None = None,
    zscore_window: int = 60,
    as_of: pd.Timestamp | None = None,
) -> dict:
    """
    Compute individual factor z-scores for both assets and check alignment.

    Returns a dict with:
        - Per-factor z-scores at latest bar (or as_of date)
        - agreement_score: fraction of factors pointing same direction (0-1)
        - direction: +1 (A expensive) or −1 (B expensive)
        - recommended_size_mult: 0.5–1.5× based on agreement
    """
    factors = {
        "yz30":    (rv_yang_zhang, {"window": 30}),
        "ts_7_30": (rv_term_structure, {"short_w": 7, "long_w": 30}),
        "ewma10":  (rv_ewma, {"span": 10}),
    }

    def _score_at(klines, dvol=None):
        scores = {}
        for name, (fn, kwargs) in factors.items():
            s = fn(klines, **kwargs)
            z = rolling_zscore(s, window=zscore_window)
            val = float(z.dropna().iloc[-1]) if as_of is None else float(z.reindex([as_of]).iloc[0]) if as_of in z.index else float("nan")
            scores[name] = val
        if dvol is not None and len(dvol.dropna()) > 10:
            rv = rv_parkinson(klines, window=30).reindex(dvol.index).ffill()
            vp = dvol.reindex(rv.index).ffill() - rv
            z_vp = rolling_zscore(vp, window=zscore_window)
            val = float(z_vp.dropna().iloc[-1]) if as_of is None else float(z_vp.reindex([as_of]).iloc[0]) if as_of in z_vp.index else float("nan")
            scores["true_vp"] = val
        return scores

    sa = _score_at(klines_a, dvol_a)
    sb = _score_at(klines_b, dvol_b)

    # Spread of each factor: positive = A expensive
    factor_spreads = {
        name: sa.get(name, float("nan")) - sb.get(name, float("nan"))
        for name in set(sa) | set(sb)
    }

    # Fraction of factors pointing same direction
    valid = [(v > 0) for v in factor_spreads.values() if not np.isnan(v)]
    if not valid:
        return {"agreement_score": 0.0, "direction": 0, "factor_spreads": factor_spreads,
                "recommended_size_mult": 0.5}

    agree_pos = sum(valid) / len(valid)
    agree_neg = 1.0 - agree_pos

    if agree_pos >= agree_neg:
        direction = +1
        agreement = agree_pos
    else:
        direction = -1
        agreement = agree_neg

    # Size multiplier: 0.5× at 50% agreement (random) → 1.5× at 100% (all aligned)
    size_mult = 0.5 + agreement

    return {
        "agreement_score":       agreement,
        "direction":             direction,
        "factor_spreads":        factor_spreads,
        "recommended_size_mult": round(size_mult, 2),
        "scores_a":              sa,
        "scores_b":              sb,
    }
