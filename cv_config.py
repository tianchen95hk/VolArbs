"""
Cross-asset vol spread strategy configuration.

Shared parameters (capital, commissions, risk limits) are loaded from
per_asset/config.py via importlib to avoid the naming conflict between
cross_vol/config.py and per_asset/config.py on sys.path.
"""
import sys
import os
import importlib.util

_HERE      = os.path.dirname(os.path.abspath(__file__))
_PER_ASSET = os.path.join(_HERE, "..", "per_asset")

# Load per_asset/config.py explicitly by file path (no sys.path collision)
_spec = importlib.util.spec_from_file_location(
    "per_asset_config", os.path.join(_PER_ASSET, "config.py")
)
_pa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pa)

INITIAL_CAPITAL      = _pa.INITIAL_CAPITAL
COMMISSION_PCT       = _pa.COMMISSION_PCT
HEDGE_COST_PCT       = _pa.HEDGE_COST_PCT
STOP_LOSS_PCT        = _pa.STOP_LOSS_PCT
MAX_DRAWDOWN_HALT_PCT= _pa.MAX_DRAWDOWN_HALT_PCT
RISK_FREE_RATE       = _pa.RISK_FREE_RATE
DERIBIT_BASE         = _pa.DERIBIT_BASE
BACKTEST_PERIODS     = _pa.BACKTEST_PERIODS

# ── Universe ─────────────────────────────────────────────────────────────────
# Both assets must have a Deribit DVOL index (Tier 1 quality)
CROSS_VOL_ASSETS = ["BTC", "ETH"]

# ── Signal ───────────────────────────────────────────────────────────────────
CROSS_VOL_SPREAD_LOOKBACK  = 75    # rolling z-score window (days)
CROSS_VOL_ZSCORE_ENTRY     = 1.2   # |z| >= threshold to enter  (was 1.5)
CROSS_VOL_ZSCORE_EXIT      = 0.4   # |z| <= threshold to exit
CROSS_VOL_SPREAD_MIN_PTS   = 3.0   # absolute floor (vol pts)   (was 5.0)
CROSS_VOL_HOLDING_DAYS_MAX = 14    # force-exit after this many days
CROSS_VOL_DTE_TARGET       = 21    # target DTE for synthetic ATM straddle

# ── Sizing / risk ─────────────────────────────────────────────────────────────
CROSS_VOL_CAPITAL_FRACTION  = 0.24  # fraction of capital allocated per pair
CROSS_VOL_MAX_GROSS_VEGA    = 0.30  # cap: |gross_vega_usd| / capital
CROSS_VOL_MAX_CONCENTRATION = 0.70  # cap: one leg's share of gross vega

# ── Data ─────────────────────────────────────────────────────────────────────
CROSS_VOL_LOOKBACK_DAYS  = 500  # DVOL + spot history depth (covers FY25 + burn-in)
CROSS_VOL_MAX_GAP_HOURS  = 4    # forward-fill threshold; warn beyond this

# ── Regime filter ────────────────────────────────────────────────────────────
# Block new entries when the BTC/ETH spot ratio is in a strong trend.
# A trending price ratio means vol spreads are likely to trend too, breaking
# the mean-reversion assumption.
CROSS_VOL_REGIME_FILTER     = True   # enable / disable
CROSS_VOL_REGIME_LOOKBACK   = 30     # days for ratio z-score rolling window
CROSS_VOL_REGIME_ZSCORE_MAX = 1.2    # block entry if |ratio_z| > this

# ── Engine ────────────────────────────────────────────────────────────────────
CROSS_VOL_USE_FULL_BSM = True   # True  → full BSM straddle repricing each bar
                                # False → vega-approximation (faster, less accurate)
CROSS_VOL_DELTA_HEDGE  = True   # charge HEDGE_COST_PCT per day when in trade

# ── Parameter pool (single-pair optimizer) ───────────────────────────────────
# Grid is intentionally mid-range focused to reduce overfit to one period while
# still exploring meaningfully different trade frequencies and holding styles.
CROSS_VOL_PARAM_POOL = {
    "spread_lookback":   [60, 75],
    "zscore_entry":      [1.1, 1.2, 1.35, 1.5],
    "zscore_exit":       [0.2, 0.3, 0.4],
    "spread_min_pts":    [2.0, 3.0, 4.0],
    "holding_days_max":  [14, 21, 28],
    "capital_fraction":  [0.12, 0.16, 0.20, 0.24],
    "dte_target":        [14, 21, 28],
    "regime_lookback":   [20, 30, 45],
    "regime_zscore_max": [1.2, 1.5],
}

# Practical limits for optimizer runtime / robustness.
CROSS_VOL_OPT_MAX_TRIALS = 180   # sampled combinations per run
CROSS_VOL_OPT_MIN_TRADES = 6     # penalise parameter sets below this avg trade count

# ── Portfolio engine (multi-pair, Binance-primary) ────────────────────────────
# Coins for the extended universe; Binance OHLCV is used for all of them.
# BTC/ETH also get Deribit DVOL overlay for higher signal quality.
PORTFOLIO_COINS           = ["BTC", "ETH", "SOL", "XRP", "DOGE"]
PORTFOLIO_MAX_PAIRS       = 2       # max simultaneous pair positions
PORTFOLIO_HISTORY_DAYS    = 500     # Binance OHLCV history depth

# Multi-factor entry gate: require ≥ this fraction of factors to agree on direction
MIN_FACTOR_AGREEMENT      = 0.60    # 0.0 = disabled, 1.0 = unanimous required

# Dynamic sizing: scale notional by z-score strength
SIZE_Z_SCALE              = True    # enable proportional sizing
SIZE_Z_BASE               = 1.2    # z at which mult = 1.0× (base notional)
SIZE_Z_MAX_THRESH         = 2.5    # z at which mult caps at MAX_SIZE_MULT
MAX_SIZE_MULT             = 1.5    # maximum multiplier
MIN_SIZE_MULT             = 0.6    # minimum multiplier (weak-signal trades)

# Autocorrelation filter: block entry when spread is trending (not mean-reverting)
AUTOCORR_FILTER           = True
AUTOCORR_WINDOW           = 20     # rolling window for lag-1 autocorr
AUTOCORR_THRESHOLD        = 0.35   # |autocorr| > this → spread is trending
