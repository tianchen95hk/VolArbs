"""
option_vol — configuration.

All strategy parameters, thresholds, and risk limits live here.
Edit this file to tune the system without touching any logic.
"""

# ── Asset ──────────────────────────────────────────────────────────────────────
CURRENCY = "BTC"           # "BTC" or "ETH"

# ── Deribit ───────────────────────────────────────────────────────────────────
DERIBIT_BASE  = "https://www.deribit.com/api/v2/public"
LOOKBACK_DAYS = 90         # history window for DVOL + spot price fetch

# ── Cache ─────────────────────────────────────────────────────────────────────
CACHE_FILE    = ".cache_option_vol.pkl"
CACHE_TTL_MIN = 60         # minutes before cache is considered stale

# ── Analytics ─────────────────────────────────────────────────────────────────
RISK_FREE_RATE    = 0.05   # annualised (decimal) used in BSM
RV_WINDOW_DAYS    = 30     # close-to-close RV lookback
EWMA_LAMBDA       = 0.94   # decay factor for EWMA vol (RiskMetrics)
ATM_MONEYNESS_PCT = 5.0    # |strike/spot − 1| × 100 < this → "ATM"
NEAR_DTE_MIN      = 7      # minimum DTE for near-term expiry band
NEAR_DTE_MAX      = 30     # maximum DTE for near-term expiry band
FAR_DTE_MIN       = 45     # minimum DTE for far-term expiry band
DELTA_25D_BAND    = (0.20, 0.30)  # delta range defining "25-delta" options

# ── Signal thresholds ─────────────────────────────────────────────────────────
VOL_PREMIUM_THRESHOLD   = 5.0    # |ATM IV − RV| in vol points to trigger
SKEW_THRESHOLD          = 4.0    # |put_iv_25d − call_iv_25d| threshold
TERM_SLOPE_THRESHOLD    = 3.0    # |back_IV − front_IV| threshold
PARITY_THRESHOLD_PCT    = 0.50   # % of spot price for parity violation net edge

# ── Portfolio / risk limits ───────────────────────────────────────────────────
INITIAL_CAPITAL       = 100_000.0   # USD
POSITION_SIZE_PCT     = 0.10        # fraction of capital per trade
MAX_OPEN_POSITIONS    = 5
MAX_NET_DELTA         = 0.10        # max |net portfolio delta| as fraction of capital
MAX_VEGA_NOTIONAL     = 0.25        # max total vega exposure as fraction of capital
MAX_GAMMA_NOTIONAL    = 0.05        # max total gamma exposure as fraction of capital
STOP_LOSS_PCT         = 0.30        # close position if MTM loss > 30% of allocated capital
MAX_DRAWDOWN_HALT_PCT = 0.20        # halt new trades if portfolio drawdown exceeds 20%

# ── Trading costs ─────────────────────────────────────────────────────────────
COMMISSION_PCT  = 0.0003   # 0.03% Deribit options taker fee (per BTC notional)
HEDGE_COST_PCT  = 0.001    # 0.1% per BTC-PERPETUAL hedge transaction

# ── Backtest ──────────────────────────────────────────────────────────────────
HEDGE_DELTA = True         # delta-hedge positions at each daily bar
BACKTEST_PERIODS = {
    "FY25":     ("2025-01-01", "2025-12-31"),
    "FY26_YTD": ("2026-01-01", "2026-03-27"),
}
