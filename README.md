# VolArbs (cross_vol)

`cross_vol` is a cross-asset volatility spread research engine for crypto options.

It focuses on **relative value in vol premium** between assets (currently BTC/ETH), with:
- live spread scanning
- event-driven daily backtests
- multi-pair portfolio backtests
- parameter-pool optimization with run logging

## Core Idea

For each asset:

`vol_premium = implied_vol_proxy - realized_vol`

For a pair `(A, B)`:

`spread = vol_premium_A - vol_premium_B`

Trade mean reversion in spread z-score:
- when spread is rich, short rich vol / long cheap vol
- exit on reversion, max holding, or stop-loss

## Main Components

- `main.py`: CLI entrypoint (`scan`, `backtest`, `portfolio`, `optimize`)
- `universe.py`: fetch + preprocess DVOL/spot/RV data
- `spread_signal.py`: spread series + entry/exit signal logic
- `sizing.py`: vega-neutral sizing and risk-limit checks
- `engine.py`: single-pair event-driven backtest engine
- `portfolio_engine.py`: multi-pair portfolio backtester
- `alpha_factors.py`: multi-factor scoring for extended universe
- `param_optimizer.py`: parameter search, scoring, and optimization logs
- `cv_config.py`: strategy defaults and parameter pools

## Quick Start

```bash
# Live scan (default pair from config)
python3 main.py

# Single-pair backtest
python3 main.py --backtest --assets BTC ETH --days 90

# Multi-pair portfolio backtest
python3 main.py --portfolio --coins BTC ETH SOL XRP DOGE --fy25

# Parameter optimization (auto-saves JSON/CSV/TXT logs)
python3 main.py --optimize --assets BTC ETH --opt-trials 180 --opt-top 10 --opt-log-dir opt_logs
```

## Current Default Strategy Parameters

Defined in `cv_config.py`:
- `CROSS_VOL_SPREAD_LOOKBACK = 75`
- `CROSS_VOL_ZSCORE_ENTRY = 1.2`
- `CROSS_VOL_ZSCORE_EXIT = 0.4`
- `CROSS_VOL_SPREAD_MIN_PTS = 3.0`
- `CROSS_VOL_HOLDING_DAYS_MAX = 14`
- `CROSS_VOL_CAPITAL_FRACTION = 0.24`
- `CROSS_VOL_DTE_TARGET = 21`
- `CROSS_VOL_REGIME_LOOKBACK = 30`
- `CROSS_VOL_REGIME_ZSCORE_MAX = 1.2`

## Optimization Logging

Each `--optimize` run is recorded to `opt_logs/<PAIR>_<UTC_TIMESTAMP>/`:
- `ranked_all.csv`: all evaluated parameter sets
- `ranked_top.csv`: top-N rows
- `summary.json`: run config, selected params, per-period validation
- `summary.txt`: human-readable summary

## Notes

- This folder depends on sibling `per_asset/` modules (analytics, data, backtest utils).
- Market data quality and API availability can materially affect results.
- Strategy outputs are for research and decision support, not autonomous execution.
