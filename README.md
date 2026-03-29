# VolArbs

VolArbs is a production-oriented crypto options volatility arbitrage scanner and backtester for Deribit BTC/ETH markets.

## What It Does

- Scans live Deribit options data and ranks opportunities
- Combines 4 signal families:
  - Vol Premium (IV vs RV)
  - 25-delta Skew
  - Term Structure
  - Put-Call Parity
- Runs historical backtests with strategy-level performance reports
- Exports equity/drawdown visualizations for quick review

## Quick Start

```bash
pip install -r requirements.txt
python3 main.py --scan
python3 main.py --backtest --days 60 --plot
```

## Project Layout

- `main.py` — CLI entrypoint for scan/backtest/continuous modes
- `signals/` — signal logic
- `analytics/` — RV, surface, BSM/parity calculations
- `backtest/` — engine, reports, and chart generation
- `data/` — Deribit API client and local cache

## Notes

- Designed for research and execution support, not autonomous order routing.
- Tier 2 backtests (`Skew Arb*`, `Term Struct*`) are approximate by design.
