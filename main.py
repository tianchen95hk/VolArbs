"""
option_vol — Crypto Options Arbitrage Scanner & Backtester

Usage
-----
  python main.py                       # single live scan (BTC, fetches fresh data)
  python main.py --scan                # same as above
  python main.py --no-fetch            # scan using cached data (faster iteration)
  python main.py --backtest            # run all strategies over BACKTEST_PERIODS
  python main.py --backtest --plot     # save equity and drawdown charts
  python main.py --backtest --fy25     # FY2025 only
  python main.py --backtest --fy26     # FY2026 YTD only
  python main.py --backtest --days 60  # custom rolling lookback from today
  python main.py --continuous          # repeat scan every 60 minutes
  python main.py --currency ETH        # scan ETH instead of BTC

Signals
-------
  1. Vol Premium      — ATM IV vs 30d realized vol (vol carry)
  2. Skew             — 25-delta put/call IV imbalance
  3. Term Structure   — Calendar spread / IV term slope vs RV slope
  4. Put-Call Parity  — Parity violations after transaction costs
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta, timezone

import numpy as np

from config import CURRENCY, BACKTEST_PERIODS
from data.cache import fetch_with_cache
from analytics.surface import parse_chain
from signals.vol_premium    import analyse_vol_premium
from signals.skew           import analyse_skew
from signals.term_structure import analyse_term_structure
from signals.parity         import analyse_parity
from scanner import rank_opportunities, print_opportunity


# ── Market context header ──────────────────────────────────────────────────────

def _print_header(currency: str, spot: float, dvol_df, spot_history) -> None:
    from analytics.rv import realised_vol_cc, realised_vol_ewma

    dvol = float(dvol_df["close"].iloc[-1]) if not dvol_df.empty else float("nan")
    prices = spot_history["mark_price"].dropna()
    rv30  = realised_vol_cc(prices, 30)
    rv7   = realised_vol_cc(prices, 7)
    ewma  = realised_vol_ewma(prices)
    premium = dvol - rv30 if not (np.isnan(dvol) or np.isnan(rv30)) else float("nan")

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print()
    print("=" * 72)
    print(f"  option_vol  |  {currency} Options Arbitrage Scanner  |  {now}")
    print("=" * 72)
    print(f"  Spot price   : ${spot:>12,.0f}")
    print(f"  DVOL (ATM IV): {dvol:>8.1f}%   (Deribit 30d IV index)")
    print(f"  RV 30d       : {rv30:>8.1f}%   RV 7d: {rv7:.1f}%   EWMA: {ewma:.1f}%")
    print(f"  Vol premium  : {premium:>+8.1f} vol pts  (DVOL − RV30d)")
    print("─" * 72)


# ── Scan ──────────────────────────────────────────────────────────────────────

def run_scan(currency: str = CURRENCY, use_cache: bool = False) -> None:
    """Fetch data, run all signals, print ranked opportunities."""

    data = fetch_with_cache(currency=currency, force_refresh=not use_cache)

    options_df   = data["options_df"]
    dvol_df      = data["dvol_df"]
    spot_history = data["spot_history"]
    futures_df   = data["futures_df"]
    spot         = data["spot_price"]

    # ── [2/4] Analytics ───────────────────────────────────────────────────────
    print(f"\n[analytics] Parsing options chain ...")
    chain = parse_chain(options_df, spot)
    print(f"  {len(chain)} contracts parsed  "
          f"({chain['expiry'].nunique() if not chain.empty else 0} expiries)")

    # ── [3/4] Signals ─────────────────────────────────────────────────────────
    print("[signals] Computing ...")

    vol_sig   = analyse_vol_premium(chain, spot_history, dvol_df)
    skew_sigs = analyse_skew(chain, spot)
    term_sig  = analyse_term_structure(chain, spot_history, spot)
    par_sigs  = analyse_parity(chain, spot, futures_df)

    sig_counts = {
        "vol_premium":    1 if vol_sig and vol_sig.direction != "neutral" else 0,
        "skew":           len(skew_sigs),
        "term_structure": 1 if term_sig and term_sig.direction != "neutral" else 0,
        "parity":         len(par_sigs),
    }
    print(f"  {sig_counts}")

    # ── [4/4] Rank & display ──────────────────────────────────────────────────
    opps = rank_opportunities(
        vol_premium=vol_sig,
        skew=skew_sigs,
        term_structure=term_sig,
        parity=par_sigs,
        spot=spot,
    )

    _print_header(currency, spot, dvol_df, spot_history)

    if not opps:
        print("\n  No actionable opportunities above thresholds at this time.")
        print("  All signals within normal ranges — monitoring mode.\n")
    else:
        print(f"\n  {len(opps)} opportunity{'s' if len(opps) != 1 else ''} found:\n")
        for opp in opps:
            print_opportunity(opp)

    print()


# ── Backtest ──────────────────────────────────────────────────────────────────

def run_backtest(
    currency: str = CURRENCY,
    period_keys: list[str] | None = None,
    days: int | None = None,
    multi_days: list[int] | None = None,
    plot: bool = False,
    plot_dir: str = "backtest/charts",
) -> None:
    """Run all strategy backtests and print performance table."""
    from backtest.engine import run_backtest as _run
    from backtest.report import print_backtest_report
    from backtest.visualize import save_backtest_charts

    data = fetch_with_cache(currency=currency, force_refresh=True)
    dvol_df      = data["dvol_df"]
    spot_history = data["spot_history"]

    if multi_days:
        now = datetime.now(timezone.utc)
        periods = {
            f"Rolling {d}d": (now - timedelta(days=d), now)
            for d in multi_days
        }
    elif days is not None:
        periods = {
            f"Rolling {days}d": (
                datetime.now(timezone.utc) - timedelta(days=days),
                datetime.now(timezone.utc),
            )
        }
    else:
        keys = period_keys or list(BACKTEST_PERIODS.keys())
        periods = {}
        for k in keys:
            if k not in BACKTEST_PERIODS:
                print(f"  [warn] Unknown period key: {k}")
                continue
            s, e = BACKTEST_PERIODS[k]
            periods[k] = (
                datetime.fromisoformat(s).replace(tzinfo=timezone.utc),
                datetime.fromisoformat(e).replace(tzinfo=timezone.utc),
            )

    all_results: dict[str, dict] = {}

    for label, (start_dt, end_dt) in periods.items():
        print(f"\n[backtest] {label}  {start_dt.date()} → {end_dt.date()}")
        stats = _run(dvol_df, spot_history, start_dt, end_dt)
        equity_curves = stats.pop("_equity", {})
        all_results[label] = stats
        print_backtest_report(stats, period_label=label)

        if plot and equity_curves:
            saved = save_backtest_charts(
                equity_curves=equity_curves,
                period_label=label,
                output_dir=plot_dir,
            )
            if saved:
                print(f"  [plot] Charts saved for {label}:")
                for name, path in saved.items():
                    print(f"    - {name}: {path}")

    # Note on Tier 2 approximate strategies
    print("  * Skew Arb and Term Struct are Tier 2 approximate backtests using")
    print("    BSM-reconstructed data. Results are indicative, not historical.\n")


# ── Continuous mode ───────────────────────────────────────────────────────────

def run_continuous(currency: str = CURRENCY, interval_min: int = 60) -> None:
    """Repeat scan every interval_min minutes."""
    print(f"[continuous] Scanning {currency} every {interval_min} min. Ctrl-C to stop.")
    while True:
        try:
            run_scan(currency=currency, use_cache=False)
            print(f"  Next scan in {interval_min} min ...")
            time.sleep(interval_min * 60)
        except KeyboardInterrupt:
            print("\n[continuous] Stopped.")
            break


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Crypto options arbitrage scanner and backtester.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--scan",       action="store_true", help="Run live scan (default)")
    p.add_argument("--no-fetch",   action="store_true", help="Use cached data")
    p.add_argument("--backtest",   action="store_true", help="Run historical backtest")
    p.add_argument("--fy25",       action="store_true", help="Backtest FY2025 only")
    p.add_argument("--fy26",       action="store_true", help="Backtest FY2026 YTD only")
    p.add_argument("--days",       type=int,            help="Rolling lookback in days")
    p.add_argument("--plot",       action="store_true", help="Save charts in backtest mode")
    p.add_argument("--plot-dir",   type=str, default="backtest/charts",
                   help="Output directory for backtest charts")
    p.add_argument("--continuous", action="store_true", help="Repeat scan on interval")
    p.add_argument("--interval",   type=int, default=60, help="Interval in minutes (default 60)")
    p.add_argument("--currency",   type=str, default=CURRENCY, help="BTC or ETH")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    currency = args.currency.upper()

    if args.continuous:
        run_continuous(currency=currency, interval_min=args.interval)
    elif args.backtest:
        period_keys = None
        if args.fy25:
            period_keys = ["FY25"]
        elif args.fy26:
            period_keys = ["FY26_YTD"]
        run_backtest(
            currency=currency,
            period_keys=period_keys,
            days=args.days,
            plot=args.plot,
            plot_dir=args.plot_dir,
        )
    else:
        run_scan(currency=currency, use_cache=args.no_fetch)


if __name__ == "__main__":
    main()
