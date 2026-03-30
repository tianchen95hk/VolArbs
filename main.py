"""
cross_vol/main.py — CLI entry point for the cross-asset vol spread strategy.

Usage:
    python main.py                        # live scan: BTC/ETH spread signal
    python main.py --scan                 # multi-asset opportunity scanner
    python main.py --scan --coins BTC ETH SOL XRP DOGE
    python main.py --backtest             # all BACKTEST_PERIODS
    python main.py --optimize             # optimize parameter pool on all periods
    python main.py --optimize --fy25      # optimize on FY2025 only
    python main.py --optimize --opt-apply-best --days 180
    python main.py --optimize --opt-log-dir opt_logs
    python main.py --backtest --fy25      # FY2025 only
    python main.py --backtest --fy26      # FY2026 YTD only
    python main.py --backtest --days 90   # rolling 90-day window
    python main.py --assets BTC ETH       # explicit asset pair
    python main.py --no-fetch             # use cached market data
    python main.py --backtest --plot      # save equity charts to charts/
"""
from __future__ import annotations

import argparse
import sys
import os
from datetime import datetime, timedelta, timezone

_PER_ASSET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "per_asset")
if _PER_ASSET not in sys.path:
    sys.path.insert(0, _PER_ASSET)

from cv_config import (
    CROSS_VOL_ASSETS,
    BACKTEST_PERIODS,
    PORTFOLIO_COINS,
)
from universe import fetch_universe, align_assets
from spread_signal import compute_spread_series, generate_signal, compute_live_signal
from engine import run_cross_vol_backtest
from report import compute_attribution, print_cross_vol_report, print_trade_log


# ── Portfolio backtest (multi-pair, Binance-primary) ─────────────────────────

def _run_portfolio_backtest(
    coins: list[str],
    period_keys: list[str] | None,
    days: int | None,
    trade_log: bool = False,
) -> None:
    """Run multi-pair alpha portfolio and print report."""
    from portfolio_engine import run_portfolio_backtest
    from report_portfolio import print_portfolio_report, print_portfolio_trade_log
    from cv_data.binance import get_spot_klines
    from cv_config import PORTFOLIO_HISTORY_DAYS

    import warnings
    now = datetime.now(timezone.utc)

    # Build periods
    if days is not None:
        periods = {f"Rolling {days}d": (now - timedelta(days=days), now)}
    elif period_keys:
        periods = {k: (
            datetime.fromisoformat(BACKTEST_PERIODS[k][0]).replace(tzinfo=timezone.utc),
            datetime.fromisoformat(BACKTEST_PERIODS[k][1]).replace(tzinfo=timezone.utc),
        ) for k in period_keys if k in BACKTEST_PERIODS}
    else:
        periods = {k: (
            datetime.fromisoformat(v[0]).replace(tzinfo=timezone.utc),
            datetime.fromisoformat(v[1]).replace(tzinfo=timezone.utc),
        ) for k, v in BACKTEST_PERIODS.items()}

    # Pre-fetch Binance OHLCV and Deribit DVOL once (reuse across periods)
    print(f"\nFetching Binance OHLCV for: {', '.join(coins)} ...")
    klines_map: dict = {}
    for coin in coins:
        try:
            klines_map[coin] = get_spot_klines(coin, days=PORTFOLIO_HISTORY_DAYS)
            print(f"  {coin}: {len(klines_map[coin])} bars")
        except Exception as e:
            print(f"  {coin}: fetch error — {e}")

    dvol_map: dict = {}
    dvol_coins = [c for c in coins if c in ("BTC", "ETH")]
    if dvol_coins:
        print(f"\nFetching Deribit DVOL overlay for: {', '.join(dvol_coins)} ...")
        from data.deribit import get_dvol_history
        for coin in dvol_coins:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    df = get_dvol_history(coin, days=PORTFOLIO_HISTORY_DAYS, resolution=86400)
                col = "dvol" if "dvol" in df.columns else df.columns[0]
                dvol_map[coin] = df[col].dropna()
                print(f"  {coin}: {len(dvol_map[coin])} DVOL bars")
            except Exception as e:
                print(f"  {coin} DVOL: {e}")

    for label, (start_dt, end_dt) in periods.items():
        print(f"\n[portfolio] {'/'.join(coins)}  {label}  "
              f"{start_dt.date()} → {end_dt.date()}")
        print("  Running multi-pair portfolio backtest ...")
        try:
            results = run_portfolio_backtest(
                coins=coins,
                start_dt=start_dt,
                end_dt=end_dt,
                klines_map={k: v.copy() for k, v in klines_map.items()},
                dvol_map=dvol_map.copy(),
                verbose=False,
            )
            print_portfolio_report(results, label, coins)
            if trade_log:
                print_portfolio_trade_log(results.get("trades", []))
        except Exception as e:
            import traceback
            print(f"  [!] Error: {e}")
            traceback.print_exc()


# ── Live scan ─────────────────────────────────────────────────────────────────

def run_scan(assets: list[str], use_cache: bool) -> None:
    """Fetch live data and print current spread signal."""
    universe = fetch_universe(currencies=assets, force_refresh=not use_cache)

    print(f"\n{'═'*68}")
    print(f"  cross_vol  |  Vol Spread Scanner  |  "
          f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'═'*68}")

    for i in range(len(assets)):
        for j in range(i + 1, len(assets)):
            a, b = assets[i].upper(), assets[j].upper()
            try:
                sig = compute_live_signal(universe, a, b)
                vp_a = float(universe[a].vol_premium.dropna().iloc[-1])
                vp_b = float(universe[b].vol_premium.dropna().iloc[-1])
                print(f"\n  {a} vol premium: {vp_a:+.1f} vol pts  "
                      f"{b} vol premium: {vp_b:+.1f} vol pts")
                print(f"  Spread ({a}−{b}): {sig.raw_spread:.1f} vol pts  "
                      f"z-score: {sig.spread_zscore:.2f}")
                print(f"  Signal: {sig.description}")
                if sig.action_items:
                    print("  Actions:")
                    for item in sig.action_items:
                        print(f"    • {item}")
            except Exception as e:
                print(f"  [{a}/{b}] Error: {e}")

    print()


# ── Backtest ──────────────────────────────────────────────────────────────────

def run_backtest(
    assets: list[str],
    period_keys: list[str] | None,
    days: int | None,
    use_cache: bool,
    plot: bool,
    plot_dir: str,
    trade_log: bool = False,
) -> None:
    """Run the cross-vol backtest and print results."""
    universe = fetch_universe(currencies=assets, force_refresh=not use_cache)

    # Build period dict
    now = datetime.now(timezone.utc)
    if days is not None:
        periods = {f"Rolling {days}d": (now - timedelta(days=days), now)}
    elif period_keys:
        periods = {k: (
            datetime.fromisoformat(BACKTEST_PERIODS[k][0]).replace(tzinfo=timezone.utc),
            datetime.fromisoformat(BACKTEST_PERIODS[k][1]).replace(tzinfo=timezone.utc),
        ) for k in period_keys if k in BACKTEST_PERIODS}
    else:
        periods = {k: (
            datetime.fromisoformat(v[0]).replace(tzinfo=timezone.utc),
            datetime.fromisoformat(v[1]).replace(tzinfo=timezone.utc),
        ) for k, v in BACKTEST_PERIODS.items()}

    # Run one pair at a time (extendable to multiple pairs)
    for i in range(len(assets)):
        for j in range(i + 1, len(assets)):
            a, b = assets[i].upper(), assets[j].upper()

            for label, (start_dt, end_dt) in periods.items():
                print(f"\n[cross-vol] {a}/{b}  {label}  "
                      f"{start_dt.date()} → {end_dt.date()}")
                print("  Running vol spread backtest ...")

                try:
                    results     = run_cross_vol_backtest(universe, a, b, start_dt, end_dt)
                    attribution = compute_attribution(results["trades"], results["equity"])
                    print_cross_vol_report(results, attribution, label, a, b)
                    if trade_log:
                        print_trade_log(results["trades"], results.get("spread_series"))

                    if plot and results["equity"] is not None:
                        try:
                            sys.path.insert(0, _PER_ASSET)
                            from backtest.visualize import save_backtest_charts
                            os.makedirs(plot_dir, exist_ok=True)
                            saved = save_backtest_charts(
                                equity_curves=results["_equity"],
                                period_label=f"CrossVol_{a}{b}_{label.replace(' ', '_')}",
                                output_dir=plot_dir,
                            )
                            for name, path in saved.items():
                                print(f"  Chart saved: {path}")
                        except Exception as e:
                            print(f"  [!] Chart generation failed: {e}")

                except ValueError as e:
                    print(f"  [!] Skipped: {e}")


def run_parameter_optimization(
    assets: list[str],
    period_keys: list[str] | None,
    days: int | None,
    use_cache: bool,
    max_trials: int,
    top_n: int,
    seed: int,
    log_dir: str,
    apply_best: bool = False,
    trade_log: bool = False,
) -> None:
    """
    Search the configured parameter pool and print top-ranked parameter sets.

    Optionally re-run the backtest with the best parameters and print full report.
    """
    from param_optimizer import (
        optimize_cross_vol_params,
        extract_param_dict,
        derive_refined_pool,
        evaluate_params_by_period,
        save_optimization_run,
    )

    universe = fetch_universe(currencies=assets, force_refresh=not use_cache)

    now = datetime.now(timezone.utc)
    if days is not None:
        periods = {f"Rolling {days}d": (now - timedelta(days=days), now)}
    elif period_keys:
        periods = {k: (
            datetime.fromisoformat(BACKTEST_PERIODS[k][0]).replace(tzinfo=timezone.utc),
            datetime.fromisoformat(BACKTEST_PERIODS[k][1]).replace(tzinfo=timezone.utc),
        ) for k in period_keys if k in BACKTEST_PERIODS}
    else:
        periods = {k: (
            datetime.fromisoformat(v[0]).replace(tzinfo=timezone.utc),
            datetime.fromisoformat(v[1]).replace(tzinfo=timezone.utc),
        ) for k, v in BACKTEST_PERIODS.items()}

    for i in range(len(assets)):
        for j in range(i + 1, len(assets)):
            a, b = assets[i].upper(), assets[j].upper()
            print(f"\n[optimize] {a}/{b}  trials={max_trials}  seed={seed}")
            print("  Searching parameter pool ...")

            ranked = optimize_cross_vol_params(
                universe=universe,
                asset_a=a,
                asset_b=b,
                periods=periods,
                max_trials=max_trials,
                seed=seed,
            )

            if ranked.empty:
                print("  [!] No valid parameter set found.")
                continue

            cols = [
                "score", "mean_sharpe", "mean_ann_ret_pct", "worst_total_ret_pct",
                "worst_max_dd_pct", "avg_trade_win_pct", "avg_trades",
                "spread_lookback", "zscore_entry", "zscore_exit", "spread_min_pts",
                "holding_days_max", "capital_fraction", "dte_target",
                "regime_lookback", "regime_zscore_max",
            ]
            show = ranked[cols].head(top_n).copy()
            pd_options = {
                "display.max_columns": None,
                "display.width": 220,
                "display.float_format": "{:,.3f}".format,
            }
            import pandas as _pd
            with _pd.option_context(
                "display.max_columns", pd_options["display.max_columns"],
                "display.width", pd_options["display.width"],
                "display.float_format", pd_options["display.float_format"],
            ):
                print("\n  Top parameter sets:")
                print(show.to_string(index=False))

            refined = derive_refined_pool(ranked, top_n=top_n)
            if refined:
                print("\n  Suggested refined parameter pool:")
                for k, v in refined.items():
                    print(f"    {k}: {v}")

            # Persist optimization process for reproducibility / audit trail.
            best = extract_param_dict(ranked.iloc[0])
            best_period_metrics = evaluate_params_by_period(
                universe=universe,
                asset_a=a,
                asset_b=b,
                periods=periods,
                params=best,
            )
            saved = save_optimization_run(
                log_root=log_dir,
                asset_a=a,
                asset_b=b,
                periods=periods,
                run_config={
                    "max_trials": int(max_trials),
                    "top_n": int(top_n),
                    "seed": int(seed),
                    "use_cache": bool(use_cache),
                    "selection_rule": "best_score",
                },
                ranked=ranked,
                top_n=top_n,
                best_params=best,
                best_period_metrics=best_period_metrics,
                refined_pool=refined,
            )
            print(f"\n  Optimization log saved: {saved['run_dir']}")
            print(f"  Summary JSON: {saved['summary_json']}")
            print(f"  Ranked CSV  : {saved['ranked_csv']}")

            if apply_best:
                print("\n  Applying best parameter set to detailed backtest report:")
                for label, (start_dt, end_dt) in periods.items():
                    print(f"\n[cross-vol] {a}/{b}  {label}  "
                          f"{start_dt.date()} → {end_dt.date()}  (best params)")
                    results = run_cross_vol_backtest(
                        universe, a, b, start_dt, end_dt, params=best
                    )
                    attribution = compute_attribution(results["trades"], results["equity"])
                    print_cross_vol_report(results, attribution, label, a, b)
                    if trade_log:
                        print_trade_log(results["trades"], results.get("spread_series"))


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cross-asset vol spread strategy — scanner and backtester"
    )
    p.add_argument("--backtest",  action="store_true", help="Run backtest")
    p.add_argument("--scan",      action="store_true",
                   help="Run multi-asset vol scanner (BTC/ETH/SOL/XRP/DOGE)")
    p.add_argument("--portfolio", action="store_true",
                   help="Run multi-pair portfolio backtest (Binance-primary alpha engine)")
    p.add_argument("--optimize",  action="store_true",
                   help="Optimize single-pair parameter pool over selected periods")
    p.add_argument("--fy25",      action="store_true", help="FY2025 period only")
    p.add_argument("--fy26",      action="store_true", help="FY2026 YTD period only")
    p.add_argument("--days",      type=int,  default=None, help="Rolling N-day window")
    p.add_argument("--assets",    type=str,  nargs="+", default=None,
                   help="Asset pair for backtest (default: BTC ETH)")
    p.add_argument("--coins",     type=str,  nargs="+", default=None,
                   help="Coins for --scan (default: BTC ETH SOL XRP DOGE)")
    p.add_argument("--no-fetch",  action="store_true", help="Use cached market data")
    p.add_argument("--plot",       action="store_true", help="Save equity charts")
    p.add_argument("--plot-dir",   type=str, default="charts", help="Chart output dir")
    p.add_argument("--trade-log",  action="store_true",
                   help="Print detailed per-trade log (signal → open → close)")
    p.add_argument("--opt-trials", type=int, default=180,
                   help="Max sampled parameter combinations for --optimize")
    p.add_argument("--opt-top",    type=int, default=10,
                   help="Show top N parameter sets for --optimize")
    p.add_argument("--opt-seed",   type=int, default=42,
                   help="Random seed for parameter sampling")
    p.add_argument("--opt-log-dir", type=str, default="opt_logs",
                   help="Directory for optimization run logs (JSON/CSV/TXT)")
    p.add_argument("--opt-apply-best", action="store_true",
                   help="After --optimize, re-run backtest with best parameter set")
    return p.parse_args()


def main() -> None:
    args   = _parse_args()
    assets = [a.upper() for a in args.assets] if args.assets else CROSS_VOL_ASSETS

    period_keys = []
    if args.fy25:
        period_keys.append("FY25")
    if args.fy26:
        period_keys.append("FY26_YTD")

    if args.portfolio:
        _run_portfolio_backtest(
            coins       = [c.upper() for c in args.coins] if args.coins else PORTFOLIO_COINS,
            period_keys = period_keys or None,
            days        = args.days,
            trade_log   = args.trade_log,
        )
    elif args.optimize:
        run_parameter_optimization(
            assets      = assets,
            period_keys = period_keys or None,
            days        = args.days,
            use_cache   = args.no_fetch,
            max_trials  = max(1, args.opt_trials),
            top_n       = max(1, args.opt_top),
            seed        = args.opt_seed,
            log_dir     = args.opt_log_dir,
            apply_best  = args.opt_apply_best,
            trade_log   = args.trade_log,
        )
    elif args.scan:
        # Multi-asset extended scanner — use importlib to avoid per_asset/scanner.py collision
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location(
            "cv_scanner",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "scanner.py"),
        )
        _mod = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        coins = [c.upper() for c in args.coins] if args.coins else None
        _mod.run_scanner(coins=coins)
    elif args.backtest:
        run_backtest(
            assets      = assets,
            period_keys = period_keys or None,
            days        = args.days,
            use_cache   = args.no_fetch,
            plot        = args.plot,
            plot_dir    = args.plot_dir,
            trade_log   = args.trade_log,
        )
    else:
        run_scan(assets=assets, use_cache=args.no_fetch)


if __name__ == "__main__":
    main()
