"""
cross_vol/param_optimizer.py — Parameter-pool search for single-pair cross-vol.

Searches parameter combinations from cv_config.CROSS_VOL_PARAM_POOL and ranks
them by a robustness-weighted score across multiple backtest periods.
"""
from __future__ import annotations

from itertools import product
from random import Random
from datetime import datetime
import warnings
import json
import os

import numpy as np
import pandas as pd

from backtest.report import compute_stats
from cv_config import (
    CROSS_VOL_PARAM_POOL,
    CROSS_VOL_OPT_MAX_TRIALS,
    CROSS_VOL_OPT_MIN_TRADES,
)
from engine import run_cross_vol_backtest
from universe import AssetVolData


PARAM_ORDER = (
    "spread_lookback",
    "zscore_entry",
    "zscore_exit",
    "spread_min_pts",
    "holding_days_max",
    "capital_fraction",
    "dte_target",
    "regime_lookback",
    "regime_zscore_max",
)


def _candidate_grid(
    param_pool: dict[str, list],
    max_trials: int,
    seed: int = 42,
) -> list[dict]:
    """Build sampled candidate list from a parameter grid."""
    values = [param_pool[k] for k in PARAM_ORDER]
    all_combos = list(product(*values))
    if len(all_combos) <= max_trials:
        chosen = all_combos
    else:
        rng = Random(seed)
        idx = sorted(rng.sample(range(len(all_combos)), max_trials))
        chosen = [all_combos[i] for i in idx]
    return [dict(zip(PARAM_ORDER, combo)) for combo in chosen]


def _is_valid_candidate(p: dict) -> bool:
    """Rule-based pruning for clearly bad / inconsistent combinations."""
    if float(p["zscore_exit"]) >= float(p["zscore_entry"]):
        return False
    if int(p["holding_days_max"]) <= 0:
        return False
    if float(p["capital_fraction"]) <= 0 or float(p["capital_fraction"]) > 0.30:
        return False
    if int(p["spread_lookback"]) < 20:
        return False
    return True


def _trade_win_rate(trades: list) -> float:
    if not trades:
        return 0.0
    wins = sum(1 for t in trades if t.total_pnl_usd > 0)
    return wins / len(trades)


def _score_rows(rows: list[dict], min_trades: float) -> dict:
    """Aggregate per-period metrics into one robustness score."""
    if not rows:
        return {}

    sharpes = np.array([r["sharpe"] for r in rows], dtype=float)
    ann_rets = np.array([r["ann_ret"] for r in rows], dtype=float)
    total_rets = np.array([r["total_ret"] for r in rows], dtype=float)
    max_dds = np.array([r["max_dd"] for r in rows], dtype=float)   # positive values
    trade_wins = np.array([r["trade_win_rate"] for r in rows], dtype=float)
    trade_counts = np.array([r["n_trades"] for r in rows], dtype=float)

    mean_sharpe = float(np.nanmean(sharpes)) if sharpes.size else 0.0
    if not np.isfinite(mean_sharpe):
        mean_sharpe = 0.0

    mean_ann = float(np.nanmean(ann_rets)) if ann_rets.size else 0.0
    worst_total = float(np.nanmin(total_rets)) if total_rets.size else 0.0
    worst_dd = float(np.nanmax(max_dds)) if max_dds.size else 0.0
    mean_trade_win = float(np.nanmean(trade_wins)) if trade_wins.size else 0.0
    avg_trades = float(np.nanmean(trade_counts)) if trade_counts.size else 0.0
    ret_stability = float(np.nanstd(total_rets)) if total_rets.size else 0.0

    # Weighted robustness score:
    # - reward Sharpe + annual return + worst-period survival
    # - penalise drawdown and unstable period returns
    score = (
        0.50 * mean_sharpe
        + 0.20 * (mean_ann * 4.0)
        + 0.25 * (worst_total * 4.0)
        + 0.05 * ((mean_trade_win - 0.5) * 2.0)
        - 0.35 * worst_dd
        - 0.10 * (ret_stability * 4.0)
    )

    if avg_trades < min_trades:
        score -= 0.08 * (min_trades - avg_trades)

    return {
        "score": score,
        "mean_sharpe": mean_sharpe,
        "mean_ann_ret": mean_ann,
        "worst_total_ret": worst_total,
        "worst_max_dd": worst_dd,
        "avg_trade_win": mean_trade_win,
        "avg_trades": avg_trades,
    }


def optimize_cross_vol_params(
    universe: dict[str, AssetVolData],
    asset_a: str,
    asset_b: str,
    periods: dict[str, tuple[datetime, datetime]],
    param_pool: dict[str, list] | None = None,
    max_trials: int = CROSS_VOL_OPT_MAX_TRIALS,
    min_trades: float = CROSS_VOL_OPT_MIN_TRADES,
    seed: int = 42,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Run cross-period parameter search and return ranked DataFrame.

    Columns include:
      score, mean_sharpe, mean_ann_ret_pct, worst_total_ret_pct, worst_max_dd_pct,
      avg_trade_win_pct, avg_trades, and all parameter values.
    """
    pool = param_pool or CROSS_VOL_PARAM_POOL
    missing = [k for k in PARAM_ORDER if k not in pool]
    if missing:
        raise ValueError(f"param_pool missing keys: {missing}")

    candidates = _candidate_grid(pool, max_trials=max_trials, seed=seed)
    rows: list[dict] = []

    for i, p in enumerate(candidates, 1):
        if not _is_valid_candidate(p):
            continue

        period_rows = []
        failed = False
        for _label, (start_dt, end_dt) in periods.items():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    bt = run_cross_vol_backtest(
                        universe=universe,
                        asset_a=asset_a,
                        asset_b=asset_b,
                        start_dt=start_dt,
                        end_dt=end_dt,
                        params=p,
                    )
            except Exception:
                failed = True
                break

            stats = compute_stats(bt["equity"])
            if not stats:
                stats = {
                    "total_ret": 0.0,
                    "ann_ret": 0.0,
                    "sharpe": 0.0,
                    "max_dd": 0.0,
                }
            period_rows.append({
                "total_ret": float(stats.get("total_ret", 0.0)),
                "ann_ret": float(stats.get("ann_ret", 0.0)),
                "sharpe": float(stats.get("sharpe", 0.0))
                if np.isfinite(stats.get("sharpe", 0.0)) else 0.0,
                "max_dd": abs(float(stats.get("max_dd", 0.0))),
                "n_trades": len(bt.get("trades", [])),
                "trade_win_rate": _trade_win_rate(bt.get("trades", [])),
            })

        if failed or not period_rows:
            continue

        agg = _score_rows(period_rows, min_trades=min_trades)
        if not agg:
            continue

        row = dict(agg)
        row.update(p)
        rows.append(row)

        if verbose and (i % 20 == 0):
            print(f"  [opt] evaluated {i}/{len(candidates)} candidates ...")

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    out["mean_ann_ret_pct"] = out["mean_ann_ret"] * 100
    out["worst_total_ret_pct"] = out["worst_total_ret"] * 100
    out["worst_max_dd_pct"] = out["worst_max_dd"] * 100
    out["avg_trade_win_pct"] = out["avg_trade_win"] * 100
    return out


def extract_param_dict(row: pd.Series | dict) -> dict:
    """Extract strategy parameters from one optimizer result row."""
    data = dict(row)
    return {k: _to_native(data[k]) for k in PARAM_ORDER}


def derive_refined_pool(
    ranked: pd.DataFrame,
    top_n: int = 12,
    min_choices: int = 2,
    max_choices: int = 4,
) -> dict[str, list]:
    """
    Derive a tighter parameter pool from the top-ranked results.

    For each parameter, keep the most frequent values among top_n rows.
    """
    if ranked.empty:
        return {}

    top = ranked.head(max(1, top_n))
    refined: dict[str, list] = {}

    int_params = {"spread_lookback", "holding_days_max", "dte_target", "regime_lookback"}

    for key in PARAM_ORDER:
        if key not in top.columns:
            continue
        vc = top[key].value_counts(dropna=True)
        if vc.empty:
            continue

        ordered_vals = list(vc.index[:max_choices])
        if len(ordered_vals) < min_choices:
            # Back-fill with globally frequent values if top_n is too concentrated.
            vc_all = ranked[key].value_counts(dropna=True)
            for v in vc_all.index:
                if v not in ordered_vals:
                    ordered_vals.append(v)
                if len(ordered_vals) >= min_choices:
                    break

        vals = ordered_vals[:max_choices]
        if key in int_params:
            vals = sorted({int(v) for v in vals})
        else:
            vals = sorted({float(v) for v in vals})
        refined[key] = vals

    return refined


def _to_native(v):
    """Convert numpy scalar types to plain Python values."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def evaluate_params_by_period(
    universe: dict[str, AssetVolData],
    asset_a: str,
    asset_b: str,
    periods: dict[str, tuple[datetime, datetime]],
    params: dict,
) -> list[dict]:
    """
    Re-run one parameter set per period and return period-level metrics.
    """
    rows: list[dict] = []
    for label, (start_dt, end_dt) in periods.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            bt = run_cross_vol_backtest(
                universe=universe,
                asset_a=asset_a,
                asset_b=asset_b,
                start_dt=start_dt,
                end_dt=end_dt,
                params=params,
            )
        st = compute_stats(bt["equity"])
        rows.append({
            "period": label,
            "start_date": start_dt.date().isoformat(),
            "end_date": end_dt.date().isoformat(),
            "total_ret_pct": float(st.get("total_ret", 0.0)) * 100.0,
            "ann_ret_pct": float(st.get("ann_ret", 0.0)) * 100.0,
            "sharpe": float(st.get("sharpe", 0.0))
            if np.isfinite(st.get("sharpe", 0.0)) else 0.0,
            "max_dd_pct": float(st.get("max_dd", 0.0)) * 100.0,
            "n_trades": len(bt.get("trades", [])),
        })
    return rows


def save_optimization_run(
    log_root: str,
    asset_a: str,
    asset_b: str,
    periods: dict[str, tuple[datetime, datetime]],
    run_config: dict,
    ranked: pd.DataFrame,
    top_n: int,
    best_params: dict,
    best_period_metrics: list[dict],
    refined_pool: dict[str, list] | None = None,
) -> dict[str, str]:
    """
    Persist one optimization run to disk (JSON + CSV + TXT).

    Returns paths dict with keys: run_dir, ranked_csv, top_csv, summary_json, summary_txt.
    """
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    pair = f"{asset_a.upper()}_{asset_b.upper()}"
    run_dir = os.path.abspath(os.path.join(log_root, f"{pair}_{ts}"))
    os.makedirs(run_dir, exist_ok=True)

    ranked_csv = os.path.join(run_dir, "ranked_all.csv")
    top_csv = os.path.join(run_dir, "ranked_top.csv")
    summary_json = os.path.join(run_dir, "summary.json")
    summary_txt = os.path.join(run_dir, "summary.txt")

    ranked.to_csv(ranked_csv, index=False)
    ranked.head(max(1, top_n)).to_csv(top_csv, index=False)

    periods_json = {
        k: {
            "start_date": v[0].date().isoformat(),
            "end_date": v[1].date().isoformat(),
        }
        for k, v in periods.items()
    }

    payload = {
        "generated_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "pair": f"{asset_a.upper()}/{asset_b.upper()}",
        "run_config": {k: _to_native(v) for k, v in run_config.items()},
        "periods": periods_json,
        "top_n": int(top_n),
        "best_params": {k: _to_native(v) for k, v in best_params.items()},
        "best_period_metrics": best_period_metrics,
        "refined_pool": refined_pool or {},
        "top_rows": [
            {k: _to_native(v) for k, v in row.items()}
            for row in ranked.head(max(1, top_n)).to_dict(orient="records")
        ],
        "files": {
            "ranked_all_csv": ranked_csv,
            "ranked_top_csv": top_csv,
            "summary_txt": summary_txt,
        },
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    lines = [
        f"Cross-Vol Optimization Summary ({asset_a.upper()}/{asset_b.upper()})",
        f"Generated (UTC): {payload['generated_at_utc']}",
        "",
        "Run Config:",
    ]
    for k, v in payload["run_config"].items():
        lines.append(f"  - {k}: {v}")
    lines.append("")
    lines.append("Best Params:")
    for k, v in payload["best_params"].items():
        lines.append(f"  - {k}: {v}")
    lines.append("")
    lines.append("Best Params Validation (per period):")
    for row in best_period_metrics:
        lines.append(
            "  - {period} [{start_date}~{end_date}] | total={total_ret_pct:+.3f}% "
            "ann={ann_ret_pct:+.3f}% sharpe={sharpe:.3f} maxdd={max_dd_pct:+.3f}% "
            "trades={n_trades}".format(**row)
        )
    lines.append("")
    lines.append(f"Saved ranked_all.csv: {ranked_csv}")
    lines.append(f"Saved ranked_top.csv: {top_csv}")
    lines.append(f"Saved summary.json: {summary_json}")

    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return {
        "run_dir": run_dir,
        "ranked_csv": ranked_csv,
        "top_csv": top_csv,
        "summary_json": summary_json,
        "summary_txt": summary_txt,
    }
