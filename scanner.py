"""
cross_vol/scanner.py — Multi-asset vol-premium scanner.

Scans the extended universe (BTC, ETH + SOL, XRP, DOGE via Bybit)
and ranks every pair by current vol-spread opportunity.

Data sources:
  BTC / ETH  — Deribit DVOL (30d implied vol index) + Binance spot RV
  SOL / XRP / DOGE — Bybit live ATM IV + Binance spot RV

Vol Premium (VP):
  VP = ATM_IV − RV30  (all in annualised %)

Spread signal:
  spread = VP_A − VP_B
  z-score computed over rolling history:
    • BTC/ETH: full rolling VP history from DVOL (high quality)
    • Any pair involving SOL/XRP/DOGE: rolling RV spread used as IV-spread
      proxy (lower confidence, flagged ⚠)

Output tables:
  1. Asset vol-premium snapshot (sorted by VP, highest first)
  2. Pair opportunities (sorted by |z-score| or |spread|, with direction)
"""
from __future__ import annotations

import sys
import os
import warnings
from datetime import datetime, timezone
from itertools import combinations

import numpy as np
import pandas as pd

_HERE      = os.path.dirname(os.path.abspath(__file__))
_PER_ASSET = os.path.join(_HERE, "..", "per_asset")
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
if _PER_ASSET not in sys.path:
    sys.path.append(_PER_ASSET)

from cv_data.binance import get_spot_klines, get_rv30_series, get_rv30, get_spot_price
from cv_data.bybit   import get_atm_iv
from data.deribit    import get_dvol_history

# ── Config ────────────────────────────────────────────────────────────────────

# Assets that have a Deribit DVOL index (high-quality IV)
DVOL_ASSETS = {"BTC", "ETH"}
# Assets where we use Bybit live ATM IV
BYBIT_ASSETS = {"SOL", "XRP", "DOGE"}

HISTORY_DAYS    = 180    # days of OHLCV / DVOL history for z-score
ZSCORE_WINDOW   = 60     # rolling window for spread z-score (days)
DTE_TARGET      = 21     # target DTE for ATM IV fetch
SPREAD_MIN_PTS  = 2.0    # minimum spread to flag as opportunity (vol points)
ZSCORE_FLAG     = 1.2    # z-score threshold for "entry zone"


# ── Data containers ───────────────────────────────────────────────────────────

class AssetSnapshot:
    __slots__ = (
        "coin", "spot", "atm_iv_pct", "rv30_pct", "vol_premium",
        "iv_source", "klines", "dvol_series",
    )
    def __init__(self, coin, spot, atm_iv_pct, rv30_pct, iv_source, klines,
                 dvol_series=None):
        self.coin         = coin
        self.spot         = spot
        self.atm_iv_pct   = atm_iv_pct
        self.rv30_pct     = rv30_pct
        self.vol_premium  = atm_iv_pct - rv30_pct
        self.iv_source    = iv_source   # "DVOL" | "Bybit-ATM"
        self.klines       = klines      # daily OHLCV from Binance
        self.dvol_series  = dvol_series # pd.Series of DVOL daily, or None


# ── Fetch universe snapshot ───────────────────────────────────────────────────

def fetch_extended_universe(
    coins: list[str] | None = None,
    history_days: int = HISTORY_DAYS,
    verbose: bool = True,
) -> dict[str, AssetSnapshot]:
    """
    Fetch current vol premium for each coin.

    Returns {coin: AssetSnapshot}.
    Silently skips coins where data is unavailable.
    """
    if coins is None:
        coins = sorted(DVOL_ASSETS | BYBIT_ASSETS)

    universe: dict[str, AssetSnapshot] = {}

    for coin in coins:
        coin = coin.upper()
        if verbose:
            print(f"  [{coin}] fetching...", end=" ", flush=True)
        try:
            # ── Spot price and RV from Binance ───────────────────────────────
            klines = get_spot_klines(coin, days=history_days)
            if len(klines) < 10:
                if verbose:
                    print("skip (insufficient spot history)")
                continue
            spot    = float(klines["close"].iloc[-1])
            rv30    = get_rv30(klines)

            # ── Implied vol ──────────────────────────────────────────────────
            if coin in DVOL_ASSETS:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    dvol_df = get_dvol_history(coin, days=history_days, resolution=86400)
                dvol_series = dvol_df["dvol"] if "dvol" in dvol_df.columns else dvol_df.iloc[:, 0]
                dvol_series = dvol_series.dropna().sort_index()
                atm_iv  = float(dvol_series.iloc[-1]) if len(dvol_series) else float("nan")
                iv_src  = "DVOL"
            else:
                dvol_series = None
                atm_iv, _dte = get_atm_iv(coin, target_dte=DTE_TARGET)
                iv_src       = "Bybit-ATM"

            snap = AssetSnapshot(
                coin=coin, spot=spot, atm_iv_pct=atm_iv,
                rv30_pct=rv30, iv_source=iv_src,
                klines=klines, dvol_series=dvol_series,
            )
            universe[coin] = snap
            if verbose:
                print(f"IV={atm_iv:.1f}%  RV30={rv30:.1f}%  VP={snap.vol_premium:+.1f} pts [{iv_src}]")
        except Exception as e:
            if verbose:
                print(f"ERROR - {e}")

    return universe


# ── Spread & z-score helpers ──────────────────────────────────────────────────

def _build_vp_series(snap: AssetSnapshot, window: int = ZSCORE_WINDOW) -> pd.Series | None:
    """
    Build a daily vol-premium (VP) time series for one asset.

    For DVOL assets: VP_t = DVOL_t − RV30_t  (both daily)
    For Bybit assets: no historical IV → VP_t ≈ RV30_t (proxy only)
                      We normalise it so the spread z-score still works.
    """
    klines = snap.klines
    rv_series = get_rv30_series(klines, window=30).dropna()

    if snap.dvol_series is not None:
        # Align DVOL with RV series
        dvol = snap.dvol_series.reindex(rv_series.index, method="ffill").dropna()
        rv_aligned = rv_series.reindex(dvol.index).dropna()
        idx = dvol.index.intersection(rv_aligned.index)
        vp = dvol.loc[idx] - rv_aligned.loc[idx]
    else:
        # Proxy: use RV as the only signal (no IV history)
        vp = rv_series.copy()
        vp.name = f"{snap.coin}_rv_proxy"

    return vp.sort_index()


def _spread_zscore(
    vp_a: pd.Series,
    vp_b: pd.Series,
    window: int = ZSCORE_WINDOW,
) -> tuple[pd.Series, pd.Series]:
    """
    Compute rolling spread and z-score for a pair.
    Returns (spread_series, zscore_series).
    """
    idx = vp_a.index.intersection(vp_b.index)
    spread = (vp_a.loc[idx] - vp_b.loc[idx]).sort_index()
    roll_mean = spread.rolling(window, min_periods=window // 2).mean()
    roll_std  = spread.rolling(window, min_periods=window // 2).std().replace(0, np.nan)
    z = (spread - roll_mean) / roll_std
    return spread, z


# ── Main scanner ──────────────────────────────────────────────────────────────

def scan_pairs(
    universe: dict[str, AssetSnapshot],
    zscore_window: int = ZSCORE_WINDOW,
    spread_min_pts: float = SPREAD_MIN_PTS,
    zscore_flag: float = ZSCORE_FLAG,
) -> pd.DataFrame:
    """
    For every pair in the universe, compute the current spread z-score.
    Returns a DataFrame sorted by |z-score| (highest first).
    """
    coins = list(universe.keys())
    vp_map = {c: _build_vp_series(universe[c], window=zscore_window) for c in coins}

    rows = []
    for coin_a, coin_b in combinations(coins, 2):
        snap_a = universe[coin_a]
        snap_b = universe[coin_b]
        vp_a   = vp_map[coin_a]
        vp_b   = vp_map[coin_b]

        if vp_a is None or vp_b is None or len(vp_a) < 10 or len(vp_b) < 10:
            continue

        spread_s, z_s = _spread_zscore(vp_a, vp_b, window=zscore_window)
        if len(z_s.dropna()) < 5:
            continue

        current_spread = float(spread_s.iloc[-1]) if len(spread_s) else float("nan")
        current_z      = float(z_s.dropna().iloc[-1]) if len(z_s.dropna()) else float("nan")

        if np.isnan(current_z):
            continue

        # Direction: z > 0 → A is historically rich → short A, long B
        if current_z >= 0:
            action = f"Long {coin_b}  ·  Short {coin_a}"
        else:
            action = f"Long {coin_a}  ·  Short {coin_b}"

        # Data quality flag
        both_dvol = coin_a in DVOL_ASSETS and coin_b in DVOL_ASSETS
        quality   = "★★★ High" if both_dvol else "★★☆ Med"

        # Entry zone flag
        in_zone = abs(current_z) >= zscore_flag and abs(current_spread) >= spread_min_pts

        rows.append({
            "Pair":          f"{coin_a}/{coin_b}",
            "VP_A":          snap_a.vol_premium,
            "VP_B":          snap_b.vol_premium,
            "Spread":        current_spread,
            "|Z|":           abs(current_z),
            "Z":             current_z,
            "Action":        action,
            "IV_src_A":      snap_a.iv_source,
            "IV_src_B":      snap_b.iv_source,
            "Quality":       quality,
            "Entry?":        "✅ YES" if in_zone else "—",
        })

    df = pd.DataFrame(rows).sort_values("|Z|", ascending=False).reset_index(drop=True)
    return df


# ── Display ───────────────────────────────────────────────────────────────────

def print_asset_table(universe: dict[str, AssetSnapshot]) -> None:
    """Print per-asset snapshot sorted by vol premium."""
    snaps = sorted(universe.values(), key=lambda s: -s.vol_premium)
    header = f"{'Asset':6s}  {'Spot':>12s}  {'ATM IV':>8s}  {'RV30':>7s}  {'VP':>7s}  {'IV Source':12s}"
    print(header)
    print("─" * len(header))
    for s in snaps:
        print(
            f"  {s.coin:<5s}  {s.spot:>12,.2f}  "
            f"{s.atm_iv_pct:>7.1f}%  {s.rv30_pct:>6.1f}%  "
            f"{s.vol_premium:>+7.1f}  {s.iv_source}"
        )


def print_pair_table(df: pd.DataFrame, top_n: int = 15) -> None:
    """Print ranked pair opportunities."""
    shown = df.head(top_n)
    for _, row in shown.iterrows():
        z_bar = "█" * min(int(abs(row["Z"]) * 4), 20)
        flag  = " ← ENTRY ZONE" if row["Entry?"] == "✅ YES" else ""
        print(
            f"  {row['Pair']:10s}  z={row['Z']:+.2f}σ  {z_bar:<20s}  {row['Action']}"
            f"  [{row['Quality']}]{flag}"
        )


def run_scanner(
    coins: list[str] | None = None,
    top_n: int = 15,
    verbose: bool = True,
) -> tuple[dict[str, AssetSnapshot], pd.DataFrame]:
    """Full scan: fetch → compute → print."""
    if coins is None:
        coins = ["BTC", "ETH", "SOL", "XRP", "DOGE"]

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"\n{'═'*70}")
    print(f"  Multi-Asset Vol Premium Scanner  —  {now}")
    print(f"  Universe: {', '.join(coins)}")
    print(f"{'═'*70}\n")

    print("Fetching data...")
    universe = fetch_extended_universe(coins, verbose=True)

    print(f"\n{'─'*70}")
    print("  Asset Vol-Premium Snapshot")
    print(f"{'─'*70}")
    print_asset_table(universe)

    print(f"\n{'─'*70}")
    print(f"  Pair Opportunities  (sorted by |z-score|, top {top_n})")
    print(f"  z-score threshold: ±{ZSCORE_FLAG:.1f}σ  |  min spread: {SPREAD_MIN_PTS:.0f} vol pts")
    print(f"{'─'*70}")
    df = scan_pairs(universe)
    print_pair_table(df, top_n=top_n)

    print(f"\n  Note: ★★★ = Deribit DVOL (index quality)  ★★☆ = Bybit mark-IV (live snapshot)")
    print(f"  For ★★☆ pairs the z-score uses RV spread as IV proxy — treat as indicative.\n")

    return universe, df


if __name__ == "__main__":
    run_scanner()
