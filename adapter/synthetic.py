"""
SYNTHETIC sample-data generator — clearly labeled, never real market data.

Purpose: let judges (and CI) run the ENTIRE pipeline offline with zero API
key. Generates a regime-switching OHLCV series plus derivatives fields
(funding rate, open interest, long/short ratio) for BTC and ETH.

EVERY artifact is marked SYNTHETIC:
  - filenames contain 'SYNTHETIC'
  - every row carries a `synthetic = True` column
  - parquet metadata carries {'data_source': 'SYNTHETIC ...'}

Results computed on this data demonstrate the PIPELINE — they are NOT
evidence of market edge and must never be presented as real results.

Generator design (seeded, deterministic):
  - 4-state Markov vol regime chain (calm / low-vol / high-vol / turbulent)
  - per-regime return vol + drift; intra-bar high/low from |return| + noise
  - funding rate mean-reverting, correlated with recent returns (crowding)
  - open interest: level series rising in trends, flushing in turbulence
  - long/short ratio: noisy contrarian crowding proxy around 1.0
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SOURCE_TAG = "SYNTHETIC (adapter/synthetic.py, seeded) — NOT real market data"

# Per-regime per-day return vol (annualized-ish crypto numbers), drift, persistence
_REGIME_VOL_D = np.array([0.012, 0.022, 0.042, 0.085])   # daily stdev of log-ret
_REGIME_DRIFT_D = np.array([0.0012, 0.0008, -0.0005, -0.004])
_TRANSMAT = np.array([
    [0.975, 0.020, 0.004, 0.001],
    [0.020, 0.955, 0.020, 0.005],
    [0.005, 0.030, 0.945, 0.020],
    [0.002, 0.008, 0.060, 0.930],
])

_ASSET_P0 = {"BTC": 60000.0, "ETH": 3000.0}
_ASSET_SEED = {"BTC": 42, "ETH": 1337}


def generate(
    asset: str = "BTC",
    bar_hours: float = 4.0,
    n_bars: int = 5400,
    end: str = "2026-06-01",
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Return a SYNTHETIC DataFrame indexed by UTC timestamps with columns:
    open, high, low, close, volume, funding_rate, open_interest,
    long_short_ratio, synthetic(=True).
    """
    seed = _ASSET_SEED.get(asset, 7) if seed is None else seed
    rng = np.random.default_rng(seed)
    bpd = 24.0 / bar_hours

    # regime chain
    states = np.empty(n_bars, dtype=int)
    states[0] = 0
    for t in range(1, n_bars):
        states[t] = rng.choice(4, p=_TRANSMAT[states[t - 1]])

    vol_bar = _REGIME_VOL_D[states] / np.sqrt(bpd)
    drift_bar = _REGIME_DRIFT_D[states] / bpd
    r = drift_bar + vol_bar * rng.standard_normal(n_bars)

    p0 = _ASSET_P0.get(asset, 100.0)
    close = p0 * np.exp(np.cumsum(r))
    open_ = np.concatenate([[p0], close[:-1]])

    # intra-bar range: proportional to bar vol with noise; ensure OHLC sanity
    spread = np.abs(rng.standard_normal(n_bars)) * vol_bar * close * 0.8
    hi_base = np.maximum(open_, close)
    lo_base = np.minimum(open_, close)
    high = hi_base + spread * rng.uniform(0.2, 1.0, n_bars)
    low = np.maximum(lo_base - spread * rng.uniform(0.2, 1.0, n_bars), 1e-9)

    volume = (np.exp(rng.normal(0, 0.4, n_bars))
              * (1.0 + 14.0 * vol_bar / vol_bar.mean() * 0.1) * 1000.0)

    # funding: mean-reverting + crowding follows trailing day return
    trail = pd.Series(r).rolling(int(bpd), min_periods=1).sum().values
    funding = np.empty(n_bars)
    funding[0] = 0.0001
    for t in range(1, n_bars):
        funding[t] = (0.92 * funding[t - 1] + 0.035 * trail[t]
                      + rng.normal(0, 2e-5))

    # open interest: trends up with |trail|, flushes in turbulence
    oi = np.empty(n_bars)
    oi[0] = 1e9
    for t in range(1, n_bars):
        flush = -0.04 if states[t] == 3 else 0.0
        oi[t] = oi[t - 1] * np.exp(0.6 * np.abs(trail[t]) * 0.1 + flush
                                   + rng.normal(0, 0.004))

    # long/short ratio: crowding proxy around 1.0, follows funding
    lsr = 1.0 + 60.0 * funding + rng.normal(0, 0.05, n_bars)
    lsr = np.clip(lsr, 0.3, 3.0)

    idx = pd.date_range(end=pd.Timestamp(end, tz="UTC"),
                        periods=n_bars, freq=pd.Timedelta(hours=bar_hours))
    df = pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close,
        "volume": volume, "funding_rate": funding, "open_interest": oi,
        "long_short_ratio": lsr, "synthetic": True,
    }, index=idx)
    df.index.name = "timestamp"
    df.attrs["data_source"] = SOURCE_TAG
    return df


def sample_path(data_dir: Path, asset: str, bar_hours: float) -> Path:
    h = int(bar_hours) if float(bar_hours).is_integer() else bar_hours
    return Path(data_dir) / f"SYNTHETIC_{asset}_{h}h.parquet"


def write_sample(df: pd.DataFrame, path: Path) -> None:
    """Write parquet with SYNTHETIC tag in file metadata."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pandas(df.reset_index())
    meta = dict(table.schema.metadata or {})
    meta[b"data_source"] = SOURCE_TAG.encode()
    pq.write_table(table.replace_schema_metadata(meta), path)


def read_sample(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path).set_index("timestamp")
    if "synthetic" not in df.columns or not df["synthetic"].all():
        raise ValueError(f"{path} does not carry the SYNTHETIC marker — refusing "
                         "to load ambiguous data as a sample.")
    df.attrs["data_source"] = SOURCE_TAG
    return df
