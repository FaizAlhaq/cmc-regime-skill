# ── Ported (one-way) from private research repo data gates ───────────────
#   source: quant-trading data-layer v3 gates (test_data_layer_v3.py invariants)
#           @ commit e67503c — reimplemented as a standalone module.
"""
Data quality gates. Every frame entering the pipeline must pass:

  1. monotonic, strictly-increasing timestamp index (no duplicates)
  2. volume > 0 on every bar (OHLCV frames)
  3. OHLC sanity: high >= max(open, close), low <= min(open, close), all > 0
  4. no NaN in required columns
  5. uniform bar spacing (gap report; hard-fail above tolerance)

`apply_gates` returns (clean_df, QualityReport). Hard violations raise.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

OHLCV_COLS = ["open", "high", "low", "close", "volume"]


@dataclass
class QualityReport:
    name: str
    n_rows_in: int
    n_rows_out: int
    n_dups_dropped: int = 0
    n_gaps: int = 0
    max_gap_bars: float = 0.0
    n_stale_flat: int = 0          # bars with close == previous close (zero return)
    stale_flat_dates: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return self.__dict__ | {"notes": list(self.notes)}


def apply_gates(
    df: pd.DataFrame,
    name: str = "frame",
    required_cols: list | None = None,
    bar_hours: float | None = None,
    max_gap_tolerance_bars: float = 24.0,
) -> tuple[pd.DataFrame, QualityReport]:
    """
    Validate and lightly clean a time-indexed frame.

    Drops exact duplicate timestamps (keep first), then HARD-FAILS on:
    non-monotonic index, non-positive volume, OHLC violations, NaN in
    required columns, or gaps larger than max_gap_tolerance_bars.
    """
    required_cols = required_cols if required_cols is not None else OHLCV_COLS
    rep = QualityReport(name=name, n_rows_in=len(df), n_rows_out=len(df))

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"[{name}] index must be a DatetimeIndex")

    # 1. duplicates → drop (keep first), then monotonic check
    dups = df.index.duplicated(keep="first")
    if dups.any():
        rep.n_dups_dropped = int(dups.sum())
        rep.notes.append(f"dropped {rep.n_dups_dropped} duplicate timestamps")
        df = df[~dups]
    if not df.index.is_monotonic_increasing:
        raise ValueError(f"[{name}] timestamps not monotonically increasing")

    # 2-4. column checks
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] missing required columns: {missing}")
    if df[required_cols].isna().any().any():
        bad = df[required_cols].isna().sum()
        raise ValueError(f"[{name}] NaN in required columns:\n{bad[bad > 0]}")

    if "volume" in required_cols:
        if (df["volume"] <= 0).any():
            n_bad = int((df["volume"] <= 0).sum())
            raise ValueError(f"[{name}] {n_bad} bars with volume <= 0")
    if set(["open", "high", "low", "close"]).issubset(required_cols):
        if (df[["open", "high", "low", "close"]] <= 0).any().any():
            raise ValueError(f"[{name}] non-positive OHLC prices")
        bad_hi = (df["high"] < df[["open", "close"]].max(axis=1)).sum()
        bad_lo = (df["low"] > df[["open", "close"]].min(axis=1)).sum()
        if bad_hi or bad_lo:
            raise ValueError(f"[{name}] OHLC violations: high<max(o,c) on "
                             f"{bad_hi} bars, low>min(o,c) on {bad_lo} bars")

    # 4b. stale-flat bars: close == previous close (zero return). Physically
    #     implausible for a daily crypto bar — almost certainly a stale/repeated
    #     close. FLAG only (not a hard fail): such a bar has rv=0, so the regime
    #     obs builder excludes it (log of zero vol is undefined) to avoid a
    #     spurious single-point HMM state. Hygiene, not tuning.
    if "close" in df.columns and len(df) > 1:
        flat = df["close"].diff() == 0.0
        rep.n_stale_flat = int(flat.sum())
        if rep.n_stale_flat:
            rep.stale_flat_dates = [str(t) for t in df.index[flat]]
            rep.notes.append(f"{rep.n_stale_flat} stale-flat (zero-return) bars: "
                             f"{rep.stale_flat_dates[:5]}"
                             f"{' ...' if rep.n_stale_flat > 5 else ''}")

    # 5. gaps
    if bar_hours is not None and len(df) > 2:
        deltas = df.index.to_series().diff().dropna()
        bar_td = pd.Timedelta(hours=bar_hours)
        gap_bars = deltas / bar_td
        gaps = gap_bars[gap_bars > 1.0001]
        rep.n_gaps = int(len(gaps))
        rep.max_gap_bars = float(gap_bars.max())
        if rep.n_gaps:
            rep.notes.append(f"{rep.n_gaps} gaps, max {rep.max_gap_bars:.1f} bars")
        if rep.max_gap_bars > max_gap_tolerance_bars:
            raise ValueError(f"[{name}] gap of {rep.max_gap_bars:.1f} bars exceeds "
                             f"tolerance {max_gap_tolerance_bars}")

    rep.n_rows_out = len(df)
    return df, rep
