# ── VENDORED (one-way) from private research repo ───────────────────────
#   source: quant-trading/src/utils/split.py @ commit e67503c (module last touched 24d0766)
#   adaptations: none (verbatim)
#   Do NOT merge changes back into the research repo.
"""
Time-based train/val/test split dengan embargo.

Prinsip:
  - Data TIDAK boleh di-shuffle. Urutan waktu harus dipertahankan.
  - Embargo (gap kosong) disisipkan di antara segmen agar fitur rolling
    dari periode train tidak bocor ke val/test.
  - Scaler HARUS di-fit hanya pada train, lalu diterapkan ke val & test.

Split default: 70% train | [embargo] | 15% val | [embargo] | 15% test

Example
-------
>>> from src.utils.split import time_split
>>> train, val, test = time_split(df)
>>> print(len(train), len(val), len(test))
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Tuple

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SplitResult:
    """Container for split indices and DataFrames."""
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    # Index positions (inclusive end)
    train_end_idx: int = 0
    val_start_idx: int = 0
    val_end_idx: int = 0
    test_start_idx: int = 0
    embargo_bars: int = 30
    meta: dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            "=== Split Summary ===",
            f"  Train : {len(self.train):>5} bars  "
            f"{self.train.index[0]}  ->  {self.train.index[-1]}",
            f"  [embargo: {self.embargo_bars} bars]",
            f"  Val   : {len(self.val):>5} bars  "
            f"{self.val.index[0]}  ->  {self.val.index[-1]}",
            f"  [embargo: {self.embargo_bars} bars]",
            f"  Test  : {len(self.test):>5} bars  "
            f"{self.test.index[0]}  ->  {self.test.index[-1]}",
            f"  Total (incl. embargo): {len(self.train) + len(self.val) + len(self.test) + 2*self.embargo_bars}",
        ]
        return "\n".join(lines)


def time_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    embargo_bars: int = 30,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Bagi DataFrame time-series menjadi train/val/test dengan embargo.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame yang sudah diurutkan berdasarkan timestamp (monoton naik).
    train_frac : float
        Fraksi data untuk train (default 0.70).
    val_frac : float
        Fraksi data untuk val (default 0.15). Test = sisa.
    embargo_bars : int
        Jumlah baris yang dikecualikan di antara segmen (default 30).
        Harus >= rolling window terpanjang yang digunakan dalam fitur.

    Returns
    -------
    (train, val, test) : tuple of DataFrames
        Tidak ada tumpang tindih, tidak ada shuffle.

    Raises
    ------
    ValueError
        Jika parameter tidak valid atau data terlalu pendek.
    """
    _validate_inputs(df, train_frac, val_frac, embargo_bars)

    n = len(df)

    # Hitung batas index
    train_end  = int(n * train_frac)
    val_start  = train_end + embargo_bars
    val_end    = val_start + int(n * val_frac)
    test_start = val_end + embargo_bars

    if test_start >= n:
        raise ValueError(
            f"Dataset terlalu pendek ({n} baris) untuk split "
            f"{train_frac}/{val_frac}/{1-train_frac-val_frac} "
            f"dengan embargo={embargo_bars}. "
            f"Perlu minimal {train_end + 2*embargo_bars + int(n*val_frac) + 1} baris."
        )

    train = df.iloc[:train_end].copy()
    val   = df.iloc[val_start:val_end].copy()
    test  = df.iloc[test_start:].copy()

    result = SplitResult(
        train=train,
        val=val,
        test=test,
        train_end_idx=train_end,
        val_start_idx=val_start,
        val_end_idx=val_end,
        test_start_idx=test_start,
        embargo_bars=embargo_bars,
        meta={
            "n_total": n,
            "n_embargo_bars": 2 * embargo_bars,
            "train_frac_actual": len(train) / n,
            "val_frac_actual": len(val) / n,
            "test_frac_actual": len(test) / n,
        },
    )

    logger.info(result.summary())
    _assert_no_overlap(result)
    _assert_chronological(result)

    return train, val, test


def time_split_result(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    embargo_bars: int = 30,
) -> SplitResult:
    """Same as time_split but returns a SplitResult with metadata."""
    train, val, test = time_split(df, train_frac, val_frac, embargo_bars)
    n = len(df)
    train_end  = int(n * train_frac)
    val_start  = train_end + embargo_bars
    val_end    = val_start + int(n * val_frac)
    test_start = val_end + embargo_bars
    return SplitResult(
        train=train, val=val, test=test,
        train_end_idx=train_end,
        val_start_idx=val_start,
        val_end_idx=val_end,
        test_start_idx=test_start,
        embargo_bars=embargo_bars,
        meta={"n_total": n},
    )


# ── Internal helpers ───────────────────────────────────────────
def _validate_inputs(df: pd.DataFrame, train_frac: float,
                     val_frac: float, embargo_bars: int) -> None:
    if not df.index.is_monotonic_increasing:
        raise ValueError(
            "DataFrame index tidak monoton naik. "
            "Urutkan dengan df.sort_index() sebelum split."
        )
    if not (0 < train_frac < 1 and 0 < val_frac < 1):
        raise ValueError("train_frac dan val_frac harus di antara 0 dan 1.")
    if train_frac + val_frac >= 1:
        raise ValueError("train_frac + val_frac harus < 1 (sisakan ruang untuk test).")
    if embargo_bars < 0:
        raise ValueError("embargo_bars tidak boleh negatif.")
    if len(df) < 100:
        raise ValueError(f"DataFrame terlalu pendek: {len(df)} baris. Minimal 100.")


def _assert_no_overlap(result: SplitResult) -> None:
    """Assert train/val/test index sets are disjoint."""
    train_idx = set(result.train.index)
    val_idx   = set(result.val.index)
    test_idx  = set(result.test.index)

    assert train_idx.isdisjoint(val_idx), "OVERLAP: train ∩ val tidak kosong!"
    assert train_idx.isdisjoint(test_idx), "OVERLAP: train ∩ test tidak kosong!"
    assert val_idx.isdisjoint(test_idx), "OVERLAP: val ∩ test tidak kosong!"


def _assert_chronological(result: SplitResult) -> None:
    """Assert train ends before val, val ends before test."""
    assert result.train.index[-1] < result.val.index[0], \
        "URUTAN SALAH: train.last >= val.first"
    assert result.val.index[-1] < result.test.index[0], \
        "URUTAN SALAH: val.last >= test.first"
    # Embargo gap must exist
    assert result.val_start_idx - result.train_end_idx >= result.embargo_bars, \
        f"Embargo train->val tidak cukup: {result.val_start_idx - result.train_end_idx} < {result.embargo_bars}"
    assert result.test_start_idx - result.val_end_idx >= result.embargo_bars, \
        f"Embargo val->test tidak cukup: {result.test_start_idx - result.val_end_idx} < {result.embargo_bars}"


# ── Walk-forward generator ─────────────────────────────────────
def walk_forward_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
    test_size: int = 500,
    embargo_bars: int = 30,
    min_train_size: int = 1000,
) -> list:
    """
    Generator untuk walk-forward (expanding window) cross-validation.

    Tiap fold: train dari awal hingga fold_start, test dari fold_start + embargo
    hingga fold_start + embargo + test_size.

    Returns list of (train_df, test_df) tuples in chronological order.
    """
    n = len(df)
    folds = []

    for i in range(n_splits):
        # Work backwards from the end: last fold uses most recent data
        test_end   = n - i * test_size
        test_start = test_end - test_size
        train_end  = test_start - embargo_bars

        if train_end < min_train_size:
            logger.warning(f"Fold {n_splits - i}: train terlalu pendek ({train_end}), dilewati.")
            continue

        train = df.iloc[:train_end].copy()
        test  = df.iloc[test_start:test_end].copy()
        folds.append((train, test))

    folds.reverse()  # chronological order
    return folds
