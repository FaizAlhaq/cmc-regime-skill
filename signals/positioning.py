"""
Derivatives-positioning signals (NEW for cmc-regime-skill; 5-stage template
follows quant-trading signal discipline — all stages strictly causal).

Two candidate scores:

  positioning_z : rolling z-score of long/short ratio (30-day window).
  oi_funding    : interaction z(Δlog OI, day) × z(funding, 30d) — rising OI
                  with extreme funding = crowded trade.

SIGN POLICY (no-lookahead): neither signal's sign is assumed. The sign is
calibrated on TRAIN ONLY via `calibrate_sign` (battery decides on TRAIN IC),
then frozen for VAL/TEST. The chosen sign is recorded in strategy_spec.json.

Causality: every rolling op uses min_periods = window; warmup rows are NaN.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.freq import BarFreq


def _roll_z(s: pd.Series, window: int) -> pd.Series:
    """Causal rolling z-score (min_periods=window)."""
    mu = s.rolling(window, min_periods=window).mean()
    sd = s.rolling(window, min_periods=window).std()
    return (s - mu) / (sd + 1e-9)


def positioning_z_score(df: pd.DataFrame, freq: BarFreq,
                        window_hours: float = 720.0) -> pd.Series:
    """
    Stage 1-4: long/short ratio → 30-day rolling z → clip(z/3, ±1).
    UNSIGNED magnitude/direction score; multiply by calibrated sign downstream.
    """
    if "long_short_ratio" not in df.columns:
        raise ValueError("positioning_z requires 'long_short_ratio' column")
    z = _roll_z(df["long_short_ratio"], freq.bars(window_hours))
    score = (z / 3.0).clip(-1.0, 1.0)
    return score.rename("positioning_z")


def oi_funding_score(df: pd.DataFrame, freq: BarFreq,
                     window_hours: float = 720.0) -> pd.Series:
    """
    Interaction: z(Δlog OI over 1 day) × z(funding, 30d), squashed to [-1, 1].
    Positive = crowded-long expansion; sign calibrated on TRAIN downstream.
    """
    for c in ("open_interest", "funding_rate"):
        if c not in df.columns:
            raise ValueError(f"oi_funding requires '{c}' column")
    w = freq.bars(window_hours)
    d_oi = np.log(df["open_interest"].clip(lower=1e-12)).diff(freq.w_day)
    inter = _roll_z(d_oi, w) * _roll_z(df["funding_rate"], w)
    score = (inter / 4.0).clip(-1.0, 1.0)
    return score.rename("oi_funding")


def calibrate_sign(score: pd.Series, fwd_return: pd.Series,
                   train_index: pd.Index) -> int:
    """
    Decide the signal sign on TRAIN ONLY: +1 if TRAIN Spearman IC >= 0 else -1.
    The returned sign must be frozen and recorded before touching VAL/TEST.
    """
    from core.battery import spearman_ic
    ic = spearman_ic(score.reindex(train_index), fwd_return.reindex(train_index))
    if np.isnan(ic):
        return +1  # no evidence — leave unflipped
    return +1 if ic >= 0 else -1


def to_direction(score: pd.Series, threshold: float = 0.5) -> pd.Series:
    """Stage 5: discrete direction {-1, 0, +1}; NaN preserved."""
    d = pd.Series(0.0, index=score.index)
    d[score > threshold] = 1.0
    d[score < -threshold] = -1.0
    d[score.isna()] = np.nan
    return d
