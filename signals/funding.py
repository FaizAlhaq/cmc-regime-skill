# ── Adapted (one-way) from private research repo ─────────────────────────
#   source: quant-trading/src/signals/funding_contrarian.py @ commit e67503c
#   adaptations: z-score window 720 bars → 720 HOURS via BarFreq; the
#                contrarian sign is no longer hard-coded — it is calibrated
#                on TRAIN via signals.positioning.calibrate_sign.
"""
Funding-rate signal (5 stages, strictly causal).

Hypothesis (direction NOT assumed): extreme funding indicates crowded
positioning. Whether that is contrarian or momentum on this dataset is
decided by TRAIN IC only, then frozen.
"""

from __future__ import annotations

import pandas as pd

from core.freq import BarFreq
from signals.positioning import _roll_z


def funding_score(df: pd.DataFrame, freq: BarFreq,
                  window_hours: float = 720.0) -> pd.Series:
    """funding_rate → 30-day rolling z → clip(z/3, ±1). Unsigned; calibrate on TRAIN."""
    if "funding_rate" not in df.columns:
        raise ValueError("funding signal requires 'funding_rate' column")
    z = _roll_z(df["funding_rate"], freq.bars(window_hours))
    return (z / 3.0).clip(-1.0, 1.0).rename("funding")
