# ── VENDORED (one-way) from private research repo ───────────────────────
#   source: quant-trading/src/risk/sizing.py @ commit e67503c (module last touched 24d0766)
#   adaptations: none (verbatim; column-name agnostic)
#   Do NOT merge changes back into the research repo.
"""
Vol-targeted position sizing with regime gating (Task R3).

Sizing formula (a priori, no tuning on test data):
    target_vol   = median(exp(har_hat[TRAIN bars]))   [rv units]
    forecast_vol = exp(har_hat[t])                     [rv units, causal]
    raw_scale[t] = clip(target_vol / forecast_vol[t], 0, max_leverage)
    scale[t]     = raw_scale[t] * REGIME_FACTOR[regime[t]]

Regime factor (K=4, fixed by R2 QLIKE analysis):
    calm (state 0)     -> 1.0
    low-vol (state 1)  -> 1.0
    high-vol (state 2) -> 0.5
    turbulent (state 3)-> 0.0  (HAR forecast unreliable in this state)

For K=2 or K=3 regimes (fallback only):
    K=2: state 0 (calm)->1.0,  state 1 (turbulent)->0.0
    K=3: state 0 (calm)->1.0,  state 1 (normal)->1.0,  state 2 (turbulent)->0.0

Causality guarantee:
    har_hat[t] uses rv_24[t], rv_168[t], rv_720[t] — all backward-looking.
    regime[t] uses the HMM forward algorithm posterior at t — causal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────
# Frozen regime factors per K (a priori, fixed before TEST)
# ─────────────────────────────────────────────────────────

_REGIME_FACTORS_4 = {0: 1.0, 1: 1.0, 2: 0.5, 3: 0.0}
_REGIME_FACTORS_3 = {0: 1.0, 1: 1.0, 2: 0.0}
_REGIME_FACTORS_2 = {0: 1.0, 1: 0.0}


def _get_regime_factors(n_states: int) -> dict[int, float]:
    if n_states == 4:
        return _REGIME_FACTORS_4
    elif n_states == 3:
        return _REGIME_FACTORS_3
    elif n_states == 2:
        return _REGIME_FACTORS_2
    else:
        raise ValueError(f"Unsupported n_states={n_states}. Expected 2, 3, or 4.")


# ─────────────────────────────────────────────────────────
# Main sizing function
# ─────────────────────────────────────────────────────────

def compute_position_scale(
    har_hat_full: pd.Series,
    train_mask: pd.Series,
    regime_full: pd.Series | None = None,
    n_states: int | None = None,
    target_vol: float | None = None,
    max_leverage: float = 3.0,
) -> pd.Series:
    """
    Compute the vol-targeted position scale for every bar.

    Parameters
    ----------
    har_hat_full : pd.Series
        HAR-RV log-vol forecast for every bar (fitted on TRAIN only).
        Index must be datetime. NaN where the HAR has no prediction.
    train_mask : pd.Series of bool
        True for TRAIN bars. Used to compute target_vol if not provided.
    regime_full : pd.Series or None
        Integer regime labels (0-indexed, ordered by ascending vol).
        If None, regime gating is skipped (scale = raw_scale only).
    n_states : int or None
        Number of HMM states (K). If None, inferred as max(regime)+1.
        Pass explicitly when some states may not appear in the data
        (e.g., all bars calm in a K=4 model).
    target_vol : float or None
        Override target volatility (in rv units). If None, computed as
        median(forecast_vol[TRAIN]) where forecast_vol = exp(har_hat).
    max_leverage : float
        Upper bound on position scale (default 3.0).

    Returns
    -------
    pd.Series
        position_scale[t] in [0, max_leverage].
        NaN where har_hat_full is NaN.
    """
    forecast_vol = np.exp(har_hat_full)  # rv units, always positive

    # ── Target vol: median of TRAIN forecast vol ──────────────────
    if target_vol is None:
        train_forecast = forecast_vol[train_mask]
        train_valid = train_forecast.dropna()
        if len(train_valid) == 0:
            raise ValueError("No valid TRAIN HAR-RV forecasts to compute target_vol.")
        target_vol = float(np.median(train_valid))

    # ── Raw scale: target_vol / forecast_vol[t] ───────────────────
    # clip to [0, max_leverage] — avoids division by zero (NaN → NaN)
    raw_scale = (target_vol / forecast_vol).clip(lower=0.0, upper=max_leverage)

    # ── Regime gating ─────────────────────────────────────────────
    if regime_full is None:
        return raw_scale.rename("position_scale")

    if n_states is None:
        n_states = int(regime_full.dropna().max()) + 1
    factors = _get_regime_factors(n_states)

    regime_factor = regime_full.map(factors)
    # NaN regimes (warmup bars) → factor = 0 (safe: do not trade without regime)
    regime_factor = regime_factor.fillna(0.0)

    scale = raw_scale * regime_factor
    return scale.clip(lower=0.0, upper=max_leverage).rename("position_scale")


def compute_target_vol(
    har_hat_full: pd.Series,
    train_mask: pd.Series,
) -> float:
    """Return median(exp(har_hat[TRAIN])) — the a priori target volatility."""
    forecast_vol = np.exp(har_hat_full)
    train_valid = forecast_vol[train_mask].dropna()
    if len(train_valid) == 0:
        raise ValueError("No valid TRAIN HAR-RV forecasts.")
    return float(np.median(train_valid))
