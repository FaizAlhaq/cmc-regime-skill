# ── VENDORED (one-way) from private research repo ───────────────────────
#   source: quant-trading/src/regime/features.py @ commit e67503c
#   adaptations: rv_24 → rv_day (window parameterized upstream in
#                core.vol_features via BarFreq); column names follow suit.
#                + fit_obs_scaler/scale_obs (StandardScaler, TRAIN-fit-only)
#                added for HMM numerical stability — see DEBUG_HMM_REAL.md.
#   Do NOT merge changes back into the research repo.
"""
Causal observation features for HMM regime detection.

Observation vector per bar: [log_return, log_rv_day]
Both strictly backward-looking (only data at or before t).

Standardization (fit_obs_scaler / scale_obs): the two obs columns live on very
different scales (at daily bars log_rv_day = log|log_return|, ~38x the std of
log_return) which makes a full-covariance GaussianHMM ill-conditioned. We
standardize with a StandardScaler that is fit on the TRAIN obs ONLY and then
applied to TRAIN/VAL/TEST alike — affine, deterministic, and lookahead-free.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

OBS_COLS = ["log_return", "log_rv_day"]


def make_regime_obs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'log_return' and 'log_rv_day' columns. Requires 'close' and 'rv_day'
    (run core.vol_features.make_vol_features first). Both columns are causal.

    A bar with rv_day == 0 (a stale/flat bar, zero realized vol) has an UNDEFINED
    log-vol — we set log_rv_day = NaN there so get_obs_array drops it, rather than
    clipping to a tiny floor (the old 1e-20 clip mapped such a bar to log_rv≈-46,
    a ~28σ outlier that became a spurious single-point HMM state). See
    DEBUG_HMM_REAL.md and the quality gate's stale-flat flag.
    """
    if "rv_day" not in df.columns:
        raise ValueError(
            "make_regime_obs requires 'rv_day' column. Run make_vol_features() first."
        )
    out = df.copy()
    out["log_return"] = np.log(df["close"]).diff()
    rv = df["rv_day"]
    out["log_rv_day"] = np.log(rv.where(rv > 0.0))   # rv<=0 -> NaN (excluded)
    return out


def get_obs_array(df: pd.DataFrame) -> tuple[np.ndarray, pd.Index]:
    """Extract valid (non-NaN) rows as (n_valid, 2) float64 array + index."""
    cols = df[OBS_COLS]
    mask = cols.notna().all(axis=1)
    valid = cols[mask]
    return valid.values.astype(np.float64), valid.index


def fit_obs_scaler(obs_train: np.ndarray) -> StandardScaler:
    """
    Fit a StandardScaler on TRAIN observations ONLY (no lookahead).

    The returned scaler's mean_/scale_ depend solely on `obs_train`; it must
    then be applied to VAL/TEST via `scale_obs` using these TRAIN statistics.
    """
    return StandardScaler().fit(obs_train)


def scale_obs(scaler: StandardScaler, obs: np.ndarray) -> np.ndarray:
    """Apply a TRAIN-fit scaler to any observation matrix (TRAIN/VAL/TEST)."""
    return scaler.transform(obs).astype(np.float64)


def assert_obs_causality(df: pd.DataFrame, t_star: int) -> None:
    """
    Assert obs at t_star unchanged when future close/rv_day are perturbed 100x.
    Raises AssertionError on causality violation.
    """
    df_full = make_regime_obs(df)
    obs_orig = df_full[OBS_COLS].iloc[t_star].values.copy()
    if np.any(np.isnan(obs_orig)):
        return  # warmup — trivially causal

    df_mut = df.copy()
    df_mut.iloc[t_star + 1:, df_mut.columns.get_loc("close")] *= 100.0
    if "rv_day" in df_mut.columns:
        df_mut.iloc[t_star + 1:, df_mut.columns.get_loc("rv_day")] *= 100.0

    obs_mut = make_regime_obs(df_mut)[OBS_COLS].iloc[t_star].values
    if not np.allclose(obs_orig, obs_mut, rtol=1e-10, equal_nan=True):
        raise AssertionError(
            f"Observation at t={t_star} changed after mutating future data: "
            f"{obs_orig} -> {obs_mut}. CAUSALITY VIOLATION."
        )
