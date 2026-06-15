# ── VENDORED (one-way) from private research repo ───────────────────────
#   source: quant-trading/src/vol/features.py + src/vol/targets.py @ commit e67503c
#   adaptations: hour-coupled windows (24/168/720 bars) parameterized via
#                core.freq.BarFreq; columns renamed rv_24/168/720 →
#                rv_day/week/month; forward target horizon parameterized.
#   Do NOT merge changes back into the research repo.
"""
Backward-looking volatility features + forward realized-vol target.

All FEATURES at bar t use only data from bars 0..t (causal, testable via
assert_feature_causality). The TARGET (rv_fwd_day) is intentionally
forward-looking — it is the label, never a feature.

Windows are defined in hours (day=24h, week=168h, month=720h) and converted
to bars through BarFreq, so the module is correct at any bar size.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.freq import BarFreq

# RiskMetrics decay factor (per-bar EWMA)
_LAMBDA = 0.94
_ALPHA_EWMA = 1.0 - _LAMBDA
_CLIP = 1e-20

VOL_FEATURE_COLS = [
    "rv_day", "rv_week", "rv_month",
    "ewma_var", "ewma_vol", "rv_parkinson_day", "rv_gk_day",
]


def make_vol_features(df: pd.DataFrame, freq: BarFreq) -> pd.DataFrame:
    """
    Add backward-looking vol features. Requires columns: open, high, low, close.

    rv_day/rv_week/rv_month : trailing realized vol = sqrt(sum r^2) over
                              24h / 168h / 720h of bars (HAR components)
    ewma_var, ewma_vol      : RiskMetrics EWMA per-bar variance / vol
    rv_parkinson_day        : Parkinson range estimator, trailing day window
    rv_gk_day               : Garman-Klass range estimator, trailing day window
    """
    w_d, w_w, w_m = freq.w_day, freq.w_week, freq.w_month

    r = np.log(df["close"]).diff()
    r_sq = r.pow(2)

    rv_day   = r_sq.rolling(w_d).sum().pipe(np.sqrt)
    rv_week  = r_sq.rolling(w_w).sum().pipe(np.sqrt)
    rv_month = r_sq.rolling(w_m).sum().pipe(np.sqrt)

    ewma_var = r_sq.ewm(alpha=_ALPHA_EWMA, adjust=False).mean()
    ewma_vol = ewma_var.pipe(np.sqrt)

    ln_hl = np.log(df["high"] / df["low"])
    pk_var_t = (1.0 / (4.0 * np.log(2.0))) * ln_hl.pow(2)
    rv_parkinson_day = pk_var_t.rolling(w_d).sum().pipe(np.sqrt)

    ln_co = np.log(df["close"] / df["open"])
    gk_var_t = 0.5 * ln_hl.pow(2) - (2.0 * np.log(2.0) - 1.0) * ln_co.pow(2)
    rv_gk_day = gk_var_t.rolling(w_d).sum().pipe(np.sqrt)

    out = df.copy()
    out["rv_day"], out["rv_week"], out["rv_month"] = rv_day, rv_week, rv_month
    out["ewma_var"], out["ewma_vol"] = ewma_var, ewma_vol
    out["rv_parkinson_day"], out["rv_gk_day"] = rv_parkinson_day, rv_gk_day
    return out


def make_rv_target(df: pd.DataFrame, freq: BarFreq) -> pd.DataFrame:
    """
    Add the forward realized-vol LABEL (intentionally forward-looking):

        rv_fwd_day[t]     = sqrt( sum_{k=1..H} log_return[t+k]^2 ),  H = bars in 24h
        log_rv_fwd_day[t] = log(rv_fwd_day[t])

    Last H rows are NaN (window extends beyond data).
    """
    H = freq.w_day
    r = np.log(df["close"]).diff()
    r_sq = r.pow(2)

    fwd_sum_sq = r_sq.rolling(H).sum().shift(-H)
    rv_fwd = fwd_sum_sq.pipe(np.sqrt)
    log_rv_fwd = np.log(rv_fwd.clip(lower=_CLIP)).where(rv_fwd.notna())

    out = df.copy()
    out["rv_fwd_day"] = rv_fwd
    out["log_rv_fwd_day"] = log_rv_fwd
    return out


def make_forward_returns(
    df: pd.DataFrame, freq: BarFreq, horizons_hours: tuple = None
) -> pd.DataFrame:
    """
    Add forward log-return LABEL columns: forward_return_{H}h for each horizon.
    Default horizons: (1 bar, 24h, 72h). Labels only — never features.
    """
    if horizons_hours is None:
        horizons_hours = (freq.bar_hours, 24.0, 72.0)

    log_close = np.log(df["close"])
    out = df.copy()
    for h in horizons_hours:
        n = freq.bars(h)
        col = forward_return_col(h)
        out[col] = log_close.shift(-n) - log_close
    return out


def forward_return_col(hours: float) -> str:
    return f"forward_return_{int(hours) if float(hours).is_integer() else hours}h"


def assert_feature_causality(
    df: pd.DataFrame,
    freq: BarFreq,
    feature_name: str,
    t_star: int,
    mutate_col: str = "close",
    rtol: float = 1e-10,
) -> None:
    """
    Assert feature_name at t_star is unchanged when future data is mutated
    (rows t_star+1..end multiplied by 1e6). Raises AssertionError on lookahead.
    """
    orig_val = make_vol_features(df, freq)[feature_name].iloc[t_star]
    if np.isnan(orig_val):
        return  # warmup rows are trivially causal

    df_mut = df.copy()
    df_mut.iloc[t_star + 1:, df_mut.columns.get_loc(mutate_col)] *= 1e6
    mutated_val = make_vol_features(df_mut, freq)[feature_name].iloc[t_star]

    assert np.isclose(orig_val, mutated_val, rtol=rtol, equal_nan=True), (
        f"Feature '{feature_name}' at t={t_star} changed after mutating future "
        f"{mutate_col}: {orig_val} -> {mutated_val}. Possible lookahead!"
    )
