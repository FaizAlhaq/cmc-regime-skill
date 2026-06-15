# ── VENDORED (one-way) from private research repo ───────────────────────
#   source: quant-trading/src/vol/baselines.py @ commit e67503c
#   adaptations: (1) HORIZON=24 bars parameterized via BarFreq (24h horizon);
#                (2) statsmodels OLS replaced by numpy lstsq (fewer pinned deps,
#                    identical point estimates); (3) GARCH(1,1) kept optional
#                    behind `arch` import, horizon parameterized;
#                (4) column names rv_24/168/720 → rv_day/week/month.
#   Do NOT merge changes back into the research repo.
"""
Volatility-forecasting baselines.

a) Persistence : log_rv_hat[t] = log(rv_day[t])
b) EWMA        : log_rv_hat[t] = 0.5*log(H * ewma_var[t]),  H = bars per day
c) HAR-RV      : OLS on TRAIN — log(rv_fwd_day) ~ log(rv_day)+log(rv_week)+log(rv_month)
d) GARCH(1,1)  : optional (requires `arch`); fit TRAIN, causal recursion full history

ALL parameters estimated from TRAIN only. Metrics: OOS R², QLIKE, Mincer-Zarnowitz.
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from core.freq import BarFreq

_CLIP = 1e-20
HAR_FEATURES = ["rv_day", "rv_week", "rv_month"]
TARGET_COL = "log_rv_fwd_day"


# ── Metrics ──────────────────────────────────────────────────────────────

def qlike(log_rv_realized: pd.Series, log_rv_hat: pd.Series) -> float:
    """QLIKE loss (Patton 2011). Lower is better."""
    mask = log_rv_realized.notna() & log_rv_hat.notna()
    y = log_rv_realized[mask].values
    m = log_rv_hat[mask].values
    return float(np.mean(2.0 * m + np.exp(2.0 * (y - m))))


def oos_r2(log_rv_realized: pd.Series, log_rv_hat: pd.Series) -> float:
    """OOS R² = 1 - MSE / Var(realized)."""
    mask = log_rv_realized.notna() & log_rv_hat.notna()
    y = log_rv_realized[mask].values
    m = log_rv_hat[mask].values
    if len(y) < 4:
        return float("nan")
    var_y = np.var(y, ddof=1)
    if var_y < _CLIP:
        return float("nan")
    return float(1.0 - np.mean((y - m) ** 2) / var_y)


def mincer_zarnowitz(
    log_rv_realized: pd.Series, log_rv_hat: pd.Series
) -> Tuple[float, float]:
    """MZ regression realized ~ a + b*forecast. Good calibration: a≈0, b≈1."""
    mask = log_rv_realized.notna() & log_rv_hat.notna()
    y = log_rv_realized[mask].values
    m = log_rv_hat[mask].values
    if len(y) < 4:
        return float("nan"), float("nan")
    X = np.column_stack([np.ones(len(m)), m])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(coef[0]), float(coef[1])


# ── Baselines ────────────────────────────────────────────────────────────

def persistence_forecast(df: pd.DataFrame) -> pd.Series:
    """log_rv_hat[t] = log(rv_day[t])."""
    return np.log(df["rv_day"].clip(lower=_CLIP))


def ewma_forecast(df: pd.DataFrame, freq: BarFreq) -> pd.Series:
    """Scale per-bar EWMA variance to the day-horizon accumulated vol."""
    H = freq.w_day
    return 0.5 * (np.log(H) + np.log(df["ewma_var"].clip(lower=_CLIP)))


class HARModel:
    """HAR-RV: log(rv_fwd_day) ~ c + b1*log(rv_day) + b2*log(rv_week) + b3*log(rv_month).

    Fit on TRAIN only (numpy least squares).
    """

    def __init__(self) -> None:
        self.coef_: Optional[np.ndarray] = None  # [const, b1, b2, b3]

    @staticmethod
    def _design(df: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(index=df.index)
        for c in HAR_FEATURES:
            X[f"log_{c}"] = np.log(df[c].clip(lower=_CLIP))
        return X

    def fit(self, df_train: pd.DataFrame) -> "HARModel":
        X = self._design(df_train)
        y = df_train[TARGET_COL]
        mask = y.notna() & X.notna().all(axis=1) & df_train[HAR_FEATURES].notna().all(axis=1)
        Xm = np.column_stack([np.ones(int(mask.sum())), X[mask].values])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.coef_, *_ = np.linalg.lstsq(Xm, y[mask].values, rcond=None)
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        if self.coef_ is None:
            raise RuntimeError("HARModel not fitted")
        X = self._design(df)
        hat = pd.Series(
            self.coef_[0] + X.values @ self.coef_[1:], index=df.index
        )
        feat_nan = df[HAR_FEATURES].isna().any(axis=1)
        hat[feat_nan] = np.nan
        return hat

    @property
    def params(self) -> dict:
        if self.coef_ is None:
            return {}
        names = ["const"] + [f"log_{c}" for c in HAR_FEATURES]
        return {n: float(v) for n, v in zip(names, self.coef_)}


# ── Optional GARCH(1,1) ──────────────────────────────────────────────────

def fit_garch(
    all_returns: pd.Series, train_end: int, freq: BarFreq
) -> Tuple[Optional[pd.Series], bool, str]:
    """
    Fit GARCH(1,1) on TRAIN returns (requires `arch`; returns (None, False, note)
    if unavailable). Causal recursion over the full history; day-horizon
    multi-step variance sum → log_rv forecast aligned to all_returns.index.
    """
    try:
        from arch import arch_model
    except ImportError:
        return None, False, "arch not installed (optional dep — GARCH skipped)"

    H = freq.w_day
    n = len(all_returns)
    r = all_returns.values.astype(float)
    r_safe = np.where(np.isfinite(r), r, 0.0)

    train_r = r_safe[:train_end]
    scale = float(np.sqrt(np.mean(train_r ** 2))) or 1.0
    r_scaled = r_safe / scale

    finite = np.isfinite(r_scaled[:train_end])
    train_series = pd.Series(
        r_scaled[:train_end][finite], index=all_returns.index[:train_end][finite]
    )

    try:
        am = arch_model(train_series, vol="Garch", p=1, q=1,
                        mean="Zero", dist="Normal", rescale=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = am.fit(disp="off", options={"ftol": 1e-9, "maxiter": 500})
        converged = res.convergence_flag == 0
        notes = "converged" if converged else f"flag={res.convergence_flag}"
        omega = float(res.params.get("omega")) * scale ** 2
        alpha = float(res.params.get("alpha[1]"))
        beta = float(res.params.get("beta[1]"))
    except Exception as exc:
        return None, False, f"arch exception: {exc}"

    uncond = max(omega / max(1.0 - alpha - beta, 1e-6),
                 float(np.mean(train_r ** 2)))
    h = np.full(n, uncond)
    for t in range(1, n):
        h[t] = omega + alpha * r_safe[t - 1] ** 2 + beta * h[t - 1]
    h1 = np.clip(omega + alpha * r_safe ** 2 + beta * h, _CLIP, None)

    p = alpha + beta
    if abs(1.0 - p) < 1e-10:
        sum_var = h1 * H
    else:
        G = (1.0 - p ** H) / (1.0 - p)
        sum_var = h1 * G + omega / (1.0 - p) * (H - G)
    log_rv_hat = 0.5 * np.log(np.clip(sum_var, _CLIP, None))

    series = pd.Series(log_rv_hat, index=all_returns.index)
    series.iloc[0] = np.nan
    return series, converged, notes
