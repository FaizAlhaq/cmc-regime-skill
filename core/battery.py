# ── VENDORED (one-way) from private research repo ───────────────────────
#   source: quant-trading/src/validation/signal_battery.py @ commit e67503c
#   adaptations: (1) HORIZONS (1h/4h/24h) and block_len=48 parameterized via
#                BarFreq — horizons default to (1 bar, 24h, 72h), block = 2 days;
#                (2) vol-tercile column realized_vol_24 → rv_day;
#                (3) import path src.utils.split → core.split.
#   Do NOT merge changes back into the research repo.
"""
Signal validation battery. Single entry point: evaluate_signal(score, df, freq).

Four checks per signal:
  1. Segment IC table: TRAIN/VAL/TEST/FULL × horizons (Spearman)
  2. Moving-block bootstrap 90% CI on full-history primary horizon (24h)
  3. Vol-tercile regime slice — thresholds fit on TRAIN only
  4. Machine-readable verdict: CONFIRMED / WEAK-MONITOR / DEAD

The battery is honest by construction: whatever it says is what ships.
"""

from __future__ import annotations

import logging
from math import ceil

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from core.freq import BarFreq
from core.split import time_split
from core.vol_features import forward_return_col

logger = logging.getLogger(__name__)

DEFAULT_N_BOOTSTRAP = 2000
DEFAULT_BOOT_SEED = 42
DEFAULT_CI_LEVEL = 0.90
DEFAULT_IC_GATE = 0.05
MIN_VALID_PAIRS = 10
PRIMARY_HORIZON_H = 24.0


def default_horizons(freq: BarFreq) -> dict:
    """{label → forward-return column} for (1 bar, 24h, 72h)."""
    hours = (freq.bar_hours, 24.0, 72.0)
    out = {}
    for h in hours:
        col = forward_return_col(h)
        out[col.replace("forward_return_", "")] = col
    return out


def spearman_ic(signal: pd.Series, fwd: pd.Series) -> float:
    """Spearman IC; NaN if < MIN_VALID_PAIRS valid pairs."""
    both = pd.DataFrame({"s": signal, "f": fwd}).dropna()
    if len(both) < MIN_VALID_PAIRS:
        return float("nan")
    rho, _ = scipy_stats.spearmanr(both["s"], both["f"])
    return float(rho)


def moving_block_bootstrap(
    signal: pd.Series,
    fwd: pd.Series,
    block_len: int,
    n_resamples: int = DEFAULT_N_BOOTSTRAP,
    seed: int = DEFAULT_BOOT_SEED,
) -> np.ndarray:
    """MBB for Spearman IC — preserves short-range autocorrelation."""
    both = pd.DataFrame({"s": signal, "f": fwd}).dropna()
    n = len(both)
    if n < block_len:
        logger.warning(f"[battery:mbb] {n} valid pairs < block_len={block_len}; all-NaN")
        return np.full(n_resamples, float("nan"))

    s_arr, f_arr = both["s"].values, both["f"].values
    rng = np.random.default_rng(seed)
    n_blocks = int(ceil(n / block_len))
    max_start = n - block_len

    ic_samples = np.empty(n_resamples)
    for i in range(n_resamples):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        rs_s = np.concatenate([s_arr[s: s + block_len] for s in starts])[:n]
        rs_f = np.concatenate([f_arr[s: s + block_len] for s in starts])[:n]
        rho, _ = scipy_stats.spearmanr(rs_s, rs_f)
        ic_samples[i] = rho
    return ic_samples


def _compute_segment_ic(score, df, train_df, val_df, test_df, horizons) -> dict:
    result: dict = {}
    for h_key, h_col in horizons.items():
        if h_col not in df.columns:
            logger.warning(f"[battery] '{h_col}' missing — skipping {h_key}")
            continue
        seg_ics = {}
        for seg_name, seg_df in [("TRAIN", train_df), ("VAL", val_df),
                                 ("TEST", test_df), ("FULL", df)]:
            seg_ics[seg_name] = spearman_ic(score.reindex(seg_df.index), seg_df[h_col])
        result[h_key] = seg_ics
    return result


def _compute_bootstrap(score, df, primary_col, block_len,
                       n_bootstrap, boot_seed, ci_level) -> dict:
    alpha_lo = (1.0 - ci_level) / 2.0
    fwd = df[primary_col]
    point_ic = spearman_ic(score, fwd)
    samples = moving_block_bootstrap(score, fwd, block_len, n_bootstrap, boot_seed)
    return {
        "point": point_ic,
        "ci_lo": float(np.nanpercentile(samples, alpha_lo * 100)),
        "ci_hi": float(np.nanpercentile(samples, (1 - alpha_lo) * 100)),
        "ci_excludes_zero": bool(
            (np.nanpercentile(samples, alpha_lo * 100) > 0)
            or (np.nanpercentile(samples, (1 - alpha_lo) * 100) < 0)
        ),
        "n": int(len(pd.DataFrame({"s": score, "f": fwd}).dropna())),
        "samples": samples,
    }


def _compute_regime_ic(score, df, train_df, primary_col) -> dict:
    """IC per vol tercile. Thresholds fit on TRAIN rv_day ONLY (no lookahead)."""
    if "rv_day" not in train_df.columns:
        logger.warning("[battery:regime] 'rv_day' not in train_df — skipping")
        return {}
    train_vol = train_df["rv_day"].dropna()
    t1 = float(np.percentile(train_vol, 100.0 / 3))
    t2 = float(np.percentile(train_vol, 200.0 / 3))

    vol = df["rv_day"]
    regime = pd.Series("mid", index=df.index, dtype=object)
    regime[vol < t1] = "low"
    regime[vol >= t2] = "high"
    regime[vol.isna()] = np.nan

    result = {"t1": t1, "t2": t2}
    for reg in ("low", "mid", "high"):
        mask = regime == reg
        result[reg] = spearman_ic(score[mask], df.loc[mask, primary_col])
        result[f"{reg}_n"] = int(mask.sum())
    return result


def _compute_verdict(seg_primary: dict, bootstrap: dict,
                     ic_gate: float = DEFAULT_IC_GATE) -> tuple[str, list[str]]:
    """First match wins: DEAD → CONFIRMED → WEAK-MONITOR."""
    reasons: list[str] = []
    full_ic = seg_primary.get("FULL", float("nan"))

    if np.isnan(full_ic):
        reasons.append("DEAD: full-history primary IC = NaN (insufficient data)")
        return "DEAD", reasons
    if full_ic <= ic_gate:
        reasons.append(f"DEAD: full-history primary IC ({full_ic:+.4f}) <= {ic_gate} gate")
        return "DEAD", reasons

    valid = {k: seg_primary[k] for k in ("TRAIN", "VAL", "TEST")
             if k in seg_primary and not np.isnan(seg_primary[k])}
    sign_consistent = len({np.sign(v) for v in valid.values()}) <= 1
    if not sign_consistent:
        reasons.append("Sign inconsistent across segments: "
                       + ", ".join(f"{k}={v:+.4f}" for k, v in valid.items()))

    ci_lo, ci_hi = bootstrap.get("ci_lo", np.nan), bootstrap.get("ci_hi", np.nan)
    ci_excl = bool((ci_lo > 0) or (ci_hi < 0))
    if not ci_excl:
        reasons.append(f"CI includes zero: [{ci_lo:+.4f}, {ci_hi:+.4f}]")

    if sign_consistent and ci_excl:
        reasons.append(f"CONFIRMED: sign-consistent across {list(valid)} "
                       f"AND CI excludes zero [{ci_lo:+.4f}, {ci_hi:+.4f}]")
        return "CONFIRMED", reasons
    reasons.insert(0, f"WEAK-MONITOR: full IC={full_ic:+.4f} > {ic_gate} gate "
                      "but fails confirmation")
    return "WEAK-MONITOR", reasons


def evaluate_signal(
    score_series: pd.Series,
    df: pd.DataFrame,
    freq: BarFreq,
    label: str = "signal",
    ic_deploy_gate: float = DEFAULT_IC_GATE,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    boot_seed: int = DEFAULT_BOOT_SEED,
    ci_level: float = DEFAULT_CI_LEVEL,
    embargo_bars: int = 30,
) -> dict:
    """
    Run the full validation battery on one signal.

    Horizons and bootstrap block length derive from `freq`
    (primary horizon = 24h forward return; block = 2 days of bars).
    Returns dict with segment_ic, bootstrap, regime_ic, verdict, split_info.
    """
    horizons = default_horizons(freq)
    primary_key = forward_return_col(PRIMARY_HORIZON_H).replace("forward_return_", "")
    primary_col = horizons[primary_key]
    block_len = freq.block_len

    missing = [c for c in list(horizons.values()) + ["rv_day"] if c not in df.columns]
    if missing:
        raise ValueError(f"[evaluate_signal:{label}] df missing columns: {missing}")
    if len(score_series) != len(df) or not score_series.index.equals(df.index):
        raise ValueError(f"[evaluate_signal:{label}] score index != df index")

    train_df, val_df, test_df = time_split(df, embargo_bars=embargo_bars)

    segment_ic = _compute_segment_ic(score_series, df, train_df, val_df, test_df, horizons)
    bootstrap = _compute_bootstrap(score_series, df, primary_col, block_len,
                                   n_bootstrap, boot_seed, ci_level)
    regime_ic = _compute_regime_ic(score_series, df, train_df, primary_col)
    verdict, verdict_reasons = _compute_verdict(
        segment_ic.get(primary_key, {}), bootstrap, ic_gate=ic_deploy_gate)

    return {
        "label": label,
        "segment_ic": segment_ic,
        "primary_horizon": primary_key,
        "bootstrap": bootstrap,
        "regime_ic": regime_ic,
        "vol_thresholds": (regime_ic.get("t1", np.nan), regime_ic.get("t2", np.nan)),
        "verdict": verdict,
        "verdict_reasons": verdict_reasons,
        "split_info": {
            "train_n": len(train_df), "val_n": len(val_df), "test_n": len(test_df),
            "train_start": str(train_df.index[0]), "train_end": str(train_df.index[-1]),
            "val_start": str(val_df.index[0]), "val_end": str(val_df.index[-1]),
            "test_start": str(test_df.index[0]), "test_end": str(test_df.index[-1]),
            "embargo_bars": embargo_bars,
        },
    }
