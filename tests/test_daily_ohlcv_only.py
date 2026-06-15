"""
Daily (D1) + OHLCV-only (D2) coverage.

Proves:
  1. causality holds at DAILY bars (features + obs invariant to future mutation);
  2. derivative signals correctly REFUSE to run without their columns;
  3. the OHLCV-only daily pipeline runs end-to-end, is segment-labeled, carries
     bars_per_year==365, makes NO alpha claim, and emits a buy-and-hold benchmark.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.freq import FREQ_1D
from core.vol_features import (make_vol_features, assert_feature_causality,
                               VOL_FEATURE_COLS)
from core.regime_obs import make_regime_obs, assert_obs_causality
from signals.positioning import positioning_z_score, oi_funding_score
from signals.funding import funding_score

ROOT = Path(__file__).resolve().parents[1]
T_STARS = [200, 800, 1500]


# ── 1. causality at DAILY ──────────────────────────────────────────────────
def test_vol_features_causal_daily(df_1d):
    for t in T_STARS:
        for feat in VOL_FEATURE_COLS:
            assert_feature_causality(df_1d, FREQ_1D, feat, t)


def test_regime_obs_causal_daily(df_1d):
    df = make_vol_features(df_1d, FREQ_1D)
    for t in T_STARS:
        assert_obs_causality(df, t)


# ── 2. derivative signals refuse to run without their columns ───────────────
@pytest.mark.parametrize("maker", [positioning_z_score, oi_funding_score,
                                   funding_score])
def test_signals_require_derivatives(df_1d_ohlcv, maker):
    with pytest.raises(ValueError):
        maker(df_1d_ohlcv, FREQ_1D)


# ── 3. end-to-end OHLCV-only daily run ─────────────────────────────────────
def test_e2e_ohlcv_only_daily(tmp_path):
    out = tmp_path / "run"
    cmd = [sys.executable, str(ROOT / "scripts" / "run_strategy.py"),
           "--offline", "--asset", "BTC", "--bar-hours", "24", "--ohlcv-only",
           "--n-restarts", "2", "--k-candidates", "2,3", "--out", str(out)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stderr[-2000:]

    spec = json.loads((out / "strategy_spec.json").read_text())

    # daily lock
    assert spec["meta"]["bars_per_year"] == 365.0
    # OHLCV-only honesty
    assert spec["meta"]["derivatives_excluded"] is True
    assert "OHLCV-only" in spec["meta"]["strategy_type"]
    assert spec["meta"]["synthetic_data"] is True
    assert "SYNTHETIC" in spec["meta"]["data_source"].upper()
    # no directional alpha signal → no battery, no alpha claim
    assert spec["signal"]["active"] == "baseline_long"
    assert spec["evidence"]["active_signal_battery"] is None
    # synthetic run must disclaim itself as non-evidence
    assert "NOT market evidence" in spec["meta"]["disclaimer"]

    # segment labeling on BOTH strategy and benchmark
    for seg in ("TRAIN", "VAL", "TEST"):
        assert "sharpe" in spec["evidence"]["backtest"][seg]
        assert "sharpe" in spec["evidence"]["benchmark_buy_and_hold"][seg]

    # causal-inference + guards intact
    assert "filtered" in spec["regime_model"]["inference"]
    assert spec["regime_model"]["fit_segment"] == "TRAIN only"
    assert spec["guards"]["live_trading"] is False

    # policy covers all states; factors within [0, 1]
    K = spec["regime_model"]["K"]
    assert len(spec["policy"]) == K
    for p in spec["policy"].values():
        assert 0.0 <= p["position_factor"] <= 1.0
        assert p["signal"] in ("baseline_long", "flat")

    # series sane
    series = pd.read_csv(out / "series.csv", index_col=0)
    assert {"close", "regime", "segment", "direction",
            "position_scale"}.issubset(series.columns)
    assert set(series["segment"].unique()) <= {"TRAIN", "VAL", "TEST", "EMBARGO"}
