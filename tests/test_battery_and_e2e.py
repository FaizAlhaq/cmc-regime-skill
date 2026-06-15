"""Battery output contract + end-to-end offline run (the release gate)."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.freq import FREQ_4H
from core.vol_features import make_vol_features, make_rv_target, make_forward_returns
from core.battery import evaluate_signal, default_horizons
from signals.funding import funding_score

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def df_feat(df_4h):
    df = make_vol_features(df_4h, FREQ_4H)
    df = make_rv_target(df, FREQ_4H)
    df = make_forward_returns(df, FREQ_4H)
    return df


def test_battery_contract(df_feat):
    score = funding_score(df_feat, FREQ_4H)
    res = evaluate_signal(score, df_feat, FREQ_4H, label="funding-test")
    assert res["verdict"] in ("CONFIRMED", "WEAK-MONITOR", "DEAD")
    for seg in ("TRAIN", "VAL", "TEST", "FULL"):
        assert seg in res["segment_ic"][res["primary_horizon"]]
    assert res["bootstrap"]["ci_lo"] <= res["bootstrap"]["ci_hi"]
    assert res["split_info"]["embargo_bars"] == 30


def test_battery_horizons_follow_freq():
    h = default_horizons(FREQ_4H)
    assert set(h) == {"4h", "24h", "72h"}
    assert h["24h"] == "forward_return_24h"


def test_battery_thresholds_train_only(df_feat):
    """Tercile thresholds must come from TRAIN — verify by mutating TEST vol."""
    score = funding_score(df_feat, FREQ_4H)
    res1 = evaluate_signal(score, df_feat, FREQ_4H)
    df_mut = df_feat.copy()
    n = len(df_mut)
    df_mut.iloc[int(n * 0.9):, df_mut.columns.get_loc("rv_day")] *= 100.0
    res2 = evaluate_signal(score, df_mut, FREQ_4H)
    assert res1["vol_thresholds"] == res2["vol_thresholds"], \
        "tercile thresholds moved when TEST data changed — lookahead"


def test_e2e_offline_run(tmp_path):
    """Fresh end-to-end offline run: spec exists, is honest, segment-labeled."""
    out = tmp_path / "run"
    cmd = [sys.executable, str(ROOT / "scripts" / "run_strategy.py"),
           "--offline", "--asset", "BTC", "--bar-hours", "4",
           "--n-restarts", "2", "--k-candidates", "2,3",
           "--out", str(out)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stderr[-2000:]

    spec = json.loads((out / "strategy_spec.json").read_text())

    # honesty guards
    assert spec["meta"]["synthetic_data"] is True
    assert "SYNTHETIC" in spec["meta"]["data_source"].upper()
    assert spec["guards"]["live_trading"] is False
    assert spec["guards"]["execution_code_present"] is False
    assert spec["regime_model"]["fit_segment"] == "TRAIN only"
    assert "filtered" in spec["regime_model"]["inference"]

    # segment labeling
    assert spec["evidence"]["segment_labeled"] is True
    for seg in ("TRAIN", "VAL", "TEST"):
        assert seg in spec["evidence"]["backtest"]
        assert "sharpe" in spec["evidence"]["backtest"][seg]
    for seg in ("TRAIN", "VAL", "TEST", "FULL"):
        assert seg in spec["evidence"]["active_signal_battery"]["ic"]

    # policy covers all states; factors within [0, 1]
    K = spec["regime_model"]["K"]
    assert len(spec["policy"]) == K
    for p in spec["policy"].values():
        assert 0.0 <= p["position_factor"] <= 1.0

    # series for the dashboard
    series = pd.read_csv(out / "series.csv", index_col=0)
    assert {"close", "regime", "segment", "direction",
            "position_scale"}.issubset(series.columns)
    assert set(series["segment"].unique()) <= {"TRAIN", "VAL", "TEST", "EMBARGO"}


def test_no_execution_code_anywhere():
    """Grep guard: no order-placement / exchange-execution code in the repo."""
    banned = ["create_order", "place_order", "submit_order", "new_order",
              "cancel_order", "/order", "binance.client", "ccxt"]
    hits = []
    for p in ROOT.rglob("*.py"):
        if "tests" in p.parts:
            continue
        text = p.read_text(encoding="utf-8", errors="ignore").lower()
        for b in banned:
            if b in text:
                hits.append((p.name, b))
    assert not hits, f"execution-like code found: {hits}"
