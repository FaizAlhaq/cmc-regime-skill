"""Backtest engine math, split integrity, quality gates."""

import numpy as np
import pandas as pd
import pytest

from core.backtest import run_backtest
from core.split import time_split
from core.sizing import compute_position_scale
from adapter.quality import apply_gates


def _idx(n):
    return pd.date_range("2025-01-01", periods=n, freq="4h", tz="UTC")


# ── engine ───────────────────────────────────────────────────────────────

def test_engine_pnl_hand_computed():
    """4 bars, known returns, known costs — check pnl bar by bar."""
    idx = _idx(4)
    direction = pd.Series([1.0, 1.0, -1.0, 0.0], index=idx)
    scale = pd.Series(1.0, index=idx)
    ret = pd.Series([0.00, 0.01, -0.02, 0.005], index=idx)
    res = run_backtest(direction, scale, ret, bars_per_year=2190.0,
                       fee_bps=10.0, slip_bps=0.0, label="TRAIN")
    c = 10.0 / 10_000.0
    expected = [
        -1.0 * c,                 # entry long
        1.0 * 0.01 - 0.0,         # hold long, no change
        1.0 * -0.02 - 2.0 * c,    # earn long on down bar, flip to short (|Δ|=2)
        -1.0 * 0.005 - 1.0 * c,   # short earns -ret, exit (|Δ|=1)
    ]
    np.testing.assert_allclose(res.pnl.values, expected, rtol=1e-12)
    np.testing.assert_allclose(res.equity.values, np.cumprod(1 + np.array(expected)),
                               rtol=1e-12)


def test_buy_and_hold_uses_simple_returns():
    """Regression (DEBUG_HMM_REAL.md): the engine compounds multiplicatively
    (cumprod(1+pnl)), so it needs SIMPLE returns. B&H of a big-move asset must
    equal the raw price ratio; feeding LOG returns understates it (the bug)."""
    idx = _idx(50)
    rng = np.random.default_rng(3)
    close = 100.0 * np.exp(np.cumsum(0.05 * rng.standard_normal(50)))  # big moves
    s = pd.Series(close, index=idx)
    log_ret = np.log(s).diff().fillna(0.0)
    simple_ret = np.expm1(log_ret)
    one = pd.Series(1.0, index=idx)
    true_bh = close[-1] / close[0] - 1.0

    res_simple = run_backtest(one, one, simple_ret, bars_per_year=365.0,
                              fee_bps=0.0, slip_bps=0.0, label="TRAIN")
    assert res_simple.total_return == pytest.approx(true_bh, rel=1e-9)
    # log returns through the SAME engine do NOT recover true B&H — the original bug
    res_log = run_backtest(one, one, log_ret, bars_per_year=365.0,
                           fee_bps=0.0, slip_bps=0.0, label="TRAIN")
    assert abs(res_log.total_return - true_bh) > 1e-3


def test_engine_position_lags_signal():
    """Direction at bar t earns bar t+1's return, never bar t's."""
    idx = _idx(3)
    # big return at bar 1; direction flips long exactly at bar 1
    direction = pd.Series([0.0, 1.0, 0.0], index=idx)
    ret = pd.Series([0.0, 0.10, 0.0], index=idx)
    res = run_backtest(direction, pd.Series(1.0, index=idx), ret,
                       bars_per_year=2190.0, fee_bps=0.0, slip_bps=0.0,
                       label="TRAIN")
    # position[0]=0 → pnl[1] must be 0 * 0.10 = 0 (no same-bar execution)
    assert res.pnl.iloc[1] == 0.0


def test_engine_requires_positive_bpy():
    idx = _idx(10)
    s = pd.Series(1.0, index=idx)
    with pytest.raises(ValueError):
        run_backtest(s, s, s * 0.001, bars_per_year=0.0, label="TRAIN")


# ── split ────────────────────────────────────────────────────────────────

def test_split_no_overlap_with_embargo():
    df = pd.DataFrame({"x": np.arange(1000.0)}, index=_idx(1000))
    train, val, test = time_split(df, embargo_bars=30)
    assert train.index.max() < val.index.min() < test.index.min()
    # embargo gap exists
    all_used = set(train.index) | set(val.index) | set(test.index)
    assert len(all_used) == len(train) + len(val) + len(test)
    assert len(df) - len(all_used) == 60  # 2 embargos of 30 bars


# ── sizing ───────────────────────────────────────────────────────────────

def test_sizing_regime_gating_and_clip():
    idx = _idx(200)
    har_hat = pd.Series(np.log(0.02), index=idx)  # constant forecast
    train_mask = pd.Series([True] * 140 + [False] * 60, index=idx)
    regime = pd.Series([0] * 50 + [1] * 50 + [2] * 50 + [3] * 50, index=idx,
                       dtype=float)
    scale = compute_position_scale(har_hat, train_mask, regime_full=regime,
                                   n_states=4, max_leverage=3.0)
    # constant forecast → raw scale 1.0; factors 1/1/0.5/0
    assert scale.iloc[10] == pytest.approx(1.0)
    assert scale.iloc[60] == pytest.approx(1.0)
    assert scale.iloc[110] == pytest.approx(0.5)
    assert scale.iloc[160] == pytest.approx(0.0)
    assert (scale.dropna() <= 3.0).all() and (scale.dropna() >= 0.0).all()


def test_sizing_nan_regime_is_flat():
    idx = _idx(100)
    har_hat = pd.Series(np.log(0.02), index=idx)
    train_mask = pd.Series(True, index=idx)
    regime = pd.Series(np.nan, index=idx)
    scale = compute_position_scale(har_hat, train_mask, regime_full=regime,
                                   n_states=4)
    assert (scale == 0.0).all(), "warmup bars without regime must not trade"


# ── quality gates ────────────────────────────────────────────────────────

def _ok_frame(n=100):
    idx = _idx(n)
    close = 100 + np.arange(n) * 0.1
    return pd.DataFrame({
        "open": close, "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": np.ones(n),
    }, index=idx)


def test_gates_pass_clean_frame():
    df, rep = apply_gates(_ok_frame(), name="t", bar_hours=4.0)
    assert rep.n_rows_out == 100 and rep.n_gaps == 0


def test_gates_fail_zero_volume():
    df = _ok_frame()
    df.iloc[5, df.columns.get_loc("volume")] = 0.0
    with pytest.raises(ValueError, match="volume"):
        apply_gates(df, name="t")


def test_gates_fail_nonmonotonic():
    df = _ok_frame()
    df = pd.concat([df.iloc[50:], df.iloc[:50]])
    with pytest.raises(ValueError, match="monoton"):
        apply_gates(df, name="t")


def test_gates_drop_duplicates():
    df = _ok_frame()
    df = pd.concat([df, df.iloc[[10]]]).sort_index()
    out, rep = apply_gates(df, name="t", bar_hours=4.0)
    assert rep.n_dups_dropped == 1 and len(out) == 100


def test_gates_fail_ohlc_violation():
    df = _ok_frame()
    df.iloc[7, df.columns.get_loc("high")] = df.iloc[7]["close"] * 0.5
    with pytest.raises(ValueError, match="OHLC"):
        apply_gates(df, name="t")
