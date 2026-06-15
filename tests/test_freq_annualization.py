"""Prove annualization and window parameterization are correct at 1H and 4H."""

import numpy as np
import pandas as pd
import pytest

from core.freq import BarFreq, FREQ_1H, FREQ_4H, FREQ_1D
from core.backtest import run_backtest
from core.vol_features import make_vol_features, make_rv_target


def test_window_bars():
    assert (FREQ_1H.w_day, FREQ_1H.w_week, FREQ_1H.w_month) == (24, 168, 720)
    assert (FREQ_4H.w_day, FREQ_4H.w_week, FREQ_4H.w_month) == (6, 42, 180)
    assert FREQ_1H.bars_per_year == 8760.0
    assert FREQ_4H.bars_per_year == 2190.0
    assert FREQ_1H.block_len == 48 and FREQ_4H.block_len == 12


def test_daily_window_bars_and_bpy():
    """D1 lock: daily bars_per_year MUST resolve to 365 (crypto trades daily)."""
    assert FREQ_1D.bars_per_year == 365.0
    assert FREQ_1D.bars_per_day == 1
    assert (FREQ_1D.w_day, FREQ_1D.w_week, FREQ_1D.w_month) == (1, 7, 30)
    assert FREQ_1D.block_len == 2
    # embargo (30 bars) fully covers the longest feature window (w_month=30)
    assert 30 >= FREQ_1D.w_month


def test_daily_sharpe_scales_vs_hourly():
    """Same pnl stream: 1H Sharpe / daily Sharpe == sqrt(8760/365) == sqrt(24)."""
    s_1h = _const_pnl_sharpe(8760.0)
    s_1d = _const_pnl_sharpe(365.0)
    assert s_1h / s_1d == pytest.approx(np.sqrt(8760.0 / 365.0), rel=1e-9)
    assert s_1h / s_1d == pytest.approx(np.sqrt(24.0), rel=1e-9)


def test_invalid_freq():
    with pytest.raises(ValueError):
        BarFreq(0.0)
    with pytest.raises(ValueError):
        BarFreq(48.0).bars(24.0)  # window shorter than one bar


def _const_pnl_sharpe(bars_per_year: float, n: int = 1000) -> float:
    """Backtest with constant per-bar return and zero costs."""
    idx = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    ret = pd.Series(0.0005 + 0.001 * rng.standard_normal(n), index=idx)
    direction = pd.Series(1.0, index=idx)
    scale = pd.Series(1.0, index=idx)
    res = run_backtest(direction, scale, ret, bars_per_year=bars_per_year,
                       fee_bps=0.0, slip_bps=0.0, label="TRAIN")
    return res.sharpe


def test_sharpe_scales_with_sqrt_bars_per_year():
    """Same pnl stream → Sharpe ratio scales exactly with sqrt(bars_per_year)."""
    s_1h = _const_pnl_sharpe(8760.0)
    s_4h = _const_pnl_sharpe(2190.0)
    assert s_1h / s_4h == pytest.approx(np.sqrt(8760.0 / 2190.0), rel=1e-9)
    assert s_1h / s_4h == pytest.approx(2.0, rel=1e-9)


def test_realized_vol_annualization_consistent():
    """An iid per-bar vol sigma must annualize to sigma*sqrt(bpy) at each freq."""
    n = 20_000
    sigma = 0.002
    rng = np.random.default_rng(1)
    for freq in (FREQ_1H, FREQ_4H, FREQ_1D):
        idx = pd.date_range("2024-01-01", periods=n,
                            freq=pd.Timedelta(hours=freq.bar_hours), tz="UTC")
        ret = pd.Series(sigma * rng.standard_normal(n), index=idx)
        res = run_backtest(pd.Series(1.0, index=idx), pd.Series(1.0, index=idx),
                           ret, bars_per_year=freq.bars_per_year,
                           fee_bps=0.0, slip_bps=0.0, label="TRAIN")
        expected = sigma * np.sqrt(freq.bars_per_year)
        assert res.realized_vol_ann == pytest.approx(expected, rel=0.05)


def test_rv_windows_span_same_clock_time(df_1h, df_4h, df_1d):
    """rv_day at 1H uses 24 bars; at 4H uses 6 bars; at daily uses 1 bar —
    all span 24 clock hours. First non-NaN at index[w_day] (log-diff costs 1 bar)."""
    f1 = make_vol_features(df_1h, FREQ_1H)
    f4 = make_vol_features(df_4h, FREQ_4H)
    fd = make_vol_features(df_1d, FREQ_1D)
    assert f1["rv_day"].notna().idxmax() == f1.index[24]
    assert f4["rv_day"].notna().idxmax() == f4.index[6]
    assert fd["rv_day"].notna().idxmax() == fd.index[1]


def test_forward_target_horizon_matches_freq(df_4h):
    out = make_rv_target(df_4h, FREQ_4H)
    # last w_day rows must be NaN — horizon extends beyond data
    assert out["rv_fwd_day"].iloc[-FREQ_4H.w_day:].isna().all()
    assert out["rv_fwd_day"].iloc[-FREQ_4H.w_day - 1:].notna().iloc[0]
