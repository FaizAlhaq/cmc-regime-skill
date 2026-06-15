"""
Causality mutation tests — the core 'is it real' proof.

Each test perturbs FUTURE data and asserts that PAST features / observations /
regimes / signals do not change. Any failure = lookahead bug.
"""

import numpy as np
import pandas as pd
import pytest

from core.freq import FREQ_4H
from core.vol_features import (make_vol_features, make_rv_target,
                               assert_feature_causality, VOL_FEATURE_COLS)
from core.regime_obs import (make_regime_obs, get_obs_array, assert_obs_causality,
                             fit_obs_scaler, scale_obs)
from core.regime_hmm import select_n_states, reorder_model_states, filtered_regimes
from signals.positioning import positioning_z_score, oi_funding_score
from signals.funding import funding_score

T_STARS = [400, 900, 1500]


def test_vol_features_causal(df_4h):
    for t in T_STARS:
        for feat in VOL_FEATURE_COLS:
            assert_feature_causality(df_4h, FREQ_4H, feat, t)


def test_regime_obs_causal(df_4h):
    df = make_vol_features(df_4h, FREQ_4H)
    for t in T_STARS:
        assert_obs_causality(df, t)


def test_obs_scaler_train_fit_only(df_4h):
    """The HMM obs StandardScaler must be fit on TRAIN obs ONLY (no lookahead).

    Its mean_/scale_ must equal the TRAIN-slice statistics and must NOT change
    when VAL/TEST (future) obs are mutated. Scaling must also be a pure affine
    transform driven solely by those TRAIN stats.
    """
    df = make_regime_obs(make_vol_features(df_4h, FREQ_4H))
    obs, _ = get_obs_array(df)
    cut = int(len(obs) * 0.70)            # TRAIN slice
    obs_train = obs[:cut]

    scaler = fit_obs_scaler(obs_train)
    # 1. scaler stats are exactly the TRAIN-slice mean/std
    np.testing.assert_allclose(scaler.mean_, obs_train.mean(axis=0), rtol=1e-12)
    np.testing.assert_allclose(scaler.scale_, obs_train.std(axis=0), rtol=1e-12)

    # 2. mutating everything AFTER the TRAIN cut must not change TRAIN stats
    obs_future_mut = obs.copy()
    obs_future_mut[cut:, :] = obs_future_mut[cut:, :] * 100.0 + 50.0
    scaler_mut = fit_obs_scaler(obs_future_mut[:cut])
    np.testing.assert_allclose(scaler.mean_, scaler_mut.mean_, rtol=1e-12)
    np.testing.assert_allclose(scaler.scale_, scaler_mut.scale_, rtol=1e-12)

    # 3. applying the TRAIN scaler is a deterministic affine map (TRAIN stats)
    z = scale_obs(scaler, obs_train)
    expected = (obs_train - scaler.mean_) / scaler.scale_
    np.testing.assert_allclose(z, expected, rtol=1e-10)


def test_target_is_forward_looking(df_4h):
    """The LABEL must depend on the future — sanity that it is a label."""
    out = make_rv_target(df_4h, FREQ_4H)
    t = 500
    df_mut = df_4h.copy()
    df_mut.iloc[t + 1:, df_mut.columns.get_loc("close")] *= 2.0
    out_mut = make_rv_target(df_mut, FREQ_4H)
    assert not np.isclose(out["rv_fwd_day"].iloc[t], out_mut["rv_fwd_day"].iloc[t]), \
        "forward target did not react to future mutation — wiring bug"
    # ...but mutating the PAST must not change the target at t
    df_mut2 = df_4h.copy()
    df_mut2.iloc[:t - FREQ_4H.w_day, df_mut2.columns.get_loc("close")] *= 2.0
    out_mut2 = make_rv_target(df_mut2, FREQ_4H)
    assert np.isclose(out["rv_fwd_day"].iloc[t], out_mut2["rv_fwd_day"].iloc[t],
                      rtol=1e-10)


@pytest.fixture(scope="module")
def fitted_hmm(df_4h):
    df = make_regime_obs(make_vol_features(df_4h, FREQ_4H))
    obs, idx = get_obs_array(df)
    train_obs = obs[: int(len(obs) * 0.7)]
    model, _ = select_n_states(train_obs, candidates=(2, 3),
                               n_restarts=2, n_iter=60)
    return reorder_model_states(model), obs


def test_filtered_regimes_causal(fitted_hmm):
    """Perturbing obs[t+1:] must not change filtered regimes[:t+1]."""
    model, obs = fitted_hmm
    t = 1200
    regimes_orig, _ = filtered_regimes(model, obs)

    obs_mut = obs.copy()
    obs_mut[t + 1:, :] += 5.0  # large perturbation of the future
    regimes_mut, _ = filtered_regimes(model, obs_mut)

    np.testing.assert_array_equal(
        regimes_orig[: t + 1], regimes_mut[: t + 1],
        err_msg="filtered regimes changed when future obs were mutated — LOOKAHEAD",
    )


def test_filtered_not_smoothed(fitted_hmm):
    """Viterbi/smoothed labels MAY differ from filtered — and future mutation
    must change smoothed-but-not-filtered past labels. Here we simply assert
    the filtered run is invariant under several future mutations."""
    model, obs = fitted_hmm
    t = 800
    base, _ = filtered_regimes(model, obs)
    rng = np.random.default_rng(0)
    for _ in range(3):
        obs_mut = obs.copy()
        obs_mut[t + 1:, :] = rng.normal(0, 3, size=obs_mut[t + 1:, :].shape)
        mut, _ = filtered_regimes(model, obs_mut)
        np.testing.assert_array_equal(base[: t + 1], mut[: t + 1])


@pytest.mark.parametrize("maker,col", [
    (positioning_z_score, "long_short_ratio"),
    (oi_funding_score, "open_interest"),
    (funding_score, "funding_rate"),
])
def test_signal_scores_causal(df_4h, maker, col):
    score = maker(df_4h, FREQ_4H)
    for t in T_STARS:
        if np.isnan(score.iloc[t]):
            continue
        df_mut = df_4h.copy()
        df_mut.iloc[t + 1:, df_mut.columns.get_loc(col)] *= 7.7
        df_mut.iloc[t + 1:, df_mut.columns.get_loc("close")] *= 3.3
        score_mut = maker(df_mut, FREQ_4H)
        assert np.isclose(score.iloc[t], score_mut.iloc[t], rtol=1e-10), \
            f"{score.name}[{t}] changed after future mutation — LOOKAHEAD"


def test_signal_warmup_nan(df_4h):
    """30-day z-window → warmup rows must be NaN (no partial-window peeking)."""
    score = funding_score(df_4h, FREQ_4H)
    w = FREQ_4H.bars(720.0)
    assert score.iloc[: w - 1].isna().all()
