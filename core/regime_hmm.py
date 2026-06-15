# ── VENDORED (one-way) from private research repo ───────────────────────
#   source: quant-trading/src/regime/hmm.py @ commit e67503c (module last touched 24d0766)
#   adaptations: numerical-stability hardening for the REAL daily run
#     (DEBUG_HMM_REAL.md): GaussianHMM defaults to covariance_type='diag' with
#     an explicit min_covar floor; _bic/select_n_states are covariance-type
#     aware; reorder_model_states is covariance-type aware and applies a
#     symmetrize + tiny-jitter PD guard; characterize_states accepts a TRAIN-fit
#     scaler to report state means in original units. Obs are standardized
#     upstream (core.regime_obs.fit_obs_scaler, TRAIN-fit-only).
#   Do NOT merge changes back into the research repo.
"""
HMM regime detection with strictly causal (filtered) inference (Task R2).

CRITICAL DESIGN PRINCIPLE:
  hmmlearn's model.predict() uses the Viterbi algorithm — this is SMOOTHING,
  meaning each state assignment uses ALL observations including future data.
  This would introduce MASSIVE lookahead in a backtest.

  The correct approach is FILTERED inference: the regime at time t is
  inferred from the FORWARD ALGORITHM posterior P(state_t | obs[0..t]).
  Only past observations influence the current regime label.

  We implement this in filtered_regimes() below.

References:
  - Corsi et al. (2005) on HMM volatility regimes
  - Hamilton (1989), original MS-AR paper
  - Rabiner (1989), forward-backward algorithm tutorial
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from scipy.stats import multivariate_normal, spearmanr
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────────────────────────
# Emission probability (using scipy for version robustness)
# ─────────────────────────────────────────────────────────

def _compute_log_emissions(model: GaussianHMM, obs: np.ndarray) -> np.ndarray:
    """
    Compute log P(obs[t] | state=k) for all t and all states k.

    Parameters
    ----------
    model : fitted GaussianHMM
    obs   : (n, d) observation array

    Returns
    -------
    log_emit : (n, K) float64 array
    """
    K = model.n_components
    n = len(obs)
    log_emit = np.zeros((n, K))
    for k in range(K):
        log_emit[:, k] = multivariate_normal.logpdf(
            obs,
            mean=model.means_[k],
            cov=model.covars_[k],
        )
    return log_emit


# ─────────────────────────────────────────────────────────
# Filtered (online) inference — the causal algorithm
# ─────────────────────────────────────────────────────────

def filtered_regimes(
    model: GaussianHMM,
    obs: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Online HMM inference using the FORWARD ALGORITHM (not Viterbi).

    At each time t, the filtered posterior is:
        P(state_t = k | obs[0..t])  ∝  alpha[t, k]

    The forward variable is updated as:
        alpha[t, k] = P(obs[t] | state_t=k) * sum_j alpha[t-1, j] * T[j, k]

    where T[j, k] = P(state_t=k | state_{t-1}=j).

    This is STRICTLY CAUSAL: the regime at t only depends on obs[0..t].

    ANTI-LOOKAHEAD TEST: perturbing obs[t+1..] must NOT change regimes[:t+1].
    This is guaranteed by the forward recursion structure.

    Parameters
    ----------
    model : fitted GaussianHMM (parameters estimated from TRAIN only)
    obs   : (n, d) array of all observations (may include val and test)

    Returns
    -------
    regimes      : (n,) int array — argmax filtered posterior at each t
    log_alpha    : (n, K) array — log-normalized filtered posteriors
    """
    n = len(obs)
    K = model.n_components

    # Log transition matrix: log_T[j, k] = log P(state_t=k | state_{t-1}=j)
    log_T = np.log(model.transmat_ + 1e-300)  # (K, K)

    # Log emission probabilities: (n, K)
    log_emit = _compute_log_emissions(model, obs)

    log_alpha = np.empty((n, K))

    # ── t = 0: initialize from prior ──────────────────────
    log_alpha[0] = np.log(model.startprob_ + 1e-300) + log_emit[0]
    log_alpha[0] -= logsumexp(log_alpha[0])  # normalize to log-probabilities

    # ── t = 1 .. n-1: forward recursion ───────────────────
    for t in range(1, n):
        # For each state k: sum over previous states j of alpha[t-1, j] * T[j, k]
        # log_alpha[t-1][:, None] shape: (K, 1)
        # log_T shape: (K, K)  — log_T[j, k]
        # Broadcasting gives (K, K) where [j, k] = log_alpha[t-1, j] + log_T[j, k]
        # logsumexp over axis=0 (sum over j) → shape (K,)
        log_pred = logsumexp(log_alpha[t - 1][:, np.newaxis] + log_T, axis=0)
        log_alpha[t] = log_pred + log_emit[t]
        log_alpha[t] -= logsumexp(log_alpha[t])  # normalize

    regimes = np.argmax(log_alpha, axis=1)
    return regimes, log_alpha


# ─────────────────────────────────────────────────────────
# BIC calculation
# ─────────────────────────────────────────────────────────

def _n_hmm_params(n_states: int, n_features: int,
                  covariance_type: str = "diag") -> int:
    """
    Free-parameter count for a GaussianHMM (used by BIC).

        startprob : K - 1
        transmat  : K * (K - 1)
        means     : K * d
        covars    : depends on covariance_type
                      full      -> K * d * (d + 1) / 2
                      diag      -> K * d
                      tied      -> d * (d + 1) / 2
                      spherical -> K
    """
    K, d = n_states, n_features
    base = (K - 1) + K * (K - 1) + K * d
    if covariance_type == "full":
        cov = K * d * (d + 1) // 2
    elif covariance_type == "diag":
        cov = K * d
    elif covariance_type == "tied":
        cov = d * (d + 1) // 2
    elif covariance_type == "spherical":
        cov = K
    else:
        raise ValueError(f"unknown covariance_type={covariance_type!r}")
    return base + cov


def _bic(log_lik: float, n_obs: int, n_states: int, n_features: int,
         covariance_type: str = "diag") -> float:
    """BIC for a GaussianHMM with the given covariance_type."""
    n_params = _n_hmm_params(n_states, n_features, covariance_type)
    return -2.0 * log_lik + n_params * np.log(n_obs)


# ─────────────────────────────────────────────────────────
# Model selection
# ─────────────────────────────────────────────────────────

def select_n_states(
    obs_train: np.ndarray,
    candidates: tuple = (2, 3, 4),
    n_iter: int = 200,
    n_restarts: int = 10,
    seed: int = 42,
    covariance_type: str = "diag",
    min_covar: float = 1e-3,
    tol: float = 1e-6,
) -> Tuple[GaussianHMM, pd.DataFrame]:
    """
    Select number of HMM states by BIC on training observations.

    For each candidate K, fits the model n_restarts times with different
    random seeds to avoid local optima, keeps the best log-likelihood run.
    Selects K with the lowest BIC.

    NOTE on numerical stability (see DEBUG_HMM_REAL.md): the obs columns are
    near-functionally-dependent at daily bars (log_rv_day = log|log_return|), so
    `covariance_type='diag'` with a `min_covar` floor is the stable default —
    full covariance lets a state collapse onto a single outlier and produce a
    non-PD covariance. Callers must pass STANDARDIZED obs (TRAIN-fit scaler).

    Parameters
    ----------
    obs_train       : (n_train, d) observation array — TRAIN ONLY (standardized)
    candidates      : tuple of n_states to try
    n_iter          : EM iterations per fit
    n_restarts      : random restarts per candidate K
    seed            : base random seed
    covariance_type : 'diag' (default, stable) / 'full' / 'tied' / 'spherical'
    min_covar       : floor added to the diagonal of the covariance (keeps PD)
    tol             : EM convergence tolerance

    Returns
    -------
    best_model : GaussianHMM with the selected K
    bic_table  : DataFrame with columns
                 [K, logL, BIC, n_params, converged, neg_ll_deltas]
    """
    rng = np.random.RandomState(seed)
    n_obs, d = obs_train.shape

    rows = []
    best_model = None
    best_bic = np.inf

    for K in candidates:
        best_ll = -np.inf
        best_local = None
        any_converged = False
        best_neg_deltas = None

        for _ in range(n_restarts):
            s = int(rng.randint(0, 99999))
            # hmmlearn reports tiny float-roundoff LL deltas at convergence via
            # the logging module (NOT warnings) as "Model is not converging".
            # We quiet that noisy logger and instead surface the real signal
            # honestly through the neg_ll_deltas column of the BIC table.
            hmm_log = logging.getLogger("hmmlearn")
            prev_level = hmm_log.level
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                hmm_log.setLevel(logging.ERROR)
                try:
                    m = GaussianHMM(
                        n_components=K,
                        covariance_type=covariance_type,
                        n_iter=n_iter,
                        random_state=s,
                        tol=tol,
                        init_params="stmc",
                        min_covar=min_covar,
                    )
                    m.fit(obs_train)
                    ll = m.score(obs_train)
                    if np.isfinite(ll) and ll > best_ll:
                        best_ll = ll
                        best_local = m
                        any_converged = m.monitor_.converged
                        hist = np.asarray(m.monitor_.history)
                        best_neg_deltas = (int((np.diff(hist) < -1e-9).sum())
                                           if len(hist) > 1 else 0)
                except Exception:
                    continue
                finally:
                    hmm_log.setLevel(prev_level)

        if best_local is None:
            continue

        K_bic = _bic(best_ll, n_obs, K, d, covariance_type)
        rows.append({
            "K": K,
            "logL": round(best_ll, 2),
            "BIC": round(K_bic, 2),
            "n_params": _n_hmm_params(K, d, covariance_type),
            "converged": any_converged,
            "neg_ll_deltas": best_neg_deltas,
        })

        if K_bic < best_bic:
            best_bic = K_bic
            best_model = best_local

    bic_table = pd.DataFrame(rows)
    return best_model, bic_table


# ─────────────────────────────────────────────────────────
# State ordering and characterization
# ─────────────────────────────────────────────────────────

def _sort_states_by_vol(model: GaussianHMM) -> np.ndarray:
    """
    Return a permutation that sorts states by mean log_rv_24 (feature index 1).
    State 0 after sorting = lowest vol (calm), last = highest vol (turbulent).
    """
    mean_log_rv = model.means_[:, 1]  # feature 1 = log_rv_24
    return np.argsort(mean_log_rv)


def reorder_model_states(model: GaussianHMM, jitter: float = 1e-6) -> GaussianHMM:
    """
    Return a NEW GaussianHMM with states reordered by ascending volatility.
    State 0 = calm (lowest vol), last state = turbulent (highest vol).
    The model is NOT re-fitted; only the parameter arrays are permuted.

    Covariance handling is covariance_type-aware. As a SECONDARY numerical-
    stability guard (the primary fix is diag covariance + standardized obs),
    full covariances are symmetrized and given a tiny diagonal jitter, and diag
    variances are floored, before assignment — so the hmmlearn PD validation in
    the covars_ setter cannot trip on a near-singular permuted covariance.
    """
    perm = _sort_states_by_vol(model)
    K = model.n_components
    cov_type = getattr(model, "covariance_type", "full")
    d = model.means_.shape[1]

    new_model = GaussianHMM(
        n_components=K,
        covariance_type=cov_type,
        n_iter=0,   # no fitting — parameters set manually
    )
    # Copy reordered parameters
    new_model.startprob_ = model.startprob_[perm]
    new_model.startprob_ /= new_model.startprob_.sum()  # renormalize (should be ~1)
    new_model.transmat_ = model.transmat_[np.ix_(perm, perm)]
    new_model.means_ = model.means_[perm]
    # Needed by hmmlearn internals for _compute_log_likelihood
    new_model.n_features = model.n_features if hasattr(model, "n_features") else d

    # model.covars_ getter always returns full (K, d, d) matrices; convert back
    # to the shape the covariance_type setter expects, with a PD guard.
    covars_full = np.asarray(model.covars_)[perm]
    if cov_type == "full":
        sym = 0.5 * (covars_full + covars_full.swapaxes(-1, -2))
        new_model.covars_ = sym + jitter * np.eye(d)
    elif cov_type == "diag":
        new_model.covars_ = np.maximum(
            np.diagonal(covars_full, axis1=-2, axis2=-1), jitter)
    elif cov_type == "tied":
        sym = 0.5 * (covars_full[0] + covars_full[0].T)
        new_model.covars_ = sym + jitter * np.eye(d)
    elif cov_type == "spherical":
        new_model.covars_ = np.maximum(covars_full[:, 0, 0], jitter)
    else:
        raise ValueError(f"unknown covariance_type={cov_type!r}")
    return new_model


@dataclass
class StateProfile:
    """Human-readable profile for one HMM state."""
    state_id: int          # 0-indexed (0 = calmest)
    label: str             # e.g. "calm", "normal", "turbulent"
    mean_return: float     # mean log_return per bar
    mean_log_rv: float     # mean log(rv_24) per bar
    mean_rv: float         # mean rv_24 per bar (exp scale)
    occupancy_train: float # fraction of TRAIN bars in this state
    self_trans_prob: float # P(stay in state k)
    avg_duration: float    # expected bars before leaving: 1/(1-self_trans)


_STATE_LABELS_2 = ["calm", "turbulent"]
_STATE_LABELS_3 = ["calm", "normal", "turbulent"]
_STATE_LABELS_4 = ["calm", "low-vol", "high-vol", "turbulent"]


def characterize_states(
    model: GaussianHMM,
    regimes_train: np.ndarray,
    scaler: Optional[StandardScaler] = None,
) -> list[StateProfile]:
    """
    Build human-readable profiles for each state.

    Parameters
    ----------
    model         : reordered GaussianHMM (state 0 = lowest vol)
    regimes_train : (n_train,) filtered regime assignments on TRAIN
    scaler        : the TRAIN-fit StandardScaler used to standardize the obs.
                    When provided, state means are inverse-transformed so that
                    mean_return / mean_log_rv / mean_rv are reported in ORIGINAL
                    units (the model itself is fit in standardized space).

    Returns
    -------
    List of StateProfile, one per state, sorted by vol (0 = calmest).
    """
    K = model.n_components

    if K == 2:
        labels = _STATE_LABELS_2
    elif K == 3:
        labels = _STATE_LABELS_3
    elif K == 4:
        labels = _STATE_LABELS_4
    else:
        labels = [f"state_{k}" for k in range(K)]

    # State means in original units (model is fit on standardized obs).
    means = (scaler.inverse_transform(model.means_)
             if scaler is not None else np.asarray(model.means_))

    profiles = []
    for k in range(K):
        occ = float(np.mean(regimes_train == k))
        self_p = float(model.transmat_[k, k])
        avg_dur = 1.0 / (1.0 - self_p) if self_p < 1.0 else np.inf

        profiles.append(StateProfile(
            state_id=k,
            label=labels[k],
            mean_return=float(means[k, 0]),
            mean_log_rv=float(means[k, 1]),
            mean_rv=float(np.exp(means[k, 1])),
            occupancy_train=occ,
            self_trans_prob=self_p,
            avg_duration=avg_dur,
        ))
    return profiles


# ─────────────────────────────────────────────────────────
# Regime series: map from obs-valid index back to full df index
# ─────────────────────────────────────────────────────────

def make_regime_series(
    regimes_valid: np.ndarray,
    valid_index: pd.Index,
    full_index: pd.Index,
) -> pd.Series:
    """
    Map filtered regime labels (on the non-NaN obs index) back to
    the full DataFrame index. NaN for warmup bars.

    Returns pd.Series of dtype float (NaN where obs was NaN, int regime otherwise).
    """
    s = pd.Series(np.nan, index=full_index, dtype=float)
    s.loc[valid_index] = regimes_valid.astype(float)
    return s


# ─────────────────────────────────────────────────────────
# Regime stability metrics
# ─────────────────────────────────────────────────────────

def regime_stability(regime_series: pd.Series) -> dict:
    """
    Compute regime stability metrics for a segment.

    Returns dict with:
        n_switches       : number of regime transitions
        switch_rate      : switches per bar
        occupancy        : {regime_k: fraction}
        mean_run_lengths : {regime_k: average consecutive bars}
    """
    valid = regime_series.dropna().astype(int)
    n = len(valid)
    if n < 2:
        return {}

    switches = int((valid.values[1:] != valid.values[:-1]).sum())

    # Occupancy
    states = sorted(valid.unique())
    occ = {int(k): float(np.mean(valid == k)) for k in states}

    # Mean run lengths per state
    run_lengths: dict[int, list[int]] = {k: [] for k in states}
    current = valid.iloc[0]
    run = 1
    for v in valid.iloc[1:]:
        if v == current:
            run += 1
        else:
            run_lengths[int(current)].append(run)
            current = v
            run = 1
    run_lengths[int(current)].append(run)

    mean_runs = {k: float(np.mean(v)) if v else float("nan")
                 for k, v in run_lengths.items()}

    return {
        "n_valid_bars": n,
        "n_switches": switches,
        "switch_rate": switches / max(n - 1, 1),
        "occupancy": occ,
        "mean_run_lengths": mean_runs,
    }
