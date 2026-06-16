#!/usr/bin/env python3
"""
run_strategy.py — end-to-end: cache → regime → switch → backtest → spec.

Pipeline (every fitted object is TRAIN-only; inference is causal):
  1. load data   (--offline → SYNTHETIC sample cache; else data/cache parquet)
  2. quality gates
  3. vol features (causal) + forward-vol target + forward-return labels
  4. time-ordered 70/15/15 split, 30-bar embargo
  5. HAR-RV fit on TRAIN → causal vol forecast for all bars
  6. HMM: BIC-select K on TRAIN obs, reorder by vol, FILTERED inference full history
  7. signals: candidate scores, sign calibrated on TRAIN, battery per candidate,
     active signal = best TRAIN |IC| among candidates
  8. policy: regime-gated direction + vol-target sizing (factors frozen a priori)
  9. backtest per segment (TRAIN / VAL / TEST separately, cost-aware)
 10. emit results/<run>/strategy_spec.json + series for reporting

Usage:
  python scripts/run_strategy.py --offline --asset BTC --bar-hours 4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.freq import BarFreq
from core.split import time_split
from core.vol_features import (make_vol_features, make_rv_target,
                               make_forward_returns, forward_return_col)
from core.vol_models import HARModel
from core.regime_obs import (make_regime_obs, get_obs_array,
                             fit_obs_scaler, scale_obs)
from core.regime_hmm import (select_n_states, reorder_model_states,
                             filtered_regimes, make_regime_series,
                             characterize_states, regime_stability)
from core.sizing import compute_position_scale, compute_target_vol, _get_regime_factors
from core.backtest import run_backtest
from core.battery import evaluate_signal
from signals.positioning import (positioning_z_score, oi_funding_score,
                                 calibrate_sign, to_direction)
from signals.funding import funding_score
from adapter.quality import apply_gates
from adapter import synthetic

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("run_strategy")

EMBARGO_BARS = 30
SPEC_VERSION = "1.1"

# OHLCV-only baseline exposure (D2: derivatives cut from the CMC path, so there
# is NO directional alpha signal). +1 = long BTC beta. The skill's contribution
# is the REGIME GATE + VOL TARGET applied on top of this baseline — NOT an alpha
# claim; results are benchmarked against buy-and-hold. This baseline direction is
# a reversible strategy choice flagged for review (see DECISIONS_FOR_FAIZ).
BASELINE_DIRECTION = 1.0


def load_data(offline: bool, asset: str, bar_hours: float) -> pd.DataFrame:
    if offline:
        p = synthetic.sample_path(ROOT / "data" / "sample", asset, bar_hours)
        if not p.exists():
            log.warning(f"{p} missing — generating SYNTHETIC sample now")
            df = synthetic.generate(asset=asset, bar_hours=bar_hours)
            p.parent.mkdir(parents=True, exist_ok=True)
            synthetic.write_sample(df, p)
        df = synthetic.read_sample(p)
        log.info(f"loaded {p.name}: {len(df)} bars — DATA IS SYNTHETIC")
        return df
    p = ROOT / "data" / "cache" / f"CMC_{asset}_{int(bar_hours)}h.parquet"
    if not p.exists():
        raise SystemExit(f"No cache at {p}. Run scripts/fetch.py first, "
                         "or use --offline for the SYNTHETIC sample.")
    df = pd.read_parquet(p)
    log.info(f"loaded {p.name}: {len(df)} bars (real CMC cache)")
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="BTC")
    ap.add_argument("--bar-hours", type=float, default=24.0,
                    help="bar size in hours; 24 = DAILY (D1, submission default)")
    ap.add_argument("--ohlcv-only", action="store_true",
                    help="force OHLCV-only mode: ignore any derivative signals and "
                         "run the regime-gated long-only vol overlay (D2 CMC path). "
                         "Auto-enabled when no derivative columns are present.")
    ap.add_argument("--offline", action="store_true",
                    help="use committed SYNTHETIC sample cache (no key/network)")
    ap.add_argument("--fee-bps", type=float, default=4.0)
    ap.add_argument("--slip-bps", type=float, default=1.0)
    ap.add_argument("--max-leverage", type=float, default=1.0,
                    help="cap on position scale. Default 1.0 = long-or-flat: this "
                         "is a LONG-ONLY vol overlay, never levered. (Was 3.0, which "
                         "silently ran up to 3x — a bug for a long-only overlay.)")
    ap.add_argument("--k-candidates", default="2,3,4")
    ap.add_argument("--n-restarts", type=int, default=6)
    ap.add_argument("--out", default=None, help="results dir (default results/<run-id>)")
    args = ap.parse_args()

    freq = BarFreq(args.bar_hours)
    # Naming: daily REAL run → results/BTC_daily_REAL (mission target).
    freq_token = "daily" if float(args.bar_hours) == 24.0 else freq.label
    source_token = "SYNTHETIC" if args.offline else "REAL"
    run_id = f"{args.asset}_{freq_token}_{source_token}"
    out_dir = Path(args.out) if args.out else ROOT / "results" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1-2. data + gates
    df = load_data(args.offline, args.asset, args.bar_hours)
    data_source = df.attrs.get("data_source",
                               "SYNTHETIC" if df.get("synthetic", pd.Series(False)).any()
                               else "CMC Data API cache")
    df, qrep = apply_gates(df, name=run_id, bar_hours=args.bar_hours)
    log.info(f"quality: {qrep.as_dict()}")

    # 3. features + labels (features causal; labels intentionally forward)
    df = make_vol_features(df, freq)
    df = make_rv_target(df, freq)
    df = make_forward_returns(df, freq)
    df = make_regime_obs(df)

    # 4. split
    train_df, val_df, test_df = time_split(df, embargo_bars=EMBARGO_BARS)
    train_mask = df.index.isin(train_df.index)
    train_mask_s = pd.Series(train_mask, index=df.index)
    seg_of = pd.Series("EMBARGO", index=df.index, dtype=object)
    seg_of[train_df.index] = "TRAIN"
    seg_of[val_df.index] = "VAL"
    seg_of[test_df.index] = "TEST"

    # 5. HAR fit on TRAIN only
    har = HARModel().fit(train_df)
    har_hat = har.predict(df)
    log.info(f"HAR params (TRAIN fit): {har.params}")

    # 6. HMM — fit/select on TRAIN obs only; filtered inference on full history.
    #    Obs are STANDARDIZED with a StandardScaler fit on TRAIN obs ONLY (no
    #    lookahead): the two obs columns differ ~38x in scale (at daily bars
    #    log_rv_day = log|log_return|), which makes full covariance ill-
    #    conditioned. We use covariance_type='diag' for numerical stability.
    #    See DEBUG_HMM_REAL.md.
    obs_train, _ = get_obs_array(train_df)
    obs_scaler = fit_obs_scaler(obs_train)        # TRAIN-fit-only
    obs_train_s = scale_obs(obs_scaler, obs_train)
    candidates = tuple(int(k) for k in args.k_candidates.split(","))
    model_raw, bic_table = select_n_states(obs_train_s, candidates=candidates,
                                           n_restarts=args.n_restarts)
    if model_raw is None:
        raise SystemExit("HMM selection failed for all K candidates")
    model = reorder_model_states(model_raw)
    K = model.n_components
    log.info(f"BIC table (TRAIN):\n{bic_table}")
    log.info(f"obs scaler (TRAIN-fit): mean={obs_scaler.mean_} "
             f"scale={obs_scaler.scale_}")

    obs_full, valid_idx = get_obs_array(df)
    obs_full_s = scale_obs(obs_scaler, obs_full)  # same TRAIN stats
    regimes_valid, _ = filtered_regimes(model, obs_full_s)
    regime_s = make_regime_series(regimes_valid, valid_idx, df.index)

    regimes_train_valid, _ = filtered_regimes(model, obs_train_s)
    profiles = characterize_states(model, regimes_train_valid, scaler=obs_scaler)

    # Per-state TRAIN point counts — REPORT degenerate regimes (do NOT mask).
    train_counts = {k: int(np.sum(regimes_train_valid == k)) for k in range(K)}
    n_train_obs = int(len(regimes_train_valid))
    degenerate_threshold = max(5, int(0.005 * n_train_obs))
    degenerate_states = [
        {"id": p.state_id, "label": p.label,
         "train_count": train_counts[p.state_id],
         "occupancy_train": p.occupancy_train,
         "mean_log_rv": p.mean_log_rv, "mean_rv": p.mean_rv}
        for p in profiles if train_counts[p.state_id] < degenerate_threshold
    ]
    log.info(f"HMM per-state TRAIN counts (filtered): {train_counts} "
             f"(n_train_obs={n_train_obs})")
    for ds in degenerate_states:
        log.warning(f"DEGENERATE regime reported (not masked): state {ds['id']} "
                    f"'{ds['label']}' has only {ds['train_count']} TRAIN bars "
                    f"(occupancy={ds['occupancy_train']:.4%}, mean_rv={ds['mean_rv']:.2e})")

    params_hash = hashlib.sha256(
        np.concatenate([model.startprob_.ravel(), model.transmat_.ravel(),
                        model.means_.ravel(), np.asarray(model.covars_).ravel()]
                       ).tobytes()).hexdigest()[:16]

    # 7. signals — derivative-positioning candidates (only if data present).
    #    D2: derivatives are unavailable on CMC, so the daily CMC path is
    #    OHLCV-only and skips this entirely.
    factors = _get_regime_factors(K)
    labels = [p.label for p in profiles]
    full_risk = {p.state_id: (factors[p.state_id] >= 1.0) for p in profiles}

    deriv_cols_present = any(c in df.columns for c in
                            ("long_short_ratio", "open_interest", "funding_rate"))
    ohlcv_only = bool(args.ohlcv_only or not deriv_cols_present)

    battery_results, signs, train_ics = {}, {}, {}
    candidates_scores: dict[str, pd.Series] = {}
    if not ohlcv_only:
        fwd24 = df[forward_return_col(24.0)]
        for name, fn in (("positioning_z", positioning_z_score),
                         ("oi_funding", oi_funding_score),
                         ("funding", funding_score)):
            try:
                candidates_scores[name] = fn(df, freq)
            except ValueError as e:
                log.warning(f"signal {name} unavailable: {e}")
        for name, score in candidates_scores.items():
            sign = calibrate_sign(score, fwd24, train_df.index)
            res = evaluate_signal(score * sign, df, freq,
                                  label=f"{name} (sign={sign:+d}, TRAIN-calibrated)",
                                  embargo_bars=EMBARGO_BARS)
            battery_results[name] = res
            signs[name] = sign
            train_ics[name] = res["segment_ic"][res["primary_horizon"]]["TRAIN"]
            log.info(f"battery[{name}]: sign={sign:+d} verdict={res['verdict']} "
                     f"TRAIN IC={train_ics[name]:+.4f}")
        if not candidates_scores:
            log.warning("no derivative signal candidates → OHLCV-only mode")
            ohlcv_only = True

    # 8. policy + direction
    if ohlcv_only:
        # OHLCV-only: no directional alpha signal. Baseline = long beta,
        # FLATTENED by the regime gate in de-risked/turbulent states.
        # Skill contribution = regime gate + vol target. NOT an alpha claim.
        active_name = "baseline_long"
        signs = {active_name: int(np.sign(BASELINE_DIRECTION)) or 1}
        active_score = pd.Series(BASELINE_DIRECTION, index=df.index, name=active_name)
        policy = {p.label: {
            "signal": "baseline_long" if full_risk[p.state_id] else "flat",
            "position_factor": factors[p.state_id],
        } for p in profiles}
        log.info(f"OHLCV-only mode: baseline_long (regime-gated, vol-targeted); "
                 "no directional alpha signal (derivatives excluded, D2).")
    else:
        # Derivative path: active signal chosen on TRAIN |IC| only.
        active_name = max(train_ics, key=lambda k: abs(train_ics[k])
                          if not np.isnan(train_ics[k]) else -1.0)
        active_score = candidates_scores[active_name] * signs[active_name]
        policy = {p.label: {
            "signal": active_name if full_risk[p.state_id] else "none",
            "position_factor": factors[p.state_id],
        } for p in profiles}
        log.info(f"active signal: {active_name} (chosen on TRAIN |IC| only)")

    # Direction: discrete {-1,0,+1} for a real signal; constant baseline for
    # OHLCV-only. Either way, gated to 0 outside full-risk regimes.
    direction_raw = (active_score.clip(-1.0, 1.0) if ohlcv_only
                     else to_direction(active_score)).fillna(0.0)
    signal_allowed = regime_s.map(
        {p.state_id: 1.0 if full_risk[p.state_id] else 0.0 for p in profiles}
    ).fillna(0.0) > 0.5
    direction = direction_raw.where(signal_allowed, 0.0)

    target_vol = compute_target_vol(har_hat, train_mask_s)
    scale = compute_position_scale(har_hat, train_mask_s, regime_full=regime_s,
                                   n_states=K, max_leverage=args.max_leverage)

    # 9. backtest per segment — every result carries its segment label.
    #    Plus a buy-and-hold benchmark (long, unit size) for honest comparison.
    #    The engine compounds multiplicatively (cumprod(1+pnl)), so it needs
    #    SIMPLE per-bar returns. df["log_return"] is a LOG return — convert with
    #    expm1 so strat and B&H share the SAME correct basis (a log return here
    #    understated B&H ~25x and mis-scaled leverage). See DEBUG_HMM_REAL.md.
    simple_ret = np.expm1(df["log_return"])
    bt, bench = {}, {}
    bh_dir = pd.Series(1.0, index=df.index)
    bh_scale = pd.Series(1.0, index=df.index)
    for seg_name, seg in (("TRAIN", train_df), ("VAL", val_df), ("TEST", test_df)):
        bt[seg_name] = run_backtest(
            direction.loc[seg.index], scale.loc[seg.index], simple_ret.loc[seg.index],
            bars_per_year=freq.bars_per_year, fee_bps=args.fee_bps,
            slip_bps=args.slip_bps, label=seg_name)
        bench[seg_name] = run_backtest(
            bh_dir.loc[seg.index], bh_scale.loc[seg.index], simple_ret.loc[seg.index],
            bars_per_year=freq.bars_per_year, fee_bps=args.fee_bps,
            slip_bps=args.slip_bps, label=f"{seg_name}-buyhold")
        log.info(f"backtest[{seg_name}]: {bt[seg_name].summary_dict()}")
        log.info(f"buyhold [{seg_name}]: {bench[seg_name].summary_dict()}")

    # 10. emit
    # split info (independent of battery so OHLCV-only mode has it too)
    split_info = {
        "train_n": len(train_df), "val_n": len(val_df), "test_n": len(test_df),
        "train_start": str(train_df.index[0]), "train_end": str(train_df.index[-1]),
        "val_start": str(val_df.index[0]), "val_end": str(val_df.index[-1]),
        "test_start": str(test_df.index[0]), "test_end": str(test_df.index[-1]),
        "embargo_bars": EMBARGO_BARS,
    }

    if ohlcv_only:
        signal_spec = {
            "active": "baseline_long",
            "mode": "OHLCV-only (regime-gated long-only vol overlay)",
            "sign": signs["baseline_long"],
            "directional_alpha_signal": None,
            "note": (f"Derivatives unavailable on CMC (D2) → no positioning/funding "
                     f"signal. Baseline exposure = long {args.asset} beta, flattened by the "
                     f"regime gate in de-risked/turbulent states. This is NOT an "
                     f"alpha claim; see the buy-and-hold benchmark."),
            "candidates_evaluated": {},
        }
        battery_evidence = None
    else:
        ab = battery_results[active_name]
        seg_ic = ab["segment_ic"][ab["primary_horizon"]]
        signal_spec = {
            "active": active_name,
            "mode": "derivative positioning signal",
            "sign": signs[active_name],
            "sign_calibration": "TRAIN Spearman IC only, frozen before VAL/TEST",
            "threshold": 0.5,
            "candidates_evaluated": {
                n: {"sign": signs[n], "verdict": battery_results[n]["verdict"]}
                for n in battery_results},
        }
        battery_evidence = {
            "verdict": ab["verdict"],
            "verdict_reasons": ab["verdict_reasons"],
            "primary_horizon": ab["primary_horizon"],
            "ic": {k: seg_ic.get(k) for k in ("TRAIN", "VAL", "TEST", "FULL")},
            "bootstrap_ci_90": [ab["bootstrap"]["ci_lo"], ab["bootstrap"]["ci_hi"]],
            "bootstrap_n": ab["bootstrap"]["n"],
        }

    spec = {
        "meta": {
            "skill": "cmc-regime-switch", "version": SPEC_VERSION,
            "asset": args.asset, "bar_freq": freq.label,
            "bars_per_year": freq.bars_per_year,
            "strategy_type": ("regime-gated long-only vol overlay (OHLCV-only)"
                              if ohlcv_only else
                              "regime-gated directional signal + vol overlay"),
            "derivatives_excluded": ohlcv_only,
            "derivatives_note": ("funding/open-interest/long-short are unavailable "
                                 "on CMC and are excluded (D2)") if ohlcv_only else None,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "data_source": data_source,
            "synthetic_data": bool(args.offline),
            "disclaimer": ("SYNTHETIC sample data — pipeline demonstration only, "
                           "NOT market evidence") if args.offline else
                          "Backtest on historical data; regime discipline + risk "
                          "transform only; NO claim of live edge or alpha. Compare "
                          "the overlay against the buy-and-hold benchmark.",
        },
        "regime_model": {
            "type": "GaussianHMM", "K": K,
            "covariance_type": model.covariance_type,
            "obs": ["log_return", "log_rv_day"],
            "obs_standardization": {
                "method": "StandardScaler",
                "fit_segment": "TRAIN only",
                "mean": [float(x) for x in obs_scaler.mean_],
                "scale": [float(x) for x in obs_scaler.scale_],
                "note": "fit on TRAIN obs only; applied to TRAIN/VAL/TEST alike "
                        "(affine, no lookahead). State means below are reported "
                        "in original units (inverse-transformed).",
            },
            "inference": "filtered (causal forward algorithm) — never Viterbi",
            "params_hash": params_hash, "fit_segment": "TRAIN only",
            "bic_table": bic_table.to_dict(orient="records"),
            "states": [{
                "id": p.state_id, "label": p.label,
                "mean_return": p.mean_return, "mean_log_rv": p.mean_log_rv,
                "occupancy_train": p.occupancy_train,
                "train_count": train_counts[p.state_id],
                "self_trans_prob": p.self_trans_prob,
                "avg_duration_bars": p.avg_duration,
            } for p in profiles],
            "degenerate_states": degenerate_states,
            "degenerate_state_threshold_train_bars": degenerate_threshold,
        },
        "policy": policy,
        "signal": signal_spec,
        "sizing": {
            "rule": f"target_vol / HAR_forecast, clip [0, {args.max_leverage}]",
            "target_vol_source": "TRAIN median of HAR forecast vol",
            "target_vol": target_vol,
            "regime_factors": {labels[k]: factors[k] for k in range(K)},
        },
        "costs": {"fee_bps": args.fee_bps, "slip_bps": args.slip_bps},
        "evidence": {
            "segment_labeled": True,
            "active_signal_battery": battery_evidence,
            "backtest": {
                seg: {
                    "sharpe": r.sharpe, "maxdd": r.max_dd,
                    "pf": (r.profit_factor if np.isfinite(r.profit_factor) else None),
                    "total_return": r.total_return,
                    "turnover_per_bar": r.turnover_per_bar,
                    "realized_vol_ann": r.realized_vol_ann, "n_bars": r.n_bars,
                } for seg, r in bt.items()},
            "benchmark_buy_and_hold": {
                seg: {
                    "sharpe": r.sharpe, "maxdd": r.max_dd,
                    "total_return": r.total_return,
                    "realized_vol_ann": r.realized_vol_ann, "n_bars": r.n_bars,
                } for seg, r in bench.items()},
            "regime_stability": {
                seg: regime_stability(regime_s.loc[sdf.index])
                for seg, sdf in (("TRAIN", train_df), ("VAL", val_df), ("TEST", test_df))},
            "split": split_info,
        },
        "guards": {
            "live_trading": False,
            "execution_code_present": False,
            "causality_mutation_tests": "see tests/ — run `pytest -q`",
            "lookahead": f"none — embargo={EMBARGO_BARS} bars, all scalers/"
                         "thresholds/HAR/HMM fit on TRAIN only, filtered HMM inference",
            "features_use_returns_not_levels": True,
        },
    }

    def _clean(o):
        """Recursively convert numpy scalars and non-str dict keys."""
        if isinstance(o, dict):
            return {str(k): _clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_clean(v) for v in o]
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating, float)):
            f = float(o)
            return f if np.isfinite(f) else None
        if isinstance(o, np.bool_):
            return bool(o)
        return o

    spec = _clean(spec)
    (out_dir / "strategy_spec.json").write_text(
        json.dumps(spec, indent=2, allow_nan=False))

    # series for report/dashboard
    series = pd.DataFrame({
        "close": df["close"], "regime": regime_s, "segment": seg_of,
        "direction": direction, "position_scale": scale,
        "score": active_score,
    })
    eq = pd.concat([bt[s].equity.rename(s) for s in ("TRAIN", "VAL", "TEST")], axis=1)
    series = series.join(eq)
    series.to_csv(out_dir / "series.csv", date_format="%Y-%m-%dT%H:%M:%SZ")

    # pointer for make_report (symlinks may not survive all filesystems)
    (ROOT / "results").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "LATEST.txt").write_text(str(out_dir))

    log.info(f"wrote {out_dir / 'strategy_spec.json'}")
    log.info(f"wrote {out_dir / 'series.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
