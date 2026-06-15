---
name: cmc-regime-switch
description: >
  Generate AND rigorously validate a regime-switching crypto strategy spec from
  CoinMarketCap daily OHLCV. Detects volatility regimes with a strictly causal
  (filtered) Gaussian HMM, applies a regime gate + volatility-target position
  sizing (long-or-flat, never levered), backtests cost-aware on embargoed
  time-ordered splits, and returns an agent-ready strategy_spec.json with
  segment-labeled (TRAIN/VAL/TEST) evidence vs a buy-and-hold benchmark. Reports
  edge OR no-edge HONESTLY and never fabricates alpha. Spec generator only —
  produces NO live-trading or execution code.
when-to-use: >
  Use when an agent or user asks to (a) build or evaluate a regime-aware crypto
  strategy or volatility risk overlay for BTC/ETH from CMC data, (b) produce a
  machine-readable position-sizing/risk policy a downstream agent can consume,
  or (c) audit a strategy idea with leakage-free backtesting discipline and an
  honest out-of-sample readout. Do NOT use for order execution, live trading, or
  price-prediction / alpha claims.
---

# cmc-regime-switch — Strategy Skill

You (the agent) produce `strategy_spec.json` + `report.md` by running the scripts
below on **daily** bars. The default and submission path is **daily, OHLCV-only**
(`--bar-hours 24 --ohlcv-only`): CMC has no funding / open-interest / long-short
history, so no derivative signal is used or claimed. Report whatever the
validation says — including **no edge**. Never fabricate or imply alpha.

## Prerequisites

```bash
pip install -r requirements.txt   # pinned; Python 3.10+
```

Optional: a CoinMarketCap key in `.env` (`CMC_API_KEY=…`). Without a key, use
`--offline` — the committed sample under `data/sample/` is **SYNTHETIC** and every
output is watermarked accordingly. Never commit `.env` or raw CMC data (ToS).

## Step 1 — Get data (real fetch, or the synthetic sample)

With a key (the only networked step):

```bash
python scripts/fetch.py --asset BTC --bar-hours 24      # real daily OHLCV → data/cache/
```

Without a key: skip this and pass `--offline` below (SYNTHETIC sample).

## Step 2 — Fit on TRAIN only, emit policy + cost-aware backtest

```bash
python scripts/run_strategy.py --asset BTC --bar-hours 24 --ohlcv-only
#   add --offline to use the SYNTHETIC sample instead of the real cache
```

This single command:

1. applies data-quality gates (volume>0, monotonic timestamps, no dups, and
   flags stale-flat zero-return bars);
2. builds causal volatility features and forward-return labels (returns, not
   levels);
3. splits 70/15/15 time-ordered with a 30-bar embargo;
4. fits a StandardScaler and HAR-RV **on TRAIN only**, then a BIC-selected
   Gaussian HMM (`covariance_type='diag'`) **on TRAIN only**, and runs
   **filtered** (forward-algorithm) regime inference over the full history;
5. applies the frozen regime policy (calm / low-vol → full size; high-vol →
   de-risk; turbulent → flat) with vol-target sizing, **clipped to [0, 1.0]
   (long-or-flat, never levered or short)**;
6. backtests each segment separately with fees + slippage and a 1-bar position
   lag, plus a **buy-and-hold benchmark on the same basis**;
7. writes `results/<run>/strategy_spec.json` and `series.csv`.

## Step 3 — Render the report and dashboard

```bash
python scripts/make_report.py --run results/latest
```

Writes `report.md` (every metric TRAIN/VAL/TEST-labeled) and `gui/data.js`;
`gui/dashboard.html` then opens by double-click — no server, no build step.

## Step 4 — Return the agent-ready JSON and read it honestly

Return `strategy_spec.json` to the caller. Contract highlights:

- `meta.synthetic_data` — if true, state plainly the evidence is a pipeline
  demonstration on synthetic data, NOT market results.
- `meta.verdict` / `meta.finding` — the honest readout (e.g. "no edge vs
  buy-and-hold; reduces drawdown") and the validated observation. Quote them.
- `regime_model` — K=4 states sorted by vol, `covariance_type`, `fit_segment:
  "TRAIN only"`, `inference: filtered (causal forward algorithm)`, params hash,
  and `degenerate_states` (report any, never hide them).
- `policy` — per-regime `{signal, position_factor}` the downstream agent consumes.
- `sizing` — `target_vol / HAR_forecast` clipped to **[0, 1.0]** (long-or-flat).
- `evidence` — per-segment backtest metrics for **strategy AND buy-and-hold**,
  plus regime stability and split info. Compare against the benchmark.
- `guards` — `live_trading: false`, lookahead controls, causality tests.

## Hard rules for the agent

1. Never present TRAIN metrics as performance. The honest out-of-sample readout
   is VAL/TEST. Always say which segment a number came from.
2. Report the verdict as-is — **including "no edge" / underperformance vs
   buy-and-hold.** Never tune parameters or flip the regime gate to manufacture a
   better-looking curve; that is overfitting.
3. Never claim live or future edge. The deliverable is a *validated spec*.
4. Never write execution/order code based on this Skill.
5. If the pipeline errors, report the error; never fabricate numbers.

## Verify

```bash
pytest -q     # 41 tests: causality mutations, annualization, engine, e2e, scaler-train-only
```
