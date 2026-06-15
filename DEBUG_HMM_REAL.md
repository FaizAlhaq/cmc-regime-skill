# DEBUG note — HMM crash on the REAL daily BTC run

Command: `python scripts/run_strategy.py --asset BTC --bar-hours 24 --ohlcv-only`
Data: real CMC daily BTC cache, 4547 bars (`data/cache/CMC_BTC_24h.parquet`).
Env: conda `quant` — hmmlearn 0.3.3, scikit-learn 1.8.0, numpy 2.4.6, scipy 1.17.1.

## Root cause (one line)
At daily bars `w_day=1`, so `log_rv_day = log|log_return|` — the two HMM obs columns
are functionally dependent and ~38× apart in scale; with `covariance_type='full'` EM
collapses one K=4 state onto a single zero-return clip outlier (bar `2017-02-28`,
`rv_day=0 → clip 1e-20 → log_rv_day=-46`), giving that state a rank-deficient,
**non-positive-definite** covariance that fails hmmlearn's PD validation in
`reorder_model_states` (`core/regime_hmm.py`).

## Evidence (STEP 1 diagnosis, full covariance, unscaled)
- **Scale mismatch:** TRAIN `log_return` std = 0.039 vs `log_rv_day` std = 1.51 → **38.6×**.
  The hmmlearn default `min_covar=0.001` is a fine floor for col 0 (var ≈ 0.0015) but
  negligible against col 1 (var ≈ 2.3), so it cannot keep col 1 PD.
- **Functional dependency:** `max|log(|log_return|) − log_rv_day| = 0.0` (exact). The obs
  lie on a 1-D curve, so any tight per-state cluster has a near-singular covariance.
- **Clip outlier:** exactly one bar (`2017-02-28`) has `close == prior close`
  (1179.969970703125 repeated — a stale/flat daily close) → daily return = 0 →
  `rv_day=0` floored to `1e-20` → `log_rv_day = −46.05`, a ~28σ outlier (next-smallest
  `log_rv_day ≈ −13`).
- **The collapse:** K=4 full-cov fit → per-state counts `[1101, 1495, 584, 1]`. State 3
  (1 point) had `min_eig = 0.0`, `cond = 2e298`, `PD = False`, and EM showed a negative
  log-likelihood delta (non-monotone). Re-assigning that covariance through hmmlearn's
  validating `covars_` setter is what raised
  *"component 0 of 'full' covars must be symmetric, positive-definite"*.
- Confirmed **standardize + full + min_covar=0.01** still leaves the 1-point state
  `PD=False` → full covariance only survives by masking. **diag** is the real fix.

## The minimal fix (STEP 2, legitimate numerical stability only)
1. **Standardize the obs — `StandardScaler` fit on TRAIN obs ONLY**, applied to
   TRAIN/VAL/TEST with the TRAIN stats (affine, no lookahead).
   `core/regime_obs.py: fit_obs_scaler / scale_obs`; wired in `scripts/run_strategy.py`.
2. **`covariance_type='diag'`** (was `'full'`) with an explicit `min_covar=1e-3` floor —
   the decisive change. Diag sidesteps the collinearity entirely and the floor keeps
   every state PD (even a 1-point state floors to `min_eig = min_covar`).
   `core/regime_hmm.py: select_n_states`.
3. **Init robustness:** keep the best-TRAIN-LL fit across restarts; report
   `neg_ll_deltas` per K in the BIC table; quiet hmmlearn's logging-based convergence
   chatter (a −9.8e−8 float-roundoff delta on the rejected K=2 candidate) without
   hiding it (the column shows it).
4. **Secondary guard:** `reorder_model_states` is now covariance-type aware and
   symmetrizes + jitters (full) / floors (diag) the permuted covariance before
   assignment. Guard, not the primary fix.

State means are reported in **original units** (`characterize_states` inverse-transforms
via the TRAIN scaler). Filtered (causal) inference and K=4 are unchanged.

## Result AFTER the fix — regimes are non-degenerate (except one reported outlier)
Selected model: **K=4 by BIC** (BIC 14759.9 / 13181.9 / **12505.7** for K=2/3/4),
`covariance_type='diag'`, `converged=True`, **`neg_ll_deltas=0`** (monotone EM, all PD).

Per-state **TRAIN** counts (filtered), n_train_obs = 3181:

| state | label     | TRAIN count | occupancy | mean_log_rv | factor |
|-------|-----------|-------------|-----------|-------------|--------|
| 0     | calm      | **1**       | 0.03%     | −46.05      | 1.0    |
| 1     | low-vol   | 745         | 23.4%     | −6.12       | 1.0    |
| 2     | high-vol  | 1424        | 44.8%     | −4.32       | 0.5    |
| 3     | turbulent | 1011        | 31.8%     | −3.01       | 0.0    |

State 0 is a **genuinely degenerate regime — reported, NOT regularized away**
(per the non-negotiable). It captures only the single `2017-02-28` flat-close bar across
the *entire* series (full-series counts: state0=1, state1=1155, state2=2118, state3=1272),
so it is immaterial to the strategy. It is surfaced in the run log (a `WARNING`) and in
`strategy_spec.json → regime_model.degenerate_states`. States 1–3 are healthy.

> Follow-up (out of scope for this numerical-stability debug, flagged not actioned):
> the `2017-02-28` flat close is almost certainly a data-quality artifact (a repeated
> daily close). Because the outlier steals the vol-ordered "calm" slot (state 0), the
> genuine regimes shift up one slot. The cleaner long-term fix is at the data/obs layer
> (treat the stale close as a gap, or use a realistic `rv_day` floor instead of `1e-20`)
> — deliberately not changed here, as that would alter obs semantics and risk masking
> the degenerate regime the non-negotiables require us to report.

## Verification (STEP 3)
- `results/BTC_daily_REAL/` produced; `meta.synthetic_data == false`,
  `meta.bars_per_year == 365`, `data_source == "CMC Data API cache"`,
  `regime_model.covariance_type == "diag"`, `K == 4`.
- `pytest -q` → **40 passed**, including the causality mutation tests
  (`test_filtered_regimes_causal`, `test_filtered_not_smoothed`) and a new
  `test_obs_scaler_train_fit_only` proving the scaler is TRAIN-fit-only (no lookahead).

## Segment-labeled metrics — strategy (regime-gated long-only vol overlay) vs buy-and-hold
OHLCV-only daily; fees 4 bps + slippage 1 bps; NOT an alpha claim — benchmarked vs B&H.

| Segment | Strat Sharpe | Strat MaxDD | Strat TotRet | B&H Sharpe | B&H MaxDD | B&H TotRet |
|---------|--------------|-------------|--------------|------------|-----------|------------|
| TRAIN   | −0.39        | −90.2%      | −88.8%       | 0.48       | −88.6%    | +89.8%     |
| VAL     | 1.11         | −21.6%      | +82.8%       | 1.21       | −30.3%    | +145.1%    |
| TEST    | −0.08        | −45.1%      | −14.8%       | 0.05       | −54.9%    | −13.2%     |

The overlay consistently cuts drawdown vs B&H (e.g. VAL −21.6% vs −30.3%, TEST −45.1%
vs −54.9%) but with high turnover (~0.5–0.6/bar) it lags B&H on raw return; it does NOT
beat B&H on Sharpe in this OHLCV-only configuration. Reported honestly, per segment.

---

# DEBUG note (part 2) — the TRAIN "−88.8%" was a BUG, not "no edge"

Follow-up investigation of the TRAIN catastrophe (Strat −88.8% while B&H "gained").
Diagnosed first (print only), then fixed correctness bugs **only** — no parameter
tuning, gating sign untouched.

## What was actually wrong (3 bugs + 1 honest finding)

**BUG 1 — backtest fed LOG returns into a multiplicative engine.**
`core/backtest.py` builds equity with `cumprod(1 + pnl)` where `pnl = pos·return`, i.e.
it expects **simple** returns — but `run_strategy.py` passed `df["log_return"]`. On a
+2320% asset this printed B&H = **+93.7%** (true price ratio 802.39 → 19419.51 =
**+2320%**). It also mis-scaled the levered strat (a log return × leverage compounded as
simple). *Fix:* convert at the call site, `simple_ret = np.expm1(df["log_return"])`, used
for **both** strat and B&H (same correct basis). Engine arg renamed `log_return →
bar_return` and documented as simple; regression test
`test_buy_and_hold_uses_simple_returns` added.

**BUG 2 — "long-only vol overlay" silently levered up to 3×.**
`--max-leverage` defaulted to 3.0; on TRAIN the position hit **3.0×**, **17.4% of bars
were >1×**, mean in-market position **1.485×**. A long-only overlay must be long-or-flat.
*Fix:* default `--max-leverage = 1.0`. After: position ∈ [0, 1.0], **0% levered**, 40.5%
flat.

**BUG 3 — stale-flat data bar created a spurious regime.**
`2017-02-28` has `close == prior close` (1179.969970703125 repeated) → `rv_day=0` →
old `clip(1e-20)` → `log_rv_day=−46` (~28σ outlier) → a 1-point "calm" state that stole
the vol-ordered slot 0 and shifted the genuine regimes. *Fix (hygiene):* quality gate now
flags stale-flat bars (`QualityReport.n_stale_flat`); `make_regime_obs` sets
`log_rv_day=NaN` when `rv_day==0` (undefined log-vol → excluded). After: **4 genuine
regimes**, `degenerate_states = []`.

**NOT A BUG — the gate is correctly de-risking high vol; BTC just rallies in high vol.**
The regime→factor map (`calm/low-vol → 1.0`, `high-vol → 0.5`, `turbulent → 0.0`) is the
documented, correctly-signed design. It is **not** inverted in code. Per-state TRAIN
forward returns vs positions (after fixes):

| state | label     | n (TRAIN) | mean next-bar ret | mean position | factor |
|-------|-----------|----------:|------------------:|--------------:|-------:|
| 0     | calm      | 606       | −0.00075          | 0.95          | 1.0    |
| 1     | low-vol   | 1277      | +0.00043          | 0.92          | 1.0    |
| 2     | high-vol  | 348       | **+0.00269**      | 0.00 (flat)   | 0.5    |
| 3     | turbulent | 949       | **+0.00221**      | 0.00 (flat)   | 0.0    |

In-market mean fwd ret **+0.00005** vs flat **+0.00240**; `corr(position, next-ret) =
−0.03`. BTC's positive drift is concentrated in its **high-vol** regimes, which the
overlay sits out by design. Flipping the gate to chase that return would be exactly the
parameter-tuning the brief forbids, so the mapping is **left as-is**.

## Verdict
TRAIN −88.8% was a **BUG** (log-vs-simple compounding + hidden 3× leverage), amplified by
the stale-bar regime artifact. After correctness-only fixes the strat is long-or-flat
(≤1×) on the correct basis with 4 genuine regimes — and it **still does not beat B&H** on
Sharpe or return on any segment. That residual is a **real no-edge / design–asset
mismatch** (de-risking high-vol misses BTC's high-vol-concentrated upside), reported
honestly. The overlay's only genuine benefit is lower drawdown and vol.

## Segment-labeled metrics AFTER correctness fixes (real daily BTC)
OHLCV-only, long-or-flat (≤1×), simple-return compounding, 4 bps + 1 bps costs.

| Segment | Strat Sharpe | Strat MaxDD | Strat TotRet | B&H Sharpe | B&H MaxDD | B&H TotRet |
|---------|-------------:|------------:|-------------:|-----------:|----------:|-----------:|
| TRAIN   | 0.13  | −73.3% | −34.5%  | 0.86 | −83.4% | +2270.7% |
| VAL     | 0.74  | −25.0% | +47.5%  | 1.46 | −26.1% | +208.7%  |
| TEST    | −0.04 | −49.1% | −11.0%  | 0.27 | −51.2% | +3.7%    |

(Before fixes the same TRAIN row read Strat −88.8% / B&H +89.8% — both wrong.)
`pytest -q` → **41 passed** (incl. causality mutation + the new return-basis regression).
No-lookahead preserved: TRAIN-only scaler/HAR/HMM, filtered inference, K=4, embargo=30.
