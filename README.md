# cmc-regime-skill

**An LLM Skill that generates *and rigorously validates* regime-switching crypto
strategies from CoinMarketCap data — and reports edge or no-edge honestly.** Built for
the BNB Hackathon Track 2 "Strategy Skills (Powered by CMC)".

The Skill detects volatility regimes with a **strictly causal (filtered) Gaussian HMM**,
applies a regime gate + volatility-target position sizing, backtests cost-aware on
time-ordered embargoed splits, and emits an agent-ready `strategy_spec.json` plus a
segment-labeled `report.md`. It is a **strategy-spec generator, not a live-trading agent** —
there is no execution or order-placement code in this repo, by design.

The worked example here is a **daily, OHLCV-only BTC overlay on real CMC data
(2014→2026)**. Its honest verdict: **no edge** — it does not beat buy-and-hold. We report
that rather than tuning it away. (Details below.)

## What this Skill actually contributes

1. **A leakage-free validation framework.** No-lookahead time splits + 30-bar embargo,
   all parameters (scaler, HAR, HMM) fit on **TRAIN only**, **filtered/causal** HMM
   inference (never Viterbi), cost-aware backtest, every metric **TRAIN/VAL/TEST-labeled**,
   and **causality-mutation tests** that perturb the future and assert the past doesn't move.
2. **One validated, honest finding** (see below): in crypto, de-risking high-volatility
   regimes *misses upside* — the opposite of the equities intuition the design encodes.

No alpha is claimed anywhere. The deliverable is a *validated spec* and an *honest readout*.

## Quickstart (offline, no API key)

```bash
pip install -r requirements.txt
python scripts/run_strategy.py --offline --asset BTC --bar-hours 24 --ohlcv-only
python scripts/make_report.py  --run results/latest
pytest -q                                   # 41 tests: causality mutation, engine, e2e
# then open gui/dashboard.html (double-click — no server, no build)
```

`--offline` uses the committed **SYNTHETIC** sample under `data/sample/` (clearly watermarked
SYNTHETIC in every output). It demonstrates the *pipeline*, never market edge. To reproduce
the **real** daily result you need a CoinMarketCap key (`CMC_API_KEY=…` in `.env`):

```bash
python scripts/fetch.py        --asset BTC --bar-hours 24      # pulls real OHLCV → data/cache/
python scripts/run_strategy.py --asset BTC --bar-hours 24 --ohlcv-only
```

> `.env` and `data/cache/` are git-ignored: never commit your key or redistribute raw CMC
> data (ToS). The real daily run writes `results/BTC_daily_REAL/`.

## Honest result — real daily BTC, OHLCV-only (2014→2026)

Daily bars (`bars_per_year=365`), derivatives excluded (CMC has no funding/OI/long-short
history), **4 genuine volatility regimes**, **long-or-flat** (`max_leverage=1.0`, never
levered or short). Strategy = regime-gated, vol-targeted overlay; benchmark = buy-and-hold.
Costs 4 bps fee + 1 bps slippage; position lagged 1 bar. Every number is segment-labeled.

| Segment | Strat Sharpe | Strat MaxDD | Strat Return | B&H Sharpe | B&H MaxDD | B&H Return |
|---------|-------------:|------------:|-------------:|-----------:|----------:|-----------:|
| **TRAIN** | 0.13  | −73.3% | −34.5%  | 0.86 | −83.4% | +2270.7% |
| **VAL**   | 0.74  | −25.0% | +47.5%  | 1.46 | −26.1% | +208.7%  |
| **TEST**  | −0.04 | −49.1% | −11.0%  | 0.27 | −51.2% | +3.7%    |

**Read it straight:** the overlay **reduces drawdown and volatility on all three segments**
(e.g. TEST MaxDD −49.1% vs −51.2%, realized vol 33.5% vs 45.7%) — but it **does not beat
buy-and-hold on Sharpe or return on any segment.** **No edge in this configuration.**

### The validated finding (why)

BTC's positive drift is **concentrated in its high-volatility regimes** (bull runs are
volatile). The overlay's risk gate de-risks exactly those regimes, so it sits out the
upside. Per-state TRAIN forward returns confirm it: the in-market (calm/low-vol) regimes are
flat/slightly negative, while the high-vol and turbulent regimes — which the gate flattens —
carry the gains. This is the **opposite of the equities intuition** that "high vol = cut
risk." The regime gate is **correctly signed for that design**; flipping it to chase the
high-vol upside would be overfitting to this sample, so **we did not**. We report the
no-edge result instead. Full diagnosis: [DEBUG_HMM_REAL.md](DEBUG_HMM_REAL.md).

## Why you can trust the numbers

1. **No lookahead.** Time-ordered 70/15/15 split, 30-bar embargo; scaler, HAR coefficients,
   and HMM parameters fit on **TRAIN only** and applied forward.
2. **Filtered HMM inference.** Regime at *t* uses the forward-algorithm posterior
   P(state_t | obs[0..t]) — never Viterbi smoothing (which leaks the future).
3. **Causality mutation tests.** The suite perturbs future bars and asserts past features,
   observations, regimes, and the TRAIN-fit scaler do not change.
4. **Segment-labeled, cost-aware.** Nothing is reported without a TRAIN/VAL/TEST label;
   fees + slippage + 1-bar execution lag are always applied; B&H uses the same engine/basis.
5. **Returns, not price levels**, in all features.

## How this maps to the judging rubric (plan §3.6)

| Rubric dimension | Where it shows up |
|---|---|
| Use of CMC data | Real daily BTC OHLCV (2014→2026) via `scripts/fetch.py` (CMC Data API). |
| Technical rigor / no leakage | TRAIN-only fits, 30-bar embargo, filtered causal HMM, causality-mutation tests, cost-aware engine — 41 tests green. |
| Innovation | Skill that *generates + validates* regime strategies and reports edge/no-edge honestly; HMM numerical-stability + return-basis correctness work documented in [DEBUG_HMM_REAL.md](DEBUG_HMM_REAL.md). |
| Honesty / anti-overfitting | No alpha claim; no-edge reported, not tuned; the gate was deliberately **not** flipped despite the observed sample upside. |
| Reproducibility | Committed SYNTHETIC sample + `--offline`; deterministic seeded runs; one-command report + single-file dashboard. |
| Documentation | This README, [SKILL.md](SKILL.md), per-run `report.md`, and the debug write-up. |

## What you get per run

- `strategy_spec.json` — machine-readable policy: regime model, per-regime position factors,
  sizing rule, costs, **segment-labeled** evidence (strategy + buy-and-hold), causality guards,
  and an honest `meta.verdict` / `meta.finding`.
- `report.md` — the same evidence, human-readable, every metric TRAIN/VAL/TEST-labeled.
- `gui/dashboard.html` — single-file dashboard (regime timeline, equity, metrics, policy);
  data-driven, shows a SYNTHETIC banner when the run is synthetic.

## Layout

```
SKILL.md            the LLM Skill (frontmatter + agent instructions)
DEBUG_HMM_REAL.md   debug log: HMM stabilization + backtest correctness + honest verdict
core/               vendored research modules (split, vol, HMM, sizing, backtest)
adapter/            CMC REST client, data-quality gates, SYNTHETIC sample generator
scripts/            fetch.py · run_strategy.py · make_report.py
data/sample/        committed SYNTHETIC cache → reproduce offline (data/cache/ is git-ignored)
tests/              causality mutation tests, engine tests, e2e offline run (41 total)
gui/                dashboard.html (single file, no build step)
```

## License

MIT — see [LICENSE](LICENSE).
