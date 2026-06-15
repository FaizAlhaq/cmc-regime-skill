# Strategy report — BTC @ 24h bars

Generated 2026-06-15T03:45:46.054553+00:00 · skill `cmc-regime-switch` v1.1

**Data source:** CMC Data API cache

## Verdict (honest)

**NO EDGE in this configuration: the overlay reduces drawdown and realized vol on all of TRAIN/VAL/TEST, but does NOT beat buy-and-hold on Sharpe or return on any segment. Reported, not tuned.**

_Finding:_ In crypto, BTC's positive drift is concentrated in HIGH-volatility regimes (bull runs are volatile), so de-risking high vol misses the upside — opposite of the equities intuition. The regime gate is correctly signed for its risk-reduction design; flipping it to chase the in-sample upside would be overfitting, so it was NOT changed.

## Regime model

GaussianHMM, K=4, obs = ['log_return', 'log_rv_day'], fit on **TRAIN only**,
inference: filtered (causal forward algorithm) — never Viterbi (params hash `a43fd80e845a8b91`).

| state | label | mean ret/bar | mean log RV | TRAIN occupancy | avg duration (bars) |
|---|---|---|---|---|---|
| 0 | calm | +0.00017 | -6.32 | 19.1% | 1.4 |
| 1 | low-vol | -0.00166 | -4.54 | 40.2% | 1.7 |
| 2 | high-vol | +0.02568 | -3.68 | 10.9% | 1.1 |
| 3 | turbulent | -0.00355 | -2.97 | 29.8% | 1.7 |

## Policy (regime switch)

| regime | signal | position factor |
|---|---|---|
| calm | baseline_long | 1.0 |
| low-vol | baseline_long | 1.0 |
| high-vol | flat | 0.5 |
| turbulent | flat | 0.0 |

**Direction:** baseline_long — OHLCV-only (regime-gated long-only vol overlay).

## Signal evidence (battery)

_Not applicable._ Derivatives unavailable on CMC (D2) → no positioning/funding signal. Baseline exposure = long BTC beta, flattened by the regime gate in de-risked/turbulent states. This is NOT an alpha claim; see the buy-and-hold benchmark.


## Backtest (cost-aware, position lagged 1 bar)

Costs: fee 4.0 bps + slip 1.0 bps. Sizing: target_vol / HAR_forecast, clip [0, 1.0] (target vol from TRAIN median of HAR forecast vol).

Strategy (regime-gated + vol-targeted) vs **buy-and-hold** benchmark, per segment:

| segment | Sharpe (ann) | MaxDD | PF | total return | turnover/bar | n bars |
|---|---|---|---|---|---|---|
| **TRAIN** strategy | 0.128 | -73.34% | 1.027 | -34.47% | 0.3649 | 3152 |
| TRAIN buy&hold | 0.864 | -83.40% | — | 2270.75% | 0.0000 | 3181 |
| **VAL** strategy | 0.741 | -24.99% | 1.144 | 47.54% | 0.3576 | 682 |
| VAL buy&hold | 1.457 | -26.12% | — | 208.73% | 0.0000 | 682 |
| **TEST** strategy | -0.036 | -49.09% | 0.994 | -10.99% | 0.3763 | 623 |
| TEST buy&hold | 0.274 | -51.21% | — | 3.71% | 0.0000 | 623 |

## Guards

- live_trading: **False** (spec generator only — no execution code)
- lookahead: none — embargo=30 bars, all scalers/thresholds/HAR/HMM fit on TRAIN only, filtered HMM inference
- causality mutation tests: see tests/ — run `pytest -q`
- features use returns, not price levels: True

*Every metric above is labeled with its segment. This report claims regime
discipline and validation rigor — it does not claim live alpha.*
