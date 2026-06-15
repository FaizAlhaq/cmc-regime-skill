# ── VENDORED (one-way) from private research repo ───────────────────────
#   source: quant-trading/src/backtest/engine.py @ commit e67503c
#   adaptations: BARS_PER_YEAR=8760 module constant replaced by an explicit
#                bars_per_year argument (annualization correct at any bar size).
#                + the per-bar return arg renamed log_return -> bar_return and
#                documented as a SIMPLE (arithmetic) return: equity is built with
#                cumprod(1+pnl), so feeding LOG returns here understates
#                compounding and mis-scales leverage. Callers must convert
#                log -> simple (expm1). See DEBUG_HMM_REAL.md.
#   Do NOT merge changes back into the research repo.
"""
Cost-aware backtesting engine.

Convention
----------
direction[t]  : signal known at END of bar t (uses data through bar t)
position[t]   : direction[t] * position_scale[t]
bar_return[t] : SIMPLE return close[t]/close[t-1] - 1 (earned DURING bar t).
                NOT a log return — equity compounds multiplicatively below, so a
                log return would be wrong (B&H of a +2320% asset would print
                +94%). Convert upstream with np.expm1(log_return).

P&L per bar t (t >= 1):
    pnl[t] = position[t-1] * bar_return[t] - |position[t]-position[t-1]| * cost_rate
At t=0: pnl[0] = -|position[0]| * cost_rate (entry cost only).
equity[t] = prod_{s<=t} (1 + pnl[s])   (multiplicative compounding)

The position therefore lags the signal by one bar — no same-bar execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    """Per-segment backtest metrics. `label` carries the segment name."""
    label: str
    n_bars: int
    sharpe: float
    max_dd: float            # fraction, e.g. -0.23 = -23%
    profit_factor: float
    turnover_per_bar: float
    realized_vol_ann: float
    total_return: float
    bars_per_year: float
    equity: pd.Series = field(repr=False, default=None)
    pnl: pd.Series = field(repr=False, default=None)

    def summary_dict(self) -> dict:
        return {
            "Segment": self.label,
            "N bars": self.n_bars,
            "Sharpe (ann)": round(self.sharpe, 4),
            "MaxDD": f"{self.max_dd*100:.2f}%",
            "PF": round(self.profit_factor, 4),
            "Turnover/bar": round(self.turnover_per_bar, 6),
            "Realised Vol (ann)": f"{self.realized_vol_ann*100:.2f}%",
            "Total Return": f"{self.total_return*100:.2f}%",
        }


def run_backtest(
    direction: pd.Series,
    position_scale: pd.Series,
    bar_return: pd.Series,
    bars_per_year: float,
    fee_bps: float = 4.0,
    slip_bps: float = 1.0,
    label: str = "backtest",
) -> BacktestResult:
    """
    Run a cost-aware backtest for one segment. `label` MUST name the segment
    (TRAIN/VAL/TEST) — unlabeled metrics are banned in this repo.

    `bar_return` must be a SIMPLE per-bar return (close[t]/close[t-1]-1), NOT a
    log return — equity compounds multiplicatively. bars_per_year drives Sharpe /
    vol annualization (8760 at 1H, 2190 at 4H, 365 daily).
    """
    if bars_per_year <= 0:
        raise ValueError(f"bars_per_year must be > 0, got {bars_per_year}")
    ann_factor = float(np.sqrt(bars_per_year))
    cost_rate = (fee_bps + slip_bps) / 10_000.0

    df = pd.DataFrame({
        "direction": direction,
        "scale": position_scale,
        "log_ret": bar_return,
    }).dropna()
    if len(df) < 2:
        raise ValueError(f"[{label}] Not enough data after dropna: {len(df)} rows.")

    pos = (df["direction"] * df["scale"]).values
    ret = df["log_ret"].values
    n = len(pos)

    pnl = np.empty(n)
    prev_pos = 0.0
    for i in range(n):
        delta = abs(pos[i] - prev_pos)
        pnl[i] = (-delta * cost_rate) if i == 0 else (prev_pos * ret[i] - delta * cost_rate)
        prev_pos = pos[i]

    equity_vals = np.cumprod(1.0 + pnl)

    pnl_mean = np.mean(pnl)
    pnl_std = np.std(pnl, ddof=1)
    sharpe = (pnl_mean / pnl_std * ann_factor) if pnl_std > 1e-12 else 0.0
    realized_vol_ann = pnl_std * ann_factor

    running_peak = np.maximum.accumulate(equity_vals)
    max_dd = float(np.min((equity_vals - running_peak) / running_peak))

    gains, losses = pnl[pnl > 0], pnl[pnl < 0]
    pf = (float(gains.sum()) / abs(float(losses.sum()))) \
        if len(losses) > 0 and losses.sum() < 0 else float("inf")

    pos_series = pd.Series(pos, index=df.index)
    delta_pos = pos_series.diff().abs()
    delta_pos.iloc[0] = abs(pos[0])
    turnover = float(delta_pos.mean())

    return BacktestResult(
        label=label,
        n_bars=n,
        sharpe=float(sharpe),
        max_dd=max_dd,
        profit_factor=float(pf),
        turnover_per_bar=turnover,
        realized_vol_ann=float(realized_vol_ann),
        total_return=float(equity_vals[-1] - 1.0),
        bars_per_year=float(bars_per_year),
        equity=pd.Series(equity_vals, index=df.index, name="equity"),
        pnl=pd.Series(pnl, index=df.index, name="pnl"),
    )
