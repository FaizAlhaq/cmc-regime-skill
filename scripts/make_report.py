#!/usr/bin/env python3
"""
make_report.py — render report.md + gui/data.js from a run directory.

Every number in the report carries its segment label (TRAIN/VAL/TEST) —
unlabeled metrics are banned in this repo. gui/data.js embeds the spec and
downsampled series so gui/dashboard.html opens by double-click (file://,
no server, no build step).

Usage:
  python scripts/make_report.py --run results/latest
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def resolve_run(run: str) -> Path:
    p = Path(run)
    if not p.is_absolute():
        p = ROOT / run
    if p.name == "latest" and not p.exists():
        txt = ROOT / "results" / "LATEST.txt"
        if txt.exists():
            p = Path(txt.read_text().strip())
    if not (p / "strategy_spec.json").exists():
        raise SystemExit(f"no strategy_spec.json under {p}")
    return p


def fmt_pct(x) -> str:
    return "n/a" if x is None else f"{x*100:.2f}%"


def fmt4(x) -> str:
    return "n/a" if x is None else f"{x:+.4f}" if isinstance(x, float) else str(x)


def render_md(spec: dict) -> str:
    m, ev = spec["meta"], spec["evidence"]
    bat = ev.get("active_signal_battery")  # None in OHLCV-only mode
    lines = [
        f"# Strategy report — {m['asset']} @ {m['bar_freq']} bars",
        "",
        f"Generated {m['generated_at']} · skill `{m['skill']}` v{m['version']}",
        "",
        f"**Data source:** {m['data_source']}",
        "",
    ]
    if m.get("synthetic_data"):
        lines += [
            "> ⚠️ **SYNTHETIC DATA.** Every number below was computed on the seeded",
            "> synthetic sample (`data/sample/`). They demonstrate that the pipeline",
            "> runs end-to-end and stays causal — they are **not** market evidence.",
            "",
        ]
    if m.get("verdict"):
        lines += [
            "## Verdict (honest)",
            "",
            f"**{m['verdict']}**",
            "",
        ]
        if m.get("finding"):
            lines += [f"_Finding:_ {m['finding']}", ""]
    rm = spec["regime_model"]
    lines += [
        "## Regime model",
        "",
        f"GaussianHMM, K={rm['K']}, obs = {rm['obs']}, fit on **{rm['fit_segment']}**,",
        f"inference: {rm['inference']} (params hash `{rm['params_hash']}`).",
        "",
        "| state | label | mean ret/bar | mean log RV | TRAIN occupancy | avg duration (bars) |",
        "|---|---|---|---|---|---|",
    ]
    for s in rm["states"]:
        lines.append(f"| {s['id']} | {s['label']} | {s['mean_return']:+.5f} | "
                     f"{s['mean_log_rv']:.2f} | {s['occupancy_train']*100:.1f}% | "
                     f"{s['avg_duration_bars']:.1f} |")
    lines += [
        "",
        "## Policy (regime switch)",
        "",
        "| regime | signal | position factor |",
        "|---|---|---|",
    ]
    for label, p in spec["policy"].items():
        lines.append(f"| {label} | {p['signal']} | {p['position_factor']} |")
    sg = spec["signal"]
    if bat is None:
        # OHLCV-only: no directional alpha signal, so no IC battery.
        lines += [
            "",
            f"**Direction:** {sg['active']} — {sg.get('mode', 'OHLCV-only')}.",
            "",
            "## Signal evidence (battery)",
            "",
            "_Not applicable._ " + sg.get("note", "OHLCV-only regime/risk overlay; "
            "no directional alpha signal (derivatives unavailable on CMC, D2). "
            "The skill contributes the regime gate + vol target only; judge it "
            "against the buy-and-hold benchmark below, not on signal IC."),
            "",
        ]
    else:
        lines += [
            "",
            f"Active signal **{sg['active']}** (sign {sg['sign']:+d}, "
            f"{sg['sign_calibration']}).",
            "",
            "## Signal evidence (battery)",
            "",
            f"Verdict: **{bat['verdict']}** on primary horizon {bat['primary_horizon']}.",
            "",
            "| segment | Spearman IC |",
            "|---|---|",
        ]
        for seg in ("TRAIN", "VAL", "TEST", "FULL"):
            lines.append(f"| {seg} | {fmt4(bat['ic'].get(seg))} |")
        ci = bat["bootstrap_ci_90"]
        lines += [
            "",
            f"Moving-block bootstrap 90% CI (FULL history): [{fmt4(ci[0])}, {fmt4(ci[1])}], "
            f"N={bat['bootstrap_n']}.",
            "",
            "Verdict reasons:",
            "",
        ]
        lines += [f"- {r}" for r in bat["verdict_reasons"]]
    lines += [
        "",
        "## Backtest (cost-aware, position lagged 1 bar)",
        "",
        f"Costs: fee {spec['costs']['fee_bps']} bps + slip {spec['costs']['slip_bps']} bps. "
        f"Sizing: {spec['sizing']['rule']} (target vol from {spec['sizing']['target_vol_source']}).",
        "",
        "Strategy (regime-gated + vol-targeted) vs **buy-and-hold** benchmark, per segment:",
        "",
        "| segment | Sharpe (ann) | MaxDD | PF | total return | turnover/bar | n bars |",
        "|---|---|---|---|---|---|---|",
    ]
    bench = ev.get("benchmark_buy_and_hold", {})
    for seg in ("TRAIN", "VAL", "TEST"):
        b = ev["backtest"][seg]
        pf = "inf" if b["pf"] is None else f"{b['pf']:.3f}"
        lines.append(f"| **{seg}** strategy | {b['sharpe']:.3f} | {fmt_pct(b['maxdd'])} | {pf} | "
                     f"{fmt_pct(b['total_return'])} | {b['turnover_per_bar']:.4f} | "
                     f"{b['n_bars']} |")
        if seg in bench:
            hb = bench[seg]
            lines.append(f"| {seg} buy&hold | {hb['sharpe']:.3f} | {fmt_pct(hb['maxdd'])} | "
                         f"— | {fmt_pct(hb['total_return'])} | 0.0000 | {hb['n_bars']} |")
    g = spec["guards"]
    lines += [
        "",
        "## Guards",
        "",
        f"- live_trading: **{g['live_trading']}** (spec generator only — no execution code)",
        f"- lookahead: {g['lookahead']}",
        f"- causality mutation tests: {g['causality_mutation_tests']}",
        f"- features use returns, not price levels: {g['features_use_returns_not_levels']}",
        "",
        "*Every metric above is labeled with its segment. This report claims regime",
        "discipline and validation rigor — it does not claim live alpha.*",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/latest")
    ap.add_argument("--max-points", type=int, default=1500,
                    help="downsample series for the dashboard")
    args = ap.parse_args()

    run_dir = resolve_run(args.run)
    spec = json.loads((run_dir / "strategy_spec.json").read_text())

    report = render_md(spec)
    (run_dir / "report.md").write_text(report, encoding="utf-8")
    print(f"wrote {run_dir / 'report.md'}")

    series = pd.read_csv(run_dir / "series.csv", index_col=0, parse_dates=True)
    step = max(1, len(series) // args.max_points)
    s = series.iloc[::step]
    def col(name, digits=6):
        import math
        out = []
        for v in s[name].tolist():
            out.append(None if (v is None or (isinstance(v, float) and math.isnan(v)))
                       else round(float(v), digits))
        return out

    # ToS-safe price: rebase to an index (start = 100), never raw USD OHLCV.
    closes = col("close", 6)
    base = next((c for c in closes if c is not None and c != 0), None)
    price_index = [None if (c is None or base is None)
                   else round(c / base * 100.0, 2) for c in closes]

    payload = {
        "spec": spec,
        "series": {
            "t": [ts.isoformat() for ts in s.index],
            "price_index": price_index,   # rebased to 100 (NOT raw USD — ToS-safe)
            "regime": col("regime", 0),
            "segment": s["segment"].tolist(),
            "equity_train": col("TRAIN"),
            "equity_val": col("VAL"),
            "equity_test": col("TEST"),
        },
    }
    data_js = "window.RUN_DATA = " + json.dumps(payload) + ";\n"
    (ROOT / "gui" / "data.js").write_text(data_js, encoding="utf-8")
    print(f"wrote {ROOT / 'gui' / 'data.js'} — open gui/dashboard.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
