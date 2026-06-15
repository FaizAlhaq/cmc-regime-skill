#!/usr/bin/env python3
"""
fetch.py — the ONLY networked step in the pipeline.

With a CMC_API_KEY in .env / env: pulls REAL daily BTC OHLCV into
data/cache/CMC_<ASSET>_<H>h.parquet and writes a per-endpoint fetch log
(endpoint, status, first_date, last_date, n_rows) to
data/cache/CMC_<ASSET>_<H>h.fetchlog.json.

DERIVATIVES ARE EXCLUDED (locked decision D2): CoinMarketCap has no
funding / open-interest / long-short history, so this script does NOT call any
derivatives endpoint. The regime + risk-overlay core is OHLCV-only by design.

With --synthetic: (re)generates the committed SYNTHETIC sample cache under
data/sample/ — no network, no key. Synthetic data is labeled SYNTHETIC in
filename, per-row column, and parquet metadata.

Usage:
  python scripts/fetch.py --asset BTC --bar-hours 24 --start 2010-01-01   # REAL daily
  python scripts/fetch.py --synthetic
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("fetch")


def load_env() -> None:
    env = ROOT / ".env"
    if env.exists():
        import os
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="BTC")
    ap.add_argument("--bar-hours", type=float, default=24.0,
                    help="bar size in hours; 24 = DAILY (D1, submission default)")
    ap.add_argument("--start", default="2010-01-01",
                    help="UTC start; CMC returns from first available bar "
                         "(BTC history on CMC begins ~2013)")
    ap.add_argument("--end", default=None,
                    help="UTC end (default: today)")
    ap.add_argument("--synthetic", action="store_true",
                    help="regenerate SYNTHETIC sample cache (no network)")
    args = ap.parse_args()
    if args.end is None:
        args.end = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    if args.synthetic:
        from adapter import synthetic
        out_dir = ROOT / "data" / "sample"
        out_dir.mkdir(parents=True, exist_ok=True)
        for asset in ("BTC", "ETH"):
            for bh in (1.0, 4.0):
                n = 5400 if bh == 4.0 else 8000
                df = synthetic.generate(asset=asset, bar_hours=bh, n_bars=n)
                p = synthetic.sample_path(out_dir, asset, bh)
                synthetic.write_sample(df, p)
                log.info(f"wrote {p.name}: {len(df)} bars [SYNTHETIC]")
        return 0

    load_env()
    from adapter.cmc_client import CMCClient, CMCError
    from adapter.quality import apply_gates

    from adapter.cmc_client import ENDPOINTS as _EP

    client = CMCClient(cache_dir=ROOT / "data" / "cache")
    log.info(f"fetching REAL OHLCV {args.asset} {args.bar_hours}h "
             f"{args.start} → {args.end} (endpoint {_EP['ohlcv']})")

    fetchlog: dict = {
        "asset": args.asset, "bar_hours": args.bar_hours,
        "requested_start": args.start, "requested_end": args.end,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "endpoints": [],
        "derivatives": "EXCLUDED — not available on CMC (locked decision D2); "
                       "no funding/OI/long-short endpoint was called.",
    }

    # ── REQUIRED: real daily OHLCV ─────────────────────────────────────────
    try:
        df = client.fetch_ohlcv(args.asset, args.bar_hours, args.start, args.end)
        fetchlog["endpoints"].append({
            "name": "ohlcv", "path": _EP["ohlcv"], "status": "OK",
            "first_date": str(df.index[0]), "last_date": str(df.index[-1]),
            "n_rows": int(len(df)), "real": True,
        })
        log.info(f"OHLCV OK: {len(df)} rows, {df.index[0]} → {df.index[-1]}")
    except CMCError as e:
        fetchlog["endpoints"].append({
            "name": "ohlcv", "path": _EP["ohlcv"], "status": f"FAIL: {e}",
            "n_rows": 0, "real": True,
        })
        _write_fetchlog(client, args, fetchlog)
        raise SystemExit(
            f"REQUIRED OHLCV fetch failed: {e}\n"
            "LAYER1_STATUS implication: BLOCKED — cannot proceed without real "
            "OHLCV. NEVER substitute synthetic for real."
        )

    # ── Quality gates (HARD-FAIL is correct; early CMC bars may have vol==0) ─
    df, rep = apply_gates(df, name=f"{args.asset}_{args.bar_hours}h",
                          bar_hours=args.bar_hours)
    fetchlog["quality"] = rep.as_dict()
    log.info(f"quality gates passed: {rep.as_dict()}")

    path = client.cache_path(args.asset, args.bar_hours)
    df.to_parquet(path)
    fetchlog["cache_path"] = str(path)
    _write_fetchlog(client, args, fetchlog)
    log.info(f"cached REAL data → {path}")
    log.info("derivatives EXCLUDED (D2): no funding/OI/long-short fetched.")
    return 0


def _write_fetchlog(client, args, fetchlog: dict) -> None:
    """Write a machine-readable per-endpoint fetch log next to the cache."""
    p = client.cache_path(args.asset, args.bar_hours).with_suffix(".fetchlog.json")
    p.write_text(json.dumps(fetchlog, indent=2))
    log.info(f"fetch log → {p}  (paste its summary into WORKLOG.md)")


if __name__ == "__main__":
    raise SystemExit(main())
