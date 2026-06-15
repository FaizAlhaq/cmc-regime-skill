"""
CoinMarketCap Data API client — fetch + parquet cache. READ-ONLY market data;
this repo contains no execution code of any kind.

OHLCV-ONLY (locked decisions D1/D2): the CMC submission path pulls DAILY OHLCV
only. CoinMarketCap exposes no funding / open-interest / long-short history, so
NO derivatives endpoint is declared or called here (we do not ship UNVERIFIED
endpoint guesses). The regime + risk-overlay core needs OHLCV only.

Design: requests session, API-key header, exponential backoff on 429/5xx AND on
connection/proxy/timeout errors, time-window pagination, parquet cache keyed by
(asset, bar_hours). Network failures surface as a clear CMCError (never a raw
traceback) so the operator gets an actionable BLOCKED message.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

BASE_URL = "https://pro-api.coinmarketcap.com"

# Endpoint registry — OHLCV only (D2: no derivatives history on CMC).
ENDPOINTS = {
    "ohlcv": "/v2/cryptocurrency/ohlcv/historical",          # documented, stable
}

_INTERVAL_FOR_HOURS = {1.0: "1h", 2.0: "2h", 4.0: "4h", 24.0: "daily"}


class CMCError(RuntimeError):
    pass


class CMCClient:
    def __init__(self, api_key: str | None = None, cache_dir: str | Path = "data/cache",
                 max_retries: int = 5, timeout: int = 30):
        self.api_key = api_key or os.environ.get("CMC_API_KEY", "")
        if not self.api_key:
            raise CMCError(
                "No CMC_API_KEY found (arg or env). For offline use, run the "
                "pipeline with --offline (uses the SYNTHETIC sample cache)."
            )
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        self.timeout = timeout
        import requests  # local import: offline path must not require requests
        self._session = requests.Session()
        self._session.headers.update({
            "X-CMC_PRO_API_KEY": self.api_key,
            "Accept": "application/json",
        })

    # ── HTTP with backoff ────────────────────────────────────────────────
    def _get(self, path: str, params: dict) -> dict:
        import requests  # for requests.exceptions
        url = BASE_URL + path
        delay = 1.0
        for attempt in range(self.max_retries):
            try:
                resp = self._session.get(url, params=params, timeout=self.timeout)
            except requests.exceptions.RequestException as e:
                # connection/proxy/timeout — retry with backoff, then fail clean.
                if attempt < self.max_retries - 1:
                    logger.warning(f"{path}: {type(e).__name__}, retry in {delay:.0f}s "
                                   f"({attempt + 1}/{self.max_retries})")
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise CMCError(
                    f"{path}: network error after {self.max_retries} attempts: "
                    f"{type(e).__name__}: {str(e)[:200]}. Check connectivity to "
                    f"{BASE_URL} (corporate proxy / firewall / sandbox allowlist)."
                ) from e
            if resp.status_code == 200:
                payload = resp.json()
                status = payload.get("status", {})
                if status.get("error_code", 0) != 0:
                    raise CMCError(f"{path}: API error {status}")
                return payload
            if resp.status_code in (429, 500, 502, 503, 504):
                logger.warning(f"{path}: HTTP {resp.status_code}, retry in {delay:.0f}s "
                               f"({attempt + 1}/{self.max_retries})")
                time.sleep(delay)
                delay *= 2
                continue
            if resp.status_code == 401:
                raise CMCError(
                    f"{path}: 401 Unauthorized — CMC_API_KEY is missing, invalid, "
                    "or not entitled to this endpoint/plan. Verify the key value "
                    "and that the Startup tier includes historical OHLCV."
                )
            if resp.status_code == 404:
                raise CMCError(
                    f"{path}: 404 — endpoint path is wrong or not enabled for "
                    "this plan. Check the CMC API docs."
                )
            raise CMCError(f"{path}: HTTP {resp.status_code}: {resp.text[:300]}")
        raise CMCError(f"{path}: exhausted {self.max_retries} retries")

    # ── Public fetchers ──────────────────────────────────────────────────
    def fetch_ohlcv(self, symbol: str, bar_hours: float,
                    time_start: str, time_end: str) -> pd.DataFrame:
        """Paginated OHLCV pull → DataFrame[open,high,low,close,volume] (UTC index)."""
        interval = _INTERVAL_FOR_HOURS.get(float(bar_hours))
        if interval is None:
            raise CMCError(f"No CMC interval for bar_hours={bar_hours}")

        rows: list[dict] = []
        cursor = pd.Timestamp(time_start, tz="UTC")
        end_ts = pd.Timestamp(time_end, tz="UTC")
        # CMC historical OHLCV caps count per call; paginate by time window.
        step = pd.Timedelta(hours=bar_hours * 400)
        while cursor < end_ts:
            window_end = min(cursor + step, end_ts)
            payload = self._get(ENDPOINTS["ohlcv"], {
                "symbol": symbol,
                "time_period": "daily" if interval == "daily" else interval,
                "interval": interval,
                "time_start": cursor.isoformat(),
                "time_end": window_end.isoformat(),
                "convert": "USD",
            })
            data = payload.get("data", {})
            quotes = (data.get("quotes")
                      or (list(data.values())[0][0].get("quotes")
                          if isinstance(data, dict) and data else []))
            for q in quotes or []:
                usd = q["quote"]["USD"]
                rows.append({
                    "timestamp": q.get("time_open") or q.get("timestamp"),
                    "open": usd["open"], "high": usd["high"],
                    "low": usd["low"], "close": usd["close"],
                    "volume": usd["volume"],
                })
            cursor = window_end
        if not rows:
            raise CMCError(f"fetch_ohlcv({symbol}): no rows returned")
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()
        df = df[~df.index.duplicated(keep="first")]
        df["synthetic"] = False
        return df

    # ── Cache ────────────────────────────────────────────────────────────
    def cache_path(self, asset: str, bar_hours: float) -> Path:
        h = int(bar_hours) if float(bar_hours).is_integer() else bar_hours
        return self.cache_dir / f"CMC_{asset}_{h}h.parquet"
