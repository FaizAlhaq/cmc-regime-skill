import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adapter import synthetic  # noqa: E402
from core.freq import FREQ_1H, FREQ_4H, FREQ_1D  # noqa: E402


@pytest.fixture(scope="session")
def df_4h() -> pd.DataFrame:
    """Small SYNTHETIC frame, 4H bars (seeded, deterministic)."""
    return synthetic.generate(asset="BTC", bar_hours=4.0, n_bars=2200, seed=7)


@pytest.fixture(scope="session")
def df_1h() -> pd.DataFrame:
    return synthetic.generate(asset="BTC", bar_hours=1.0, n_bars=3000, seed=7)


@pytest.fixture(scope="session")
def df_1d() -> pd.DataFrame:
    """SYNTHETIC DAILY frame (seeded) — the CMC submission frequency (D1)."""
    return synthetic.generate(asset="BTC", bar_hours=24.0, n_bars=2000, seed=7)


@pytest.fixture(scope="session")
def df_1d_ohlcv() -> pd.DataFrame:
    """SYNTHETIC DAILY frame with derivative columns dropped — simulates the
    OHLCV-only CMC feed (no funding/OI/long-short)."""
    df = synthetic.generate(asset="BTC", bar_hours=24.0, n_bars=2000, seed=7)
    return df.drop(columns=["funding_rate", "open_interest", "long_short_ratio"])


@pytest.fixture(scope="session")
def freq_4h():
    return FREQ_4H


@pytest.fixture(scope="session")
def freq_1h():
    return FREQ_1H


@pytest.fixture(scope="session")
def freq_1d():
    return FREQ_1D
