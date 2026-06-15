"""
Bar-frequency parameterization (NEW for cmc-regime-skill).

The research modules silently assumed 1H bars (BARS_PER_YEAR=8760, rv_24/168/720
windows, 24-bar forward horizon, 48-bar bootstrap blocks). This module threads a
single source of truth — `BarFreq(bar_hours)` — through every hour-coupled
constant so the pipeline is correct at 1H, 4H, or any bar size.

Window semantics are defined in HOURS and converted to bars:
    day = 24h, week = 168h, month = 720h.

DAILY (D1, the CMC submission path): BarFreq(24.0) resolves to
    bars_per_year = 8760/24 = 365   (NOT 252 — crypto trades every day),
    w_day = 1, w_week = 7, w_month = 30, block_len = 2.
At daily, rv_day = sqrt(sum r^2 over 1 bar) = |daily log-return| — a valid
single-bar vol proxy. The 30-bar embargo then fully covers the 30-bar monthly
feature window, so the daily split has no rolling-window leakage.
"""

from __future__ import annotations

from dataclasses import dataclass

HOURS_PER_YEAR = 8760.0
DAY_H, WEEK_H, MONTH_H = 24.0, 168.0, 720.0


@dataclass(frozen=True)
class BarFreq:
    """Single source of truth for everything derived from bar size."""
    bar_hours: float

    def __post_init__(self) -> None:
        if self.bar_hours <= 0:
            raise ValueError(f"bar_hours must be > 0, got {self.bar_hours}")

    @property
    def bars_per_day(self) -> int:
        return self.bars(DAY_H)

    @property
    def bars_per_year(self) -> float:
        return HOURS_PER_YEAR / self.bar_hours

    @property
    def label(self) -> str:
        h = self.bar_hours
        return f"{int(h)}h" if float(h).is_integer() else f"{h}h"

    def bars(self, hours: float) -> int:
        """Number of bars spanning `hours` (rounded, min 1)."""
        n = int(round(hours / self.bar_hours))
        if n < 1:
            raise ValueError(
                f"window of {hours}h is shorter than one {self.bar_hours}h bar"
            )
        return n

    # Canonical windows used across the pipeline
    @property
    def w_day(self) -> int:    # realized-vol short window / forward-vol horizon
        return self.bars(DAY_H)

    @property
    def w_week(self) -> int:   # HAR weekly component
        return self.bars(WEEK_H)

    @property
    def w_month(self) -> int:  # HAR monthly component / z-score windows
        return self.bars(MONTH_H)

    @property
    def block_len(self) -> int:  # MBB bootstrap block = 2 days
        return self.bars(2 * DAY_H)


FREQ_1H = BarFreq(1.0)
FREQ_4H = BarFreq(4.0)
FREQ_1D = BarFreq(24.0)  # DAILY — CMC submission path (D1). bars_per_year == 365.
