# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.simulation.core.minute_grid
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Weather for the live tick on whole minutes, whatever cadence its sources deliver.
"""

from __future__ import annotations

from typing import Collection

import pandas as pd
from lories.components.weather import Weather

__all__ = ["to_minute_grid"]

STEP = pd.Timedelta(minutes=1)

# Bright Sky's rates and means cover the hour before their timestamp
MAX_PERIOD = pd.Timedelta(hours=1)

# Values for the period before their timestamp; every other column is a value at its timestamp. Hourly rates hold
# for at most that hour, so a missing hour stays without rain instead of repeating the next one.
HOURLY_RATES = frozenset({Weather.PRECIPITATION, Weather.SUNSHINE})
PERIOD_VALUES = HOURLY_RATES | {Weather.GHI, Weather.WIND_SPEED, Weather.WIND_SPEED_GUST, Weather.WIND_DIRECTION}


def to_minute_grid(
    frame: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cover: Collection[str],
) -> pd.DataFrame:
    """The weather in ``frame`` on the whole minutes in ``(start, end]``, ending earlier at the last record of any
    ``cover`` column that has data, so no minute is left that a later record still has to fill."""
    frame = frame.sort_index()
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    for key in cover:
        if key in frame.columns and frame[key].notna().any():
            end = min(end, frame[key].last_valid_index())
    grid = pd.date_range(start.floor(STEP) + STEP, end.floor(STEP), freq=STEP)
    columns = {key: _to_grid(key, frame[key].dropna(), grid) for key in frame.columns}
    return pd.DataFrame(columns, index=grid)


def _to_grid(key: str, values: pd.Series, grid: pd.DatetimeIndex) -> pd.Series:
    if key in HOURLY_RATES:
        return _hold_within(values, grid, MAX_PERIOD)
    joined = values.reindex(values.index.union(grid))
    if key in PERIOD_VALUES or not pd.api.types.is_numeric_dtype(values):
        return joined.bfill().reindex(grid)
    return joined.interpolate(method="time").reindex(grid)


def _hold_within(values: pd.Series, grid: pd.DatetimeIndex, period: pd.Timedelta) -> pd.Series:
    record = values.index.to_series().reindex(grid, method="bfill")
    held = pd.Series(values.reindex(record).to_numpy(), index=grid)
    return held.where(record - grid.to_series() < period)
