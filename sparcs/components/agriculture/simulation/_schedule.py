# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.simulation._schedule
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tick-schedule config parsing and wall-clock slot alignment, factored out of
``FieldSimulation``/``SoilPredictor`` so the ``interval``/``offset``
vocabulary has exactly one home.

Both components -- and lories' own ``WeatherForecast`` (forecast.py) --
share the same schedule vocabulary: ``interval`` (minutes) is the wall-clock
cadence a schedule is aligned to, and ``offset`` (minutes, ``0 <= offset <
interval``) shifts that alignment within the interval. Before this module
the validation and the ``floor_date``-based slot math were duplicated
between ``FieldSimulation._parse_tick_schedule``/``_next_slot`` (validated,
rolls FORWARD to the slot strictly after ``now``) and
``SoilPredictor._current_boundary`` (unvalidated, rolls BACK to the
most-recent slot at-or-before ``now``) -- two independent copies of the same
absolute-alignment idea, one of them missing validation entirely (finding
17: a misconfigured ``[soil_predictor]`` interval/offset failed silently or
only at the first predict tick, never at configure). ``parse_tick_schedule``
hoists the validated parse (``section_name`` parameterizes the error
messages so each caller's config section is still named correctly in the
raised ``ValueError``); ``slot_ceil``/``slot_floor`` hoist the two roll
directions. Under validated inputs (``0 <= offset < interval``) the two
directions are mathematically equivalent restated the other way, but each
keeps its own exact expression here so the pre-extraction methods' pinned
exact-timestamp outputs stay bit-identical.
"""

from __future__ import annotations

import pandas as pd
from lories.typing import Configurations
from lories.util import floor_date

__all__ = ["parse_tick_schedule", "slot_ceil", "slot_floor"]


def parse_tick_schedule(
    configs: Configurations,
    *,
    default_interval: int,
    default_offset: int,
    section_name: str,
) -> tuple[int, int]:
    """Parse and validate ``interval``/``offset`` (minutes) from ``configs``.

    ``interval`` must be >= 1 minute; ``offset`` must be in ``[0,
    interval)``. ``section_name`` only names the config section in the
    raised ``ValueError`` messages (e.g. ``"field_simulation"`` or
    ``"soil_predictor"``); it does not affect which keys are read.
    """
    interval = int(configs.get("interval", default=default_interval))
    offset = int(configs.get("offset", default=default_offset))
    if interval < 1:
        raise ValueError(f"[{section_name}] interval must be >= 1 minute, got {interval}")
    if not 0 <= offset < interval:
        raise ValueError(f"[{section_name}] offset must be in [0, interval), got {offset}")
    return interval, offset


def slot_ceil(now: pd.Timestamp, tz, interval_min: int, offset_min: int) -> pd.Timestamp:
    """First aligned slot strictly after ``now``.

    Alignment is absolute (``floor_date`` on ``tz`` + ``offset_min``), not
    relative to any activation time, so restarts do not shift the schedule.
    Exact expression preserved from ``FieldSimulation._next_slot``.
    """
    slot = floor_date(now, tz, freq=f"{interval_min}min")
    slot += pd.Timedelta(minutes=offset_min)
    while slot <= now:
        slot += pd.Timedelta(minutes=interval_min)
    return slot


def slot_floor(now: pd.Timestamp, tz, interval_min: int, offset_min: int) -> pd.Timestamp:
    """Most-recent aligned slot at or before ``now``.

    Same absolute alignment as ``slot_ceil``, rolled back instead of
    forward. Exact expression preserved from
    ``SoilPredictor._current_boundary``.
    """
    boundary = floor_date(now, tz, freq=f"{interval_min}min") + pd.Timedelta(minutes=offset_min)
    if boundary > now:
        boundary -= pd.Timedelta(minutes=interval_min)
    return boundary
