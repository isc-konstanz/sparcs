# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.simulation.core.schedule
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Wall-clock slot alignment: ``interval`` (minutes) is the cadence a schedule
is aligned to, and ``offset`` (minutes, ``0 <= offset < interval``) shifts
that alignment within the interval. Alignment is absolute (``floor_date`` on
``tz`` plus the offset), so restarts never shift a schedule.
"""

from __future__ import annotations

import pandas as pd
from lories.util import floor_date

__all__ = ["slot_floor"]


def slot_floor(now: pd.Timestamp, tz, interval_min: int, offset_min: int) -> pd.Timestamp:
    """Most-recent aligned slot at or before ``now``."""
    boundary = floor_date(now, tz, freq=f"{interval_min}min") + pd.Timedelta(minutes=offset_min)
    if boundary > now:
        boundary -= pd.Timedelta(minutes=interval_min)
    return boundary
