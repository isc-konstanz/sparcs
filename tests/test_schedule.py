# -*- coding: utf-8 -*-
"""sparcs.tests.test_schedule
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Unit tests for ``simulation.core.schedule.slot_floor``, the absolute wall-clock
slot alignment used by ``simulation.core.candidates`` and the field runner.
Pure function, exercised directly with pinned exact-timestamp cases.
"""

import pytest

import pandas as pd

_schedule = pytest.importorskip("sparcs.components.agriculture.simulation.core.schedule")
slot_floor = _schedule.slot_floor


# --- slot_floor ------------------------------------------------------------


def test_slot_floor_matches_current_boundary_pin_daily_time_past_midnight():
    tz = "Europe/Berlin"
    now = pd.Timestamp("2026-07-03 10:00", tz=tz)
    assert slot_floor(now, tz, 1440, 60) == pd.Timestamp("2026-07-03 01:00", tz=tz)


def test_slot_floor_matches_current_boundary_pin_before_offset_falls_back():
    tz = "Europe/Berlin"
    now = pd.Timestamp("2026-07-03 00:30", tz=tz)
    assert slot_floor(now, tz, 1440, 60) == pd.Timestamp("2026-07-02 01:00", tz=tz)


def test_slot_floor_matches_current_boundary_pin_custom_interval_and_offset():
    tz = "Europe/Berlin"
    assert slot_floor(pd.Timestamp("2026-07-03 10:20", tz=tz), tz, 60, 15) == pd.Timestamp("2026-07-03 10:15", tz=tz)
