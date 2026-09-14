# -*- coding: utf-8 -*-
"""sparcs.tests.test_schedule
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Unit tests for ``_schedule`` -- the hoisted ``interval``/``offset`` config
validation (``parse_tick_schedule``) and absolute wall-clock slot alignment
(``slot_ceil``/``slot_floor``) shared by ``FieldSimulation`` and
``SoilPredictor`` (issue 17-w1-6-schedule-unification). Pure functions,
exercised directly.

The four ``parse_tick_schedule`` tests pin the corrected ``[soil_predictor]``
failure modes from finding 17: interval/offset now fails at ``configure()``
instead of mis-scheduling silently (offset >= interval), defeating the
``predict()`` dedup gate (interval == 0), or exploding only at the first
predict tick (negative interval) -- plus one passing case (the kob fixture's
60/0). No existing predictor test calls ``SoilPredictor.configure()`` (the
gate-level tests in test_soil_predictor_scheduling_gate.py inject fields on
``object.__new__`` instances), so this exercises ``parse_tick_schedule``
directly with ``section_name="soil_predictor"``, matching how
``SoilPredictor.configure()`` will call it.

The six ``slot_ceil``/``slot_floor`` tests re-run pinned exact-timestamp
cases from ``test_field_simulation_tick.py`` (``_next_slot``) and
``test_soil_predictor_scheduling_gate.py`` (``_current_boundary``) directly
against the extracted functions, so a regression in the shared module would
show up here even if a caller's own wrapper test were ever weakened.
"""

import pytest

import pandas as pd

_schedule = pytest.importorskip("sparcs.components.agriculture.simulation._schedule")
parse_tick_schedule = _schedule.parse_tick_schedule
slot_ceil = _schedule.slot_ceil
slot_floor = _schedule.slot_floor

from lories import Configurations  # noqa: E402


def _configs(tmp_path, **values) -> Configurations:
    return Configurations.load("test.conf", conf_dir=str(tmp_path), require=False, **values)


# --- parse_tick_schedule: corrected soil_predictor failure modes (finding 17) -----


def test_soil_predictor_rejects_offset_gte_interval(tmp_path):
    """offset >= interval used to yield a boundary in the FUTURE of now (the
    single-if roll-back in the old unvalidated ``_current_boundary``); now
    caught at configure time."""
    with pytest.raises(ValueError, match="offset"):
        parse_tick_schedule(
            _configs(tmp_path, interval=60, offset=90),
            default_interval=1440,
            default_offset=60,
            section_name="soil_predictor",
        )


def test_soil_predictor_rejects_zero_interval(tmp_path):
    """interval=0 used to return ``now`` itself every tick, defeating the
    ``predict()`` dedup gate; now caught at configure time."""
    with pytest.raises(ValueError, match="interval"):
        parse_tick_schedule(
            _configs(tmp_path, interval=0),
            default_interval=1440,
            default_offset=60,
            section_name="soil_predictor",
        )


def test_soil_predictor_rejects_negative_interval(tmp_path):
    """A negative interval used to explode only at the first predict tick
    (``ValueError: Invalid frequency format``); now caught at configure time."""
    with pytest.raises(ValueError, match="interval"):
        parse_tick_schedule(
            _configs(tmp_path, interval=-5),
            default_interval=1440,
            default_offset=60,
            section_name="soil_predictor",
        )


def test_soil_predictor_valid_schedule_passes(tmp_path):
    """The kob fixture's own [soil_predictor] schedule (60/0) is valid --
    gaining validation must not break it."""
    assert parse_tick_schedule(
        _configs(tmp_path, interval=60, offset=0),
        default_interval=1440,
        default_offset=60,
        section_name="soil_predictor",
    ) == (60, 0)


# --- slot_ceil / slot_floor: equivalence with the pre-extraction methods ----------


def test_slot_ceil_matches_next_slot_pin_aligns_to_interval_plus_offset():
    now = pd.Timestamp("2026-07-12 10:20", tz="UTC")
    assert slot_ceil(now, None, 60, 5) == pd.Timestamp("2026-07-12 11:05", tz="UTC")


def test_slot_ceil_matches_next_slot_pin_strictly_future_when_now_is_on_slot():
    now = pd.Timestamp("2026-07-12 10:05", tz="UTC")
    assert slot_ceil(now, None, 60, 5) == pd.Timestamp("2026-07-12 11:05", tz="UTC")


def test_slot_ceil_matches_next_slot_pin_sub_hourly():
    now = pd.Timestamp("2026-07-12 10:07", tz="UTC")
    assert slot_ceil(now, None, 15, 0) == pd.Timestamp("2026-07-12 10:15", tz="UTC")


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
