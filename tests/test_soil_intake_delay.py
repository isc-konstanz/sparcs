# -*- coding: utf-8 -*-
"""sparcs.tests.test_soil_intake_delay
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pure unit tests for the ``intake_delay`` feature on ``FieldSimulation``: the
frame-clipping helper that holds the whole chain behind wall-clock so it only
advances over replicated data, and the ``[field_simulation] intake_delay``
config parse.

Importing ``base`` pulls the full lories + FiPy/Gmsh stack; ``importorskip``
keeps this out of environments that lack it (the full check runs on the box).
The targets are a ``@staticmethod`` and a config parse, so no ``Component`` /
PDE instantiation or solver run is needed; the tick's wall-clock reads are
covered in ``test_field_simulation_tick``.
"""

import pytest

import pandas as pd

_base = pytest.importorskip("sparcs.components.agriculture.simulation.base")
FieldSimulation = _base.FieldSimulation

from lories import Configurations  # noqa: E402


def _weather_frame() -> pd.DataFrame:
    # tz-aware UTC index, matching what lories channels stamp (Channel.set).
    index = pd.date_range("2026-07-07 10:00", periods=5, freq="15min", tz="UTC")
    return pd.DataFrame({"ghi": [100.0, 200.0, 300.0, 400.0, 500.0]}, index=index)


def _configs(tmp_path, **values) -> Configurations:
    return Configurations.load("test.conf", conf_dir=str(tmp_path), require=False, **values)


# --- _clip_to_cutoff --------------------------------------------------------


def test_clip_to_cutoff_none_returns_frame_unchanged():
    """intake_delay == 0 -> cutoff None -> the exact same frame object, untouched."""
    frame = _weather_frame()
    clipped = FieldSimulation._clip_to_cutoff(frame, None)
    assert clipped is frame


def test_clip_to_cutoff_interior_drops_later_rows():
    frame = _weather_frame()
    cutoff = pd.Timestamp("2026-07-07 10:30", tz="UTC")  # the third of five rows
    clipped = FieldSimulation._clip_to_cutoff(frame, cutoff)
    assert len(clipped) == 3
    assert clipped.index[-1] == cutoff  # now stays the latest row at or before the cutoff


def test_clip_to_cutoff_keeps_row_exactly_at_cutoff():
    """The keep is inclusive, so a cutoff landing on the last row keeps all rows."""
    frame = _weather_frame()
    cutoff = frame.index[-1]
    clipped = FieldSimulation._clip_to_cutoff(frame, cutoff)
    assert len(clipped) == len(frame)
    assert clipped.index[-1] == cutoff


def test_clip_to_cutoff_before_frame_returns_empty():
    """A cutoff before the whole frame empties it, so the callback no-ops the tick."""
    frame = _weather_frame()
    cutoff = pd.Timestamp("2026-07-07 09:00", tz="UTC")  # before the first row
    clipped = FieldSimulation._clip_to_cutoff(frame, cutoff)
    assert clipped.empty


# --- config parse -----------------------------------------------------------


def test_intake_delay_absent_parses_to_zero(tmp_path):
    configs = _configs(tmp_path)
    assert FieldSimulation._parse_intake_delay(configs) == pd.Timedelta(0)


def test_intake_delay_parses_duration_string(tmp_path):
    configs = _configs(tmp_path, intake_delay="30min")
    assert FieldSimulation._parse_intake_delay(configs) == pd.Timedelta(minutes=30)


def test_intake_delay_default_attr_is_zero_on_field():
    """The class default is the off-state, so a field whose config omits the key
    runs the byte-for-byte pre-feature path."""
    assert FieldSimulation._intake_delay == pd.Timedelta(0)
