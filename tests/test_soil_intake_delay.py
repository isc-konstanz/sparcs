# -*- coding: utf-8 -*-
"""sparcs.tests.test_soil_intake_delay
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pure unit tests for the ``intake_delay`` feature on ``SoilBase``: the
frame-clipping helper that holds the live sim behind wall-clock so it only
advances over replicated data, and the ``[soil_simulation] intake_delay``
config parse.

Importing ``_soil`` pulls the full lories + FiPy/Gmsh stack; ``importorskip``
keeps this out of environments that lack it (the full check runs on the box).
The targets are a ``@staticmethod`` and a config parse, so no ``Component`` /
PDE instantiation or solver run is needed -- the wall-clock read in
``_replication_cutoff`` is deliberately kept out of the assertions.
"""

import pytest

import pandas as pd

_soil = pytest.importorskip("sparcs.components.agriculture.simulation._soil")
SoilBase = _soil.SoilBase

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
    clipped = SoilBase._clip_to_cutoff(frame, None)
    assert clipped is frame


def test_clip_to_cutoff_interior_drops_later_rows():
    frame = _weather_frame()
    cutoff = pd.Timestamp("2026-07-07 10:30", tz="UTC")  # the third of five rows
    clipped = SoilBase._clip_to_cutoff(frame, cutoff)
    assert len(clipped) == 3
    assert clipped.index[-1] == cutoff  # now stays the latest row at or before the cutoff


def test_clip_to_cutoff_keeps_row_exactly_at_cutoff():
    """The keep is inclusive, so a cutoff landing on the last row keeps all rows."""
    frame = _weather_frame()
    cutoff = frame.index[-1]
    clipped = SoilBase._clip_to_cutoff(frame, cutoff)
    assert len(clipped) == len(frame)
    assert clipped.index[-1] == cutoff


def test_clip_to_cutoff_before_frame_returns_empty():
    """A cutoff before the whole frame empties it, so the callback no-ops the tick."""
    frame = _weather_frame()
    cutoff = pd.Timestamp("2026-07-07 09:00", tz="UTC")  # before the first row
    clipped = SoilBase._clip_to_cutoff(frame, cutoff)
    assert clipped.empty


# --- config parse -----------------------------------------------------------


def test_intake_delay_absent_parses_to_zero(tmp_path):
    configs = _configs(tmp_path)
    assert SoilBase._parse_intake_delay(configs) == pd.Timedelta(0)


def test_intake_delay_parses_duration_string(tmp_path):
    configs = _configs(tmp_path, intake_delay="30min")
    assert SoilBase._parse_intake_delay(configs) == pd.Timedelta(minutes=30)


def test_intake_delay_default_attr_is_zero_on_base():
    """The base default keeps components that never parse the key (the predictor) safe."""
    assert SoilBase._intake_delay == pd.Timedelta(0)
