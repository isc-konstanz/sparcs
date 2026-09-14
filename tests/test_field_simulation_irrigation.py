# -*- coding: utf-8 -*-
"""Irrigation flow enters the soil PDE as a time series read from its connector, not a latch.

The wall-clock tick reads the flow channel's history over the same span as
weather (a ranged connector read) and aligns it onto the weather timesteps
(backward ffill), so a tick's water forcing depends only on the recorded flow,
never on when the tick runs. NULL flow rows mean "not watering" and must become
0.0, never NaN, in the aligned series -- the series successor of the latch guard
from commit 20431c7.

Importing ``FieldSimulation`` pulls the full lories + soil (FiPy) stack;
``importorskip`` keeps this out of environments that lack it (the full check
runs on the box). The alignment helpers only touch ``self`` attributes, so bare
``object.__new__`` instances exercise them without the Component machinery.
"""

import pytest

import numpy as np
import pandas as pd

base = pytest.importorskip("sparcs.components.agriculture.simulation.base")
FieldSimulation = base.FieldSimulation


def _weather_index(start="2026-05-01 10:00", periods=4, freq="15min"):
    return pd.date_range(start, periods=periods, freq=freq, tz="UTC")


def _flow_frame(points: dict[str, float]) -> pd.DataFrame:
    index = pd.DatetimeIndex([pd.Timestamp(ts, tz="UTC") for ts in points])
    return pd.DataFrame({"flow": list(points.values())}, index=index)


# --- _align_flow --------------------------------------------------------------


def test_align_backward_fills_between_flow_rows():
    """Each weather timestep gets the most recent flow value at or before it."""
    index = _weather_index()  # 10:00, 10:15, 10:30, 10:45
    frame = _flow_frame({"2026-05-01 10:00": 50.0, "2026-05-01 10:20": 0.0})
    aligned = FieldSimulation._align_flow(frame, index)
    assert list(aligned) == [50.0, 50.0, 0.0, 0.0]


def test_align_null_flow_reads_as_zero():
    """A NULL/NaN flow row -> 0.0 from that row on, never NaN into the PDE source."""
    index = _weather_index()
    frame = _flow_frame({"2026-05-01 10:00": 50.0, "2026-05-01 10:20": np.nan})
    aligned = FieldSimulation._align_flow(frame, index)
    assert list(aligned) == [50.0, 50.0, 0.0, 0.0]
    assert not aligned.isna().any()


def test_align_leading_gap_reads_as_zero():
    """No flow row at or before the first weather timestep -> not watering."""
    index = _weather_index()
    frame = _flow_frame({"2026-05-01 10:20": 30.0})
    aligned = FieldSimulation._align_flow(frame, index)
    assert list(aligned) == [0.0, 0.0, 30.0, 30.0]


def test_align_empty_history_is_all_zero():
    index = _weather_index()
    aligned = FieldSimulation._align_flow(pd.DataFrame(), index)
    assert list(aligned) == [0.0, 0.0, 0.0, 0.0]


def test_align_sub_tick_window_covers_only_its_timesteps():
    """A 20-minute watering window inside an hour waters exactly the timesteps it
    covers, not the whole hour (the distortion the latched scalar had)."""
    index = _weather_index(periods=12, freq="5min")  # 10:00 .. 10:55
    frame = _flow_frame({"2026-05-01 10:10": 40.0, "2026-05-01 10:30": 0.0})
    aligned = FieldSimulation._align_flow(frame, index)
    watered = aligned[aligned > 0.0]
    assert list(watered.index) == list(pd.date_range("2026-05-01 10:10", "2026-05-01 10:25", freq="5min", tz="UTC"))


def test_align_replays_history_not_current_value():
    """Catch-up over a gap uses the flow that was logged then; a burst logged
    after the span cannot leak backwards."""
    index = _weather_index()  # ends 10:45
    frame = _flow_frame({"2026-05-01 10:00": 0.0, "2026-05-01 11:30": 99.0})
    aligned = FieldSimulation._align_flow(frame, index)
    assert list(aligned) == [0.0, 0.0, 0.0, 0.0]
