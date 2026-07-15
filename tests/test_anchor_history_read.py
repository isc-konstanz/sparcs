# -*- coding: utf-8 -*-
"""The contemporaneous tension read shared by the live tick and soil_tuning.

``latest_reading_at`` is the single lookup both backends call per step: given a
per-sensor tension series (range-read once per tick), return the reading at or
before the step time so it anchors at its own timestamp -- never the latest value
smeared onto a different step (the catch-up back-dating bug). ``_read_history_tension``
is the live binding over ``SoilSimulation._anchor_history``.
"""

import math
import types

import numpy as np
import pandas as pd
from sparcs.components.agriculture.simulation._anchor import AnchorSensor, latest_reading_at


def _series(pairs):
    idx = pd.to_datetime([t for t, _ in pairs])
    return pd.Series([v for _, v in pairs], index=idx)


T0 = pd.Timestamp("2026-07-14 12:00")


def test_returns_reading_at_or_before_now():
    s = _series([(T0, 100.0), (T0 + pd.Timedelta("1h"), 200.0), (T0 + pd.Timedelta("2h"), 300.0)])
    # Between the 1h and 2h readings -> the 1h one (contemporaneous, not the latest).
    ts, val = latest_reading_at(s, T0 + pd.Timedelta("90min"))
    assert ts == T0 + pd.Timedelta("1h") and val == 200.0
    ts, val = latest_reading_at(s, T0 + pd.Timedelta("5h"))
    assert ts == T0 + pd.Timedelta("2h") and val == 300.0


def test_each_reading_lands_at_its_own_time():
    """Stepping the frontier forward hands each reading out exactly once, at its time."""
    s = _series([(T0, 100.0), (T0 + pd.Timedelta("1h"), 200.0)])
    assert latest_reading_at(s, T0 - pd.Timedelta("1min"))[0] is None
    assert latest_reading_at(s, T0)[1] == 100.0
    assert latest_reading_at(s, T0 + pd.Timedelta("30min"))[1] == 100.0
    assert latest_reading_at(s, T0 + pd.Timedelta("1h"))[1] == 200.0


def test_empty_or_missing_series_reads_as_missing():
    for ts, val in (latest_reading_at(None, T0), latest_reading_at(_series([]), T0)):
        assert ts is None and math.isnan(val)


def test_nothing_before_now_reads_as_missing():
    s = _series([(T0 + pd.Timedelta("1h"), 200.0)])
    ts, val = latest_reading_at(s, T0)
    assert ts is None and math.isnan(val)


def test_nonfinite_latest_value_reads_as_missing():
    s = _series([(T0, 100.0), (T0 + pd.Timedelta("1h"), float("nan"))])
    ts, val = latest_reading_at(s, T0 + pd.Timedelta("2h"))
    assert ts is None and math.isnan(val)


def _fake_sim(history):
    return types.SimpleNamespace(_anchor_history=history)


def test_read_history_tension_binds_to_now():
    from sparcs.components.agriculture.simulation.soil import SoilSimulation

    s = _series([(T0, 100.0), (T0 + pd.Timedelta("1h"), 200.0)])
    sim = _fake_sim({"s1": s})
    sensor = AnchorSensor(key="s1", x_offset_cm=0.0, depth_cm=30.0)
    ts, val = SoilSimulation._read_history_tension(sim, sensor, T0 + pd.Timedelta("90min"))
    assert ts == T0 + pd.Timedelta("1h") and val == 200.0


def test_read_history_tension_unknown_sensor_reads_as_missing():
    from sparcs.components.agriculture.simulation.soil import SoilSimulation

    sim = _fake_sim({})
    sensor = AnchorSensor(key="s1", x_offset_cm=0.0, depth_cm=30.0)
    ts, val = SoilSimulation._read_history_tension(sim, sensor, T0)
    assert ts is None and np.isnan(val)
