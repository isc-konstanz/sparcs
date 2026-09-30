# -*- coding: utf-8 -*-
"""
tests.test_fieldsim_anchor_history
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``latest_reading_at`` is the lookup every assimilation backend calls per step:
given a per-sensor tension series (range-read once per tick), return the
reading at or before the step time, so it anchors at its own timestamp instead
of smearing the latest value onto a different step.
"""

import math

import pandas as pd
from sparcs.components.agriculture.simulation.core.anchor import latest_reading_at


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
