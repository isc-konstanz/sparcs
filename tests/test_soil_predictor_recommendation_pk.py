# -*- coding: utf-8 -*-
"""sparcs.tests.test_soil_predictor_recommendation_pk
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Regression tests for the recommendation write path.

The recommendation lives in its OWN table (``_RECOMMEND_TABLE_NAME``), keyed by run
time only, so ``_publish_recommendation`` is a plain auto-logged write: one row per
run, no ``timestamp_creation`` composite-PK partner. That decoupling is what avoids
the "Unable to prepare datetime index from None" failure the shared soil_predictor
table used to hit (the auto-logger flushes per update-batch and would drop the PK
partner from the recommend_* batch). These tests pin the invariant: the method sets
exactly the recommend_* channels at run_timestamp and never touches
``timestamp_creation``. The predict-wiring tests stub the method out, so they cannot.
"""

import pytest

import pandas as pd

soil_predictor = pytest.importorskip("sparcs.components.agriculture.simulation.soil_predictor")
SoilPredictor = soil_predictor.SoilPredictor

_TZ = "Europe/Berlin"


class _FakeChannel:
    def __init__(self):
        self.sets = []

    def set(self, timestamp, value):
        self.sets.append((timestamp, value))


class _FakeData(dict):
    """Minimal ``self.data``: hands back a recording channel per key on first access."""

    def __getitem__(self, key):
        if key not in self:
            self[key] = _FakeChannel()
        return dict.__getitem__(self, key)


def _make_predictor(max_windows=4):
    p = object.__new__(SoilPredictor)
    p._name = "test_predictor"
    p._recommend_window_keys = [f"recommend_w{i}_min" for i in range(max_windows)]
    # ``data`` is a read-only property backing onto Component's name-mangled field.
    p._Component__data = _FakeData()
    return p


def test_publish_recommendation_writes_one_row_at_run_timestamp():
    """Every recommend_* channel (windows, total, status) is set exactly once, at
    run_timestamp -- one coherent auto-logged row per run."""
    predictor = _make_predictor(max_windows=4)
    run_timestamp = pd.Timestamp("2026-07-03 01:30", tz=_TZ)
    forecast_creation = pd.Timestamp("2026-07-03 00:00", tz=_TZ)

    predictor._publish_recommendation(
        (pd.Timedelta(minutes=45), pd.Timedelta(minutes=15)),
        "ok",
        run_timestamp,
        forecast_creation,
    )

    assert predictor.data["recommend_w0_min"].sets == [(run_timestamp, 45.0)]
    assert predictor.data["recommend_w1_min"].sets == [(run_timestamp, 15.0)]
    # Unconfigured windows (index >= len(chosen)) get 0.0, not the -1 sentinel.
    assert predictor.data["recommend_w2_min"].sets == [(run_timestamp, 0.0)]
    assert predictor.data["recommend_w3_min"].sets == [(run_timestamp, 0.0)]
    assert predictor.data[SoilPredictor._RECOMMEND_TOTAL_KEY].sets == [(run_timestamp, 60.0)]
    assert predictor.data[SoilPredictor._RECOMMEND_STATUS_KEY].sets == [(run_timestamp, "ok")]


def test_publish_recommendation_never_touches_timestamp_creation():
    """The decoupling invariant: the recommendation is keyed by run time in its own
    table, so it must NOT touch the forecast table's timestamp_creation PK partner.
    Touching it is what reintroduces the composite-PK write failure."""
    predictor = _make_predictor(max_windows=4)
    run_timestamp = pd.Timestamp("2026-07-03 01:30", tz=_TZ)
    forecast_creation = pd.Timestamp("2026-07-03 00:00", tz=_TZ)

    predictor._publish_recommendation((pd.Timedelta(0),), "none_needed", run_timestamp, forecast_creation)

    touched = set(predictor.data.keys())
    assert SoilPredictor._TIMESTAMP_CREATION_KEY not in touched, (
        "recommendation must not write timestamp_creation; it lives in its own "
        f"run-time-keyed table. Touched: {sorted(touched)}"
    )
