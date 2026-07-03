# -*- coding: utf-8 -*-
"""sparcs.tests.test_soil_predictor_predict_wiring
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Integration tests for ``SoilPredictor.predict()``'s control flow (issue 07).

Unit tests of the individual pure methods (flow schedule, ladder, selector,
trajectory frame) cannot catch a mistake in how ``predict()`` *wires them
together* -- e.g. calling a collaborator with an attribute that does not exist
on the instance. Because the grid block is wrapped in a defensive try/except,
such a bug degrades silently. These tests drive ``predict()`` end-to-end with
stubbed collaborators and assert the grid block is actually invoked (and that
its try/except did not swallow an exception), plus that the cold-start and
missing-forecast guards short-circuit before any roll.
"""

import logging

import pytest

import pandas as pd

soil_predictor = pytest.importorskip("sparcs.components.agriculture.simulation.soil_predictor")
SoilPredictor = soil_predictor.SoilPredictor

_TZ = "Europe/Berlin"


class _Loc:
    timezone = _TZ


class _Soil:
    def __init__(self, last_simulated_at):
        self._last_simulated_at = last_simulated_at

    def get_rel_sat_snapshot(self):
        return "IC"


class _Pde:
    # Only ever passed through to the (stubbed) _select as an argument; the real
    # attribute name matters -- predict() reads self._pde.soil_model, so a rename
    # of either would break the wiring and fail these tests.
    soil_model = object()


class _Ctx:
    def __init__(self, soil):
        self.location = _Loc()
        self.soil_simulation = soil

    def _run_chain(self, forecast, publish=False):
        idx = pd.DatetimeIndex(
            [
                pd.Timestamp("2026-07-03 01:00", tz=_TZ),
                pd.Timestamp("2026-07-03 02:00", tz=_TZ),
            ],
            name="timestamp",
        )
        return pd.DataFrame({"x": [0.0, 0.0]}, index=idx), {}


def _make_predictor(windows, calls, last_simulated_at=pd.Timestamp("2026-07-03 00:00", tz=_TZ)):
    """A bare SoilPredictor whose heavy collaborators are stubbed to record calls,
    so predict()'s gate -> guards -> legacy roll -> grid wiring can run PDE-free."""
    p = object.__new__(SoilPredictor)
    p._name = "test_predictor"
    p._Registrator__context = _Ctx(_Soil(last_simulated_at))
    p._interval_min = 1440
    p._offset_min = 60
    p._last_boundary_run = None
    p._last_predicted_key = None
    p._horizon = pd.Timedelta("24h")
    p._probes = []
    p._pde = _Pde()

    p._windows = windows
    p._ladder = [(pd.Timedelta(0),)]
    p._flow_m3s = 1.0e-5
    p._decision_probes = []
    p._threshold_hpa = 300.0
    p._grid_mode = "fill_order"

    p._fetch_forecast = lambda now: pd.DataFrame({"f": [0.0]}, index=pd.DatetimeIndex([now]))

    def _solve(ic, et_data, seg_et):
        calls.append("solve")
        return [et_data.index[0]], {}, {}, {}

    def _rollout(*_a, **_k):
        calls.append("rollout")
        return {}

    def _select(*_a, **_k):
        calls.append("select")
        return (pd.Timedelta(0),), "none_needed"

    def _build_frame(*_a, **_k):
        calls.append("build_frame")
        return pd.DataFrame()

    p._solve = _solve
    p._publish_results = lambda *a, **k: calls.append("publish_results")
    p._rollout_ladder = _rollout
    p._select = _select
    p._publish_recommendation = lambda *a, **k: calls.append("publish_recommendation")
    p._build_trajectory_frame = _build_frame
    p._write_trajectory_table = lambda *a, **k: calls.append("write_table")
    return p


def test_predict_grid_block_invokes_all_collaborators(caplog):
    """The happy path: the legacy roll runs, then the grid block invokes the
    ladder roll-out, selector, recommendation publish, frame build, and direct
    write -- and its try/except swallows nothing. This fails if predict()
    references a non-existent attribute (e.g. an accidental self._soil_model
    instead of self._pde.soil_model)."""
    calls = []
    predictor = _make_predictor([object()], calls)
    now = pd.Timestamp("2026-07-03 01:30", tz=_TZ)

    with caplog.at_level(logging.ERROR):
        predictor.predict(now, forecast_creation=now)

    for step in (
        "solve",
        "publish_results",
        "rollout",
        "select",
        "publish_recommendation",
        "build_frame",
        "write_table",
    ):
        assert step in calls, f"{step!r} not called; predict() grid wiring is broken. calls={calls}"
    assert "watering-grid" not in caplog.text.lower(), f"grid try/except swallowed an error: {caplog.text}"


def test_predict_grid_block_skipped_without_windows():
    """No windows configured -> the predictor stays a pure zero-flow forecaster:
    the legacy roll still publishes, but no grid collaborator runs."""
    calls = []
    predictor = _make_predictor([], calls)
    now = pd.Timestamp("2026-07-03 01:30", tz=_TZ)

    predictor.predict(now, forecast_creation=now)

    assert "publish_results" in calls
    for grid_step in ("rollout", "select", "publish_recommendation", "build_frame", "write_table"):
        assert grid_step not in calls


def test_predict_skips_on_cold_start():
    """No live soil state yet -> predict() returns at the cold-start guard before
    any roll (neither the legacy roll nor the grid runs)."""
    calls = []
    predictor = _make_predictor([object()], calls, last_simulated_at=None)
    now = pd.Timestamp("2026-07-03 01:30", tz=_TZ)

    predictor.predict(now, forecast_creation=now)

    assert calls == []


def test_predict_missing_forecast_does_not_roll_or_claim_boundary():
    """A missing forecast -> predict() returns at the forecast guard, without
    running any roll and WITHOUT claiming the boundary (so the next tick retries)."""
    calls = []
    predictor = _make_predictor([object()], calls)
    predictor._fetch_forecast = lambda now: None
    now = pd.Timestamp("2026-07-03 01:30", tz=_TZ)

    predictor.predict(now, forecast_creation=now)

    assert calls == []
    assert predictor._last_boundary_run is None


def test_predict_short_forecast_chain_skips():
    """A forecast whose chain replay yields fewer than 2 rows -> predict() skips
    (a single-row horizon cannot be integrated)."""
    calls = []
    predictor = _make_predictor([object()], calls)
    single = pd.DataFrame(
        {"x": [0.0]},
        index=pd.DatetimeIndex([pd.Timestamp("2026-07-03 01:00", tz=_TZ)], name="timestamp"),
    )
    predictor._Registrator__context._run_chain = lambda forecast, publish=False: (single, {})
    now = pd.Timestamp("2026-07-03 01:30", tz=_TZ)

    predictor.predict(now, forecast_creation=now)

    assert calls == []


def test_predict_grid_failure_does_not_abort_legacy_forecast(caplog):
    """A grid/DB failure must never abort the tick: the legacy forecast is
    published, then a raising ladder roll-out is caught and logged, and predict()
    returns normally."""
    calls = []
    predictor = _make_predictor([object()], calls)

    def _boom(*_a, **_k):
        raise RuntimeError("simulated grid failure")

    predictor._rollout_ladder = _boom
    now = pd.Timestamp("2026-07-03 01:30", tz=_TZ)

    with caplog.at_level(logging.ERROR):
        predictor.predict(now, forecast_creation=now)  # must not raise

    assert "publish_results" in calls  # legacy forecast still published
    assert "write_table" not in calls  # grid aborted after the failure
    assert "watering-grid" in caplog.text.lower()  # failure was logged, not silent
