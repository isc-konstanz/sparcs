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


def _make_predictor(windows, calls, last_simulated_at=pd.Timestamp("2026-07-03 00:00", tz=_TZ), parallel=False):
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
    # Execution path: the sequential caterpillar (_rollout_ladder stub below) by
    # default; parallel=True routes through _rollout_parallel (stubbed too).
    p._parallel = parallel

    p._fetch_forecast = lambda now: pd.DataFrame({"f": [0.0]}, index=pd.DatetimeIndex([now]))

    def _solve(ic, et_data, seg_et):
        calls.append("solve")
        return [et_data.index[0]], {}, {}, {}

    def _solve_candidate(ic, candidate, et_data, seg_et, horizon_start, horizon_end):
        calls.append("solve_candidate")
        return [et_data.index[0]], {}, {}, {}

    def _rollout(*_a, **_k):
        calls.append("rollout")
        return {}

    def _select(*_a, **_k):
        calls.append("select")
        return (pd.Timedelta(0),)

    def _build_header_frame(*_a, **_k):
        calls.append("build_header_frame")
        return pd.DataFrame()

    def _build_detail_frame(*_a, **_k):
        calls.append("build_detail_frame")
        return pd.DataFrame()

    def _build_irrigation_frame(*_a, **_k):
        calls.append("build_irrigation_frame")
        return pd.DataFrame()

    def _rollout_parallel(*_a, **_k):
        calls.append("rollout_parallel")
        return {}

    p._solve = _solve
    p._solve_candidate = _solve_candidate
    p._publish_results = lambda *a, **k: calls.append("publish_results")
    p._rollout_ladder = _rollout
    p._rollout_parallel = _rollout_parallel
    p._select = _select
    p._build_header_frame = _build_header_frame
    p._write_header_table = lambda *a, **k: calls.append("write_header_table")
    p._build_detail_frame = _build_detail_frame
    p._write_detail_table = lambda *a, **k: calls.append("write_detail_table")
    p._build_irrigation_frame = _build_irrigation_frame
    p._write_irrigation_table = lambda *a, **k: calls.append("write_irrigation_table")
    return p


def test_predict_grid_block_invokes_all_collaborators(caplog):
    """The happy path: the legacy roll runs, then the grid block invokes the
    ladder roll-out, selector, and the header/detail/irrigation frame build +
    direct write -- and its try/except swallows nothing. This fails if predict()
    calls a grid collaborator with a non-existent attribute or a wrong argument:
    the resulting exception is caught by the grid try/except, so the collaborator
    spies never fire and the 'watering-grid' error is logged -- both asserted
    below."""
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
        "build_header_frame",
        "write_header_table",
        "build_detail_frame",
        "write_detail_table",
        "build_irrigation_frame",
        "write_irrigation_table",
    ):
        assert step in calls, f"{step!r} not called; predict() grid wiring is broken. calls={calls}"
    assert "watering-grid" not in caplog.text.lower(), f"grid try/except swallowed an error: {caplog.text}"
    # The irrigation plan is published after the header/detail tables.
    assert calls.index("write_detail_table") < calls.index("build_irrigation_frame")


def test_predict_writes_image_table_when_publish_returns_a_render(caplog):
    """When _publish_results returns a rendered (plot_index, pngs) tuple (plotting
    on), predict() persists it via _build_image_frame -> _write_image_table, after
    the irrigation write (each table under its own per-table try since W2.6). With
    no render (the default stub returns None) the image write is skipped (covered
    by the happy path above, where 'write_image_table' never appears)."""
    calls = []
    predictor = _make_predictor([object()], calls)
    save_index = pd.DatetimeIndex([pd.Timestamp("2026-07-03 02:00", tz=_TZ)], name="timestamp")
    predictor._publish_results = lambda *a, **k: calls.append("publish_results") or (save_index, [b"png"])
    predictor._build_image_frame = lambda *a, **k: calls.append("build_image_frame") or pd.DataFrame()
    predictor._write_image_table = lambda *a, **k: calls.append("write_image_table")
    now = pd.Timestamp("2026-07-03 01:30", tz=_TZ)

    with caplog.at_level(logging.ERROR):
        predictor.predict(now, forecast_creation=now)

    assert "build_image_frame" in calls
    assert "write_image_table" in calls
    assert (
        "-table write failed" not in caplog.text.lower()
    ), f"a per-table write try/except swallowed an error: {caplog.text}"
    assert calls.index("write_irrigation_table") < calls.index("write_image_table")


def test_predict_header_write_failure_does_not_skip_later_tables(caplog):
    """W2.6: one try PER table -- a header build/write failure must not drop the
    detail, irrigation, or image writes (previously a single try skipped them all)."""
    calls = []
    predictor = _make_predictor([object()], calls)
    save_index = pd.DatetimeIndex([pd.Timestamp("2026-07-03 02:00", tz=_TZ)], name="timestamp")
    predictor._publish_results = lambda *a, **k: calls.append("publish_results") or (save_index, [b"png"])
    predictor._build_image_frame = lambda *a, **k: calls.append("build_image_frame") or pd.DataFrame()
    predictor._write_image_table = lambda *a, **k: calls.append("write_image_table")

    def _boom_header(*_a, **_k):
        calls.append("write_header_table")
        raise RuntimeError("header connector down")

    predictor._write_header_table = _boom_header
    now = pd.Timestamp("2026-07-03 01:30", tz=_TZ)

    with caplog.at_level(logging.ERROR):
        predictor.predict(now, forecast_creation=now)

    for step in ("write_header_table", "write_detail_table", "write_irrigation_table", "write_image_table"):
        assert step in calls, f"{step!r} skipped; per-table isolation broken. calls={calls}"
    header_errors = [r for r in caplog.records if "header-table write failed" in r.getMessage()]
    assert len(header_errors) == 1


def test_predict_build_failure_does_not_skip_later_tables(caplog):
    """W2.6: the BUILD half of a table's try -- a detail-frame build failure is
    contained exactly like a write failure, and later tables still write."""
    calls = []
    predictor = _make_predictor([object()], calls)

    def _boom_build(*_a, **_k):
        calls.append("build_detail_frame")
        raise RuntimeError("frame construction bug")

    predictor._build_detail_frame = _boom_build
    now = pd.Timestamp("2026-07-03 01:30", tz=_TZ)

    with caplog.at_level(logging.ERROR):
        predictor.predict(now, forecast_creation=now)

    assert "write_detail_table" not in calls  # its own write is skipped with the build
    assert "write_irrigation_table" in calls  # later tables are not
    assert [r for r in caplog.records if "detail-table write failed" in r.getMessage()]


def test_predict_irrigation_write_failure_does_not_skip_image_write(caplog):
    """W2.6 twin: the table whose failure used to drop the image write."""
    calls = []
    predictor = _make_predictor([object()], calls)
    save_index = pd.DatetimeIndex([pd.Timestamp("2026-07-03 02:00", tz=_TZ)], name="timestamp")
    predictor._publish_results = lambda *a, **k: calls.append("publish_results") or (save_index, [b"png"])
    predictor._build_image_frame = lambda *a, **k: calls.append("build_image_frame") or pd.DataFrame()
    predictor._write_image_table = lambda *a, **k: calls.append("write_image_table")

    def _boom_irrigation(*_a, **_k):
        calls.append("write_irrigation_table")
        raise RuntimeError("irrigation frame bug")

    predictor._write_irrigation_table = _boom_irrigation
    now = pd.Timestamp("2026-07-03 01:30", tz=_TZ)

    with caplog.at_level(logging.ERROR):
        predictor.predict(now, forecast_creation=now)

    assert "write_image_table" in calls, f"image write dropped with the irrigation failure. calls={calls}"
    assert [r for r in caplog.records if "irrigation-table write failed" in r.getMessage()]


def test_predict_publishes_resolved_chosen_when_recommendation_is_nonzero(caplog):
    """When the selector picks a NON-zero candidate, predict() re-solves that single
    candidate (to recover its snapshots + diagnostics) and publishes ITS roll on the
    main channels. The re-solve happens before the main publish, which happens before
    the secondary header/detail writes."""
    calls = []
    predictor = _make_predictor([object()], calls)

    def _select_nonzero(*_a, **_k):
        calls.append("select")
        return (pd.Timedelta(minutes=30),)

    predictor._select = _select_nonzero
    now = pd.Timestamp("2026-07-03 01:30", tz=_TZ)

    with caplog.at_level(logging.ERROR):
        predictor.predict(now, forecast_creation=now)

    assert "solve_candidate" in calls, f"chosen candidate not re-solved. calls={calls}"
    assert calls.index("solve_candidate") < calls.index("publish_results")
    assert calls.index("publish_results") < calls.index("write_header_table")
    assert "watering-grid" not in caplog.text.lower()


def test_predict_zero_recommendation_reuses_zero_flow_solve_without_resolve(caplog):
    """When the selector picks the all-0min rung, predict() reuses the held zero-flow
    solve for the main channels -- it must NOT re-solve (the zero-flow roll already IS
    that candidate's roll)."""
    calls = []
    predictor = _make_predictor([object()], calls)  # default _select returns the zero rung
    now = pd.Timestamp("2026-07-03 01:30", tz=_TZ)

    with caplog.at_level(logging.ERROR):
        predictor.predict(now, forecast_creation=now)

    assert "publish_results" in calls
    assert "solve_candidate" not in calls, f"zero rung must not re-solve. calls={calls}"


def test_predict_degrades_to_caterpillar_when_parallel_roll_raises(caplog):
    """parallel=True but the parallel executor raises (e.g. the pool cannot be
    created): predict() must still produce the forecast via the sequential
    caterpillar, log the fallback, and NOT let the failure reach the outer grid
    try/except (which would skip the header + detail + irrigation write)."""
    calls = []
    predictor = _make_predictor([object()], calls, parallel=True)

    def _boom(*_a, **_k):
        raise RuntimeError("cannot create process pool")

    predictor._rollout_parallel = _boom
    now = pd.Timestamp("2026-07-03 01:30", tz=_TZ)

    with caplog.at_level(logging.ERROR):
        predictor.predict(now, forecast_creation=now)

    # The full grid chain still runs via the caterpillar fallback ("rollout");
    # the parallel spy never appended, because it raised first.
    for step in (
        "solve",
        "publish_results",
        "rollout",
        "select",
        "build_header_frame",
        "write_header_table",
        "build_detail_frame",
        "write_detail_table",
        "build_irrigation_frame",
        "write_irrigation_table",
    ):
        assert step in calls, f"{step!r} not called; parallel-degrade wiring is broken. calls={calls}"
    assert "rollout_parallel" not in calls
    assert "parallel roll-out failed" in caplog.text.lower()
    # The degrade is handled inside _rollout_dispatch, so the OUTER grid
    # try/except (its "watering-grid" error) must not have fired.
    assert "watering-grid" not in caplog.text.lower()


def test_predict_grid_block_skipped_without_windows():
    """No windows configured -> the predictor stays a pure zero-flow forecaster:
    the legacy roll still publishes, but no grid collaborator runs."""
    calls = []
    predictor = _make_predictor([], calls)
    now = pd.Timestamp("2026-07-03 01:30", tz=_TZ)

    predictor.predict(now, forecast_creation=now)

    assert "publish_results" in calls
    for grid_step in (
        "rollout",
        "select",
        "build_header_frame",
        "write_header_table",
        "build_detail_frame",
        "write_detail_table",
        "build_irrigation_frame",
        "write_irrigation_table",
    ):
        assert grid_step not in calls


def test_predict_skips_on_cold_start_without_claiming_the_boundary():
    """No live soil state yet -> predict() returns at the cold-start guard before
    any roll, and (like the missing-forecast guard) does NOT claim the boundary, so
    the next tick retries once the live solver has state."""
    calls = []
    predictor = _make_predictor([object()], calls, last_simulated_at=None)
    now = pd.Timestamp("2026-07-03 01:30", tz=_TZ)

    predictor.predict(now, forecast_creation=now)

    assert calls == []
    assert predictor._last_boundary_run is None


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


def test_predict_grid_failure_falls_back_to_zero_flow_forecast(caplog):
    """A grid/DB failure must never abort the tick: a raising ladder roll-out is
    caught and logged, the held zero-flow roll is published as the fallback, and the
    secondary header/detail/irrigation writes are skipped. predict() returns
    normally."""
    calls = []
    predictor = _make_predictor([object()], calls)

    def _boom(*_a, **_k):
        raise RuntimeError("simulated grid failure")

    predictor._rollout_ladder = _boom
    now = pd.Timestamp("2026-07-03 01:30", tz=_TZ)

    with caplog.at_level(logging.ERROR):
        predictor.predict(now, forecast_creation=now)  # must not raise

    assert "publish_results" in calls  # zero-flow fallback still published
    assert "write_header_table" not in calls  # secondary writes skipped after the failure
    assert "write_detail_table" not in calls
    assert "build_header_frame" not in calls
    assert "build_irrigation_frame" not in calls
    assert "write_irrigation_table" not in calls
    assert "watering-grid" in caplog.text.lower()  # failure was logged, not silent
