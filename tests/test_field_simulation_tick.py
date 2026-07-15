# -*- coding: utf-8 -*-
"""sparcs.tests.test_field_simulation_tick
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Unit tests for the wall-clock tick scaffolding on ``FieldSimulation``:
config parse (``[field_simulation] interval``/``offset``), absolute slot
alignment, the injected clock, overrun skip, and thread teardown.

No real time is slept on: the clock is injected via ``_now`` and the loop's
``Event.wait`` chunk is shrunk through ``_TICK_WAIT_MAX_S``. As in the
sibling field-simulation tests, instances come from ``object.__new__`` so no
Component/PDE stack is instantiated.
"""

import threading
import time
import types

import pytest

import pandas as pd

_base = pytest.importorskip("sparcs.components.agriculture.simulation.base")
FieldSimulation = _base.FieldSimulation

from lories import Configurations  # noqa: E402


def _configs(tmp_path, **values) -> Configurations:
    return Configurations.load("test.conf", conf_dir=str(tmp_path), require=False, **values)


def _sim(interval_min: int = 60, offset_min: int = 0) -> FieldSimulation:
    sim = object.__new__(FieldSimulation)
    sim._name = "test_field_simulation"
    sim.location = None
    sim._interval_min = interval_min
    sim._offset_min = offset_min
    return sim


# --- config parse -----------------------------------------------------------


def test_tick_schedule_defaults(tmp_path):
    assert FieldSimulation._parse_tick_schedule(_configs(tmp_path)) == (60, 0)


def test_tick_schedule_parses_values(tmp_path):
    assert FieldSimulation._parse_tick_schedule(_configs(tmp_path, interval=15, offset=5)) == (15, 5)


def test_tick_schedule_rejects_zero_interval(tmp_path):
    with pytest.raises(ValueError, match="interval"):
        FieldSimulation._parse_tick_schedule(_configs(tmp_path, interval=0))


def test_tick_schedule_rejects_offset_outside_interval(tmp_path):
    with pytest.raises(ValueError, match="offset"):
        FieldSimulation._parse_tick_schedule(_configs(tmp_path, interval=60, offset=60))


def test_tick_schedule_rejects_negative_offset(tmp_path):
    with pytest.raises(ValueError, match="offset"):
        FieldSimulation._parse_tick_schedule(_configs(tmp_path, offset=-5))


# --- slot alignment ---------------------------------------------------------


def test_next_slot_aligns_to_interval_plus_offset():
    sim = _sim(interval_min=60, offset_min=5)
    now = pd.Timestamp("2026-07-12 10:20", tz="UTC")
    assert sim._next_slot(now) == pd.Timestamp("2026-07-12 11:05", tz="UTC")


def test_next_slot_is_strictly_future_when_now_is_on_slot():
    sim = _sim(interval_min=60, offset_min=5)
    now = pd.Timestamp("2026-07-12 10:05", tz="UTC")
    assert sim._next_slot(now) == pd.Timestamp("2026-07-12 11:05", tz="UTC")


def test_next_slot_sub_hourly():
    sim = _sim(interval_min=15, offset_min=0)
    now = pd.Timestamp("2026-07-12 10:07", tz="UTC")
    assert sim._next_slot(now) == pd.Timestamp("2026-07-12 10:15", tz="UTC")


def test_next_slot_depends_only_on_now():
    """Alignment is absolute wall clock, so a restart cannot shift the schedule."""
    now = pd.Timestamp("2026-07-12 10:20", tz="UTC")
    assert _sim()._next_slot(now) == _sim()._next_slot(now)


# --- span computation (_on_tick) --------------------------------------------


def _tick_sim(frontier, intake_delay=pd.Timedelta(0), interval_min: int = 60):
    """A sim wired for _on_tick with a stub soil child and recording span reads."""
    sim = _sim(interval_min=interval_min)
    sim._intake_delay = intake_delay
    sim._required_weather_keys = ()
    sim.evapotranspiration = object()
    sim._weather_channels = object()
    sim.soil_simulation = types.SimpleNamespace(
        _last_simulated_at=frontier,
        advance=lambda et, now, seg: None,
        simulate_loop=lambda et, seg: None,
        load_anchor_history=lambda start, end: None,
    )
    sim._spans = []

    def _record_span(start, end):
        sim._spans.append((start, end))
        return pd.DataFrame()

    sim._read_weather_span = _record_span
    return sim


def test_on_tick_span_is_frontier_to_now_minus_intake_delay():
    sim = _tick_sim(
        frontier=pd.Timestamp("2026-07-12 10:00", tz="UTC"),
        intake_delay=pd.Timedelta(minutes=30),
    )
    sim._on_tick(pd.Timestamp("2026-07-12 12:00", tz="UTC"))
    assert sim._spans == [(pd.Timestamp("2026-07-12 10:00", tz="UTC"), pd.Timestamp("2026-07-12 11:30", tz="UTC"))]


def test_on_tick_without_frontier_reads_one_interval_back():
    sim = _tick_sim(frontier=None, interval_min=60)
    sim._on_tick(pd.Timestamp("2026-07-12 12:00", tz="UTC"))
    assert sim._spans == [(pd.Timestamp("2026-07-12 11:00", tz="UTC"), pd.Timestamp("2026-07-12 12:00", tz="UTC"))]


def test_on_tick_is_noop_when_frontier_reaches_cutoff():
    sim = _tick_sim(
        frontier=pd.Timestamp("2026-07-12 12:00", tz="UTC"),
        intake_delay=pd.Timedelta(minutes=30),
    )
    sim._on_tick(pd.Timestamp("2026-07-12 12:00", tz="UTC"))
    assert sim._spans == []


def test_on_tick_chunks_backlog_by_day():
    sim = _tick_sim(frontier=pd.Timestamp("2026-07-10 06:00", tz="UTC"))
    sim._on_tick(pd.Timestamp("2026-07-12 12:00", tz="UTC"))
    assert sim._spans == [
        (pd.Timestamp("2026-07-10 06:00", tz="UTC"), pd.Timestamp("2026-07-11 00:00", tz="UTC")),
        (pd.Timestamp("2026-07-11 00:00", tz="UTC"), pd.Timestamp("2026-07-12 00:00", tz="UTC")),
        (pd.Timestamp("2026-07-12 00:00", tz="UTC"), pd.Timestamp("2026-07-12 12:00", tz="UTC")),
    ]


def test_on_tick_advances_only_to_the_logged_data_frontier():
    """Weather ends before the cutoff: the chain runs on what exists, no filling."""
    sim = _tick_sim(frontier=pd.Timestamp("2026-07-12 10:00", tz="UTC"))
    index = pd.date_range("2026-07-12 10:15", periods=3, freq="15min", tz="UTC")  # ends 10:45 < 12:00
    weather = pd.DataFrame({"ghi": [100.0, 200.0, 300.0]}, index=index)
    sim._read_weather_span = lambda start, end: weather
    sim._run_chain = lambda frame: (frame, {})
    advanced = []
    sim.soil_simulation.simulate_loop = lambda et, seg: advanced.append(et.index[-1])

    sim._on_tick(pd.Timestamp("2026-07-12 12:00", tz="UTC"))
    assert advanced == [pd.Timestamp("2026-07-12 10:45", tz="UTC")]


def test_on_tick_cold_start_uses_single_advance():
    sim = _tick_sim(frontier=None)
    index = pd.date_range("2026-07-12 11:15", periods=3, freq="15min", tz="UTC")
    weather = pd.DataFrame({"ghi": [100.0, 200.0, 300.0]}, index=index)
    sim._read_weather_span = lambda start, end: weather
    sim._run_chain = lambda frame: (frame, {})
    calls = []
    sim.soil_simulation.advance = lambda et, now, seg: calls.append(("advance", now))
    sim.soil_simulation.simulate_loop = lambda et, seg: calls.append(("loop", et.index[-1]))

    sim._on_tick(pd.Timestamp("2026-07-12 12:00", tz="UTC"))
    assert calls == [("advance", pd.Timestamp("2026-07-12 11:45", tz="UTC"))]


# --- span reads come from the connector, not a logger -------------------------


class _RecordingData:
    """Stub ``self.data`` recording whether a span was read via the connector
    (``read``) or a logger (``read_logged``)."""

    def __init__(self):
        self.calls = []

    def read(self, channels, start=None, end=None, unique=False):
        self.calls.append(("read", start, end, unique))
        return pd.DataFrame()

    def read_logged(self, *args, **kwargs):
        self.calls.append(("read_logged",))
        return pd.DataFrame()


def test_read_weather_span_reads_connector_not_logger():
    """Weather already lives in its source (kob_tracker station / Brightsky), so
    the tick reads the span from the connector; it must not go through a logger."""
    sim = _sim()
    sim._weather_channels = object()
    sim._evapo_rename = {}
    sim._Component__data = _RecordingData()  # backs the read-only `data` property

    start = pd.Timestamp("2026-07-12 10:00", tz="UTC")
    end = pd.Timestamp("2026-07-12 11:00", tz="UTC")
    sim._read_weather_span(start, end)

    assert sim.data.calls == [("read", start, end, True)]


def test_read_flow_span_reads_connector_not_logger():
    """Irrigation flow is read from its connector over the (lookback, end] span too."""
    sim = _sim()
    sim._irrigation_flow_channel = types.SimpleNamespace(id="irrigation_flow")
    sim._Component__data = _RecordingData()  # backs the read-only `data` property

    index = pd.date_range("2026-07-12 10:00", periods=2, freq="30min", tz="UTC")
    series = sim._read_flow_span(index[0], index[-1], index)

    assert [c[0] for c in sim.data.calls] == ["read"]
    assert list(series) == [0.0, 0.0]  # empty history aligns to zeros


# --- predictor on the tick ----------------------------------------------------


def _predicting_sim(frontier):
    """A _tick_sim whose soil stub moves its frontier on advance, with a spy predictor."""
    sim = _tick_sim(frontier=frontier)
    sim._read_forecast_epoch = lambda: None
    predictions = []
    sim.soil_predictor = types.SimpleNamespace(predict=lambda now, forecast_creation=None: predictions.append(now))
    return sim, predictions


def test_predict_runs_after_a_tick_that_advanced():
    frontier = pd.Timestamp("2026-07-12 10:00", tz="UTC")
    sim, predictions = _predicting_sim(frontier)
    index = pd.date_range("2026-07-12 10:15", periods=3, freq="15min", tz="UTC")
    sim._read_weather_span = lambda start, end: pd.DataFrame({"ghi": [1.0, 2.0, 3.0]}, index=index)
    sim._run_chain = lambda frame: (frame, {})

    def _advance_frontier(et, seg):
        sim.soil_simulation._last_simulated_at = et.index[-1]

    sim.soil_simulation.simulate_loop = _advance_frontier

    sim._on_tick(pd.Timestamp("2026-07-12 12:00", tz="UTC"))
    assert predictions == [pd.Timestamp("2026-07-12 10:45", tz="UTC")]


def test_predict_not_called_after_a_noop_tick():
    """No new logged data -> no advance -> the predictor is not invoked."""
    frontier = pd.Timestamp("2026-07-12 10:00", tz="UTC")
    sim, predictions = _predicting_sim(frontier)  # _read_weather_span returns empty

    sim._on_tick(pd.Timestamp("2026-07-12 12:00", tz="UTC"))
    assert predictions == []


# --- day chunking and span trim ----------------------------------------------


def test_iter_day_chunks_splits_at_midnight():
    chunks = list(
        FieldSimulation._iter_day_chunks(
            pd.Timestamp("2026-07-10 06:00", tz="UTC"),
            pd.Timestamp("2026-07-11 03:00", tz="UTC"),
        )
    )
    assert chunks == [
        (pd.Timestamp("2026-07-10 06:00", tz="UTC"), pd.Timestamp("2026-07-11 00:00", tz="UTC")),
        (pd.Timestamp("2026-07-11 00:00", tz="UTC"), pd.Timestamp("2026-07-11 03:00", tz="UTC")),
    ]


def test_iter_day_chunks_single_chunk_within_day():
    chunks = list(
        FieldSimulation._iter_day_chunks(
            pd.Timestamp("2026-07-12 06:00", tz="UTC"),
            pd.Timestamp("2026-07-12 09:00", tz="UTC"),
        )
    )
    assert chunks == [(pd.Timestamp("2026-07-12 06:00", tz="UTC"), pd.Timestamp("2026-07-12 09:00", tz="UTC"))]


def test_trim_span_renames_and_keeps_half_open_interval():
    sim = _sim()
    sim._evapo_rename = {"weather.ghi": "ghi"}
    index = pd.date_range("2026-07-12 10:00", periods=5, freq="15min", tz="UTC")
    frame = pd.DataFrame({"weather.ghi": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=index)

    trimmed = sim._trim_span(frame, index[0], pd.Timestamp("2026-07-12 10:45", tz="UTC"))
    assert list(trimmed.columns) == ["ghi"]
    assert trimmed.index[0] == pd.Timestamp("2026-07-12 10:15", tz="UTC")  # start row excluded
    assert trimmed.index[-1] == pd.Timestamp("2026-07-12 10:45", tz="UTC")  # end row kept


def test_weather_frame_valid_flags_missing_required_column():
    sim = _sim()
    sim._required_weather_keys = ("ghi", "temp_air")
    frame = pd.DataFrame(
        {"ghi": [1.0]},
        index=pd.DatetimeIndex([pd.Timestamp("2026-07-12 10:00", tz="UTC")]),
    )
    assert sim._weather_frame_valid(frame) is False
    frame["temp_air"] = 20.0
    assert sim._weather_frame_valid(frame) is True


# --- overrun skip -----------------------------------------------------------


def test_tick_skips_slot_while_previous_run_holds_the_lock():
    sim = _sim()
    sim._now = lambda: pd.Timestamp("2026-07-12 12:00", tz="UTC")
    sim._tick_lock = threading.Lock()
    calls = []
    sim._on_tick = calls.append

    sim._tick_lock.acquire()
    try:
        sim._tick()
    finally:
        sim._tick_lock.release()
    assert calls == []

    sim._tick()
    assert len(calls) == 1


def test_tick_releases_lock_after_failure():
    sim = _sim()
    sim._now = lambda: pd.Timestamp("2026-07-12 12:00", tz="UTC")
    sim._tick_lock = threading.Lock()

    def _boom(now):
        raise RuntimeError("chain failed")

    sim._on_tick = _boom
    sim._tick()  # must not raise and must not leave the lock held
    assert sim._tick_lock.acquire(blocking=False)
    sim._tick_lock.release()


# --- tick loop + teardown ---------------------------------------------------


class _SteppingClock:
    """Each ``now()`` call advances a fake clock by ``step``; thread-safe enough
    for one reader."""

    def __init__(self, start: pd.Timestamp, step: pd.Timedelta):
        self._current = start
        self._step = step

    def __call__(self) -> pd.Timestamp:
        now = self._current
        self._current = now + self._step
        return now


def test_tick_loop_fires_on_slot_and_stops_on_interrupt():
    sim = _sim(interval_min=1, offset_min=0)
    sim._TICK_WAIT_MAX_S = 0.005
    sim._now = _SteppingClock(pd.Timestamp("2026-07-12 10:00:10", tz="UTC"), pd.Timedelta(seconds=20))
    fired = threading.Event()
    ticks = []

    def _record(now):
        ticks.append(now)
        fired.set()

    sim._on_tick = _record
    sim._start_tick_thread()
    try:
        assert fired.wait(timeout=5.0), "tick did not fire"
    finally:
        sim._stop_tick_thread()

    assert sim._tick_thread is None
    assert ticks, "no tick recorded"
    # The loop only runs a slot once the fake clock passed 10:01:00.
    assert ticks[0] >= pd.Timestamp("2026-07-12 10:01:00", tz="UTC")


def test_stop_tick_thread_joins_promptly():
    sim = _sim(interval_min=60, offset_min=0)
    sim._TICK_WAIT_MAX_S = 0.005
    sim._now = lambda: pd.Timestamp("2026-07-12 10:00:10", tz="UTC")
    sim._on_tick = lambda now: None

    sim._start_tick_thread()
    thread = sim._tick_thread
    started = time.monotonic()
    sim._stop_tick_thread()
    assert time.monotonic() - started < 5.0
    assert not thread.is_alive()
