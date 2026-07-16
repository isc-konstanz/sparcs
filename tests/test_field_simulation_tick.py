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
        advance=lambda et, now, seg, cancel=None: None,
        simulate_loop=lambda et, seg, cancel=None: None,
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


def test_on_tick_aligns_a_location_tz_frontier_to_the_utc_clock():
    """A frontier inherited from a site-local weather index (Brightsky indexes its
    rows in ``location.timezone``) must be aligned to the UTC tick clock before the
    window is sliced. Otherwise a chunk pairs a ``+02:00`` start with the ``+00:00``
    cutoff, and Brightsky's ``source_data.loc[start:end]`` raises "Both dates must
    have the same UTC offset"."""
    frontier = pd.Timestamp("2026-07-14 00:00", tz="Europe/Berlin")  # 2026-07-13 22:00 UTC
    sim = _tick_sim(frontier=frontier)
    sim._on_tick(pd.Timestamp("2026-07-14 12:00", tz="UTC"))

    assert sim._spans, "expected at least one span read"
    for start, end in sim._spans:
        assert start.utcoffset() == end.utcoffset() == pd.Timedelta(0)


def test_on_tick_advances_only_to_the_logged_data_frontier():
    """Weather ends before the cutoff: the chain runs on what exists, no filling."""
    sim = _tick_sim(frontier=pd.Timestamp("2026-07-12 10:00", tz="UTC"))
    index = pd.date_range("2026-07-12 10:15", periods=3, freq="15min", tz="UTC")  # ends 10:45 < 12:00
    weather = pd.DataFrame({"ghi": [100.0, 200.0, 300.0]}, index=index)
    sim._read_weather_span = lambda start, end: weather
    sim._run_chain = lambda frame: (frame, {})
    advanced = []
    sim.soil_simulation.simulate_loop = lambda et, seg, cancel=None: advanced.append(et.index[-1])

    sim._on_tick(pd.Timestamp("2026-07-12 12:00", tz="UTC"))
    assert advanced == [pd.Timestamp("2026-07-12 10:45", tz="UTC")]


def test_on_tick_cold_start_uses_single_advance():
    sim = _tick_sim(frontier=None)
    index = pd.date_range("2026-07-12 11:15", periods=3, freq="15min", tz="UTC")
    weather = pd.DataFrame({"ghi": [100.0, 200.0, 300.0]}, index=index)
    sim._read_weather_span = lambda start, end: weather
    sim._run_chain = lambda frame: (frame, {})
    calls = []
    sim.soil_simulation.advance = lambda et, now, seg, cancel=None: calls.append(("advance", now))
    sim.soil_simulation.simulate_loop = lambda et, seg, cancel=None: calls.append(("loop", et.index[-1]))

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

    def _advance_frontier(et, seg, cancel=None):
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


def test_on_tick_threads_live_interrupt_callable_into_soil_calls():
    """B8: _on_tick must hand advance/simulate_loop the LIVE bound method
    _tick_interrupt.is_set (not its called-once bool, not None) so a shutdown
    signalled mid-walk is observed by walk_window's cancel checks."""
    sim = _tick_sim(frontier=pd.Timestamp("2026-07-12 10:00", tz="UTC"))
    sim._tick_interrupt = threading.Event()
    index = pd.date_range("2026-07-12 10:15", periods=2, freq="15min", tz="UTC")
    weather = pd.DataFrame({"ghi": [100.0, 200.0]}, index=index)
    sim._read_weather_span = lambda start, end: weather
    sim._run_chain = lambda frame: (frame, {})
    received = []
    sim.soil_simulation.simulate_loop = lambda et, seg, cancel=None: received.append(cancel)

    sim._on_tick(pd.Timestamp("2026-07-12 12:00", tz="UTC"))

    # Bound methods have no stable identity (each access creates a new object),
    # so pin the binding (__self__ is the live Event) plus live behavior --
    # exactly what an accidental `.is_set()` (frozen bool) would break.
    assert received
    for c in received:
        assert c.__self__ is sim._tick_interrupt
    assert received[0]() is False  # live callable: reads the event's CURRENT state
    sim._tick_interrupt.set()
    assert received[0]() is True


# --- slot-boundary watchdog + overrun summary --------------------------------


def test_tick_fast_run_leaves_no_watchdog_or_summary_log(caplog):
    """A tick that finishes before the next slot boundary: no watchdog log, no
    overrun summary, and the armed Timer is cancelled rather than leaked."""
    sim = _sim(interval_min=60, offset_min=0)
    sim._tick_lock = threading.Lock()
    sim._now = lambda: pd.Timestamp("2026-07-12 12:00", tz="UTC")
    sim._on_tick = lambda now: None

    with caplog.at_level("WARNING"):
        sim._tick()

    assert sim._watchdog_timer is None  # armed, then cancelled -- not leaked
    assert not any("still running; slot skipped" in m for m in caplog.messages)
    assert not any("overran its slot" in m for m in caplog.messages)


def test_tick_watchdog_fires_at_boundary_and_logs_overrun_summary(caplog):
    """A tick that outlasts its own slot: the watchdog logs once the real,
    ms-scale ``threading.Timer`` fires at the boundary (the fake clock only
    drives the boundary computation, not the Timer's wait), and the post-tick
    summary reports the overrun once the stubbed ``_on_tick`` has advanced the
    fake clock past it."""
    sim = _sim(interval_min=1, offset_min=0)
    sim._tick_lock = threading.Lock()

    clock = types.SimpleNamespace(value=pd.Timestamp("2026-07-12 12:00:59.99", tz="UTC"))
    sim._now = lambda: clock.value

    def _slow_on_tick(now):
        deadline = time.monotonic() + 2.0
        while not any("still running; slot skipped" in m for m in caplog.messages):
            assert time.monotonic() < deadline, "watchdog did not fire before the boundary"
            time.sleep(0.001)
        clock.value = pd.Timestamp("2026-07-12 12:02:05", tz="UTC")  # past 1 slot boundary

    sim._on_tick = _slow_on_tick

    with caplog.at_level("WARNING"):
        sim._tick()

    assert any("still running; slot skipped" in m for m in caplog.messages)
    assert any("overran its slot" in m and "slots_skipped=1" in m for m in caplog.messages)
    # Cancelled on completion; the cancel-vs-re-arm TOCTOU window makes this
    # probabilistic in theory (worst case: an orphaned no-op daemon Timer),
    # empirically stable in repeated runs.
    assert sim._watchdog_timer is None


def test_tick_runs_even_when_watchdog_arming_fails(caplog):
    """A watchdog-arming failure is logged and the tick still runs -- the
    visibility feature must never suppress the simulation itself."""
    sim = _sim(interval_min=60, offset_min=0)
    sim._tick_lock = threading.Lock()
    sim._now = lambda: pd.Timestamp("2026-07-12 12:00", tz="UTC")
    calls = []
    sim._on_tick = calls.append

    def _boom_arm(reference, done):
        raise RuntimeError("no threads left")

    sim._arm_watchdog = _boom_arm

    with caplog.at_level("ERROR"):
        sim._tick()

    assert len(calls) == 1  # the tick ran despite the arming failure
    assert any("failed to arm the slot watchdog" in m for m in caplog.messages)


# --- run-chain shading freshness ---------------------------------------------


def _shading_sim(stale: dict[str, float], fresh: dict[str, float]) -> tuple[FieldSimulation, dict]:
    """A minimal ``_run_chain`` harness: stubbed weather/vegetation passthrough,
    a GroundShading stub returning ``fresh``, and an ET stub capturing the
    segments it was handed. ``stale`` seeds the stored live-tick factors."""
    sim = object.__new__(FieldSimulation)
    sim._name = "test_field_simulation"
    sim._segment_shade = dict(stale)
    sim.bare_lai = 1.0
    sim.bare_roughness = 0.002
    sim.bare_plant_height = 0.1
    sim.bare_ndvi = 0.25
    sim._prepare_weather = lambda weather: weather
    sim._populate_vegetation = lambda df, publish: df

    sim.ground_shading = types.SimpleNamespace(evaluate=lambda df, publish: dict(fresh))
    sim.soil_simulation = types.SimpleNamespace(
        top_segment_names=lambda: ["PlantTopLeftSegment"],
        segment_face_length=lambda name: 1.0,
    )

    captured: dict = {}

    def _et_evaluate(df, segments, publish):
        captured["segments"] = segments
        return df, {}

    sim.evapotranspiration = types.SimpleNamespace(evaluate=_et_evaluate)
    return sim, captured


def _veg_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            FieldSimulation.LAI: [1.0],
            FieldSimulation.PLANT_HEIGHT: [0.5],
            FieldSimulation.NDVI: [0.3],
            FieldSimulation.ROUGHNESS: [0.01],
        },
        index=[pd.Timestamp("2026-07-12 12:00", tz="UTC")],
    )


def test_run_chain_replay_uses_fresh_shading_and_keeps_stored_state():
    sim, captured = _shading_sim(stale={"PlantTopLeftSegment": 0.2}, fresh={"PlantTopLeftSegment": 0.9})

    sim._run_chain(_veg_frame(), publish=False)

    (segment,) = captured["segments"]
    assert segment.shade_factor == pytest.approx(0.9)
    assert sim._segment_shade == {"PlantTopLeftSegment": 0.2}


def test_run_chain_publish_uses_and_stores_fresh_shading():
    sim, captured = _shading_sim(stale={"PlantTopLeftSegment": 0.2}, fresh={"PlantTopLeftSegment": 0.9})

    sim._run_chain(_veg_frame(), publish=True)

    (segment,) = captured["segments"]
    assert segment.shade_factor == pytest.approx(0.9)
    assert sim._segment_shade == {"PlantTopLeftSegment": 0.9}


def test_run_chain_empty_factors_fall_back_to_stored_shading():
    sim, captured = _shading_sim(stale={"PlantTopLeftSegment": 0.2}, fresh={})

    sim._run_chain(_veg_frame(), publish=False)

    (segment,) = captured["segments"]
    assert segment.shade_factor == pytest.approx(0.2)
    assert sim._segment_shade == {"PlantTopLeftSegment": 0.2}


def test_run_chain_without_ground_shading_uses_stored_shading():
    sim, captured = _shading_sim(stale={"PlantTopLeftSegment": 0.2}, fresh={})
    sim.ground_shading = None

    sim._run_chain(_veg_frame(), publish=False)

    (segment,) = captured["segments"]
    assert segment.shade_factor == pytest.approx(0.2)
    assert sim._segment_shade == {"PlantTopLeftSegment": 0.2}
