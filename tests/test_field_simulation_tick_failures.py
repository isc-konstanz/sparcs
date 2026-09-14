# -*- coding: utf-8 -*-
"""sparcs.tests.test_field_simulation_tick_failures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Issue 20 (W2.2): FieldSimulation._tick's catch-all used to treat a one-off
transient and a deterministic every-tick crash identically -- both landed as
an identical ``logger.exception("... tick failed.")`` with no way to tell
them apart (the brightsky mixed-tz bug surfaced this way as a daily logged
exception nobody saw). ``_tick`` now counts consecutive failures
(``_tick_failures``, reset to 0 the moment ``_on_tick`` returns without
raising) and logs a distinct ERROR from the 2nd consecutive failure on, in
addition to the unchanged exception log. The escalation must never itself
escape ``_tick`` (hard pin, shared with test_field_simulation_tick.py::
test_tick_releases_lock_after_failure): the lock must always be released.

Per Q6 (same recipe as issue 19/W2.1) the count is mirrored onto
``soil_simulation._tick_failures`` next to the weather-stall mirror near the
top of ``_on_tick`` (the value PRECEDING this tick, since the reset happens
after ``_on_tick`` returns) and persisted via
``SoilSimulation._record_diagnostics`` -- a failed tick commits no row
itself, so only the NEXT successful tick's row carries the count. A
shutdown-cancelled walk returns normally (B8), so it counts as success.

Instances mirror test_field_simulation_tick.py's ``_sim`` fixture
(``object.__new__``, no ``self.data``, ``soil_simulation`` = class-default
``None``). The ``_record_diagnostics`` tests mirror
test_field_simulation_weather_stall.py's ``_bare_soil``/``_RecordingData``
precedent.
"""

import logging
import threading
import types

import pytest

import pandas as pd

_base = pytest.importorskip("sparcs.components.agriculture.simulation.base")
FieldSimulation = _base.FieldSimulation

soil = pytest.importorskip("sparcs.components.agriculture.simulation.soil")
SoilSimulation = soil.SoilSimulation

_soil_core = pytest.importorskip("sparcs.components.agriculture.simulation._soil")
FluxRates = _soil_core.FluxRates
ClipDiagnostics = _soil_core.ClipDiagnostics


def _sim(interval_min: int = 60, offset_min: int = 0) -> FieldSimulation:
    """Mirrors test_field_simulation_tick.py's ``_sim`` helper plus the two
    extra attributes every test here needs (``_now``, ``_tick_lock``)."""
    sim = object.__new__(FieldSimulation)
    sim._name = "test_field_simulation"
    sim.location = None
    sim._interval_min = interval_min
    sim._offset_min = offset_min
    sim._now = lambda: pd.Timestamp("2026-07-12 12:00", tz="UTC")
    sim._tick_lock = threading.Lock()
    return sim


# --- _tick: consecutive-failure counter + escalation -------------------------


def test_two_consecutive_failures_log_distinct_error_not_one(caplog):
    """The 2nd consecutive failure escalates with a distinct ERROR (in
    addition to the unchanged `tick failed` exception log); the 1st failure
    alone does not -- matches test_tick_releases_lock_after_failure's
    single-failure, no-escalation pin."""
    sim = _sim()

    def _boom(now):
        raise RuntimeError("chain failed")

    sim._on_tick = _boom

    with caplog.at_level(logging.ERROR):
        sim._tick()  # failure #1: no escalation yet
    assert not [r for r in caplog.records if "consecutive times" in r.getMessage()]
    assert sim._tick_failures == 1

    with caplog.at_level(logging.ERROR):
        sim._tick()  # failure #2: escalates
    escalations = [r for r in caplog.records if "consecutive times" in r.getMessage()]
    assert len(escalations) == 1
    assert escalations[0].levelno == logging.ERROR
    message = escalations[0].getMessage()
    assert "still running; slot skipped" not in message
    assert "overran its slot" not in message
    assert sim._tick_failures == 2

    with caplog.at_level(logging.ERROR):
        sim._tick()  # failure #3: keeps escalating (>= threshold, not a one-shot ==)
    escalations = [r for r in caplog.records if "consecutive times" in r.getMessage()]
    assert len(escalations) == 2
    assert sim._tick_failures == 3


def test_success_between_failures_resets_the_counter_no_escalation(caplog):
    """failure, success, failure: the intervening success resets the counter
    (contract: reset happens immediately after _on_tick returns, inside the
    try), so the 2nd (non-consecutive) failure does not escalate."""
    sim = _sim()
    calls = {"n": 0}

    def _flaky(now):
        calls["n"] += 1
        if calls["n"] != 2:
            raise RuntimeError("chain failed")

    sim._on_tick = _flaky

    with caplog.at_level(logging.ERROR):
        sim._tick()  # failure #1 -> counter = 1
        sim._tick()  # success -> counter reset to 0
        sim._tick()  # failure -> counter = 1 (not consecutive with #1)

    assert sim._tick_failures == 1
    assert not [r for r in caplog.records if "consecutive times" in r.getMessage()]


def test_lock_skip_and_arming_failure_do_not_increment():
    """Neither the lock-skip early return nor a watchdog-arming failure (inner
    catch) touches the counter -- only the outer except (a genuine _on_tick
    failure) does."""
    sim = _sim()
    sim._on_tick = lambda now: None

    sim._tick_lock.acquire()
    try:
        sim._tick()  # lock-skip: returns before the try block
    finally:
        sim._tick_lock.release()
    assert sim._tick_failures == 0

    def _boom_arm(reference, done):
        raise RuntimeError("no threads left")

    sim._arm_watchdog = _boom_arm
    sim._tick()  # arming fails (inner catch); _on_tick still runs and succeeds
    assert sim._tick_failures == 0


def test_escalation_survives_a_raising_soil_simulation_mirror():
    """The tick-failures mirror this unit adds (``self.soil_simulation.
    _tick_failures = ...`` in _on_tick, next to 2.1's weather-stall mirror)
    can raise -- a channel-backed soil_simulation is not a bare test stub in
    production. Two consecutive raises (so escalation actually fires) must
    still leave _tick non-raising with the lock released (same hard pin as
    test_tick_releases_lock_after_failure, exercised through the escalation
    path)."""

    class _RaisingSoil:
        def __setattr__(self, name, value):
            if name == "_tick_failures":
                raise RuntimeError("soil_simulation is unhappy")
            object.__setattr__(self, name, value)

    sim = _sim()
    sim.soil_simulation = _RaisingSoil()
    sim.evapotranspiration = object()
    sim._weather_channels = object()
    sim._intake_delay = pd.Timedelta(0)

    sim._tick()  # failure #1 (raised from the new mirror's __setattr__)
    sim._tick()  # failure #2: escalation must not escape _tick either

    assert sim._tick_lock.acquire(blocking=False)
    sim._tick_lock.release()
    assert sim._tick_failures == 2


def test_mirror_carries_preceding_failure_count_into_healing_tick():
    """End-to-end through the real _tick/_on_tick pair (mirrors W2.1's
    test_mirror_carries_preceding_count_into_next_processed_tick): two ticks
    fail for real (raising weather read), the third heals -- its _on_tick
    mirrors the PRE-reset count (2.0) onto soil_simulation, and only then
    does _tick's reset zero the live counter."""
    sim = _sim()
    sim._intake_delay = pd.Timedelta(0)
    sim._required_weather_keys = ()
    sim.evapotranspiration = object()
    sim._weather_channels = object()
    sim.soil_simulation = types.SimpleNamespace(
        _last_simulated_at=pd.Timestamp("2026-07-12 09:00", tz="UTC"),
        advance=lambda et, now, seg, cancel=None: None,
        simulate_loop=lambda et, seg, cancel=None: None,
        load_anchor_history=lambda start, end: None,
    )

    def _raise_span(start, end):
        raise RuntimeError("weather source down")

    sim._read_weather_span = _raise_span
    sim._tick()  # failure #1
    sim._tick()  # failure #2
    assert sim._tick_failures == 2

    index = pd.date_range("2026-07-12 11:15", periods=2, freq="15min", tz="UTC")
    sim._read_weather_span = lambda start, end: pd.DataFrame({"ghi": [1.0, 2.0]}, index=index)
    sim._run_chain = lambda frame: (frame, {})

    sim._tick()  # heals

    assert sim.soil_simulation._tick_failures == 2.0  # count that PRECEDED the healing tick
    assert sim._tick_failures == 0


# --- SoilSimulation._record_diagnostics: merges the tick_failures key -------


class _RecordingChannel:
    def __init__(self, channel_id: str):
        self.id = channel_id
        self.calls: list[tuple] = []

    def set(self, timestamp, value) -> None:
        self.calls.append((timestamp, value))


class _RecordingData:
    """``self.data`` stand-in: auto-creates a recording channel per key (these
    tests bypass ``configure()``, so there is no ``add()`` call to seed the map)."""

    def __init__(self):
        self._channels: dict = {}

    def __getitem__(self, key: str) -> _RecordingChannel:
        return self._channels.setdefault(key, _RecordingChannel(key))


def _bare_soil(monkeypatch):
    sim = object.__new__(SoilSimulation)
    sim._name = "test_soil_simulation"
    sim._probes = []  # _sample_probes short-circuits on empty, no _soil_model needed
    sim._pde = types.SimpleNamespace(
        segment_face_len={"WateringTopSegment": 1.0, "GroundBottomSegment": 1.0},
        top_segment_names=[],
        rain_face_len=1.0,
        bottom_drainage_estimate=lambda: 0.0,
    )
    fake = _RecordingData()
    monkeypatch.setattr(SoilSimulation, "data", property(lambda self: fake))
    return sim, fake


def test_record_diagnostics_merges_tick_failures_key_default_zero(monkeypatch):
    sim, fake = _bare_soil(monkeypatch)
    now = pd.Timestamp("2026-07-16 10:00", tz="UTC")
    rates = FluxRates(seg_evap={}, seg_transp={}, flow_m3s=0.0, rain_flux=0.0)
    clip = ClipDiagnostics()

    diagnostics = sim._record_diagnostics(rates, now, 0.0, 600.0, clip, skipped_s=0.0)

    assert fake["tick_failures"].calls == [(now, 0.0)]
    assert diagnostics["tick_failures"] == 0.0


def test_record_diagnostics_merges_mirrored_tick_failures_count(monkeypatch):
    sim, fake = _bare_soil(monkeypatch)
    sim._tick_failures = 2.0  # mirrored down by FieldSimulation._on_tick
    now = pd.Timestamp("2026-07-16 10:00", tz="UTC")
    rates = FluxRates(seg_evap={}, seg_transp={}, flow_m3s=0.0, rain_flux=0.0)
    clip = ClipDiagnostics()

    diagnostics = sim._record_diagnostics(rates, now, 0.0, 600.0, clip, skipped_s=0.0)

    assert fake["tick_failures"].calls == [(now, 2.0)]
    assert diagnostics["tick_failures"] == 2.0
