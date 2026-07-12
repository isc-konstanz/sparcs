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


# --- injected clock ---------------------------------------------------------


def test_replication_cutoff_uses_injected_clock():
    sim = _sim()
    sim._intake_delay = pd.Timedelta(minutes=30)
    frozen = pd.Timestamp("2026-07-12 12:00", tz="UTC")
    sim._now = lambda: frozen
    assert sim._replication_cutoff() == frozen - pd.Timedelta(minutes=30)


def test_replication_cutoff_zero_delay_stays_none():
    sim = _sim()
    sim._intake_delay = pd.Timedelta(0)
    sim._now = lambda: pd.Timestamp("2026-07-12 12:00", tz="UTC")
    assert sim._replication_cutoff() is None


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
