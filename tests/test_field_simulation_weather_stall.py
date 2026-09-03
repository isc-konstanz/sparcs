# -*- coding: utf-8 -*-
"""sparcs.tests.test_field_simulation_weather_stall
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Issue 19 (W2.1): an empty/invalid weather chunk used to fall through
``_on_tick``'s ``continue`` (base.py:635-638) with zero operator-visible
signal -- a dead weather connector stalls the frontier forever and nothing
logs it. ``_on_tick`` now logs a WARNING once per fully-stalled tick (>=1
chunk read, every one empty/invalid) naming the window and frontier, with the
missing-column detail folded in when invalid frames (not just empty ones)
were among the causes, and escalates to a single ERROR at the Nth
(``_WEATHER_STALL_ERROR_TICKS``) consecutive stall. The continue/self-heal
semantics are unchanged.

Per Q6 the stall tally is SIM-side and persists via the existing
diagnostics -> ``agri_field_simulation`` path: ``SoilSimulation
._record_diagnostics`` never runs during a stall (``advance()``/
``simulate_loop()`` are the only commit path), so ``_on_tick`` mirrors the
PRE-tick tally onto ``soil_simulation._weather_stall_ticks`` (plain setattr)
before any chunk in that tick can commit a row; whatever row a later,
healing tick commits then carries the count that preceded it.

Instances mirror ``test_field_simulation_tick.py``'s ``_tick_sim`` fixture:
``object.__new__`` with a ``SimpleNamespace`` soil child (no ``self.data``),
so the mirror write must be a plain attribute set. The ``_record_diagnostics``
tests mirror ``test_soil_skipped_s_diagnostics.py``'s bare-instance +
recording-``data`` precedent.
"""

import logging
import types

import pytest

import pandas as pd

_base = pytest.importorskip("sparcs.components.agriculture.simulation.base")
FieldSimulation = _base.FieldSimulation
_WEATHER_STALL_ERROR_TICKS = _base._WEATHER_STALL_ERROR_TICKS

soil = pytest.importorskip("sparcs.components.agriculture.simulation.soil")
SoilSimulation = soil.SoilSimulation

_soil_core = pytest.importorskip("sparcs.components.agriculture.simulation._soil")
FluxRates = _soil_core.FluxRates
ClipDiagnostics = _soil_core.ClipDiagnostics


# --- FieldSimulation._on_tick: stall detection, WARNING/ERROR, mirror -------


def _sim(interval_min: int = 60, offset_min: int = 0) -> FieldSimulation:
    sim = object.__new__(FieldSimulation)
    sim._name = "test_field_simulation"
    sim.location = None
    sim._interval_min = interval_min
    sim._offset_min = offset_min
    return sim


def _tick_sim(frontier, intake_delay=pd.Timedelta(0), interval_min: int = 60, required_weather_keys=()):
    """Mirrors test_field_simulation_tick.py's ``_tick_sim``: a sim wired for
    ``_on_tick`` with a stub soil child and recording span reads. ``soil_simulation``
    is a bare ``SimpleNamespace`` -- no ``self.data`` -- so the stall-counter
    mirror write must be a plain setattr."""
    sim = _sim(interval_min=interval_min)
    sim._intake_delay = intake_delay
    sim._required_weather_keys = required_weather_keys
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


def test_all_empty_tick_warns_once_and_increments_counter(caplog):
    sim = _tick_sim(frontier=pd.Timestamp("2026-07-12 10:00", tz="UTC"))

    with caplog.at_level(logging.WARNING):
        sim._on_tick(pd.Timestamp("2026-07-12 12:00", tz="UTC"))

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "still running; slot skipped" not in warnings[0].getMessage()
    assert "overran its slot" not in warnings[0].getMessage()
    assert sim._weather_stall_ticks == 1
    assert sim.soil_simulation._weather_stall_ticks == 0.0  # mirrors the tally from BEFORE this tick


def test_partial_tick_with_one_good_chunk_does_not_warn_and_resets_counter(caplog):
    sim = _tick_sim(frontier=pd.Timestamp("2026-07-10 06:00", tz="UTC"))  # 3-day backlog -> 3 day-chunks
    sim._weather_stall_ticks = 2  # two prior stalled ticks

    index = pd.date_range("2026-07-12 00:15", periods=2, freq="15min", tz="UTC")
    good_weather = pd.DataFrame({"ghi": [1.0, 2.0]}, index=index)
    reads = {"n": 0}

    def _partial_read(start, end):
        reads["n"] += 1
        return good_weather if reads["n"] == 3 else pd.DataFrame()  # only the last day-chunk has data

    sim._read_weather_span = _partial_read
    sim._run_chain = lambda frame: (frame, {})

    with caplog.at_level(logging.WARNING):
        sim._on_tick(pd.Timestamp("2026-07-12 12:00", tz="UTC"))

    assert not [r for r in caplog.records if r.levelno == logging.WARNING]
    assert sim._weather_stall_ticks == 0


def test_pre_chunk_early_return_does_not_touch_the_stall_counter(caplog):
    sim = _tick_sim(frontier=pd.Timestamp("2026-07-12 10:00", tz="UTC"))
    with caplog.at_level(logging.WARNING):
        sim._on_tick(pd.Timestamp("2026-07-12 12:00", tz="UTC"))  # stalls: counter -> 1
    assert sim._weather_stall_ticks == 1
    caplog.clear()

    # Frontier has caught up to cutoff: pre-chunk early return, no chunk is read.
    sim.soil_simulation._last_simulated_at = pd.Timestamp("2026-07-12 12:00", tz="UTC")
    with caplog.at_level(logging.WARNING):
        sim._on_tick(pd.Timestamp("2026-07-12 12:00", tz="UTC"))

    assert sim._spans == [(pd.Timestamp("2026-07-12 10:00", tz="UTC"), pd.Timestamp("2026-07-12 12:00", tz="UTC"))]
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]
    assert sim._weather_stall_ticks == 1  # unchanged by the no-op tick


def test_frontier_none_stall_does_not_raise(caplog):
    sim = _tick_sim(frontier=None)

    with caplog.at_level(logging.WARNING):
        sim._on_tick(pd.Timestamp("2026-07-12 12:00", tz="UTC"))  # must not raise

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "none" in warnings[0].getMessage().lower()
    assert sim._weather_stall_ticks == 1


def test_error_logs_exactly_once_at_the_stall_crossing(caplog):
    sim = _tick_sim(frontier=pd.Timestamp("2026-07-12 09:00", tz="UTC"))

    with caplog.at_level(logging.WARNING):
        for hour in (10, 11, 12, 13):
            sim._on_tick(pd.Timestamp(f"2026-07-12 {hour}:00", tz="UTC"))

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(warnings) == 4  # one per stalled tick, including after the crossing
    assert len(errors) == 1  # only at the n == N crossing, not every tick after
    assert sim._weather_stall_ticks == 4


def test_invalid_frame_missing_column_detail_surfaces_at_warning(caplog):
    sim = _tick_sim(
        frontier=pd.Timestamp("2026-07-12 10:00", tz="UTC"),
        required_weather_keys=("ghi", "temp_air"),
    )
    index = pd.date_range("2026-07-12 10:15", periods=2, freq="15min", tz="UTC")
    weather = pd.DataFrame({"ghi": [1.0, 2.0]}, index=index)  # temp_air missing
    sim._read_weather_span = lambda start, end: weather

    with caplog.at_level(logging.WARNING):
        sim._on_tick(pd.Timestamp("2026-07-12 12:00", tz="UTC"))

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "temp_air" in warnings[0].getMessage()
    assert sim._weather_stall_ticks == 1


def test_mirror_carries_preceding_count_into_next_processed_tick():
    """After N stalled ticks, the tick that finally processes a chunk mirrors
    the PRECEDING (not post-reset) tally onto soil_simulation -- the row that
    tick commits (via _record_diagnostics, which never runs during a stall
    itself) is the one that must carry it."""
    sim = _tick_sim(frontier=pd.Timestamp("2026-07-12 09:00", tz="UTC"))

    sim._on_tick(pd.Timestamp("2026-07-12 10:00", tz="UTC"))  # stall #1
    sim._on_tick(pd.Timestamp("2026-07-12 11:00", tz="UTC"))  # stall #2
    assert sim._weather_stall_ticks == 2

    index = pd.date_range("2026-07-12 11:15", periods=2, freq="15min", tz="UTC")
    weather = pd.DataFrame({"ghi": [1.0, 2.0]}, index=index)
    sim._read_weather_span = lambda start, end: weather
    sim._run_chain = lambda frame: (frame, {})

    sim._on_tick(pd.Timestamp("2026-07-12 12:00", tz="UTC"))  # heals

    assert sim.soil_simulation._weather_stall_ticks == 2.0  # tally that PRECEDED the healing tick
    assert sim._weather_stall_ticks == 0  # reset after processing >=1 chunk


# --- SoilSimulation._record_diagnostics: merges the weather_stall channel ---


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


def test_record_diagnostics_merges_weather_stall_key_default_zero(monkeypatch):
    sim, fake = _bare_soil(monkeypatch)
    now = pd.Timestamp("2026-07-16 10:00", tz="UTC")
    rates = FluxRates(seg_evap={}, seg_transp={}, flow_m3s=0.0, rain_flux=0.0)
    clip = ClipDiagnostics()

    diagnostics = sim._record_diagnostics(rates, now, 0.0, 600.0, clip, skipped_s=0.0)

    assert fake["weather_stall"].calls == [(now, 0.0)]
    assert diagnostics["weather_stall"] == 0.0


def test_record_diagnostics_merges_mirrored_weather_stall_count(monkeypatch):
    sim, fake = _bare_soil(monkeypatch)
    sim._weather_stall_ticks = 3.0  # mirrored down by FieldSimulation._on_tick
    now = pd.Timestamp("2026-07-16 10:00", tz="UTC")
    rates = FluxRates(seg_evap={}, seg_transp={}, flow_m3s=0.0, rain_flux=0.0)
    clip = ClipDiagnostics()

    diagnostics = sim._record_diagnostics(rates, now, 0.0, 600.0, clip, skipped_s=0.0)

    assert fake["weather_stall"].calls == [(now, 3.0)]
    assert diagnostics["weather_stall"] == 3.0
