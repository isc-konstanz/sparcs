# -*- coding: utf-8 -*-
"""
tests.test_fieldsim_runtime
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``FrameInputs``, ``Recorder`` and the ``FieldRunner`` tick policy, driven directly and through
``ScenarioRunner``. Engine, shading and ET are stubs; everything between them is real.
"""

import datetime as dt
import io
import itertools
import logging
from types import SimpleNamespace

import pytest

import numpy as np
import pandas as pd
import pytz
from sparcs.components.agriculture.simulation.core.chain import WeatherChain
from sparcs.components.agriculture.simulation.core.config import FieldConfig, FieldSetup, PlannerConfig, SoilConfig
from sparcs.components.agriculture.simulation.core.evapotranspiration import SegmentProperties
from sparcs.components.agriculture.simulation.core.simulation import Simulation
from sparcs.components.agriculture.simulation.core.state import (
    Forcing,
    Plan,
    SoilState,
    StepResult,
    decode_state_blob,
    encode_state_blob,
)
from sparcs.components.agriculture.simulation.runtime.memory import FrameInputs, Recorder
from sparcs.components.agriculture.simulation.runtime.runner import FieldRunner
from sparcs.components.agriculture.simulation.runtime.scenario import ScenarioRunner

UTC = dt.timezone.utc
PLUS_TWO = dt.timezone(dt.timedelta(hours=2))
BERLIN = "Europe/Berlin"
RUNNER_LOGGER = "sparcs.components.agriculture.simulation.runtime.runner"


class _Probe:
    def __init__(self, channel_id: str) -> None:
        self.channel_id = channel_id


class _Engine:
    cold_start_s = 0.0

    def initial_state(self, at):
        return SoilState(np.full(3, 0.5), np.full(3, 0.5), {}, at)

    def load(self, state):
        if state.se.shape != (3,):
            raise ValueError(f"state carries {state.se.shape} cells but the mesh has (3,)")

    def advance(self, state, forcing, *, cancel=None):
        return StepResult(SoilState(state.se * 0.99, state.se, {}, forcing.end), {"water": float(state.se.sum())})

    def tension_at(self, state, probe):
        return -100.0 / float(state.se[0])


class _Shading:
    def evaluate(self, weather, *, remember=True):
        return pd.DataFrame({"top": 1.0}, index=weather.index)

    def render_inputs_at(self, ts):
        return [], [], (90.0, 0.0, None)

    def envelope(self, mesh_height):
        return None


class _ET:
    def evaluate(self, weather, segments):
        bulk = pd.DataFrame({"et_top": 0.1}, index=weather.index)
        seg_et = {"top": pd.DataFrame({"et": 0.1, "evap": 0.1, "transp": 0.0}, index=weather.index)}
        return bulk, seg_et


class _Chain(WeatherChain):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.flows = []

    def _prepare_weather(self, weather):
        return weather

    def _segments(self, weather, shading):
        return [SegmentProperties("top", lai=1.0)]

    def _forcings(self, weather, shading, seg_et, irrigation_lpm, *, frontier, first_dt_s=0.0):
        self.flows.append(irrigation_lpm)
        cutoff = pd.Timestamp(frontier) if frontier is not None else None
        forcings = []
        previous = cutoff
        for ts in weather.index:
            if cutoff is not None and ts <= cutoff:
                continue
            dt_s = first_dt_s if previous is None else (ts - previous).total_seconds()
            previous = ts
            forcings.append(Forcing(at=ts.to_pydatetime(), dt_s=dt_s))
        return forcings

    def horizon_inputs(self, forecast):
        return forecast, {}


class _NoAssimilation:
    enabled = False


class _Planner:
    def plan(self, state, weather, seg_et, horizon_start, horizon_end, *, run_timestamp, weather_creation=None):
        return Plan(
            chosen=("06:00", 30),
            trajectories={},
            header=pd.DataFrame(),
            detail=pd.DataFrame(),
            irrigation=pd.DataFrame(),
        )


class _RecordingInputs(FrameInputs):
    """``FrameInputs`` that keeps every weather and irrigation window the runner asked for."""

    def __init__(self, state=None, **frames) -> None:
        super().__init__(state=state, **frames)
        self.windows = []

    def weather(self, start, end):
        self.windows.append(("weather", start, end))
        return super().weather(start, end)

    def irrigation(self, start, end):
        self.windows.append(("irrigation", start, end))
        return super().irrigation(start, end)


class _UndecodableInputs(FrameInputs):
    def load_state(self):
        raise ValueError("Object arrays cannot be loaded when allow_pickle=False")


class _TruncatedBlobInputs(FrameInputs):
    """A persisted blob cut short mid-write."""

    def load_state(self):
        at = dt.datetime(2026, 9, 20, 10, tzinfo=UTC)
        return SoilState.from_blob(_state_at(at).to_blob()[:200], at)


class _FlakyRecorder(Recorder):
    """``Recorder`` whose ``steps`` raises for the ``fail_at``-th chunk."""

    def __init__(self, fail_at: int) -> None:
        super().__init__()
        self.fail_at = fail_at
        self.seen = 0

    def steps(self, results):
        self.seen += 1
        if self.seen == self.fail_at:
            raise RuntimeError("sink unavailable")
        super().steps(results)


class _ChainFailingRecorder(Recorder):
    def chain(self, now, result):
        raise RuntimeError("shading sink unavailable")


def _setup(planner=True, intake_delay="0min") -> FieldSetup:
    soil = SoilConfig.from_dict({"mesh": {}})
    return FieldSetup(
        field=FieldConfig.from_dict({"interval": 360, "intake_delay": intake_delay}),
        soil=soil,
        shading=None,
        planner=PlannerConfig.from_dict({"horizon": "6h"}) if planner else None,
    )


def _simulation(setup: FieldSetup) -> Simulation:
    return Simulation(
        setup, _Engine(), _Chain(setup, _Shading(), _ET()), _NoAssimilation(), _Planner(), probes=[_Probe("soil_30cm")]
    )


def _weather(start: str, end: str) -> pd.DataFrame:
    idx = pd.date_range(start, end, freq="1h", inclusive="left", tz=UTC)
    return pd.DataFrame({"ghi": 100.0}, index=idx)


def _state_at(moment: dt.datetime) -> SoilState:
    return SoilState(np.full(3, 0.4), np.full(3, 0.4), {}, moment)


def _runner(setup, frames, outputs, state=None, inputs=None) -> FieldRunner:
    return FieldRunner(setup, _simulation(setup), inputs or FrameInputs(**frames, state=state), outputs)


# --------------------------------------------------------------------------- memory adapters


def test_frame_inputs_slices_left_open_and_handles_missing_keys():
    inputs = FrameInputs(weather=_weather("2026-09-20", "2026-09-21"))
    got = inputs.weather(dt.datetime(2026, 9, 20, 6, tzinfo=UTC), dt.datetime(2026, 9, 20, 9, tzinfo=UTC))
    assert list(got.index.hour) == [7, 8, 9]
    assert inputs.forecast(dt.datetime(2026, 9, 20, tzinfo=UTC), dt.datetime(2026, 9, 21, tzinfo=UTC)).empty
    assert inputs.irrigation(dt.datetime(2026, 9, 20, tzinfo=UTC), dt.datetime(2026, 9, 21, tzinfo=UTC)).empty
    assert inputs.load_state() is None


def test_frame_inputs_forecast_is_inclusive_and_irrigation_reaches_back():
    flow = pd.Series(1.0, index=pd.date_range("2026-09-19 00:00", "2026-09-20 12:00", freq="6h", tz=UTC))
    inputs = FrameInputs(forecast=_weather("2026-09-20", "2026-09-21"), irrigation=flow)
    start, end = dt.datetime(2026, 9, 20, 6, tzinfo=UTC), dt.datetime(2026, 9, 20, 9, tzinfo=UTC)

    assert list(inputs.forecast(start, end).index.hour) == [6, 7, 8, 9]
    assert list(inputs.irrigation(start, end).index) == list(flow.index[2:6])  # (start - 1 day, end]


def test_recorder_to_frames():
    rec = Recorder()
    t0 = dt.datetime(2026, 9, 20, 1, tzinfo=UTC)
    rec.steps(
        [
            StepResult(
                SoilState(np.zeros(1), np.zeros(1), {}, t0 + dt.timedelta(hours=i)),
                {"water": float(i)},
                {"p": -10.0 * i},
            )
            for i in range(3)
        ]
    )
    frames = rec.to_frames()
    assert frames["diagnostics"].shape == (3, 1) and list(frames["diagnostics"]["water"]) == [0.0, 1.0, 2.0]
    assert frames["tension"]["p"].iloc[-1] == -20.0
    assert "shading" not in frames


# --------------------------------------------------------------------------- scenario over the real runner


def test_scenario_runs_catch_up_in_day_chunks_and_plans_once_per_slot():
    setup = _setup()
    sim = _simulation(setup)
    frames = {
        "weather": _weather("2026-09-20", "2026-09-23"),
        "forecast": _weather("2026-09-23", "2026-09-23 06:00"),
    }
    rec = ScenarioRunner(setup, sim).run(
        frames, start=dt.datetime(2026, 9, 20, 6, tzinfo=UTC), end=dt.datetime(2026, 9, 23, 0, tzinfo=UTC)
    )
    # 72 hourly rows; the row at the very start of the first window falls outside
    # (start, end] and the next one is the cold-start anchor.
    assert len(rec.rows) == 70
    assert len(rec.chains) == 12  # 6-hourly ticks, one chunk each
    # Only the last two ticks' horizons reach the forecast; both frontiers (18:00,
    # 23:00) sit in the daily slot opened at 01:00, so the second one does not plan.
    assert len(rec.plans) == 1
    assert len(rec.saved) == 12  # one state per chunk
    snap = sim.snapshot()
    assert snap.frontier == dt.datetime(2026, 9, 22, 23, tzinfo=UTC)
    assert snap.last_plan is not None and snap.last_plan.chosen == ("06:00", 30)
    assert set(snap.last_step.probe_tension) == {"soil_30cm"}


def test_repeated_tick_at_same_instant_is_a_noop():
    setup = _setup(planner=False)
    runner = _runner(setup, {"weather": _weather("2026-09-20", "2026-09-21")}, Recorder())
    now = dt.datetime(2026, 9, 20, 12, tzinfo=UTC)
    assert runner.run_tick(now) is True
    assert runner.run_tick(now) is False


def test_resume_from_persisted_state_skips_history():
    setup = _setup(planner=False)
    rec = Recorder()
    runner = _runner(
        setup,
        {"weather": _weather("2026-09-20", "2026-09-21")},
        rec,
        state=_state_at(dt.datetime(2026, 9, 20, 10, tzinfo=UTC)),
    )
    assert runner.run_tick(dt.datetime(2026, 9, 20, 12, tzinfo=UTC)) is True
    assert [s.state.at.hour for s in rec.rows] == [11, 12]


def test_frontier_in_another_zone_is_aligned_to_the_cutoff():
    setup = _setup(planner=False)
    inputs = _RecordingInputs(
        weather=_weather("2026-09-20", "2026-09-21"),
        state=_state_at(dt.datetime(2026, 9, 20, 12, tzinfo=PLUS_TWO)),
    )
    rec = Recorder()
    runner = _runner(setup, None, rec, inputs=inputs)

    assert runner.run_tick(dt.datetime(2026, 9, 20, 12, tzinfo=UTC)) is True
    assert [s.state.at.hour for s in rec.rows] == [11, 12]
    assert all(start.utcoffset() == dt.timedelta(0) for _, start, _ in inputs.windows)
    assert all(end.utcoffset() == dt.timedelta(0) for _, _, end in inputs.windows)


def test_cancel_stops_after_the_current_chunk():
    setup = _setup(planner=False)
    rec = Recorder()
    runner = _runner(setup, {"weather": _weather("2026-09-18", "2026-09-21")}, rec)
    calls = {"n": 0}

    def cancel() -> bool:
        calls["n"] += 1
        return True  # cancel as soon as the runner asks

    assert runner.run_tick(dt.datetime(2026, 9, 20, 12, tzinfo=UTC), cancel=cancel) is True
    assert 0 < len(rec.rows) < 60  # one chunk committed, the rest left for the next tick


def test_a_failing_steps_write_does_not_stop_the_frontier():
    setup = _setup(planner=False)
    rec = _FlakyRecorder(fail_at=1)
    frontier = dt.datetime(2026, 9, 19, 20, tzinfo=UTC)
    runner = _runner(setup, {"weather": _weather("2026-09-19", "2026-09-21")}, rec, state=_state_at(frontier))

    assert runner.run_tick(dt.datetime(2026, 9, 20, 3, tzinfo=UTC)) is True
    assert [s.state.at.hour for s in rec.rows] == [1, 2, 3]  # the first chunk's steps write raised
    assert [s.at for s in rec.saved] == [
        dt.datetime(2026, 9, 20, 0, tzinfo=UTC),
        dt.datetime(2026, 9, 20, 3, tzinfo=UTC),
    ]
    assert runner.simulation.snapshot().frontier == dt.datetime(2026, 9, 20, 3, tzinfo=UTC)


def test_a_failing_chain_write_does_not_stop_the_rows():
    setup = _setup(planner=False)
    rec = _ChainFailingRecorder()
    runner = _runner(setup, {"weather": _weather("2026-09-20", "2026-09-21")}, rec)

    assert runner.run_tick(dt.datetime(2026, 9, 20, 12, tzinfo=UTC)) is True
    assert [s.state.at.hour for s in rec.rows] == [8, 9, 10, 11, 12]
    assert [s.at.hour for s in rec.saved] == [12]


def test_restore_is_applied_on_the_next_tick():
    setup = _setup(planner=False)
    rec = Recorder()
    runner = _runner(setup, {"weather": _weather("2026-09-20", "2026-09-21")}, rec)
    runner.run_tick(dt.datetime(2026, 9, 20, 12, tzinfo=UTC))
    runner.restore(_state_at(dt.datetime(2026, 9, 20, 20, tzinfo=UTC)))
    rec.rows.clear()

    assert runner.run_tick(dt.datetime(2026, 9, 20, 22, tzinfo=UTC)) is True
    assert [s.state.at.hour for s in rec.rows] == [21, 22]


@pytest.mark.parametrize("hour", [8, 12], ids=["older", "at-the-frontier"])
def test_restore_not_newer_than_the_frontier_is_dropped(caplog, hour):
    setup = _setup(planner=False)
    rec = Recorder()
    runner = _runner(setup, {"weather": _weather("2026-09-20", "2026-09-21")}, rec)
    runner.run_tick(dt.datetime(2026, 9, 20, 12, tzinfo=UTC))
    runner.restore(_state_at(dt.datetime(2026, 9, 20, hour, tzinfo=UTC)))
    rec.rows.clear()

    with caplog.at_level(logging.INFO, logger=RUNNER_LOGGER):
        assert runner.run_tick(dt.datetime(2026, 9, 20, 13, tzinfo=UTC)) is True
    assert [s.state.at.hour for s in rec.rows] == [13]
    assert sum("not newer than the frontier" in r.message for r in caplog.records) == 1


def test_only_the_newest_queued_restore_is_applied():
    """Two blobs arrive between ticks (the listener fires per read); the tick
    resumes from the newest, never replays the superseded one."""
    setup = _setup(planner=False)
    rec = Recorder()
    runner = _runner(setup, {"weather": _weather("2026-09-20", "2026-09-21")}, rec)
    runner.run_tick(dt.datetime(2026, 9, 20, 12, tzinfo=UTC))
    runner.restore(_state_at(dt.datetime(2026, 9, 20, 16, tzinfo=UTC)))
    runner.restore(_state_at(dt.datetime(2026, 9, 20, 20, tzinfo=UTC)))
    rec.rows.clear()

    assert runner.run_tick(dt.datetime(2026, 9, 20, 22, tzinfo=UTC)) is True
    assert [s.state.at.hour for s in rec.rows] == [21, 22]


def test_intake_delay_holds_the_frontier_back():
    """Every tick advances only to ``now - intake_delay``, so a source that
    back-fills its latest rows is never overtaken."""
    setup = _setup(planner=False, intake_delay="30min")
    rec = Recorder()
    runner = _runner(setup, {"weather": _weather("2026-09-20", "2026-09-21")}, rec)

    assert runner.run_tick(dt.datetime(2026, 9, 20, 12, tzinfo=UTC)) is True
    assert rec.rows[-1].state.at == dt.datetime(2026, 9, 20, 11, tzinfo=UTC)


def test_a_noop_tick_does_not_count_as_a_weather_stall():
    """The frontier has caught up to the cutoff: no chunk is read, so the
    stall tally must stay where it was."""
    setup = _setup(planner=False)
    runner = _runner(setup, {"weather": _weather("2026-09-20", "2026-09-21")}, Recorder())
    now = dt.datetime(2026, 9, 20, 12, tzinfo=UTC)

    assert runner.run_tick(now) is True
    assert runner.run_tick(now) is False
    assert runner.weather_stall_ticks == 0.0


def test_every_stalled_tick_warns_and_the_crossing_errors_once(caplog):
    """A dead weather feed from a cold start: one WARNING per stalled tick and a
    single ERROR at the crossing, never a raise."""
    setup = _setup(planner=False)
    runner = _runner(setup, {}, Recorder())

    with caplog.at_level(logging.WARNING, logger=RUNNER_LOGGER):
        for hour in (10, 11, 12, 13):
            assert runner.run_tick(dt.datetime(2026, 9, 20, hour, tzinfo=UTC)) is False

    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 4
    assert len([r for r in caplog.records if r.levelno == logging.ERROR]) == 1
    assert runner.weather_stall_ticks == 4.0


def test_stall_and_failure_tallies_reach_the_diagnostics():
    setup = _setup(planner=False)
    rec = Recorder()
    runner = _runner(setup, {"weather": _weather("2026-09-20 12:00", "2026-09-20 18:00")}, rec)
    runner.tick_failures = 2.0

    assert runner.run_tick(dt.datetime(2026, 9, 20, 6, tzinfo=UTC)) is False
    assert runner.weather_stall_ticks == 1.0

    assert runner.run_tick(dt.datetime(2026, 9, 20, 18, tzinfo=UTC)) is True
    assert runner.weather_stall_ticks == 0.0
    assert rec.rows and all(s.diagnostics["weather_stall"] == 1.0 for s in rec.rows)
    assert all(s.diagnostics["tick_failures"] == 2.0 for s in rec.rows)


def test_planner_warning_is_latched_once_per_slot(caplog):
    setup = _setup()
    runner = _runner(setup, {"weather": _weather("2026-09-20", "2026-09-21")}, Recorder())
    with caplog.at_level(logging.WARNING, logger=RUNNER_LOGGER):
        for hour in (6, 12, 18):
            runner.run_tick(dt.datetime(2026, 9, 20, hour, tzinfo=UTC))
    assert sum("planner skipped" in r.message for r in caplog.records) == 1


def test_planner_plans_once_per_interval_offset_slot():
    """``interval = 30, offset = 20`` opens a slot at :20 and :50; a ten-minute
    tick plans on entering each, twice within the hour."""
    setup = FieldSetup(
        field=FieldConfig.from_dict({"interval": 10, "intake_delay": "0min"}),
        soil=SoilConfig.from_dict({"mesh": {}}),
        shading=None,
        planner=PlannerConfig.from_dict({"horizon": "1h", "interval": 30, "offset": 20}),
    )
    weather = pd.DataFrame(
        {"ghi": 100.0}, index=pd.date_range("2026-09-20 09:00", "2026-09-20 12:00", freq="10min", tz=UTC)
    )
    rec = Recorder()
    runner = _runner(
        setup,
        {"weather": weather, "forecast": _weather("2026-09-20 10:00", "2026-09-20 13:00")},
        rec,
        state=_state_at(dt.datetime(2026, 9, 20, 10, tzinfo=UTC)),
    )

    for minute in (20, 30, 40, 50, 60, 70):
        runner.run_tick(dt.datetime(2026, 9, 20, 10, tzinfo=UTC) + dt.timedelta(minutes=minute))

    assert len(rec.plans) == 2


def test_planner_slot_is_aligned_in_the_site_timezone():
    """The daily slot (interval 1440, offset 60) opens at 01:00 site time: in
    Berlin summer time that is 23:00 UTC, so a frontier at 23:00 UTC plans again."""
    setup = FieldSetup(
        field=FieldConfig.from_dict({"interval": 60, "intake_delay": "0min"}),
        soil=SoilConfig.from_dict({"mesh": {}}),
        shading=None,
        planner=PlannerConfig.from_dict({"horizon": "6h"}),
        location=SimpleNamespace(timezone=pytz.timezone(BERLIN)),
    )
    rec = Recorder()
    runner = _runner(
        setup,
        {"weather": _weather("2026-09-20", "2026-09-22"), "forecast": _weather("2026-09-20", "2026-09-22")},
        rec,
        state=_state_at(dt.datetime(2026, 9, 20, 20, tzinfo=UTC)),
    )

    for hour in (22, 23):
        runner.run_tick(dt.datetime(2026, 9, 20, hour, tzinfo=UTC))

    assert len(rec.plans) == 2


@pytest.mark.parametrize("hours", [1, 25, 49])
def test_day_chunks_cover_span_without_gaps(hours):
    start = dt.datetime(2026, 9, 20, 18, tzinfo=UTC)
    end = start + dt.timedelta(hours=hours)
    chunks = list(FieldRunner._day_chunks(start, end))
    assert chunks[0][0] == pd.Timestamp(start) and chunks[-1][1] == pd.Timestamp(end)
    assert all(a[1] == b[0] for a, b in zip(chunks, chunks[1:]))
    assert all((b - a) <= dt.timedelta(days=1) for a, b in chunks)


def test_day_chunks_break_at_midnight():
    chunks = list(
        FieldRunner._day_chunks(dt.datetime(2026, 9, 20, 18, tzinfo=UTC), dt.datetime(2026, 9, 22, 6, tzinfo=UTC))
    )
    assert [c[1] for c in chunks] == [
        pd.Timestamp("2026-09-21", tz=UTC),
        pd.Timestamp("2026-09-22", tz=UTC),
        pd.Timestamp("2026-09-22 06:00", tz=UTC),
    ]


def test_day_chunks_cross_the_autumn_dst_change():
    """2026-10-25 has 25 hours in Berlin: a chunk boundary one day after local
    midnight must land on the next local midnight, not on the same one again."""
    start = pd.Timestamp("2026-10-24 18:00", tz=BERLIN)
    end = pd.Timestamp("2026-10-26 06:00", tz=BERLIN)

    chunks = list(itertools.islice(FieldRunner._day_chunks(start, end), 10))

    assert [c[1] for c in chunks] == [pd.Timestamp("2026-10-25", tz=BERLIN), pd.Timestamp("2026-10-26", tz=BERLIN), end]
    assert chunks[0][0] == start
    assert all(a[1] == b[0] for a, b in zip(chunks, chunks[1:]))


def test_cancel_is_honoured_while_the_chunks_are_empty():
    setup = _setup(planner=False)
    inputs = _RecordingInputs(state=_state_at(dt.datetime(2026, 9, 17, 12, tzinfo=UTC)))
    runner = _runner(setup, None, Recorder(), inputs=inputs)

    assert runner.run_tick(dt.datetime(2026, 9, 20, 12, tzinfo=UTC), cancel=lambda: True) is False
    assert [key for key, _, _ in inputs.windows] == ["weather"]  # three more empty chunks left unread


def test_irrigation_samples_are_aligned_onto_the_weather_rows():
    """The sample in force before the chunk start carries into it; each row gets
    the last sample at or before it."""
    setup = _setup(planner=False)
    flow = pd.Series([2.0, 0.0], index=pd.DatetimeIndex(["2026-09-20 05:30", "2026-09-20 08:30"], tz=UTC))
    runner = _runner(
        setup,
        {"weather": _weather("2026-09-20", "2026-09-21"), "irrigation": flow},
        Recorder(),
        state=_state_at(dt.datetime(2026, 9, 20, 6, tzinfo=UTC)),
    )

    runner.run_tick(dt.datetime(2026, 9, 20, 9, tzinfo=UTC))

    assert list(runner.simulation.chain.flows[-1]) == [2.0, 2.0, 0.0]


@pytest.mark.parametrize("cause", ["cell count", "undecodable", "truncated"])
def test_an_incompatible_persisted_state_cold_starts_with_a_warning(caplog, cause):
    setup = _setup(planner=False)
    rec = Recorder()
    weather = _weather("2026-09-20", "2026-09-21")
    if cause == "cell count":
        stale = SoilState(np.full(5, 0.4), np.full(5, 0.4), {}, dt.datetime(2026, 9, 20, 10, tzinfo=UTC))
        inputs = FrameInputs(weather=weather, state=stale)
    elif cause == "undecodable":
        inputs = _UndecodableInputs(weather=weather)
    else:
        inputs = _TruncatedBlobInputs(weather=weather)
    runner = _runner(setup, None, rec, inputs=inputs)

    with caplog.at_level(logging.WARNING, logger=RUNNER_LOGGER):
        assert runner.run_tick(dt.datetime(2026, 9, 20, 12, tzinfo=UTC)) is True

    assert sum("incompatible" in r.message for r in caplog.records) == 1
    assert [s.state.at.hour for s in rec.rows] == [8, 9, 10, 11, 12]  # cold start anchored at 07:00


def test_an_incompatible_restore_is_dropped_with_a_warning(caplog):
    setup = _setup(planner=False)
    rec = Recorder()
    runner = _runner(setup, {"weather": _weather("2026-09-20", "2026-09-21")}, rec)
    runner.run_tick(dt.datetime(2026, 9, 20, 12, tzinfo=UTC))
    runner.restore(SoilState(np.full(5, 0.4), np.full(5, 0.4), {}, dt.datetime(2026, 9, 20, 20, tzinfo=UTC)))
    rec.rows.clear()

    with caplog.at_level(logging.WARNING, logger=RUNNER_LOGGER):
        assert runner.run_tick(dt.datetime(2026, 9, 20, 14, tzinfo=UTC)) is True

    assert sum("incompatible" in r.message for r in caplog.records) == 1
    assert [s.state.at.hour for s in rec.rows] == [13, 14]


def _blob_without_rel_sat() -> bytes:
    buf = io.BytesIO()
    np.savez(buf, rel_sat_old=np.full(3, 0.4))
    return buf.getvalue()


def _single_array_npy() -> bytes:
    buf = io.BytesIO()
    np.save(buf, np.full(3, 0.4))
    return buf.getvalue()


@pytest.mark.parametrize(
    "blob",
    [
        pytest.param(encode_state_blob(np.full(3, 0.4), np.full(3, 0.4), {})[:200], id="truncated"),
        pytest.param(_blob_without_rel_sat(), id="no rel_sat"),
        pytest.param(_single_array_npy(), id="npy not npz"),
    ],
)
def test_decode_state_blob_raises_value_error_for_an_undecodable_blob(blob):
    with pytest.raises(ValueError):
        decode_state_blob(blob)


def test_an_empty_chunk_inside_the_span_does_not_stop_the_tick():
    setup = _setup(planner=False)
    rec = Recorder()
    weather = pd.concat([_weather("2026-09-18", "2026-09-19"), _weather("2026-09-20 06:00", "2026-09-21")])
    runner = _runner(setup, {"weather": weather}, rec)
    state = SoilState(np.full(3, 0.4), np.full(3, 0.4), {}, dt.datetime(2026, 9, 18, 12, tzinfo=UTC))
    runner.restore(state)

    assert runner.run_tick(dt.datetime(2026, 9, 20, 12, tzinfo=UTC)) is True
    assert runner.weather_stall_ticks == 0.0
    assert rec.rows[-1].state.at == dt.datetime(2026, 9, 20, 12, tzinfo=UTC)
    assert not any(
        dt.datetime(2026, 9, 19, 0, tzinfo=UTC) < s.state.at < dt.datetime(2026, 9, 20, 6, tzinfo=UTC) for s in rec.rows
    )
