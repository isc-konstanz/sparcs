# -*- coding: utf-8 -*-
"""sparcs.tests.test_fieldsim_runtime
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Runtime layer of the ``fieldsim`` skeleton: ``FrameInputs`` slicing,
``Recorder`` frames, and the live tick policy of ``FieldRunner`` -- frontier
alignment, midnight-aligned chunks, isolated writes, the queued warm-start
restore and the stall/failure tallies -- driven through ``ScenarioRunner``
and directly. The engine, shading and ET are stubs; everything between them
is the real code path.
"""

import datetime as dt
import logging

import pytest

import numpy as np
import pandas as pd
from sparcs.components.agriculture.fieldsim.core.chain import WeatherChain
from sparcs.components.agriculture.fieldsim.core.config import FieldConfig, FieldSetup, PlannerConfig, SoilConfig
from sparcs.components.agriculture.fieldsim.core.evapotranspiration import SegmentProperties
from sparcs.components.agriculture.fieldsim.core.simulation import Simulation
from sparcs.components.agriculture.fieldsim.core.state import Forcing, Plan, SoilState, StepResult
from sparcs.components.agriculture.fieldsim.runtime.memory import FrameInputs, Recorder
from sparcs.components.agriculture.fieldsim.runtime.ports import InputKey
from sparcs.components.agriculture.fieldsim.runtime.runner import FieldRunner
from sparcs.components.agriculture.fieldsim.runtime.scenario import ScenarioRunner

UTC = dt.timezone.utc
PLUS_TWO = dt.timezone(dt.timedelta(hours=2))
RUNNER_LOGGER = "sparcs.components.agriculture.fieldsim.runtime.runner"


class _Probe:
    def __init__(self, channel_id: str) -> None:
        self.channel_id = channel_id


class _Engine:
    cold_start_s = 0.0

    def initial_state(self, at):
        return SoilState(np.full(3, 0.5), np.full(3, 0.5), {}, at)

    def advance(self, state, forcing, *, cancel=None):
        return StepResult(SoilState(state.se * 0.99, state.se, {}, forcing.end), {"water": float(state.se.sum())})

    def tension_at(self, state, probe):
        return -100.0 / float(state.se[0])


class _Shading:
    def evaluate(self, weather):
        return pd.DataFrame({"top": 1.0}, index=weather.index)


class _ET:
    def evaluate(self, weather, segments):
        bulk = pd.DataFrame({"et_top": 0.1}, index=weather.index)
        seg_et = {"top": pd.DataFrame({"et": 0.1, "evap": 0.1, "transp": 0.0}, index=weather.index)}
        return bulk, seg_et


class _Chain(WeatherChain):
    def _prepare_weather(self, weather):
        return weather

    def _segments(self, weather, shading):
        return [SegmentProperties("top", lai=1.0)]

    def _forcings(self, weather, shading, seg_et, irrigation_lpm, *, frontier, first_dt_s=0.0):
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
    """``FrameInputs`` that keeps every read window the runner asked for."""

    def __init__(self, frames, state=None) -> None:
        super().__init__(frames, state=state)
        self.windows = []

    def read(self, key, start, end):
        self.windows.append((key, start, end))
        return super().read(key, start, end)


class _FlakyRecorder(Recorder):
    """``Recorder`` whose ``step`` raises on the ``fail_at``-th row."""

    def __init__(self, fail_at: int) -> None:
        super().__init__()
        self.fail_at = fail_at
        self.seen = 0

    def step(self, result):
        self.seen += 1
        if self.seen == self.fail_at:
            raise RuntimeError("sink unavailable")
        super().step(result)


def _setup(planner=True) -> FieldSetup:
    soil = SoilConfig.from_dict({"mesh": {}})
    soil.probe_specs = (_Probe("soil_30cm"),)
    return FieldSetup(
        field=FieldConfig.from_dict({"interval": 360, "intake_delay": "0min"}),
        soil=soil,
        shading=None,
        planner=PlannerConfig.from_dict({"horizon": "6h"}) if planner else None,
    )


def _simulation(setup: FieldSetup) -> Simulation:
    return Simulation(setup, _Engine(), _Chain(setup, _Shading(), _ET()), _NoAssimilation(), _Planner())


def _weather(start: str, end: str) -> pd.DataFrame:
    idx = pd.date_range(start, end, freq="1h", inclusive="left", tz=UTC)
    return pd.DataFrame({"ghi": 100.0}, index=idx)


def _state_at(moment: dt.datetime) -> SoilState:
    return SoilState(np.full(3, 0.4), np.full(3, 0.4), {}, moment)


def _runner(setup, frames, outputs, state=None, inputs=None) -> FieldRunner:
    return FieldRunner(setup, _simulation(setup), inputs or FrameInputs(frames, state=state), outputs)


# --------------------------------------------------------------------------- memory adapters


def test_frame_inputs_slices_left_open_and_handles_missing_keys():
    inputs = FrameInputs({InputKey.WEATHER: _weather("2026-09-20", "2026-09-21")})
    got = inputs.read(
        InputKey.WEATHER, dt.datetime(2026, 9, 20, 6, tzinfo=UTC), dt.datetime(2026, 9, 20, 9, tzinfo=UTC)
    )
    assert list(got.index.hour) == [7, 8, 9]
    assert inputs.read(
        InputKey.FORECAST, dt.datetime(2026, 9, 20, tzinfo=UTC), dt.datetime(2026, 9, 21, tzinfo=UTC)
    ).empty
    assert inputs.load_state() is None


def test_recorder_to_frames():
    rec = Recorder()
    t0 = dt.datetime(2026, 9, 20, 1, tzinfo=UTC)
    for i in range(3):
        rec.step(
            StepResult(
                SoilState(np.zeros(1), np.zeros(1), {}, t0 + dt.timedelta(hours=i)),
                {"water": float(i)},
                {"p": -10.0 * i},
            )
        )
    frames = rec.to_frames()
    assert frames["diagnostics"].shape == (3, 1) and list(frames["diagnostics"]["water"]) == [0.0, 1.0, 2.0]
    assert frames["tension"]["p"].iloc[-1] == -20.0
    assert "shading" not in frames


# --------------------------------------------------------------------------- scenario over the real runner


def test_scenario_runs_catch_up_in_day_chunks_and_plans_once_per_day():
    setup = _setup()
    sim = _simulation(setup)
    frames = {
        InputKey.WEATHER: _weather("2026-09-20", "2026-09-23"),
        InputKey.FORECAST: _weather("2026-09-23", "2026-09-23 06:00"),
    }
    rec = ScenarioRunner(setup, sim).run(
        frames, start=dt.datetime(2026, 9, 20, 6, tzinfo=UTC), end=dt.datetime(2026, 9, 23, 0, tzinfo=UTC)
    )
    # 72 hourly rows; the row at the very start of the first window falls outside
    # (start, end] and the next one is the cold-start anchor.
    assert len(rec.steps) == 70
    assert len(rec.chains) == 12  # 6-hourly ticks, one chunk each
    assert len(rec.plans) == 2  # the two ticks whose horizon reaches the forecast, one per date
    assert len(rec.saved) == 70
    snap = sim.snapshot()
    assert snap.frontier == dt.datetime(2026, 9, 22, 23, tzinfo=UTC)
    assert snap.last_plan is not None and snap.last_plan.chosen == ("06:00", 30)
    assert set(snap.last_step.probe_tension) == {"soil_30cm"}


def test_repeated_tick_at_same_instant_is_a_noop():
    setup = _setup(planner=False)
    runner = _runner(setup, {InputKey.WEATHER: _weather("2026-09-20", "2026-09-21")}, Recorder())
    now = dt.datetime(2026, 9, 20, 12, tzinfo=UTC)
    assert runner.run_tick(now) is True
    assert runner.run_tick(now) is False


def test_resume_from_persisted_state_skips_history():
    setup = _setup(planner=False)
    rec = Recorder()
    runner = _runner(
        setup,
        {InputKey.WEATHER: _weather("2026-09-20", "2026-09-21")},
        rec,
        state=_state_at(dt.datetime(2026, 9, 20, 10, tzinfo=UTC)),
    )
    assert runner.run_tick(dt.datetime(2026, 9, 20, 12, tzinfo=UTC)) is True
    assert [s.state.at.hour for s in rec.steps] == [11, 12]


def test_frontier_in_another_zone_is_aligned_to_the_cutoff():
    setup = _setup(planner=False)
    inputs = _RecordingInputs(
        {InputKey.WEATHER: _weather("2026-09-20", "2026-09-21")},
        state=_state_at(dt.datetime(2026, 9, 20, 12, tzinfo=PLUS_TWO)),
    )
    rec = Recorder()
    runner = _runner(setup, None, rec, inputs=inputs)

    assert runner.run_tick(dt.datetime(2026, 9, 20, 12, tzinfo=UTC)) is True
    assert [s.state.at.hour for s in rec.steps] == [11, 12]
    assert all(start.utcoffset() == dt.timedelta(0) for _, start, _ in inputs.windows)
    assert all(end.utcoffset() == dt.timedelta(0) for _, _, end in inputs.windows)


def test_cancel_stops_after_the_current_chunk():
    setup = _setup(planner=False)
    rec = Recorder()
    runner = _runner(setup, {InputKey.WEATHER: _weather("2026-09-18", "2026-09-21")}, rec)
    calls = {"n": 0}

    def cancel() -> bool:
        calls["n"] += 1
        return True  # cancel as soon as the runner asks

    assert runner.run_tick(dt.datetime(2026, 9, 20, 12, tzinfo=UTC), cancel=cancel) is True
    assert 0 < len(rec.steps) < 60  # one chunk committed, the rest left for the next tick


def test_a_failing_row_write_does_not_stop_the_frontier():
    setup = _setup(planner=False)
    rec = _FlakyRecorder(fail_at=2)
    runner = _runner(setup, {InputKey.WEATHER: _weather("2026-09-20", "2026-09-21")}, rec)

    assert runner.run_tick(dt.datetime(2026, 9, 20, 12, tzinfo=UTC)) is True
    assert [s.state.at.hour for s in rec.steps] == [8, 10, 11, 12]  # the 09:00 write raised
    assert [s.at.hour for s in rec.saved] == [8, 10, 11, 12]
    assert runner.simulation.snapshot().frontier == dt.datetime(2026, 9, 20, 12, tzinfo=UTC)


def test_restore_is_applied_on_the_next_tick():
    setup = _setup(planner=False)
    rec = Recorder()
    runner = _runner(setup, {InputKey.WEATHER: _weather("2026-09-20", "2026-09-21")}, rec)
    runner.run_tick(dt.datetime(2026, 9, 20, 12, tzinfo=UTC))
    runner.restore(_state_at(dt.datetime(2026, 9, 20, 20, tzinfo=UTC)))
    rec.steps.clear()

    assert runner.run_tick(dt.datetime(2026, 9, 20, 22, tzinfo=UTC)) is True
    assert [s.state.at.hour for s in rec.steps] == [21, 22]


def test_restore_older_than_the_frontier_is_dropped(caplog):
    setup = _setup(planner=False)
    rec = Recorder()
    runner = _runner(setup, {InputKey.WEATHER: _weather("2026-09-20", "2026-09-21")}, rec)
    runner.run_tick(dt.datetime(2026, 9, 20, 12, tzinfo=UTC))
    runner.restore(_state_at(dt.datetime(2026, 9, 20, 8, tzinfo=UTC)))
    rec.steps.clear()

    with caplog.at_level(logging.INFO, logger=RUNNER_LOGGER):
        assert runner.run_tick(dt.datetime(2026, 9, 20, 13, tzinfo=UTC)) is True
    assert [s.state.at.hour for s in rec.steps] == [13]
    assert sum("not newer than the frontier" in r.message for r in caplog.records) == 1


def test_stall_and_failure_tallies_reach_the_diagnostics():
    setup = _setup(planner=False)
    rec = Recorder()
    runner = _runner(setup, {InputKey.WEATHER: _weather("2026-09-20 12:00", "2026-09-20 18:00")}, rec)
    runner.tick_failures = 2.0

    assert runner.run_tick(dt.datetime(2026, 9, 20, 6, tzinfo=UTC)) is False
    assert runner.weather_stall_ticks == 1.0

    assert runner.run_tick(dt.datetime(2026, 9, 20, 18, tzinfo=UTC)) is True
    assert runner.weather_stall_ticks == 0.0
    assert rec.steps and all(s.diagnostics["weather_stall"] == 1.0 for s in rec.steps)
    assert all(s.diagnostics["tick_failures"] == 2.0 for s in rec.steps)


def test_planner_warning_is_latched_once_per_day(caplog):
    setup = _setup()
    runner = _runner(setup, {InputKey.WEATHER: _weather("2026-09-20", "2026-09-21")}, Recorder())
    with caplog.at_level(logging.WARNING, logger=RUNNER_LOGGER):
        for hour in (6, 12, 18):
            runner.run_tick(dt.datetime(2026, 9, 20, hour, tzinfo=UTC))
    assert sum("planner skipped" in r.message for r in caplog.records) == 1


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


def test_an_empty_chunk_inside_the_span_does_not_stop_the_tick():
    setup = _setup(planner=False)
    rec = Recorder()
    weather = pd.concat([_weather("2026-09-18", "2026-09-19"), _weather("2026-09-20 06:00", "2026-09-21")])
    runner = _runner(setup, {InputKey.WEATHER: weather}, rec)
    state = SoilState(np.full(3, 0.4), np.full(3, 0.4), {}, dt.datetime(2026, 9, 18, 12, tzinfo=UTC))
    runner.restore(state)

    assert runner.run_tick(dt.datetime(2026, 9, 20, 12, tzinfo=UTC)) is True
    assert runner.weather_stall_ticks == 0.0
    assert rec.steps[-1].state.at == dt.datetime(2026, 9, 20, 12, tzinfo=UTC)
    assert not any(
        dt.datetime(2026, 9, 19, 0, tzinfo=UTC) < s.state.at < dt.datetime(2026, 9, 20, 6, tzinfo=UTC)
        for s in rec.steps
    )
