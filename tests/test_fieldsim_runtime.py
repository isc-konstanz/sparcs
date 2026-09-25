# -*- coding: utf-8 -*-
"""sparcs.tests.test_fieldsim_runtime
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Runtime layer of the ``fieldsim`` skeleton: ``FrameInputs`` slicing,
``Recorder`` frames, and ``ScenarioRunner`` driving the production
``FieldRunner`` + ``Simulation`` over stub models. The engine, shading and
ET are stubs; everything between them is the real code path.
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


class _Probe:
    def __init__(self, key: str) -> None:
        self.key = key


class _Engine:
    def initial_state(self, at):
        return SoilState(np.full(3, 0.5), np.full(3, 0.5), {}, at)

    def advance(self, state, forcing, cancel=None):
        return StepResult(SoilState(state.se * 0.99, state.se, {}, forcing.end), {"water": float(state.se.sum())})

    def tension_at(self, state, probe):
        return -100.0 / float(state.se[0])


class _Shading:
    def evaluate(self, weather):
        return pd.DataFrame({"top": 1.0}, index=weather.index)


class _ET:
    def evaluate(self, weather, segments):
        return pd.DataFrame({"et_top": 0.1}, index=weather.index)


class _Chain(WeatherChain):
    def _prepare_weather(self, weather):
        return weather

    def _segments(self, weather, shading):
        return [SegmentProperties("top", lai=1.0)]

    def _forcings(self, weather, shading, et, irrigation_lpm):
        return [Forcing(t.to_pydatetime(), 3600.0) for t in weather.index]


class _NoAssimilation:
    enabled = False


class _Planner:
    def plan(self, state, horizon):
        return Plan(
            chosen=("06:00", 30),
            trajectories={},
            header=pd.DataFrame(),
            detail=pd.DataFrame(),
            irrigation=pd.DataFrame(),
        )


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


# --------------------------------------------------------------------------- memory adapters


def test_frame_inputs_slices_half_open_and_handles_missing_keys():
    inputs = FrameInputs({InputKey.WEATHER: _weather("2026-09-20", "2026-09-21")})
    got = inputs.read(
        InputKey.WEATHER, dt.datetime(2026, 9, 20, 6, tzinfo=UTC), dt.datetime(2026, 9, 20, 9, tzinfo=UTC)
    )
    assert list(got.index.hour) == [6, 7, 8]
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


def test_scenario_runs_catch_up_in_day_chunks_and_plans_once():
    setup = _setup()
    sim = _simulation(setup)
    frames = {
        InputKey.WEATHER: _weather("2026-09-20", "2026-09-23"),
        InputKey.FORECAST: _weather("2026-09-23", "2026-09-23 06:00"),
    }
    rec = ScenarioRunner(setup, sim).run(
        frames, start=dt.datetime(2026, 9, 20, 6, tzinfo=UTC), end=dt.datetime(2026, 9, 23, 0, tzinfo=UTC)
    )
    assert len(rec.steps) == 72  # three days hourly, first tick backfills from cutoff - interval
    assert len(rec.chains) == 12  # 6-hourly ticks, day-chunked reads
    assert len(rec.plans) == 1  # only the tick whose horizon has forecast rows
    assert len(rec.saved) == 72
    snap = sim.snapshot()
    assert snap.frontier == dt.datetime(2026, 9, 23, 0, tzinfo=UTC)
    assert snap.last_plan is not None and snap.last_plan.chosen == ("06:00", 30)
    assert set(snap.last_step.probe_tension) == {"soil_30cm"}


def test_repeated_tick_at_same_instant_is_a_noop():
    setup = _setup(planner=False)
    runner = FieldRunner(
        setup, _simulation(setup), FrameInputs({InputKey.WEATHER: _weather("2026-09-20", "2026-09-21")}), Recorder()
    )
    now = dt.datetime(2026, 9, 20, 12, tzinfo=UTC)
    assert runner.run_tick(now) is True
    assert runner.run_tick(now) is False


def test_resume_from_persisted_state_skips_history():
    setup = _setup(planner=False)
    state = SoilState(np.full(3, 0.4), np.full(3, 0.4), {}, dt.datetime(2026, 9, 20, 10, tzinfo=UTC))
    rec = Recorder()
    runner = FieldRunner(
        setup,
        _simulation(setup),
        FrameInputs({InputKey.WEATHER: _weather("2026-09-20", "2026-09-21")}, state=state),
        rec,
    )
    assert runner.run_tick(dt.datetime(2026, 9, 20, 12, tzinfo=UTC)) is True
    assert [s.state.at.hour for s in rec.steps] == [11, 12]


def test_cancel_stops_after_the_current_chunk():
    setup = _setup(planner=False)
    rec = Recorder()
    runner = FieldRunner(
        setup, _simulation(setup), FrameInputs({InputKey.WEATHER: _weather("2026-09-18", "2026-09-21")}), rec
    )
    calls = {"n": 0}

    def cancel() -> bool:
        calls["n"] += 1
        return True  # cancel as soon as the runner asks

    assert runner.run_tick(dt.datetime(2026, 9, 20, 12, tzinfo=UTC), cancel=cancel) is True
    assert 0 < len(rec.steps) < 60  # one chunk committed, the rest left for the next tick


def test_planner_warning_is_latched_once_per_day(caplog):
    setup = _setup()
    runner = FieldRunner(
        setup, _simulation(setup), FrameInputs({InputKey.WEATHER: _weather("2026-09-20", "2026-09-21")}), Recorder()
    )
    with caplog.at_level(logging.WARNING, logger="sparcs.components.agriculture.fieldsim.runtime.runner"):
        for hour in (6, 12, 18):
            runner.run_tick(dt.datetime(2026, 9, 20, hour, tzinfo=UTC))
    assert sum("planner skipped" in r.message for r in caplog.records) == 1


@pytest.mark.parametrize("hours", [1, 25, 49])
def test_day_chunks_cover_span_without_gaps(hours):
    start = dt.datetime(2026, 9, 20, tzinfo=UTC)
    end = start + dt.timedelta(hours=hours)
    chunks = list(FieldRunner._day_chunks(start, end))
    assert chunks[0][0] == start and chunks[-1][1] == end
    assert all(a[1] == b[0] for a, b in zip(chunks, chunks[1:]))
    assert all((b - a) <= dt.timedelta(days=1) for a, b in chunks)
