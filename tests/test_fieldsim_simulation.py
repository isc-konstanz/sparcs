# -*- coding: utf-8 -*-
"""
tests.test_fieldsim_simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Simulation.run``, ``add_sensors`` and ``plan`` over the real ``WeatherChain`` and a stub engine.
"""

import logging
import types

import pytest

import numpy as np
import pandas as pd
from lories.components.weather import Weather
from sparcs.components.agriculture.simulation.core.anchor import AnchorSensor
from sparcs.components.agriculture.simulation.core.chain import WeatherChain
from sparcs.components.agriculture.simulation.core.config import FieldConfig, FieldSetup, SoilConfig
from sparcs.components.agriculture.simulation.core.evapotranspiration import ETModel
from sparcs.components.agriculture.simulation.core.shading import ShadingConfig, ShadingModel
from sparcs.components.agriculture.simulation.core.simulation import Simulation
from sparcs.components.agriculture.simulation.core.state import SoilState, StepResult

_HOUR = 3600.0


def _weather_frame(hours: int = 3, start: str = "2026-06-21") -> pd.DataFrame:
    idx = pd.date_range(start, periods=hours, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            Weather.GHI: 0.0,
            Weather.TEMP_AIR: 20.0,
            Weather.HUMIDITY_REL: 55.0,
            Weather.WIND_SPEED: 2.0,
            Weather.CLEAR_SKY_INDEX: 0.8,
            Weather.PRECIPITATION: 0.0,
        },
        index=idx,
    )


class _StubEngine:
    """Records the forcings it is handed and returns a state stamped at each window's end."""

    def __init__(self, cold_start_s: float = 0.0, cancel_after: int = -1) -> None:
        self.cold_start_s = cold_start_s
        self.cancel_after = cancel_after
        self.calls: list = []

    def initial_state(self, at) -> SoilState:
        return SoilState(se=np.zeros(1), se_old=np.zeros(1), surface_h={}, at=at)

    def advance(self, state: SoilState, forcing, *, cancel=None) -> StepResult:
        self.calls.append(forcing)
        if 0 <= self.cancel_after < len(self.calls):
            return StepResult(state=state, diagnostics={}, cancelled=True)
        advanced = SoilState(se=state.se, se_old=state.se_old, surface_h={}, at=forcing.end)
        return StepResult(state=advanced, diagnostics={"water_total": float(len(self.calls))})

    def tension_at(self, state: SoilState, probe) -> float:
        return -100.0

    def probe_from_sensor(self, sensor):
        return types.SimpleNamespace(channel_id=sensor.key, at=(sensor.x_offset_cm, sensor.depth_cm))


class _StubAssimilator:
    """Enabled once anchoring is on and sensors exist; ``update`` leaves the state as it is."""

    def __init__(self, anchor_enabled: bool = False) -> None:
        self.anchor_enabled = anchor_enabled
        self.sensors: list = []
        self.last_result = None

    @property
    def enabled(self) -> bool:
        return self.anchor_enabled and bool(self.sensors)

    def set_sensors(self, sensors) -> None:
        self.sensors = list(sensors)

    def update(self, state: SoilState, now) -> SoilState:
        return state


def _simulation(
    cold_start_s: float = 0.0,
    probes=(),
    cancel_after: int = -1,
    *,
    anchor_enabled: bool = False,
    discover_sensor_probes: bool = False,
) -> Simulation:
    setup = FieldSetup(
        field=FieldConfig.from_dict({"bay_width": 3.5}),
        soil=SoilConfig.from_dict({"mesh": {}, "discover_sensor_probes": discover_sensor_probes}),
        shading=None,
    )
    shading_config = ShadingConfig.from_dict({"mode": "free_field"}).derive(bay_width=3.5, segment_ranges=None)
    chain = WeatherChain(setup, ShadingModel(shading_config), ETModel(), top_segment_names=(), segment_face_length={})
    engine = _StubEngine(cold_start_s=cold_start_s, cancel_after=cancel_after)
    assimilator = _StubAssimilator(anchor_enabled)
    return Simulation(setup, engine, chain, assimilator, probes=probes)


def _run(sim: Simulation, weather: pd.DataFrame, **kwargs):
    return sim.run(weather, pd.Series(0.0, index=weather.index), **kwargs)


def test_cold_start_spins_up_over_the_configured_window():
    sim = _simulation(cold_start_s=3 * _HOUR)
    weather = _weather_frame(hours=3)

    results, _ = _run(sim, weather)

    assert [f.dt_s for f in sim.engine.calls] == [3 * _HOUR, _HOUR, _HOUR]
    assert pd.Timestamp(sim.engine.calls[0].at) == weather.index[0]
    assert len(results) == 3
    assert results[0].state.at == weather.index[0]
    assert sim.state.at == weather.index[-1]


def test_zero_cold_start_anchors_the_state_without_advancing():
    sim = _simulation(cold_start_s=0.0)
    weather = _weather_frame(hours=3)

    results, _ = _run(sim, weather)

    assert [pd.Timestamp(f.at) for f in sim.engine.calls] == list(weather.index[1:])
    assert len(results) == 2
    assert sim.state.at == weather.index[-1]


def test_rows_at_or_before_the_frontier_are_not_advanced_again():
    sim = _simulation(cold_start_s=0.0)
    first = _weather_frame(hours=3)
    _run(sim, first)
    sim.engine.calls.clear()

    overlapping = _weather_frame(hours=5)
    results, _ = _run(sim, overlapping)

    assert [pd.Timestamp(f.at) for f in sim.engine.calls] == list(overlapping.index[3:])
    assert all(f.dt_s == _HOUR for f in sim.engine.calls)
    assert len(results) == 2


def test_extra_diagnostics_are_merged_into_every_step():
    sim = _simulation(cold_start_s=_HOUR)
    weather = _weather_frame(hours=3)

    results, _ = _run(sim, weather, extra_diagnostics={"weather_stall": 2.0, "tick_failures": 1.0})

    assert len(results) == 3
    for result in results:
        assert result.diagnostics["weather_stall"] == 2.0
        assert result.diagnostics["tick_failures"] == 1.0
        assert "water_total" in result.diagnostics


def test_probe_tension_is_keyed_by_channel_id():
    probe = types.SimpleNamespace(channel_id="strip", key="ignored")
    sim = _simulation(cold_start_s=0.0, probes=[probe])

    results, _ = _run(sim, _weather_frame(hours=2))

    assert results[-1].probe_tension == {"strip": pytest.approx(-100.0)}


def test_cancelled_cold_start_spin_up_leaves_no_state():
    """A spin-up cancelled mid-walk has not produced a state: the next run must
    cold-start again, not continue from an initial condition that never advanced."""
    sim = _simulation(cold_start_s=3 * _HOUR, cancel_after=0)

    results, _ = _run(sim, _weather_frame(hours=3))

    assert results == []
    assert sim.state is None


class _AnchoringAssimilator(_StubAssimilator):
    """Anchors on the rows listed in ``anchor_at``, with fixed innovations."""

    def __init__(self, anchor_at) -> None:
        super().__init__(anchor_enabled=True)
        self.sensors = [_sensor("s1")]
        self.anchor_at = set(anchor_at)

    def update(self, state: SoilState, now) -> SoilState:
        if pd.Timestamp(now) not in self.anchor_at:
            return state
        self.last_result = types.SimpleNamespace(innovations={"s1": 0.03, "s2": -0.01})
        return SoilState(se=state.se + 0.1, se_old=state.se + 0.1, surface_h={}, at=now)


def test_each_row_carries_its_own_anchor_increment():
    weather = _weather_frame(hours=4)
    sim = _simulation(cold_start_s=0.0)
    sim.assimilator = _AnchoringAssimilator(anchor_at=[weather.index[2]])

    results, _ = _run(sim, weather)

    assert [r.diagnostics["anchor"] for r in results] == [0.0, pytest.approx(0.02), 0.0]


def test_no_anchor_increment_without_anchoring():
    results, _ = _run(_simulation(cold_start_s=0.0), _weather_frame(hours=3))

    assert all("anchor" not in r.diagnostics for r in results)


def test_cancelled_advance_keeps_the_committed_rows():
    sim = _simulation(cold_start_s=0.0, cancel_after=1)
    weather = _weather_frame(hours=4)

    results, _ = _run(sim, weather)

    assert len(results) == 1
    assert sim.state.at == weather.index[1]


def _sensor(key: str) -> AnchorSensor:
    return AnchorSensor(key=key, x_offset_cm=25.0, depth_cm=60.0)


def test_add_sensors_without_anchor_or_discovery_keeps_them_off_the_probe_list():
    sim = _simulation(probes=[types.SimpleNamespace(channel_id="strip")])

    sim.add_sensors([_sensor("s1")])

    assert [s.key for s in sim.assimilator.sensors] == ["s1"]
    assert [p.channel_id for p in sim.probes] == ["strip"]


@pytest.mark.parametrize(
    "anchor_enabled, discover_sensor_probes",
    [pytest.param(True, False, id="anchor"), pytest.param(False, True, id="discover")],
)
def test_add_sensors_samples_each_sensor_as_a_probe(anchor_enabled, discover_sensor_probes):
    sim = _simulation(
        probes=[types.SimpleNamespace(channel_id="strip")],
        anchor_enabled=anchor_enabled,
        discover_sensor_probes=discover_sensor_probes,
    )

    sim.add_sensors([_sensor("s1")])
    results, _ = _run(sim, _weather_frame(hours=2))

    assert [p.channel_id for p in sim.probes] == ["strip", "s1"]
    assert sim.probes[-1].at == (25.0, 60.0)
    assert set(results[-1].probe_tension) == {"strip", "s1"}


class _RecordingPlanner:
    """Keeps the keyword arguments of every ``plan`` call."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def plan(self, state, weather, seg_et, horizon_start, horizon_end, **kwargs):
        self.calls.append(kwargs)


_CREATION_KEY = "timestamp_creation"
_ISSUED = pd.Timestamp("2026-06-20 23:00", tz="UTC")


def _planning_simulation() -> Simulation:
    sim = _simulation()
    sim.planner = _RecordingPlanner()
    sim.state = sim.engine.initial_state(pd.Timestamp("2026-06-21 00:00", tz="UTC"))
    return sim


def _simulation_warnings(caplog) -> list:
    return [r for r in caplog.records if r.name == Simulation.__module__ and r.levelno == logging.WARNING]


def test_plan_hands_the_latest_forecast_issue_time_to_the_planner():
    sim = _planning_simulation()
    forecast = _weather_frame(hours=3)
    forecast[_CREATION_KEY] = [_ISSUED, _ISSUED - pd.Timedelta(hours=6), None]

    sim.plan(forecast)

    (call,) = sim.planner.calls
    assert call["weather_creation"] == _ISSUED
    assert call["run_timestamp"] == pd.Timestamp(sim.state.at)


@pytest.mark.parametrize(
    "creation",
    [pytest.param(None, id="missing"), pytest.param([None, None, None], id="null")],
)
def test_plan_falls_back_to_the_run_time_with_one_warning_without_an_issue_time(caplog, creation):
    sim = _planning_simulation()
    forecast = _weather_frame(hours=3)
    if creation is not None:
        forecast[_CREATION_KEY] = creation

    with caplog.at_level(logging.WARNING):
        sim.plan(forecast)
        sim.plan(forecast)

    run_timestamp = pd.Timestamp(sim.state.at)
    assert [call["weather_creation"] for call in sim.planner.calls] == [run_timestamp, run_timestamp]
    (warning,) = _simulation_warnings(caplog)
    assert _CREATION_KEY in warning.getMessage()
