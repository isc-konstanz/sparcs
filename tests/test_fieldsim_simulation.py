# -*- coding: utf-8 -*-
"""sparcs.tests.test_fieldsim_simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Simulation.run`` over the real ``WeatherChain`` and a stub engine: cold-start
spin-up, the already-simulated guard and the ``extra_diagnostics`` merge; and
``Simulation.add_sensors`` handing discovered tensiometers to the assimilator.
"""

import types

import pytest

import numpy as np
import pandas as pd
from lories.components.weather import Weather
from sparcs.components.agriculture.fieldsim.core.anchor import AnchorSensor
from sparcs.components.agriculture.fieldsim.core.chain import WeatherChain
from sparcs.components.agriculture.fieldsim.core.config import FieldConfig, FieldSetup, SoilConfig
from sparcs.components.agriculture.fieldsim.core.evapotranspiration import ETModel
from sparcs.components.agriculture.fieldsim.core.shading import ShadingConfig, ShadingModel
from sparcs.components.agriculture.fieldsim.core.simulation import Simulation
from sparcs.components.agriculture.fieldsim.core.state import SoilState, StepResult

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
        return types.SimpleNamespace(channel_id=sensor.key, at=(sensor.x_offset, sensor.depth))


class _StubAssimilator:
    """Enabled once anchoring is on and sensors exist; ``update`` leaves the state as it is."""

    def __init__(self, anchor_enabled: bool = False) -> None:
        self.anchor_enabled = anchor_enabled
        self.sensors: list = []

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
