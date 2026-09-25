# -*- coding: utf-8 -*-
"""tests.test_fieldsim_components
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The lories layer of the fieldsim skeleton: ``Simulation.build`` assembling
engine/chain/assimilator/planner from a configured ``FieldSetup`` (against
the tmp-mesh recipe shared with ``test_fieldsim_planner.py``), and
``ChannelInputs``/``ChannelOutputs`` as adapters over fakes -- no lories
Component is instantiated here, only the plain config/value objects and
hand-built fakes that stand in for one.
"""

import datetime as dt
from types import SimpleNamespace

import pytest

import numpy as np
import pandas as pd
from lories.components.weather import Weather
from lories.data import Channels
from sparcs.components.agriculture.fieldsim import components
from sparcs.components.agriculture.fieldsim.core.anchor import AnchorResult, AnchorSensor
from sparcs.components.agriculture.fieldsim.core.config import FieldConfig, FieldSetup, PlannerConfig, SoilConfig
from sparcs.components.agriculture.fieldsim.core.simulation import Simulation
from sparcs.components.agriculture.fieldsim.core.state import ChainResult, Plan, SoilState, StepResult
from sparcs.components.agriculture.fieldsim.runtime.ports import InputKey

UTC = dt.timezone.utc

_MESH_KW = {
    "dl": 0.2,
    "width": 3.0,
    "height": 1.5,
    "plant_width": 1.0,
    "plant_height": 0.5,
    "watering_width": 0.5,
    "d_x": 0.5,
}


def _soil_config(tmp_path, filename: str) -> SoilConfig:
    soil = SoilConfig.from_dict(
        {
            "mesh": {**_MESH_KW, "filename": str(tmp_path / filename)},
            "pde": {"dt": "600s", "dt_min": "30s"},
            "probes": {"points": {"strip": {"x_offset": 0.0, "depth": 30.0}}},
        }
    )
    soil.mesh.derive(bay_width=3.0)
    return soil


def _weather_frame(hours: int = 3) -> pd.DataFrame:
    idx = pd.date_range("2026-06-21 08:00", periods=hours, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            Weather.GHI: 500.0,
            Weather.TEMP_AIR: 22.0,
            Weather.HUMIDITY_REL: 55.0,
            Weather.WIND_SPEED: 2.0,
            Weather.CLEAR_SKY_INDEX: 0.8,
            Weather.PRECIPITATION: 0.0,
        },
        index=idx,
    )


# --------------------------------------------------------------------------- Simulation.build


@pytest.mark.slow  # builds a real Gmsh mesh and runs FiPy
def test_simulation_build_wires_engine_chain_assimilator_planner(tmp_path):
    soil = _soil_config(tmp_path, "components_build.msh")
    planner_cfg = PlannerConfig.from_dict(
        {
            "windows": {"w0": {"start": "08:10"}, "w1": {"start": "10:10"}},
            "durations_min": [0, 5],
            "grid_mode": "ladder",
            "decision_probes": ["strip"],
            "threshold_hpa": 5.0,
            "max_windows": 4,
        }
    )
    field = FieldConfig.from_dict({"bay_width": 3.0})
    setup = FieldSetup(field=field, soil=soil, shading=components.ShadingConfig.from_dict(), planner=planner_cfg)

    simulation = Simulation.build(setup, rel_sat_name="Se_components_build")

    assert simulation.engine is not None
    assert simulation.assimilator.enabled is False  # no [anchor] block, no sensors
    assert simulation.assimilator.sensors == []
    assert simulation.planner is not None
    assert simulation.planner._probe_ids == ["strip"]

    assert [p.channel_id for p in setup.soil.probe_specs] == ["strip"]

    assert tuple(simulation.chain._top_segment_names) == tuple(simulation.engine.top_segment_names)
    assert len(simulation.chain._top_segment_names) > 0

    weather = _weather_frame(3)
    irrigation_lpm = pd.Series(0.0, index=weather.index)
    results, chain_result = simulation.run(weather, irrigation_lpm)

    assert len(results) == 3
    for result in results:
        assert "strip" in result.probe_tension
        assert np.isfinite(result.probe_tension["strip"])
    assert chain_result.evapotranspiration.index.equals(weather.index)


# --------------------------------------------------------------------------- ChannelInputs


class _FakeChannel:
    def __init__(self, *, connector: bool = True, valid: bool = False, value=None, timestamp=None):
        self._connector = connector
        self._valid = valid
        self.value = value
        self.timestamp = timestamp

    def has_connector(self) -> bool:
        return self._connector

    def is_valid(self) -> bool:
        return self._valid


class _FakeData:
    """Stands in for ``Component.data``: routes ``.read(channels, ...)`` by
    identity (the weather channels list) or by contained channel object (a
    fresh ``Channels([one_channel])`` per irrigation/tension read)."""

    def __init__(self, weather_channels=None, weather_frame=None):
        self.weather_channels = weather_channels if weather_channels is not None else []
        self.weather_frame = weather_frame if weather_frame is not None else pd.DataFrame()
        self.by_channel: dict[int, pd.DataFrame] = {}

    def bind(self, channel, frame: pd.DataFrame) -> None:
        self.by_channel[id(channel)] = frame

    def read(self, channels, start=None, end=None, unique=True) -> pd.DataFrame:
        if channels is self.weather_channels:
            return self.weather_frame.copy()
        chans = list(channels)
        if len(chans) == 1 and id(chans[0]) in self.by_channel:
            return self.by_channel[id(chans[0])].copy()
        return pd.DataFrame()


def _fake_field(**overrides) -> SimpleNamespace:
    weather_channels = Channels([])
    weather_frame = _weather_frame(3)
    field = SimpleNamespace(
        name="fieldsim_test",
        data=_FakeData(weather_channels, weather_frame),
        weather=SimpleNamespace(forecast=None),
        soil=None,
        simulation=None,
        setup=SimpleNamespace(soil=SoilConfig.from_dict({"mesh": {}})),
        _weather_channels=weather_channels,
        _evapo_rename={},
        _required_weather_keys=components.ETModel.REQUIRED_WEATHER_COLUMNS,
        _irrigation_flow_channel=None,
        _irrigation_state_channel=None,
        _anchor_sensors=(),
        _anchor_channels={},
        _anchor_data={},
    )
    for key, value in overrides.items():
        setattr(field, key, value)
    return field


def test_channel_inputs_weather_trims_and_validates():
    field = _fake_field()
    inputs = components.ChannelInputs(field)

    start = field.data.weather_frame.index[0]
    end = field.data.weather_frame.index[1]
    got = inputs.read(InputKey.WEATHER, start.to_pydatetime(), end.to_pydatetime())
    assert list(got.index) == [end]  # (start, end]

    field.data.weather_frame = field.data.weather_frame.drop(columns=[Weather.GHI])
    got_invalid = inputs.read(InputKey.WEATHER, start.to_pydatetime(), end.to_pydatetime())
    assert got_invalid.empty
    assert Weather.GHI in inputs._last_invalid_weather_columns


def test_channel_inputs_irrigation_measured_then_state_then_empty():
    weather = _weather_frame(2)
    flow_channel = _FakeChannel()
    state_channel = _FakeChannel()
    soil = SoilConfig.from_dict({"mesh": {}, "drip": {"nozzle_count": 5, "nozzle_flow_lph": 60.0}})
    assert soil.drip.explicit and soil.drip.design_flow_lpm == pytest.approx(5.0)

    data = _FakeData(Channels([]), weather)
    field = _fake_field(
        data=data,
        _weather_channels=data.weather_channels,
        setup=SimpleNamespace(soil=soil),
        _irrigation_flow_channel=flow_channel,
        _irrigation_state_channel=state_channel,
    )
    inputs = components.ChannelInputs(field)
    start, end = weather.index[0].to_pydatetime(), weather.index[-1].to_pydatetime()

    # measured flow present -> wins
    measured_frame = pd.DataFrame({"flow": [1.2]}, index=[weather.index[0]])
    data.bind(flow_channel, measured_frame)
    got = inputs.read(InputKey.IRRIGATION, start, end)
    assert list(got.columns) == ["irrigation_flow_lpm"]
    assert (got["irrigation_flow_lpm"] == 1.2).all()

    # measured absent -> state x design flow
    data.bind(flow_channel, pd.DataFrame())
    state_frame = pd.DataFrame({"state": [1.0]}, index=[weather.index[0]])
    data.bind(state_channel, state_frame)
    got = inputs.read(InputKey.IRRIGATION, start, end)
    assert got["irrigation_flow_lpm"].tolist() == pytest.approx([5.0])

    # state wired but not explicit -> never fabricate a forcing from it: empty
    soil.drip.explicit = False
    got = inputs.read(InputKey.IRRIGATION, start, end)
    assert got.empty

    # no meter and no state channel at all -> empty
    field._irrigation_state_channel = None
    got = inputs.read(InputKey.IRRIGATION, start, end)
    assert got.empty


def test_channel_inputs_tension_one_column_per_sensor_nan_where_absent():
    s1 = _FakeChannel()
    s2 = _FakeChannel()
    idx1 = pd.DatetimeIndex(["2026-06-21 08:00", "2026-06-21 09:00"], tz="UTC")
    frame1 = pd.DataFrame({"tension": [-50.0, -55.0]}, index=idx1)
    data = _FakeData()
    data.bind(s1, frame1)
    data.bind(s2, pd.DataFrame())  # sensor 2 produced nothing this span

    field = _fake_field(
        data=data,
        simulation=SimpleNamespace(assimilator=SimpleNamespace(config=None)),
        _anchor_sensors=[
            AnchorSensor(key="s1", x_offset_cm=0.0, depth_cm=10.0),
            AnchorSensor(key="s2", x_offset_cm=0.0, depth_cm=20.0),
        ],
        _anchor_channels={"s1": s1, "s2": s2},
        _anchor_data={"s1": data, "s2": data},
    )
    inputs = components.ChannelInputs(field)

    got = inputs.read(
        InputKey.TENSION, dt.datetime(2026, 6, 21, 7, tzinfo=UTC), dt.datetime(2026, 6, 21, 10, tzinfo=UTC)
    )
    assert set(got.columns) == {"s1", "s2"}
    assert got["s1"].dropna().tolist() == [-50.0, -55.0]
    assert got["s2"].isna().all()


def test_channel_inputs_forecast_spans_start_end():
    idx = pd.date_range("2026-06-21 00:00", periods=6, freq="1h", tz="UTC")
    forecast_frame = pd.DataFrame({Weather.GHI: 100.0}, index=idx)
    forecast_sub = SimpleNamespace(
        data=SimpleNamespace(to_frame=lambda unique=False: forecast_frame),
        is_enabled=lambda: True,
    )
    field = _fake_field(weather=SimpleNamespace(forecast=forecast_sub))
    inputs = components.ChannelInputs(field)

    got = inputs.read(
        InputKey.FORECAST, dt.datetime(2026, 6, 21, 2, tzinfo=UTC), dt.datetime(2026, 6, 21, 4, tzinfo=UTC)
    )
    assert list(got.index.hour) == [2, 3]


def test_channel_inputs_load_state_from_blob():
    state = SoilState(
        se=np.array([0.5, 0.6]), se_old=np.array([0.5, 0.6]), surface_h={}, at=dt.datetime(2026, 6, 21, tzinfo=UTC)
    )
    blob = state.to_blob()
    channel = _FakeChannel(valid=True, value=blob, timestamp=state.at)
    field = _fake_field(soil=SimpleNamespace(data={"simulation_state": channel}))
    inputs = components.ChannelInputs(field)

    got = inputs.load_state()
    assert got is not None
    np.testing.assert_allclose(got.se, state.se)
    assert got.at == state.at

    field.soil = None
    assert inputs.load_state() is None


# --------------------------------------------------------------------------- ChannelOutputs


class _RecordingChannel:
    def __init__(self):
        self.calls: list[tuple] = []

    def set(self, ts, value) -> None:
        self.calls.append((ts, value))


class _RecordingData(dict):
    """A dict of pre-registered ``_RecordingChannel``s; an unregistered key
    raises ``KeyError``, mirroring a channel that was never added (e.g. a
    sensor-derived probe with no logged channel)."""


class _NS:
    def __init__(self, keys):
        self.data = _RecordingData({k: _RecordingChannel() for k in keys})


class _FakeConnector:
    def __init__(self, raise_on_column: str):
        self.raise_on_column = raise_on_column
        self.written: list[pd.DataFrame] = []

    def write(self, frame: pd.DataFrame) -> None:
        if self.raise_on_column in frame.columns:
            raise RuntimeError("boom")
        self.written.append(frame)


def test_channel_outputs_step_writes_diagnostics_probes_and_anchor():
    soil_ns = _NS(
        [
            "top_in",
            "top_out",
            "bottom_out",
            "transpiration",
            "runoff",
            "demand_unmet",
            "balance_residual",
            "skipped_s",
            "retries",
            "anchor",
            "strip",
        ]
    )
    assimilator = SimpleNamespace(
        last_result=AnchorResult(se_new=np.array([0.5]), anchored_at={}, innovations={"strip": 0.07})
    )
    field = SimpleNamespace(simulation=SimpleNamespace(assimilator=assimilator))
    outputs = components.ChannelOutputs(field, shading=None, et=None, soil=soil_ns, predictor=None)

    ts = dt.datetime(2026, 6, 21, 9, tzinfo=UTC)
    result = StepResult(
        state=SoilState(se=np.array([0.5]), se_old=np.array([0.5]), surface_h={}, at=ts),
        diagnostics={
            "top_in": 1.0,
            "top_out": 2.0,
            "bottom_out": 3.0,
            "transpiration": 4.0,
            "runoff": 5.0,
            "demand_unmet": 6.0,
            "balance_residual": 7.0,
            "skipped_s": 0.0,
            "retries": 0.0,
            "water_total": 999.0,  # not a channel key; must be ignored, not raise
            "walk_ok": 1.0,
        },
        probe_tension={"strip": -120.0, "sensor_only": -80.0},  # sensor_only has no registered channel
    )

    outputs.step(result)

    assert soil_ns.data["top_in"].calls == [(ts, 1.0)]
    assert soil_ns.data["balance_residual"].calls == [(ts, 7.0)]
    assert soil_ns.data["strip"].calls == [(ts, -120.0)]
    assert soil_ns.data["anchor"].calls == [(ts, 0.07)]


def test_channel_outputs_save_state_writes_simulation_state():
    soil_ns = _NS(["simulation_state"])
    outputs = components.ChannelOutputs(SimpleNamespace(), shading=None, et=None, soil=soil_ns, predictor=None)

    state = SoilState(se=np.array([0.4]), se_old=np.array([0.4]), surface_h={}, at=dt.datetime(2026, 6, 21, tzinfo=UTC))
    outputs.save_state(state)

    assert len(soil_ns.data["simulation_state"].calls) == 1
    ts, blob = soil_ns.data["simulation_state"].calls[0]
    assert ts == state.at
    assert blob == state.to_blob()


def test_channel_outputs_chain_writes_shading_et_and_vegetation_per_row():
    shading_ns = _NS(["shading_factor", "shading_progress_image"])
    et_ns = _NS(list(components._ET_CHANNEL_KEYS))
    field_ns = _NS(["seg_ghi", "lai", "roughness", "plant_height", "ndvi"])
    field_ns.simulation = SimpleNamespace(engine=SimpleNamespace(top_segment_names=["seg1", "seg2"]))
    outputs = components.ChannelOutputs(field_ns, shading=shading_ns, et=et_ns, soil=None, predictor=None)

    idx = pd.date_range("2026-06-21 08:00", periods=2, freq="1h", tz="UTC")
    shading_df = pd.DataFrame(
        {
            "seg1": [1.0, 0.8],
            "seg2": [1.0, 0.6],
            "ghi_seg1": [500.0, 400.0],
            "ghi_seg2": [500.0, 300.0],
            "open_sky_ghi": [500.0, 500.0],
        },
        index=idx,
    )
    et_df = pd.DataFrame(
        {
            "evapotranspiration": [0.1, 0.2],
            "lai": [1.0, 1.0],
            "roughness": [0.002, 0.002],
            "plant_height": [0.1, 0.1],
            "ndvi": [0.25, 0.25],
        },
        index=idx,
    )

    outputs.chain(idx[-1].to_pydatetime(), ChainResult(shading=shading_df, evapotranspiration=et_df, image=b"png"))

    assert [v for _, v in shading_ns.data["shading_factor"].calls] == [pytest.approx(1.0), pytest.approx(0.7)]
    assert shading_ns.data["shading_progress_image"].calls == [(idx[-1].to_pydatetime(), b"png")]
    assert [v for _, v in field_ns.data["seg_ghi"].calls] == [[500.0, 500.0], [400.0, 300.0]]
    assert [v for _, v in et_ns.data["evapotranspiration"].calls] == [0.1, 0.2]
    assert [v for _, v in field_ns.data["lai"].calls] == [1.0, 1.0]


def test_channel_outputs_plan_writes_four_tables_best_effort():
    connector = _FakeConnector(raise_on_column="irrigation_state")
    predictor = _NS([])
    predictor.connectors = {"db": connector}
    field = SimpleNamespace(setup=SimpleNamespace(planner=SimpleNamespace(logger="db")))
    outputs = components.ChannelOutputs(field, shading=None, et=None, soil=None, predictor=predictor)

    ts = pd.Timestamp("2026-06-21 08:00", tz="UTC")
    plan = Plan(
        chosen=None,
        trajectories={},
        header=pd.DataFrame({"forecast_id": [0]}, index=[ts]),
        detail=pd.DataFrame({"traj_strip": [-100.0]}, index=[ts]),
        irrigation=pd.DataFrame({"irrigation_state": [True]}, index=[ts]),
        image=pd.DataFrame({"predict_image": [b"png"]}, index=[ts]),
    )

    outputs.plan(plan)  # must not raise even though the irrigation write raises

    assert len(connector.written) == 3  # header, detail, image -- irrigation raised and was skipped
    written_columns = [set(f.columns) for f in connector.written]
    assert {"forecast_id"} in written_columns
    assert {"traj_strip"} in written_columns
    assert {"predict_image"} in written_columns


# --------------------------------------------------------------------------- CHANNELS completeness


def test_ground_shading_channels_match_live():
    try:
        from sparcs.components.agriculture.simulation.ground_shading import GroundShading as LiveGroundShading

        expected = {c.key for c in LiveGroundShading.CHANNELS} | {"shading_progress_image", "plot_strikes"}
    except ImportError:
        expected = {"shading_factor", "shading_progress_image", "plot_strikes"}
    assert {spec.key for spec in components.GroundShading.CHANNELS} == expected


def test_evapotranspiration_channels_match_live():
    try:
        from sparcs.components.agriculture.simulation.evapotranspiration import Evapotranspiration as LiveET

        expected = {c.key for c in LiveET.CHANNELS}
    except ImportError:
        expected = {
            "sat_vapor_pressure",
            "ground_vapor_pressure",
            "vaporization_heat",
            "slope_sat_vapor_pressure",
            "net_irradiance",
            "aerodynamic_resistance",
            "soil_heat_flow",
            "resistance_surface",
            "radiation_term",
            "aerodynamic_term",
            "evapotranspiration",
        }
    assert {spec.key for spec in components.Evapotranspiration.CHANNELS} == expected


def test_soil_simulation_channels_match_live():
    try:
        from sparcs.components.agriculture.simulation.soil import SoilSimulation as LiveSoil

        expected = {
            c.key
            for c in (
                LiveSoil.WATER_TOP_IN,
                LiveSoil.WATER_TOP_OUT,
                LiveSoil.WATER_BOTTOM,
                LiveSoil.WATER_TRANSP,
                LiveSoil.WATER_RUNOFF,
                LiveSoil.WATER_DEMAND_UNMET,
                LiveSoil.WATER_BALANCE_RESIDUAL,
                LiveSoil.WATER_ANCHOR,
                LiveSoil.WALK_SKIPPED_S,
                LiveSoil.WALK_RETRIES,
                LiveSoil.WEATHER_STALL,
                LiveSoil.TICK_FAILURES,
            )
        } | {LiveSoil.SIMULATION_STATE.key, LiveSoil.SOIL_PROGRESS_IMAGE.key, "plot_strikes"}
    except ImportError:
        expected = {
            "top_in",
            "top_out",
            "bottom_out",
            "transpiration",
            "runoff",
            "demand_unmet",
            "balance_residual",
            "anchor",
            "skipped_s",
            "retries",
            "weather_stall",
            "tick_failures",
            "simulation_state",
            "soil_progress_image",
            "plot_strikes",
        }
    assert {spec.key for spec in components.SoilSimulation.CHANNELS} == expected


def test_soil_predictor_channels_empty_tables_only():
    assert components.SoilPredictor.CHANNELS == ()
