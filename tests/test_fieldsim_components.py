# -*- coding: utf-8 -*-
"""
tests.test_fieldsim_components
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Simulation.build``, ``ChannelInputs``/``ChannelOutputs`` and the offline ``FieldSimulation.simulate`` session.
No lories Component is configured here; plain config objects and hand-built fakes stand in for one.
"""

import datetime as dt
import io
from types import SimpleNamespace

import pytest
from conftest import MESH_KW, load_configs

import numpy as np
import pandas as pd
from lories import Constant
from lories.components.weather import Weather, WeatherProvider
from lories.core import ConfigurationError
from lories.data import Channels
from sparcs.components.agriculture.simulation import components
from sparcs.components.agriculture.simulation.core.anchor import AnchorSensor
from sparcs.components.agriculture.simulation.core.assimilator import parse_anchor_config
from sparcs.components.agriculture.simulation.core.config import FieldConfig, FieldSetup, PlannerConfig, SoilConfig
from sparcs.components.agriculture.simulation.core.simulation import Simulation
from sparcs.components.agriculture.simulation.core.state import ChainResult, Plan, SoilState, StepResult

UTC = dt.timezone.utc


def _soil_config(tmp_path, filename: str, **extra) -> SoilConfig:
    soil = SoilConfig.from_dict(
        {
            "mesh": {**MESH_KW, "filename": str(tmp_path / filename)},
            "pde": {"dt": "600s", "dt_min": "30s"},
            "probes": {"points": {"strip": {"x_offset": 0.0, "depth": 30.0}}},
            **extra,
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
            "windows": {
                "w0": {"start": "08:10", "durations": ["0min", "5min"]},
                "w1": {"start": "10:10", "durations": ["0min", "5min"]},
            },
            "grid_mode": "fill_order",
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

    assert [p.channel_id for p in simulation.probes] == ["strip"]

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


@pytest.mark.slow  # builds a real Gmsh mesh
def test_simulation_build_reads_an_enabled_anchor_table_and_samples_its_sensor(tmp_path):
    anchor = {"enabled": True, "sensors": {"bay1_30cm": {"r_vertical": 0.1}}}
    soil = _soil_config(tmp_path, "components_anchor.msh", anchor=anchor)
    setup = FieldSetup(
        field=FieldConfig.from_dict({"bay_width": 3.0}), soil=soil, shading=components.ShadingConfig.from_dict()
    )
    sensor = AnchorSensor(key="bay1_30cm", x_offset_cm=0.0, depth_cm=30.0)

    simulation = Simulation.build(setup, anchor_sensors=[sensor], rel_sat_name="Se_components_anchor")

    assert simulation.assimilator.config.enabled is True
    assert simulation.assimilator.config.sensors["bay1_30cm"].r_vertical == 0.1
    assert simulation.assimilator.enabled is True
    assert [p.channel_id for p in simulation.probes] == ["strip", "bay1_30cm"]


# --------------------------------------------------------------------------- ChannelInputs


class _FakeChannel:
    def __init__(self, *, id: str = "fake", connector: bool = True, valid: bool = False, value=None, timestamp=None):
        self.id = id
        self._connector = connector
        self._valid = valid
        self.value = value
        self.timestamp = timestamp

    def has_connector(self) -> bool:
        return self._connector

    def is_valid(self) -> bool:
        return self._valid


class _FakeData:
    """Stands in for ``Component.data``: ``.read`` matches the weather channels list by identity,
    otherwise returns one column per bound, non-empty channel, named by its id."""

    def __init__(self, weather_channels=None, weather_frame=None):
        self.weather_channels = weather_channels if weather_channels is not None else []
        self.weather_frame = weather_frame if weather_frame is not None else pd.DataFrame()
        self.by_channel: dict[int, pd.DataFrame] = {}

    def bind(self, channel, frame: pd.DataFrame) -> None:
        self.by_channel[id(channel)] = frame

    def read(self, channels, start=None, end=None, unique=False) -> pd.DataFrame:
        if channels is self.weather_channels:
            return self.weather_frame.copy()
        columns = [
            self.by_channel[id(c)].iloc[:, 0].rename(c.id)
            for c in channels
            if id(c) in self.by_channel and not self.by_channel[id(c)].empty
        ]
        return pd.concat(columns, axis=1) if columns else pd.DataFrame()


def _fake_field(**overrides) -> SimpleNamespace:
    weather_channels = Channels([])
    weather_frame = _weather_frame(3)
    field = SimpleNamespace(
        name="fieldsim_test",
        data=_FakeData(weather_channels, weather_frame),
        weather=SimpleNamespace(forecast=None),
        soil_simulation=None,
        simulation=None,
        setup=SimpleNamespace(soil=SoilConfig.from_dict({"mesh": {}})),
        _weather_channels=weather_channels,
        _required_weather_keys=components.ETModel.REQUIRED_WEATHER_COLUMNS,
        _irrigation_flow_channel=None,
        _irrigation_state_channel=None,
        _anchor_channels={},
    )
    for key, value in overrides.items():
        setattr(field, key, value)
    return field


def test_channel_inputs_weather_trims_and_validates():
    field = _fake_field()
    inputs = components.ChannelInputs(field)

    start = field.data.weather_frame.index[0]
    end = field.data.weather_frame.index[1]
    got = inputs.weather(start.to_pydatetime(), end.to_pydatetime())
    assert list(got.index) == [end]  # (start, end]

    field.data.weather_frame = field.data.weather_frame.drop(columns=[Weather.GHI])
    got_invalid = inputs.weather(start.to_pydatetime(), end.to_pydatetime())
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

    # measured flow present -> wins, as the raw samples
    measured_frame = pd.DataFrame({"flow": [1.2]}, index=[weather.index[0]])
    data.bind(flow_channel, measured_frame)
    got = inputs.irrigation(start, end)
    assert got.tolist() == [1.2]

    # measured absent -> state x design flow
    data.bind(flow_channel, pd.DataFrame())
    state_frame = pd.DataFrame({"state": [1.0]}, index=[weather.index[0]])
    data.bind(state_channel, state_frame)
    got = inputs.irrigation(start, end)
    assert got.tolist() == pytest.approx([5.0])

    # state wired but not explicit -> never fabricate a forcing from it: empty
    soil.drip.explicit = False
    assert inputs.irrigation(start, end).empty

    # no meter and no state channel at all -> empty (the runner reads it as 0.0)
    field._irrigation_state_channel = None
    assert inputs.irrigation(start, end).empty


def test_channel_inputs_tension_one_column_per_sensor_nan_where_absent():
    s1 = _FakeChannel(id="field.soil_1.water_tension")
    s2 = _FakeChannel(id="field.soil_2.water_tension")
    idx1 = pd.DatetimeIndex(["2026-06-21 08:00", "2026-06-21 09:00"], tz="UTC")
    frame1 = pd.DataFrame({"tension": [-50.0, -55.0]}, index=idx1)
    data = _FakeData()
    data.bind(s1, frame1)
    data.bind(s2, pd.DataFrame())  # sensor 2 produced nothing this span

    field = _fake_field(
        data=data,
        simulation=SimpleNamespace(assimilator=SimpleNamespace(config=parse_anchor_config({}))),
        _anchor_channels={"s1": s1, "s2": s2},
    )
    inputs = components.ChannelInputs(field)

    got = inputs.tension(dt.datetime(2026, 6, 21, 7, tzinfo=UTC), dt.datetime(2026, 6, 21, 10, tzinfo=UTC))
    assert set(got.columns) == {"s1", "s2"}
    assert got["s1"].dropna().tolist() == [-50.0, -55.0]
    assert got["s2"].isna().all()


def test_channel_inputs_forecast_spans_start_to_end_inclusive():
    idx = pd.date_range("2026-06-21 00:00", periods=6, freq="1h", tz="UTC")
    forecast_frame = pd.DataFrame({Weather.GHI: 100.0}, index=idx)
    forecast_sub = SimpleNamespace(
        data=SimpleNamespace(to_frame=lambda unique=False: forecast_frame),
        is_enabled=lambda: True,
    )
    provider = object.__new__(WeatherProvider)
    provider._WeatherProvider__forecast = forecast_sub
    field = _fake_field(weather=provider)
    inputs = components.ChannelInputs(field)

    got = inputs.forecast(dt.datetime(2026, 6, 21, 2, tzinfo=UTC), dt.datetime(2026, 6, 21, 4, tzinfo=UTC))
    assert list(got.index.hour) == [2, 3, 4]


def test_channel_inputs_load_state_from_blob():
    state = SoilState(
        se=np.array([0.5, 0.6]), se_old=np.array([0.5, 0.6]), surface_h={}, at=dt.datetime(2026, 6, 21, tzinfo=UTC)
    )
    blob = state.to_blob()
    channel = _FakeChannel(valid=True, value=blob, timestamp=state.at)
    field = _fake_field(soil_simulation=SimpleNamespace(data={"simulation_state": channel}))
    inputs = components.ChannelInputs(field)

    got = inputs.load_state()
    assert got is not None
    np.testing.assert_allclose(got.se, state.se)
    assert got.at == state.at

    field.soil_simulation = None
    assert inputs.load_state() is None


def test_channel_inputs_load_state_refuses_a_blob_that_needs_pickle():
    buf = io.BytesIO()
    np.savez(buf, rel_sat=np.array([0.5, 0.6]), surface_names=np.array(["a"], dtype=object), surface_h=[0.0])
    at = dt.datetime(2026, 6, 21, tzinfo=UTC)
    channel = _FakeChannel(valid=True, value=buf.getvalue(), timestamp=at)
    inputs = components.ChannelInputs(_fake_field(soil_simulation=SimpleNamespace(data={"simulation_state": channel})))

    with pytest.raises(ValueError, match="allow_pickle"):
        inputs.load_state()


# --------------------------------------------------------------------------- ChannelOutputs


class _RecordingChannel:
    def __init__(self):
        self.calls: list[tuple] = []

    def set(self, ts, value) -> None:
        self.calls.append((ts, value))


def _rows(channel) -> list:
    """Each recorded set as ``(timestamp, value)`` rows; a Series value expands to its rows."""
    return [list(value.items()) if isinstance(value, pd.Series) else [(ts, value)] for ts, value in channel.calls]


class _RecordingData(dict):
    """A dict of pre-registered ``_RecordingChannel``s; an unregistered key raises ``KeyError``
    like a channel that was never added."""


class _NS:
    """A channel namespace with plotting off, as a child whose ``[plot]`` block is disabled."""

    name = "ns"
    plot_config = None
    _last_plot_ts = None
    _plot_strikes = 0

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


def test_channel_outputs_steps_write_diagnostics_probes_and_anchor():
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
    outputs = components.ChannelOutputs(SimpleNamespace(), shading=None, et=None, soil=soil_ns, predictor=None)

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
            "anchor": 0.07,
            "water_total": 999.0,  # not a channel key; must be ignored, not raise
            "walk_ok": 1.0,
        },
        probe_tension={"strip": -120.0, "sensor_only": -80.0},  # sensor_only has no registered channel
    )

    outputs.steps([result])

    assert _rows(soil_ns.data["top_in"]) == [[(ts, 1.0)]]
    assert _rows(soil_ns.data["balance_residual"]) == [[(ts, 7.0)]]
    assert _rows(soil_ns.data["strip"]) == [[(ts, -120.0)]]
    assert _rows(soil_ns.data["anchor"]) == [[(ts, 0.07)]]


def test_channel_outputs_steps_set_each_channel_once_per_chunk_with_every_row():
    """Two rows of one chunk reach each channel as ONE set whose Series carries
    both rows, stamped with the first; a NaN row is left out."""
    soil_ns = _NS(["top_in", "runoff", "strip"])
    outputs = components.ChannelOutputs(SimpleNamespace(), shading=None, et=None, soil=soil_ns, predictor=None)
    t0, t1 = pd.Timestamp("2026-06-21 09:00", tz="UTC"), pd.Timestamp("2026-06-21 10:00", tz="UTC")
    results = [
        StepResult(
            state=SoilState(se=np.array([0.5]), se_old=np.array([0.5]), surface_h={}, at=t.to_pydatetime()),
            diagnostics={"top_in": value, "runoff": runoff},
            probe_tension={"strip": -100.0 - value},
        )
        for t, value, runoff in ((t0, 1.0, 0.0), (t1, 2.0, float("nan")))
    ]

    outputs.steps(results)

    assert [ts for ts, _ in soil_ns.data["top_in"].calls] == [t0]
    assert _rows(soil_ns.data["top_in"]) == [[(t0, 1.0), (t1, 2.0)]]
    assert _rows(soil_ns.data["strip"]) == [[(t0, -101.0), (t1, -102.0)]]
    assert _rows(soil_ns.data["runoff"]) == [[(t0, 0.0)]]


def test_channel_outputs_steps_write_the_stall_and_failure_tallies():
    """The runner's per-tick tallies ride the row's diagnostics, so the row a
    healing tick commits carries the count that preceded it."""
    soil_ns = _NS(["weather_stall", "tick_failures"])
    outputs = components.ChannelOutputs(SimpleNamespace(), shading=None, et=None, soil=soil_ns, predictor=None)

    ts = dt.datetime(2026, 6, 21, 9, tzinfo=UTC)
    outputs.steps(
        [
            StepResult(
                state=SoilState(se=np.array([0.5]), se_old=np.array([0.5]), surface_h={}, at=ts),
                diagnostics={"weather_stall": 3.0, "tick_failures": 2.0},
            )
        ]
    )

    assert _rows(soil_ns.data["weather_stall"]) == [[(ts, 3.0)]]
    assert _rows(soil_ns.data["tick_failures"]) == [[(ts, 2.0)]]


def test_channel_outputs_save_state_writes_simulation_state():
    soil_ns = _NS(["simulation_state"])
    outputs = components.ChannelOutputs(SimpleNamespace(), shading=None, et=None, soil=soil_ns, predictor=None)

    state = SoilState(se=np.array([0.4]), se_old=np.array([0.4]), surface_h={}, at=dt.datetime(2026, 6, 21, tzinfo=UTC))
    outputs.save_state(state)

    assert len(soil_ns.data["simulation_state"].calls) == 1
    ts, blob = soil_ns.data["simulation_state"].calls[0]
    assert ts == state.at
    assert blob == state.to_blob()


def test_channel_outputs_chain_writes_one_series_per_channel_and_the_last_segment_ghi():
    shading_ns = _NS(["shading_factor", "shading_progress_image"])
    et_ns = _NS(list(components._ET_CHANNEL_KEYS))
    field_ns = _NS(["seg_ghi", "lai", "roughness", "plant_height", "ndvi"])
    field_ns.simulation = SimpleNamespace(engine=SimpleNamespace(top_segment_names=["seg1", "seg2"]))
    outputs = components.ChannelOutputs(field_ns, shading=shading_ns, et=et_ns, soil=None, predictor=None)

    idx = pd.date_range("2026-06-21 08:00", periods=2, freq="1h", tz="UTC")
    shading_df = pd.DataFrame(
        {
            "shading_factor": [1.0, 0.65],
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

    outputs.chain(idx[-1].to_pydatetime(), ChainResult(shading=shading_df, evapotranspiration=et_df))

    assert _rows(shading_ns.data["shading_factor"]) == [[(idx[0], pytest.approx(1.0)), (idx[1], pytest.approx(0.65))]]
    assert shading_ns.data["shading_progress_image"].calls == []  # plotting off: no frame
    assert field_ns.data["seg_ghi"].calls == [(idx[-1], [400.0, 300.0])]
    assert _rows(et_ns.data["evapotranspiration"]) == [[(idx[0], 0.1), (idx[1], 0.2)]]
    assert _rows(field_ns.data["lai"]) == [[(idx[0], 1.0), (idx[1], 1.0)]]


class _FakePredictorData(dict):
    """Auto-creates a recording channel per key with ``id`` equal to the key (the key -> id rename is the identity)
    and ``logger`` None (the connector hop falls back to the id lookups)."""

    def __missing__(self, key):
        channel = _RecordingChannel()
        channel.id = key
        channel.logger = None
        self[key] = channel
        return channel


class _FakePredictor:
    """The surface ``ForecastTablePublisher`` reads off a predictor."""

    _HEADER_FORECAST_ID_KEY = components.SoilPredictor._HEADER_FORECAST_ID_KEY
    _HEADER_IS_RECOMMENDED_KEY = components.SoilPredictor._HEADER_IS_RECOMMENDED_KEY
    _HEADER_TOTAL_MIN_KEY = components.SoilPredictor._HEADER_TOTAL_MIN_KEY
    _HEADER_WEATHER_CREATION_KEY = components.SoilPredictor._HEADER_WEATHER_CREATION_KEY
    _IRRIGATION_STATE_KEY = components.SoilPredictor._IRRIGATION_STATE_KEY
    _IRRIGATION_TIMESTAMP_CREATION_KEY = components.SoilPredictor._IRRIGATION_TIMESTAMP_CREATION_KEY
    _IMAGE_KEY = components.SoilPredictor._IMAGE_KEY
    _IMAGE_TIMESTAMP_CREATION_KEY = components.SoilPredictor._IMAGE_TIMESTAMP_CREATION_KEY
    _header_window_min_keys = ()
    _header_window_start_keys = ()
    _traj_channel_keys = {"strip": "traj_strip"}
    _detail_creation_keys = {}
    _detail_forecast_id_keys = {}

    tables = components.SoilPredictor.tables
    _write_direct_frame = components.SoilPredictor._write_direct_frame
    _resolve_logger_connector = components.SoilPredictor._resolve_logger_connector
    _bump_write_failure = components.SoilPredictor._bump_write_failure

    def __init__(self, connector):
        self.name = "fieldsim_test.soil_predictor"
        self.path = ["fieldsim_test", "soil_predictor"]
        self._logger_id = "db"
        self._write_failures = None
        self.connectors = SimpleNamespace(context={"db": connector})
        self.data = _FakePredictorData()


def test_channel_outputs_plan_writes_four_tables_best_effort():
    connector = _FakeConnector(raise_on_column="irrigation_state")
    predictor = _FakePredictor(connector)
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

    assert len(connector.written) == 3  # header, detail, image; irrigation raised and was skipped
    written_columns = [set(f.columns) for f in connector.written]
    assert {"forecast_id"} in written_columns
    assert {"traj_strip"} in written_columns
    assert {"predict_image"} in written_columns

    failures = components._WRITE_FAILURE_CHANNELS["irrigation table"]
    assert predictor._write_failures == {"irrigation table": 1}
    assert [v for _, v in predictor.data[failures].calls] == [1.0]


def test_channel_outputs_plan_skips_a_connector_without_write():
    predictor = _FakePredictor(connector=object())
    field = SimpleNamespace(setup=SimpleNamespace(planner=SimpleNamespace(logger="db")))
    outputs = components.ChannelOutputs(field, shading=None, et=None, soil=None, predictor=predictor)

    ts = pd.Timestamp("2026-06-21 08:00", tz="UTC")
    plan = Plan(
        chosen=None,
        trajectories={},
        header=pd.DataFrame({"forecast_id": [0]}, index=[ts]),
        detail=pd.DataFrame(),
        irrigation=pd.DataFrame(),
        image=None,
    )

    outputs.plan(plan)  # skipped, not raised

    assert predictor._write_failures is None


# --------------------------------------------------------------------------- FieldSimulation.simulate


class _OfflineSession:
    """What ``Simulation.build`` returns in the simulate tests: each ``run``
    advances one hour per weather row, counting on from its own last row."""

    def __init__(self):
        self.rows = 0
        self.engine = SimpleNamespace(top_segment_names=["seg1", "seg2"])

    def run(self, weather, irrigation_lpm):
        results = []
        for ts in weather.index:
            self.rows += 1
            state = SoilState(se=np.zeros(1), se_old=np.zeros(1), surface_h={}, at=ts.to_pydatetime())
            results.append(StepResult(state=state, diagnostics={"top_in": float(self.rows), "water_total": 9.0}))
        shading = pd.DataFrame(
            {"ghi_seg1": np.arange(len(weather), dtype=float) + 1.0, "ghi_seg2": 10.0}, index=weather.index
        )
        return results, ChainResult(shading=shading, evapotranspiration=pd.DataFrame())


def _offline_field(monkeypatch, segments=False):
    builds: list = []

    def build(setup, **kwargs):
        builds.append(_OfflineSession())
        return builds[-1]

    monkeypatch.setattr(components.Simulation, "build", staticmethod(build))
    field = object.__new__(components.FieldSimulation)
    field._name = "field_sim"
    field.setup = object()
    field.simulation = SimpleNamespace(run=lambda *a, **k: pytest.fail("simulate must not run the live session"))
    field.soil_simulation = SimpleNamespace(data={"top_in": SimpleNamespace(id="agri.field_1.soil.top_in")})
    data = {"seg_ghi": SimpleNamespace(id="agri.field_1.seg_ghi")} if segments else {}
    monkeypatch.setattr(components.FieldSimulation, "data", property(lambda self: data))
    return field, builds


def test_simulate_returns_the_soil_channel_ids_and_drops_the_rest(monkeypatch):
    field, builds = _offline_field(monkeypatch)

    frame = field.simulate(_weather_frame(2))

    assert list(frame.columns) == ["agri.field_1.soil.top_in"]
    assert frame["agri.field_1.soil.top_in"].tolist() == [1.0, 2.0]
    assert len(builds) == 1


def test_simulate_continues_its_session_only_for_a_chained_slice(monkeypatch):
    field, builds = _offline_field(monkeypatch)
    first, second = _weather_frame(2), _weather_frame(4).iloc[2:]

    chained = field.simulate(second, prior=field.simulate(first))
    fresh = field.simulate(first)

    assert chained.iloc[:, 0].tolist() == [3.0, 4.0]
    assert fresh.iloc[:, 0].tolist() == [1.0, 2.0]
    assert len(builds) == 2


def test_simulate_adds_the_segment_ghi_lists_aligned_to_the_soil_rows(monkeypatch):
    field, _ = _offline_field(monkeypatch, segments=True)

    frame = field.simulate(_weather_frame(2))

    assert list(frame.columns) == ["agri.field_1.soil.top_in", "agri.field_1.seg_ghi"]
    assert frame["agri.field_1.seg_ghi"].tolist() == [[1.0, 10.0], [2.0, 10.0]]
    assert frame.index.equals(pd.DatetimeIndex(_weather_frame(2).index))
    assert frame["agri.field_1.soil.top_in"].notna().all()


def test_simulate_gives_a_segment_missing_from_the_shading_frame_zero(monkeypatch):
    field, builds = _offline_field(monkeypatch, segments=True)
    field.simulate(_weather_frame(1))
    builds[0].engine.top_segment_names = ["seg1", "seg3", "seg2"]

    frame = field.simulate(_weather_frame(2), prior=pd.DataFrame())

    assert frame["agri.field_1.seg_ghi"].tolist() == [[1.0, 0.0, 10.0], [2.0, 0.0, 10.0]]


def test_simulate_without_the_segment_channel_returns_only_the_soil_columns(monkeypatch):
    field, _ = _offline_field(monkeypatch, segments=False)

    frame = field.simulate(_weather_frame(2))

    assert list(frame.columns) == ["agri.field_1.soil.top_in"]


def test_simulate_chained_slice_carries_the_segment_ghi_column(monkeypatch):
    field, _ = _offline_field(monkeypatch, segments=True)
    first, second = _weather_frame(2), _weather_frame(4).iloc[2:]

    chained = field.simulate(second, prior=field.simulate(first))

    assert list(chained.columns) == ["agri.field_1.soil.top_in", "agri.field_1.seg_ghi"]
    assert chained["agri.field_1.seg_ghi"].tolist() == [[1.0, 10.0], [2.0, 10.0]]
    assert chained.index.equals(pd.DatetimeIndex(second.index))


# --------------------------------------------------------------------------- CHANNELS


def _declared(namespace) -> set:
    return {str(c) for c in (*namespace.CHANNELS, *namespace.PLOT_CHANNELS)}


def test_ground_shading_channels_are_the_canonical_keys():
    assert components.GroundShading.CHANNELS[0].key == "shading_factor"
    assert [c.key for c in components.GroundShading.PLOT_CHANNELS] == ["shading_progress_image"]
    assert _declared(components.GroundShading) == {"shading_factor", "shading_progress_image", "plot_strikes"}


def test_evapotranspiration_channels_are_the_canonical_keys():
    assert [c.key for c in components.Evapotranspiration.CHANNELS] == [
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
    ]
    assert components.Evapotranspiration.PLOT_CHANNELS == ()


def test_soil_simulation_channels_are_the_canonical_keys():
    expected = {
        "simulation_state",
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
        "plot_strikes",
        "soil_progress_image",
    }
    assert _declared(components.SoilSimulation) == expected
    assert [c.key for c in components.SoilSimulation.PLOT_CHANNELS] == ["soil_progress_image"]
    assert all(isinstance(c, Constant) for c in components.SoilSimulation.CHANNELS)


def test_field_level_channels_are_the_canonical_keys():
    assert [c.key for c in components._VEGETATION_CHANNELS] == [
        "temp_ground",
        "lai",
        "roughness",
        "plant_height",
        "ndvi",
    ]
    assert [c.key for c in components._SEGMENT_CHANNELS] == [
        "seg_ghi",
        "seg_evapotranspiration",
        "seg_temp_ground",
    ]
    assert components._SOIL_DIAGNOSTIC_CHANNELS[0].key == "top_in"


def test_soil_predictor_channels_are_the_write_failure_counters():
    assert components.SoilPredictor.CHANNELS == tuple(components._WRITE_FAILURE_CHANNELS.values())
    assert components.SoilPredictor._HEADER_TABLE_NAME == "agri_field_forecast"


class _RecordingAdds:
    """Stands in for ``Component.data`` during channel registration."""

    def __init__(self):
        self.calls: list[tuple] = []

    def add(self, key, **configs) -> None:
        self.calls.append((key, configs))


def _added(namespace, **kwargs) -> dict:
    data = _RecordingAdds()
    namespace.register_channels(data, **kwargs)
    assert all(isinstance(key, Constant) for key, _ in data.calls)
    return {str(key): configs for key, configs in data.calls}


def test_register_channels_passes_constants_with_the_live_kwargs():
    soil = _added(components.SoilSimulation)
    assert soil["top_in"] == {"aggregate": "mean", "logger": {"enabled": True}}
    assert soil["weather_stall"] == {"aggregate": "mean", "logger": {"enabled": True}}
    assert soil["simulation_state"] == {"aggregate": "last", "logger": {"enabled": True, "column": "state"}}
    assert soil["soil_progress_image"] == {"aggregate": "last", "logger": {"enabled": True, "column": "image"}}
    assert soil["plot_strikes"] == {"aggregate": "last", "logger": {"enabled": False}}

    shading = _added(components.GroundShading)
    assert shading["shading_factor"] == {"aggregate": "mean", "logger": {"enabled": False}}
    assert shading["shading_progress_image"] == {"aggregate": "last", "logger": {"enabled": True}}

    et = _added(components.Evapotranspiration)
    assert all(configs == {"aggregate": "mean", "logger": {"enabled": False}} for configs in et.values())

    predictor = _added(components.SoilPredictor)
    assert predictor["header_write_failures"] == {"aggregate": "last", "logger": {"enabled": False}}


def test_register_channels_skips_the_image_channels_when_plot_is_disabled():
    for namespace in (components.GroundShading, components.SoilSimulation):
        added = _added(namespace, plot_enabled=False)
        assert set(added) == {str(c) for c in namespace.CHANNELS}
        assert set(added).isdisjoint({str(c) for c in namespace.PLOT_CHANNELS})


# --------------------------------------------------------------------------- [plot]


def _plot_config(tmp_path, **child):
    return components.ChannelNamespace._plot_config(load_configs(tmp_path, "child.conf", **child))


def test_plot_config_reads_the_childs_own_plot_block(tmp_path):
    config = _plot_config(tmp_path, plot={"enabled": True, "interval": "30min", "disable_after_failures": 2})

    assert config.interval == pd.Timedelta(minutes=30)
    assert config.disable_after_failures == 2


def test_plot_config_is_none_when_the_block_is_disabled(tmp_path):
    assert _plot_config(tmp_path, plot={"enabled": False, "interval": "30min"}) is None


def test_plot_config_defaults_to_hourly_frames_without_a_block(tmp_path):
    config = _plot_config(tmp_path)

    assert config.interval == pd.Timedelta(hours=1)
    assert config.disable_after_failures == 3


def test_plot_config_rejects_an_unknown_key(tmp_path):
    with pytest.raises(ConfigurationError, match="unknown configuration keys"):
        _plot_config(tmp_path, plot={"every": "30min"})
