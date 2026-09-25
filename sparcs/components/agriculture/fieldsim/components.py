# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.fieldsim.components
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The lories layer: the only module in this package that imports lories
components and channels.

``FieldSimulation`` configures its own section, lets each ``.d`` child
configure its own, bundles them into a frozen ``FieldSetup``, builds the
``Simulation``, the runner and the ``Ticker``, and exposes the ``Snapshot``
for dash. Each child is a ``ChannelNamespace``: a TYPE, the lories
``Constant``s it registers, and the ``Config`` class its file is validated
against. The constants are the live classes' own objects, imported rather
than redeclared; only channels the live classes register by bare key are
declared here, under ``context="fieldsim"``.

Not registered as a component type; the live ``simulation.FieldSimulation``
keeps the ``field_simulation`` type.
"""

from __future__ import annotations

import datetime as dt
import logging
from copy import deepcopy
from dataclasses import replace
from typing import Any, ClassVar, Mapping, Optional, Sequence, Type, TypeVar

import pandas as pd
from lories import Constant
from lories.components import Component
from lories.components.weather import Weather
from lories.core import Configurations, ConfigurationUnavailableError
from lories.data import Channels
from sparcs.components.agriculture.irrigation import Irrigation
from sparcs.components.agriculture.simulation import Evapotranspiration as LiveEvapotranspiration
from sparcs.components.agriculture.simulation import FieldSimulation as LiveFieldSimulation
from sparcs.components.agriculture.simulation import GroundShading as LiveGroundShading
from sparcs.components.agriculture.simulation import SoilPredictor as LiveSoilPredictor
from sparcs.components.agriculture.simulation import SoilSimulation as LiveSoilSimulation
from sparcs.components.agriculture.simulation._anchor_runtime import _walk_components
from sparcs.components.agriculture.simulation._predictor_tables import ForecastTablePublisher

from .core.anchor import AnchorSensor
from .core.assimilator import parse_anchor_config
from .core.config import Config, FieldConfig, FieldSetup, PlannerConfig, PlotConfig, SoilConfig
from .core.evapotranspiration import ETModel
from .core.shading import ShadingConfig
from .core.simulation import Simulation
from .core.state import ChainResult, Plan, Snapshot, SoilState, StepResult
from .runtime.ports import InputKey
from .runtime.runner import FieldRunner
from .runtime.scheduler import Ticker

logger = logging.getLogger(__name__)

_C = TypeVar("_C", bound="ChannelNamespace")

# Weather keys the chain fills with a default instead of requiring a feed.
_OPTIONAL_WEATHER_KEYS: frozenset = frozenset({Weather.CLEAR_SKY_INDEX, Weather.HUMIDITY_REL})

# The live components register these by bare key, so they have no live Constant.
PLOT_STRIKES = Constant(float, "plot_strikes", "Plot Strikes", context="fieldsim")

# Keyed by the table label the live publisher passes to _bump_write_failure.
_WRITE_FAILURE_CHANNELS: Mapping[str, Constant] = {
    "header table": Constant(float, "header_write_failures", "Header Write Failures", context="fieldsim"),
    "detail table": Constant(float, "detail_write_failures", "Detail Write Failures", context="fieldsim"),
    "irrigation table": Constant(float, "irrigation_write_failures", "Irrigation Write Failures", context="fieldsim"),
    "image table": Constant(float, "image_write_failures", "Image Write Failures", context="fieldsim"),
}

# ``data.add`` kwargs per channel shape; ``data.add`` expands the rest from
# ``Constant.to_dict()``.
_MEAN_LOGGED: Mapping[str, Any] = {"aggregate": "mean", "logger": {"enabled": True}}
_MEAN_MEMORY: Mapping[str, Any] = {"aggregate": "mean", "logger": {"enabled": False}}
_LAST_MEMORY: Mapping[str, Any] = {"aggregate": "last", "logger": {"enabled": False}}
# A list[float] bundle carries no aggregate at all.
_BUNDLE_MEMORY: Mapping[str, Any] = {"logger": {"enabled": False}}
_STATE_BLOB: Mapping[str, Any] = {"aggregate": "last", "logger": {"enabled": True, "column": "state"}}
_SOIL_IMAGE: Mapping[str, Any] = {"aggregate": "last", "logger": {"enabled": True, "column": "image"}}
_SHADING_IMAGE: Mapping[str, Any] = {"aggregate": "last", "logger": {"enabled": True}}


class ChannelNamespace(Component):
    """A Component that exists to own channels under its id and to validate
    its ``.d`` file against one ``Config`` class. No logic."""

    CHANNELS: ClassVar[Sequence[Constant]] = ()
    # Registered only when the child's own [plot] block is enabled.
    PLOT_CHANNELS: ClassVar[Sequence[Constant]] = ()
    CHANNEL_CONFIGS: ClassVar[Mapping[str, Mapping[str, Any]]] = {}
    DEFAULT_CONFIGS: ClassVar[Mapping[str, Any]] = _MEAN_MEMORY
    CONFIG: ClassVar[Optional[Type[Config]]] = None

    config: Optional[Config] = None
    plot_enabled: bool = True

    def configure(self, configs: Configurations) -> None:
        super().configure(configs)
        self.plot_enabled = self._plot_enabled(configs)
        if self.CONFIG is not None:
            self.config = self.CONFIG()
            # A copy: the section was already configured against this object.
            self.config.configure(configs.copy())
        self.register_channels(self.data, plot_enabled=self.plot_enabled)

    @classmethod
    def register_channels(cls, data: Any, *, plot_enabled: bool = True) -> None:
        """One ``data.add(constant, **kwargs)`` per declared channel."""
        constants = list(cls.CHANNELS)
        if plot_enabled:
            constants.extend(cls.PLOT_CHANNELS)
        for constant in constants:
            data.add(constant, **deepcopy(dict(cls.CHANNEL_CONFIGS.get(constant, cls.DEFAULT_CONFIGS))))

    @staticmethod
    def _plot_enabled(configs: Configurations) -> bool:
        return configs.get_member("plot", defaults={}, ensure_exists=True).get_bool("enabled", default=True)

    @classmethod
    def schema(cls) -> dict[str, Any]:
        return cls.CONFIG.schema() if cls.CONFIG is not None else {}


class GroundShading(ChannelNamespace):
    TYPE: str = "ground_shading"
    CONFIG = ShadingConfig
    CHANNELS = (LiveGroundShading.SHADING_FACTOR, PLOT_STRIKES)
    PLOT_CHANNELS = (LiveGroundShading.SHADING_PROGRESS_IMAGE,)
    CHANNEL_CONFIGS = {
        PLOT_STRIKES: _LAST_MEMORY,
        LiveGroundShading.SHADING_PROGRESS_IMAGE: _SHADING_IMAGE,
    }


class Evapotranspiration(ChannelNamespace):
    TYPE: str = "evapotranspiration"
    CONFIG = None  # no keys of its own today; its .d file carries only channels
    CHANNELS = tuple(LiveEvapotranspiration.CHANNELS)


class SoilSimulation(ChannelNamespace):
    TYPE: str = "soil_simulation"
    CONFIG = SoilConfig
    CHANNELS = (
        LiveSoilSimulation.SIMULATION_STATE,
        LiveSoilSimulation.WATER_TOP_IN,
        LiveSoilSimulation.WATER_TOP_OUT,
        LiveSoilSimulation.WATER_BOTTOM,
        LiveSoilSimulation.WATER_TRANSP,
        LiveSoilSimulation.WATER_RUNOFF,
        LiveSoilSimulation.WATER_DEMAND_UNMET,
        LiveSoilSimulation.WATER_BALANCE_RESIDUAL,
        LiveSoilSimulation.WATER_ANCHOR,
        LiveSoilSimulation.WALK_SKIPPED_S,
        LiveSoilSimulation.WALK_RETRIES,
        LiveSoilSimulation.WEATHER_STALL,
        LiveSoilSimulation.TICK_FAILURES,
        PLOT_STRIKES,
    )
    PLOT_CHANNELS = (LiveSoilSimulation.SOIL_PROGRESS_IMAGE,)
    DEFAULT_CONFIGS = _MEAN_LOGGED
    CHANNEL_CONFIGS = {
        LiveSoilSimulation.SIMULATION_STATE: _STATE_BLOB,
        LiveSoilSimulation.SOIL_PROGRESS_IMAGE: _SOIL_IMAGE,
        PLOT_STRIKES: _LAST_MEMORY,
    }

    def register_probe(self, probe: Any) -> None:
        """One float channel per resolved config probe. Sensor-derived probes
        stay sample-only."""
        self.data.add(
            probe.channel_id,
            type=float,
            name=probe.name,
            unit="hPa",
            aggregate="mean",
            logger={"enabled": True, "table": "agri_soil_simulation", "column": "water_tension"},
        )


class SoilPredictor(ChannelNamespace):
    """The forecast tables' schema and write path are the live
    ``ForecastTablePublisher``'s; this class supplies what it reads."""

    TYPE: str = "soil_predictor"
    CONFIG = PlannerConfig
    CHANNELS = tuple(_WRITE_FAILURE_CHANNELS.values())
    DEFAULT_CONFIGS = _LAST_MEMORY

    _HEADER_TABLE_NAME = LiveSoilPredictor._HEADER_TABLE_NAME
    _HEADER_FORECAST_ID_KEY = LiveSoilPredictor._HEADER_FORECAST_ID_KEY
    _HEADER_IS_RECOMMENDED_KEY = LiveSoilPredictor._HEADER_IS_RECOMMENDED_KEY
    _HEADER_TOTAL_MIN_KEY = LiveSoilPredictor._HEADER_TOTAL_MIN_KEY
    _HEADER_WEATHER_CREATION_KEY = LiveSoilPredictor._HEADER_WEATHER_CREATION_KEY
    _DETAIL_TABLE_NAME = LiveSoilPredictor._DETAIL_TABLE_NAME
    _DETAIL_TIMESTAMP_CREATION_SUFFIX = LiveSoilPredictor._DETAIL_TIMESTAMP_CREATION_SUFFIX
    _DETAIL_FORECAST_ID_SUFFIX = LiveSoilPredictor._DETAIL_FORECAST_ID_SUFFIX
    _IRRIGATION_TABLE_NAME = LiveSoilPredictor._IRRIGATION_TABLE_NAME
    _IRRIGATION_STATE_KEY = LiveSoilPredictor._IRRIGATION_STATE_KEY
    _IRRIGATION_TIMESTAMP_CREATION_KEY = LiveSoilPredictor._IRRIGATION_TIMESTAMP_CREATION_KEY
    _IMAGE_TABLE_NAME = LiveSoilPredictor._IMAGE_TABLE_NAME
    _IMAGE_KEY = LiveSoilPredictor._IMAGE_KEY
    _IMAGE_COLUMN = LiveSoilPredictor._IMAGE_COLUMN
    _IMAGE_TIMESTAMP_CREATION_KEY = LiveSoilPredictor._IMAGE_TIMESTAMP_CREATION_KEY

    _logger_id: Optional[str] = None
    _max_windows: int = 0
    _write_failures: Optional[dict[str, int]] = None
    _header_window_min_keys: Sequence[str] = ()
    _header_window_start_keys: Sequence[str] = ()
    _traj_channel_keys: Mapping[str, str] = {}
    _detail_creation_keys: Mapping[str, str] = {}
    _detail_forecast_id_keys: Mapping[str, str] = {}

    def configure(self, configs: Configurations) -> None:
        super().configure(configs)
        if self.config is not None:
            self._logger_id = self.config.logger
            self._max_windows = self.config.max_windows

    def tables(self) -> ForecastTablePublisher:
        return ForecastTablePublisher(self)

    def register_forecast_tables(self, soil_configs: Configurations, probes: list) -> None:
        """Register the four persisted forecast tables' channels."""
        if self._logger_id is None:
            return
        tables = self.tables()
        self._header_window_min_keys, self._header_window_start_keys = tables.register_header_channels()
        identities = tables.resolve_probe_identities(soil_configs, probes)
        detail = tables.register_detail_channels(probes, identities)
        self._traj_channel_keys, self._detail_creation_keys, self._detail_forecast_id_keys = detail
        tables.register_irrigation_channels()
        if self.plot_enabled:
            tables.register_image_channels()

    def _resolve_logger_connector(self, logger_id: str) -> Optional[Any]:
        return self.tables().resolve_logger_connector(logger_id)

    def _write_direct_frame(self, frame: pd.DataFrame, id_by_key_fn: Any, table_label: str) -> None:
        self.tables().write_direct_frame(frame, id_by_key_fn, table_label)

    def _logger_connector_from_channel(self) -> Optional[Any]:
        """The connector the header's ``forecast_id`` channel already resolved
        against the root context; ``None`` degrades to the id fallbacks."""
        try:
            return self.data[self._HEADER_FORECAST_ID_KEY].logger._get_registrator()
        except Exception:  # noqa: BLE001
            return None

    def _bump_write_failure(self, table_label: str) -> None:
        if self._write_failures is None:
            self._write_failures = {}
        count = self._write_failures.get(table_label, 0) + 1
        self._write_failures[table_label] = count
        constant = _WRITE_FAILURE_CHANNELS.get(table_label)
        if constant is None:
            return
        try:
            self.data[constant].set(pd.Timestamp.now(tz="UTC"), float(count))
        except Exception:  # noqa: BLE001
            logger.debug("%s: write-failure channel '%s' unavailable; count=%d.", self.name, constant, count)


# --------------------------------------------------------------------------- field-level channels

# TEMP_GROUND is registered but not written: only Evapotranspiration.evaluate's
# per-segment publish path derives it, which ETModel does not reproduce.
_VEGETATION_CHANNELS = tuple(LiveFieldSimulation.VEGETATION_CHANNELS)
_VEGETATION_WRITE_CHANNELS = (
    LiveFieldSimulation.LAI,
    LiveFieldSimulation.ROUGHNESS,
    LiveFieldSimulation.PLANT_HEIGHT,
    LiveFieldSimulation.NDVI,
)

# Only SEG_GHI is derivable from ChainResult; the other two need seg_et.
_SEGMENT_CHANNELS = tuple(LiveFieldSimulation.SEGMENT_CHANNELS)

_ET_CHANNEL_KEYS = tuple(str(c) for c in LiveEvapotranspiration.CHANNELS)
_SOIL_DIAGNOSTIC_CHANNELS = (
    LiveSoilSimulation.WATER_TOP_IN,
    LiveSoilSimulation.WATER_TOP_OUT,
    LiveSoilSimulation.WATER_BOTTOM,
    LiveSoilSimulation.WATER_TRANSP,
    LiveSoilSimulation.WATER_RUNOFF,
    LiveSoilSimulation.WATER_DEMAND_UNMET,
    LiveSoilSimulation.WATER_BALANCE_RESIDUAL,
    LiveSoilSimulation.WALK_SKIPPED_S,
    LiveSoilSimulation.WALK_RETRIES,
)


class ChannelInputs:
    """``Inputs`` over lories connector reads. Every key is one ranged read;
    no decisions live here."""

    _FLOW_LOOKBACK: dt.timedelta = dt.timedelta(days=1)

    def __init__(self, field: "FieldSimulation") -> None:
        self.field = field
        self._last_invalid_weather_columns: list[str] = []

    def read(self, key: InputKey, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
        if key is InputKey.WEATHER:
            return self._read_weather(start, end)
        if key is InputKey.IRRIGATION:
            return self._read_irrigation(start, end)
        if key is InputKey.TENSION:
            return self._read_tension(start, end)
        if key is InputKey.FORECAST:
            return self._read_forecast(start, end)
        raise ValueError(f"unknown input key {key!r}")

    def load_state(self) -> Optional[SoilState]:
        soil = self.field.soil
        if soil is None:
            return None
        try:
            channel = soil.data[LiveSoilSimulation.SIMULATION_STATE]
        except Exception:  # noqa: BLE001
            return None
        if not channel.is_valid():
            return None
        blob = channel.value
        if not blob:
            return None
        return SoilState.from_blob(blob, at=channel.timestamp)

    # -- WEATHER -----------------------------------------------------------

    def _read_weather(self, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
        field = self.field
        frame = field.data.read(field._weather_channels, start=start, end=end, unique=True)
        frame = self._trim_span(frame, start, end)
        if not self._weather_frame_valid(frame):
            return frame.iloc[0:0]
        return frame

    def _trim_span(self, frame: pd.DataFrame, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
        if frame.empty:
            return frame
        frame = frame.rename(columns=self.field._evapo_rename)
        return frame.loc[(frame.index > start) & (frame.index <= end)]

    def _weather_frame_valid(self, frame: pd.DataFrame) -> bool:
        if frame.empty:
            return False
        missing = [
            k
            for k in self.field._required_weather_keys
            if k not in _OPTIONAL_WEATHER_KEYS and (k not in frame.columns or frame[k].isna().all())
        ]
        if missing:
            self._last_invalid_weather_columns = missing
            return False
        return True

    # -- IRRIGATION ----------------------------------------------------------

    def _read_irrigation(self, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
        weather = self._read_weather(start, end)
        index = weather.index
        column = LiveSoilSimulation.IRRIGATION_FLOW_LPM
        if len(index) == 0:
            return pd.DataFrame(columns=[column])
        field = self.field
        measured = self._read_measured_flow(start, end, index)
        if measured is not None:
            series = measured
        elif field._irrigation_state_channel is not None and field.setup.soil.drip.explicit:
            series = self._read_state_span(start, end, index) * field.setup.soil.drip.design_flow_lpm
        else:
            # Unwired: not watering, never a fabricated forcing.
            series = pd.Series(0.0, index=index)
        return series.to_frame(column)

    def _read_measured_flow(self, start: dt.datetime, end: dt.datetime, index: pd.Index) -> Optional[pd.Series]:
        field = self.field
        if field._irrigation_flow_channel is None:
            return None
        frame = field.data.read(
            Channels([field._irrigation_flow_channel]), start=start - self._FLOW_LOOKBACK, end=end, unique=True
        )
        if frame.empty or frame.iloc[:, 0].isna().all():
            return None
        return self._align_flow(frame, index)

    def _read_state_span(self, start: dt.datetime, end: dt.datetime, index: pd.Index) -> pd.Series:
        field = self.field
        if field._irrigation_state_channel is None:
            return pd.Series(0.0, index=index)
        frame = field.data.read(
            Channels([field._irrigation_state_channel]), start=start - self._FLOW_LOOKBACK, end=end, unique=True
        )
        return self._align_flow(frame, index)

    @staticmethod
    def _align_flow(frame: pd.DataFrame, index: pd.Index) -> pd.Series:
        if frame.empty:
            return pd.Series(0.0, index=index)
        series = frame.iloc[:, 0].sort_index()
        series = series[~series.index.duplicated(keep="last")]
        aligned = series.reindex(index, method="ffill")
        return aligned.fillna(0.0).astype(float)

    # -- TENSION ---------------------------------------------------------------

    def _read_tension(self, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
        field = self.field
        sensors = getattr(field, "_anchor_sensors", ())
        if not sensors:
            return pd.DataFrame()
        assimilator = getattr(getattr(field, "simulation", None), "assimilator", None)
        cfg = getattr(assimilator, "config", None)
        lookback = (
            max((cfg.sensor_staleness(s.key) for s in sensors), default=dt.timedelta(0)) if cfg else dt.timedelta(0)
        )
        read_start = start - lookback

        # One column per sensor even when its read is empty, so an absent
        # reading is NaN once another sensor's rows widen the shared index.
        series_by_key: dict[str, pd.Series] = {}
        for sensor in sensors:
            series_by_key[sensor.key] = pd.Series(dtype=float)
            data = getattr(field, "_anchor_data", {}).get(sensor.key)
            channel = getattr(field, "_anchor_channels", {}).get(sensor.key)
            if data is None or channel is None:
                continue
            try:
                frame = data.read(Channels([channel]), start=read_start, end=end, unique=True)
            except Exception:  # noqa: BLE001
                logger.exception(
                    "%s: anchor history read failed for %s", getattr(field, "name", "fieldsim"), sensor.key
                )
                continue
            if frame is None or frame.empty:
                continue
            series = frame.iloc[:, 0].dropna().sort_index()
            series = series[~series.index.duplicated(keep="last")]
            if not series.empty:
                series_by_key[sensor.key] = series
        return pd.concat(series_by_key, axis=1)

    # -- FORECAST -----------------------------------------------------------

    def _read_forecast(self, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
        field = self.field
        weather = field.weather
        forecast_sub = getattr(weather, "forecast", None) if weather is not None else None
        if forecast_sub is None or not forecast_sub.is_enabled():
            return pd.DataFrame()
        try:
            frame = forecast_sub.data.to_frame(unique=False)
        except Exception as e:  # noqa: BLE001
            logger.warning("%s: forecast read failed: %s", getattr(field, "name", "fieldsim"), e)
            return pd.DataFrame()
        if frame.empty:
            return frame
        return frame.loc[(frame.index >= start) & (frame.index < end)]


class ChannelOutputs:
    """``Outputs`` over channel sets and logger tables. Fans a ``ChainResult``
    out per row; the forecast tables go through the live publisher."""

    def __init__(
        self,
        field: "FieldSimulation",
        shading: Optional[GroundShading],
        et: Optional[Evapotranspiration],
        soil: SoilSimulation,
        predictor: Optional[SoilPredictor],
    ) -> None:
        self.field = field
        self.shading = shading
        self.et = et
        self.soil = soil
        self.predictor = predictor

    def chain(self, now: dt.datetime, result: ChainResult) -> None:
        shading_df = result.shading
        et_df = result.evapotranspiration

        if self.shading is not None and not shading_df.empty:
            seg_cols = [c for c in shading_df.columns if c != "open_sky_ghi" and not c.startswith("ghi_")]
            for ts in shading_df.index:
                factor = float(shading_df.loc[ts, seg_cols].mean()) if seg_cols else 1.0
                self.shading.data[LiveGroundShading.SHADING_FACTOR].set(ts, factor)
            if result.image is not None:
                self.shading.data[LiveGroundShading.SHADING_PROGRESS_IMAGE].set(now, result.image)

            ghi_cols = {c[len("ghi_") :]: c for c in shading_df.columns if c.startswith("ghi_")}
            segment_names = list(self._top_segment_names())
            if ghi_cols and segment_names:
                for ts in shading_df.index:
                    values = [
                        float(shading_df.loc[ts, ghi_cols[name]]) if name in ghi_cols else 0.0 for name in segment_names
                    ]
                    self.field.data[LiveFieldSimulation.SEG_GHI].set(ts, values)

        if self.et is not None and not et_df.empty:
            for key in _ET_CHANNEL_KEYS:
                if key not in et_df.columns:
                    continue
                for ts in et_df.index:
                    self.et.data[key].set(ts, float(et_df.loc[ts, key]))
            for constant in _VEGETATION_WRITE_CHANNELS:
                if constant not in et_df.columns:
                    continue
                for ts in et_df.index:
                    self.field.data[constant].set(ts, float(et_df.loc[ts, constant]))

    def step(self, result: StepResult) -> None:
        if self.soil is None:
            return
        ts = result.state.at
        for constant in _SOIL_DIAGNOSTIC_CHANNELS:
            if constant in result.diagnostics:
                self.soil.data[constant].set(ts, float(result.diagnostics[constant]))
        for probe_id, tension in result.probe_tension.items():
            try:
                self.soil.data[probe_id].set(ts, float(tension))
            except Exception:  # noqa: BLE001
                continue  # sensor-derived probe: sample-only, no registered channel

        assimilator = getattr(getattr(self.field, "simulation", None), "assimilator", None)
        last_result = getattr(assimilator, "last_result", None)
        if last_result is not None and last_result.innovations:
            increment = float(sum(last_result.innovations.values()))
            self.soil.data[LiveSoilSimulation.WATER_ANCHOR].set(ts, increment)

    def plan(self, plan: Plan) -> None:
        if self.predictor is None:
            return
        tables = self.predictor.tables()
        for frame, write in (
            (plan.header, tables.write_header_table),
            (plan.detail, tables.write_detail_table),
            (plan.irrigation, tables.write_irrigation_table),
            (plan.image, tables.write_image_table),
        ):
            if frame is not None:
                write(frame)

    def save_state(self, state: SoilState) -> None:
        if self.soil is None:
            return
        self.soil.data[LiveSoilSimulation.SIMULATION_STATE].set(state.at, state.to_blob())

    def _top_segment_names(self) -> Sequence[str]:
        simulation = getattr(self.field, "simulation", None)
        engine = getattr(simulation, "engine", None)
        return getattr(engine, "top_segment_names", ()) or ()


class FieldSimulation(Component):
    """Configure own section, let children configure theirs, bundle, assemble, own the thread."""

    TYPE: str = "field_simulation"
    CHILDREN: ClassVar[Sequence[Type[ChannelNamespace]]] = (
        GroundShading,
        Evapotranspiration,
        SoilSimulation,
        SoilPredictor,
    )
    INCLUDES = [c.TYPE for c in CHILDREN]

    location: Any = None
    weather: Any = None
    irrigation: Any = None

    def configure(self, configs: Configurations) -> None:
        super().configure(configs)
        field = FieldConfig()
        field.configure(configs)  # own keys; child sections and [plot] are allowed, not resolved here

        # [model] and [plot] cascade into every child as its defaults.
        defaults = Component._build_defaults(configs, includes=["model", "plot"], strict=True)
        self.ground_shading = self._child(GroundShading, configs, defaults)
        self.evapotranspiration = self._child(Evapotranspiration, configs, defaults)
        self.soil = self._child(SoilSimulation, configs, defaults)
        self.predictor = self._child(SoilPredictor, configs, defaults)
        if self.soil is None or self.soil.config is None:
            raise ValueError(f"{self.id}: [soil_simulation] block is required")

        plots = None
        if configs.has_member("plot"):
            plot_member = configs.get_member("plot")
            if plot_member.get_bool("enabled", default=True):
                plots = PlotConfig()
                plots.configure(plot_member)

        soil: SoilConfig = self.soil.config
        soil.mesh.derive(bay_width=field.bay_width)
        shading = self.ground_shading.config if self.ground_shading is not None else ShadingConfig.from_dict()
        self.setup = FieldSetup(
            field=field,
            soil=soil,
            shading=shading,
            planner=self.predictor.config if self.predictor is not None else None,
            plots=plots,
        )

        for constant in _VEGETATION_CHANNELS:
            self.data.add(constant, **deepcopy(dict(_MEAN_MEMORY)))
        if self._top_segment_names_from_config():
            for constant in _SEGMENT_CHANNELS:
                self.data.add(constant, **deepcopy(dict(_BUNDLE_MEMORY)))

        self.simulation: Optional[Simulation] = None
        self.runner: Optional[FieldRunner] = None
        self.ticker: Optional[Ticker] = None

    def activate(self) -> None:
        super().activate()

        system = self.context.context.context
        self.location = getattr(system, "location", None)
        self.weather = getattr(system, "weather", None)
        self.irrigation = getattr(self.context, "irrigation", None)

        if self.evapotranspiration is None or self.soil is None:
            return
        if self.weather is None:
            logger.warning("%s: no Weather component resolved; chain will never tick.", self.name)
            return

        self._weather_channels = Channels(list(self.weather.data.values()))
        self._required_weather_keys = ETModel.REQUIRED_WEATHER_COLUMNS
        self._evapo_rename = {c.id: c.key for c in self._weather_channels}
        self._warn_unwired_weather_channels()

        self.setup = replace(self.setup, location=self.location)

        anchor_cfg = parse_anchor_config(self.setup.soil.anchor)
        discover_enabled = self.setup.soil.discover_sensor_probes or anchor_cfg.enabled
        self._anchor_sensors, self._anchor_channels, self._anchor_data = self._discover_and_validate_sensors(
            anchor_cfg.enabled, discover_enabled
        )

        self._irrigation_flow_channel = self._resolve_irrigation_channel(Irrigation.FLOW)
        self._irrigation_state_channel = self._resolve_irrigation_channel(Irrigation.STATE)
        self._validate_irrigation_input()

        self.simulation = Simulation.build(self.setup, anchor_sensors=self._anchor_sensors, name=self.name)

        config_probes = []
        if self.setup.soil.configs is not None:
            probes_block = self.setup.soil.configs.get_member("probes", defaults={}, ensure_exists=True)
            config_probes = self.simulation.engine.probes(probes_block)
        for probe in config_probes:
            self.soil.register_probe(probe)

        if self.predictor is not None:
            self.predictor.register_forecast_tables(self.soil.configs, list(self.setup.soil.probe_specs))

        self.inputs = ChannelInputs(self)
        self.outputs = ChannelOutputs(self, self.ground_shading, self.evapotranspiration, self.soil, self.predictor)
        self.runner = FieldRunner(self.setup, self.simulation, self.inputs, self.outputs)
        self._register_state_listener()
        self.ticker = Ticker(self.setup, self.runner, tz=getattr(self.location, "timezone", None), name=self.name)
        self.ticker.start()

    def deactivate(self) -> None:
        if self.ticker is not None:
            self.ticker.stop()
        super().deactivate()

    @property
    def snapshot(self) -> Snapshot:
        """What dash reads. Never a channel round-trip."""
        return self.simulation.snapshot()

    @classmethod
    def schema(cls) -> dict[str, Any]:
        """The whole tree of needed and allowed keys: own section, [plot],
        and one entry per child type."""
        tree: dict[str, Any] = dict(FieldConfig.schema())
        tree["plot"] = {"type": "group", "children": PlotConfig.schema()}
        for child in cls.CHILDREN:
            tree[child.TYPE] = {"type": "component", "children": child.schema()}
        return tree

    def _child(self, cls: Type[_C], configs: Configurations, defaults: dict[str, Any]) -> Optional[_C]:
        """Build one channel namespace from its ``.d`` member, if present."""
        if not configs.has_member(cls.TYPE, includes=True):
            return None
        child = cls(self, configs.get_member(cls.TYPE, defaults=defaults))
        self.components.add(child)
        return child

    def _top_segment_names_from_config(self) -> list[str]:
        """Top-segment names from the resolved mesh config, needed before the
        FiPy engine exists to decide whether the per-segment channels exist."""
        from sparcs.components.agriculture.simulation._soil import top_segment_names_from_mesh

        mesh = self.setup.soil.mesh
        if mesh.width is None:
            return []
        return top_segment_names_from_mesh(mesh)

    def _warn_unwired_weather_channels(self) -> None:
        for channel in self._weather_channels:
            key = channel.key
            if key not in self._required_weather_keys or key in _OPTIONAL_WEATHER_KEYS:
                continue
            if channel.has_connector():
                continue
            logger.warning(
                "%s: required weather channel '%s' has no connector configured; "
                "the wall-clock tick reads its span from the connector and will never see it.",
                self.name,
                key,
            )

    # -- warm start ---------------------------------------------------------------

    def _register_state_listener(self) -> None:
        soil_data = self.soil.data
        if not self._check_state_channel_warm_start(soil_data):
            return
        soil_data.register(
            self._on_state,
            soil_data[LiveSoilSimulation.SIMULATION_STATE],
            how="any",
            unique=True,
        )

    def _check_state_channel_warm_start(self, soil_data: Any) -> bool:
        """Warn about a warm-start-breaking state-channel config; return whether
        a read-side connector is present, so the listener is worth registering."""
        state_channel = soil_data[LiveSoilSimulation.SIMULATION_STATE]
        if not state_channel.has_logger():
            logger.warning(
                "%s: SIMULATION_STATE has no logger configured; soil state will not "
                "persist across restarts. Configure a logger on the channel to enable "
                "warm starts.",
                self.name,
            )
        elif not state_channel.has_connector():
            logger.warning(
                "%s: SIMULATION_STATE has a logger but no read-side connector "
                "configured; soil state will be written but never restored on "
                "restart. Configure a connector on the channel to enable warm "
                "starts.",
                self.name,
            )
        return state_channel.has_connector()

    def _on_state(self, data: pd.DataFrame) -> None:
        if data.empty or self.runner is None:
            return
        self.runner.restore(SoilState.from_blob(data.iloc[0, 0], data.index[0]))

    # -- irrigation input (flow meter, with state x design-flow fallback) --------

    def _resolve_irrigation_channel(self, constant: Constant) -> Any:
        if self.irrigation is None:
            return None
        try:
            return self.irrigation.data[constant]
        except KeyError:
            return None

    def _validate_irrigation_input(self) -> None:
        if self.irrigation is None:
            return
        flow_wired = self._irrigation_flow_channel is not None and self._irrigation_flow_channel.has_connector()
        state_wired = (
            self._irrigation_state_channel is not None
            and self._irrigation_state_channel.has_connector()
            and self.setup.soil.drip.explicit
        )
        if not (flow_wired or state_wired):
            raise ConfigurationUnavailableError(
                f"{self.name}: irrigation is configured but no usable input is wired. "
                "Wire the metered feed ([irrigation.data.channels.flow] with a connector), "
                "or the on/off state feed ([irrigation.data.channels.state] with a connector "
                "PLUS a [soil_simulation.drip] block giving nozzle_count and nozzle_flow_lph). "
                "Refusing to start on a silent 0 l/min fallback."
            )

    # -- anchor sensor discovery --------------------------------------------------

    def _discover_sensors(
        self,
    ) -> tuple[list[AnchorSensor], dict[str, Any], dict[str, Any], list[tuple[str, Exception]]]:
        """One ``AnchorSensor`` per enabled, tension-measured ``SoilMoisture``
        sensor in this field, walked from this component's own parent field."""
        from sparcs.components.agriculture.soil.moisture import SoilMoisture

        field = getattr(self, "context", None)
        sensors: list[AnchorSensor] = []
        channels: dict[str, Any] = {}
        data: dict[str, Any] = {}
        failures: list[tuple[str, Exception]] = []
        if field is None:
            return sensors, channels, data, failures
        for comp in _walk_components(field):
            if not isinstance(comp, SoilMoisture):
                continue
            try:
                if not comp.has_measured_tension:
                    continue
                sensors.append(AnchorSensor(key=comp.key, x_offset_cm=comp.x_offset, depth_cm=comp.depth))
                channels[comp.key] = comp.data[SoilMoisture.WATER_TENSION]
                data[comp.key] = comp.data
            except Exception as e:  # noqa: BLE001
                logger.exception("%s: failed to derive an anchor sensor from %s", self.name, getattr(comp, "key", "?"))
                failures.append((str(getattr(comp, "key", "?")), e))
        return sensors, channels, data, failures

    def _discover_and_validate_sensors(
        self, anchor_enabled: bool, discover_enabled: bool
    ) -> tuple[list[AnchorSensor], dict[str, Any], dict[str, Any]]:
        """With ``[anchor]`` enabled, a discovery failure or zero discovered
        sensors refuses startup; with only ``discover_sensor_probes`` on,
        failures are logged and the sim runs with whatever was found."""
        if not discover_enabled:
            return [], {}, {}
        strict = anchor_enabled
        try:
            sensors, channels, data, failures = self._discover_sensors()
        except Exception as e:  # noqa: BLE001
            if strict:
                raise ConfigurationUnavailableError(
                    f"{self.name}: [anchor] is enabled but sensor-probe discovery failed: {e}. "
                    "Verify the field's SoilMoisture sensor wiring before starting."
                ) from e
            logger.exception("%s: sensor-probe discovery failed; continuing without sensor probes", self.name)
            return [], {}, {}
        if not strict:
            return sensors, channels, data
        if failures:
            failed = ", ".join(key for key, _ in failures)
            raise ConfigurationUnavailableError(
                f"{self.name}: [anchor] is enabled but probe derivation failed for sensor(s): {failed}. "
                "See the exception log above; fix the sensor geometry/mesh wiring before starting."
            ) from failures[0][1]
        if not sensors:
            raise ConfigurationUnavailableError(
                f"{self.name}: [anchor] is enabled but no tension-measured SoilMoisture sensor was "
                "discovered in this field (a sensor counts when its water_tension channel has a "
                "connector). Wire a tensiometer or disable [anchor]."
            )
        return sensors, channels, data
