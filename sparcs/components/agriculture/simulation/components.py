# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.simulation.components
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The lories layer: ``FieldSimulation``, its ``ChannelNamespace`` children and the channel ports.
``FieldSimulation`` builds the ``Simulation`` at configure; the runner and the ``Ticker`` start at activate.
"""

from __future__ import annotations

import datetime as dt
import logging
from copy import deepcopy
from typing import Any, Callable, ClassVar, Mapping, Optional, Sequence, Type, TypeVar

import pandas as pd
from lories import Constant, System
from lories.components import Component
from lories.components.weather import Weather, WeatherProvider
from lories.core import Configurations, ConfigurationUnavailableError
from lories.data import Channels
from lories.util import get_context
from sparcs.components.agriculture.irrigation import Irrigation

from .core import plots
from .core.anchor import AnchorSensor
from .core.config import (
    Config,
    EvapotranspirationConfig,
    FieldConfig,
    FieldSetup,
    PlannerConfig,
    PlotConfig,
    SoilConfig,
)
from .core.evapotranspiration import ETModel
from .core.shading import MODE_FREE_FIELD, ShadingConfig
from .core.simulation import Simulation
from .core.state import ChainResult, Plan, Snapshot, SoilState, StepResult
from .forecast_tables import ForecastTablePublisher
from .runtime.ports import IRRIGATION_LOOKBACK
from .runtime.runner import FieldRunner
from .runtime.scheduler import Ticker

logger = logging.getLogger(__name__)

_C = TypeVar("_C", bound="ChannelNamespace")

# Weather keys the chain fills with a default instead of requiring a feed.
_OPTIONAL_WEATHER_KEYS: frozenset = frozenset({Weather.CLEAR_SKY_INDEX, Weather.HUMIDITY_REL})

PLOT_STRIKES = Constant(float, "plot_strikes", "Plot Strikes", context="fieldsim")

# Keyed by the table label ForecastTablePublisher passes to _bump_write_failure.
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
    """A Component that owns channels under its id; ``FieldSimulation`` validates its ``.d`` file against ``CONFIG``.
    ``plot_config`` is its own ``[plot]`` block over the field-level defaults, None once disabled for the process."""

    CHANNELS: ClassVar[Sequence[Constant]] = ()
    # Registered only when the child's own [plot] block is enabled.
    PLOT_CHANNELS: ClassVar[Sequence[Constant]] = ()
    CHANNEL_CONFIGS: ClassVar[Mapping[str, Mapping[str, Any]]] = {}
    DEFAULT_CONFIGS: ClassVar[Mapping[str, Any]] = _MEAN_MEMORY
    CONFIG: ClassVar[Optional[Type[Config]]] = None

    config: Optional[Config] = None
    plot_enabled: bool = True
    plot_config: Optional[PlotConfig] = None
    _last_plot_ts: Optional[pd.Timestamp] = None
    _plot_strikes: int = 0

    def configure(self, configs: Configurations) -> None:
        super().configure(configs)
        self.plot_config = self._plot_config(configs)
        self.plot_enabled = self.plot_config is not None
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
    def _plot_config(configs: Configurations) -> Optional[PlotConfig]:
        plot = configs.get_member("plot", defaults={}, ensure_exists=True)
        if not plot.get_bool("enabled", default=True):
            return None
        config = PlotConfig()
        config.configure(plot)
        return config


class GroundShading(ChannelNamespace):
    TYPE: str = "ground_shading"

    SHADING_FACTOR = Constant(float, "shading_factor", "Mean Ground Shading Factor", "-")
    SHADING_PROGRESS_IMAGE = Constant(bytes, "shading_progress_image", "Ground Shading Progress Image", "png")
    CONFIG = ShadingConfig
    CHANNELS = (SHADING_FACTOR, PLOT_STRIKES)
    PLOT_CHANNELS = (SHADING_PROGRESS_IMAGE,)
    CHANNEL_CONFIGS = {
        PLOT_STRIKES: _LAST_MEMORY,
        SHADING_PROGRESS_IMAGE: _SHADING_IMAGE,
    }


class Evapotranspiration(ChannelNamespace):
    TYPE: str = "evapotranspiration"

    SVP = Constant(float, "sat_vapor_pressure", "Saturation Vapor Pressure", "kPa")
    GVP = Constant(float, "ground_vapor_pressure", "Vapor Pressure on the Ground Surface", "kPa")
    VAP_HEAT = Constant(float, "vaporization_heat", "Latent Heat of Vaporization", "J/kg")
    SVP_SLOPE = Constant(float, "slope_sat_vapor_pressure", "Saturation Vapor Pressure Slope", "kPa/K")
    NET_IRR = Constant(float, "net_irradiance", "Net Irradiance", "W/m^2")
    AIR_RES = Constant(float, "aerodynamic_resistance", "Aerodynamic Resistance", "s/m")
    SOIL_HEAT_FLOW = Constant(float, "soil_heat_flow", "Soil Heat Flow", "W/m^2")
    SURFACE_RES = Constant(float, "resistance_surface", "Surface Resistance", "s/m")
    RAD_TERM = Constant(float, "radiation_term", "Radiation Term", "(kPa*W)/(K*m^2)")
    AER_TERM = Constant(float, "aerodynamic_term", "Aerodynamic Term", "(kPa*J)/(m^2*K*s)")
    EVAPOTRANSPIRATION = Constant(float, "evapotranspiration", "Evapotranspiration", "kg/(m^2*h)")
    CONFIG = EvapotranspirationConfig
    CHANNELS = (
        SVP,
        GVP,
        VAP_HEAT,
        SVP_SLOPE,
        NET_IRR,
        AIR_RES,
        SOIL_HEAT_FLOW,
        SURFACE_RES,
        RAD_TERM,
        AER_TERM,
        EVAPOTRANSPIRATION,
    )


class SoilSimulation(ChannelNamespace):
    TYPE: str = "soil_simulation"

    SIMULATION_STATE = Constant(bytes, "simulation_state", "Soil Simulation State", "-")
    SOIL_PROGRESS_IMAGE = Constant(bytes, "soil_progress_image", "Soil Simulation Progress Image", "png")
    WATER_TOP_IN = Constant(float, "top_in", "Top Water Input (Irrigation + Rain)", "kg/(m^2*h)", context="water")
    WATER_TOP_OUT = Constant(float, "top_out", "Top Water Output (Evaporation)", "kg/(m^2*h)", context="water")
    WATER_BOTTOM = Constant(float, "bottom_out", "Bottom Water Output (Drainage)", "kg/(m^2*h)", context="water")
    WATER_TRANSP = Constant(float, "transpiration", "Plant Transpiration", "kg/(m^2*h)", context="water")
    WATER_RUNOFF = Constant(float, "runoff", "Rejected Top Influx (Runoff)", "kg/(m^2*h)", context="water")
    WATER_DEMAND_UNMET = Constant(float, "demand_unmet", "Unmet Evap+Transp Demand", "kg/(m^2*h)", context="water")
    WATER_BALANCE_RESIDUAL = Constant(
        float,
        "balance_residual",
        "Mass-Balance Residual (integral - direct)",
        "kg/(m^2*h)",
        context="water",
    )
    WATER_ANCHOR = Constant(float, "anchor", "Anchor Assimilation Increment", "kg/m", context="water")
    WALK_SKIPPED_S = Constant(float, "skipped_s", "Skipped Walk Duration (dt_min Unsolvable)", "s", context="water")
    WALK_RETRIES = Constant(float, "retries", "Walk Substep Retries", "-", context="water")
    WEATHER_STALL = Constant(float, "weather_stall", "Consecutive Weather-Stall Ticks", "-", context="water")
    TICK_FAILURES = Constant(float, "tick_failures", "Consecutive Tick Failures", "-", context="water")
    IRRIGATION_FLOW_LPM: str = "irrigation_flow_lpm"
    CONFIG = SoilConfig
    CHANNELS = (
        SIMULATION_STATE,
        WATER_TOP_IN,
        WATER_TOP_OUT,
        WATER_BOTTOM,
        WATER_TRANSP,
        WATER_RUNOFF,
        WATER_DEMAND_UNMET,
        WATER_BALANCE_RESIDUAL,
        WATER_ANCHOR,
        WALK_SKIPPED_S,
        WALK_RETRIES,
        WEATHER_STALL,
        TICK_FAILURES,
        PLOT_STRIKES,
    )
    PLOT_CHANNELS = (SOIL_PROGRESS_IMAGE,)
    DEFAULT_CONFIGS = _MEAN_LOGGED
    CHANNEL_CONFIGS = {
        SIMULATION_STATE: _STATE_BLOB,
        SOIL_PROGRESS_IMAGE: _SOIL_IMAGE,
        PLOT_STRIKES: _LAST_MEMORY,
    }

    # The config probes, set by FieldSimulation before this component configures.
    probes: Sequence[Any] = ()

    def configure(self, configs: Configurations) -> None:
        super().configure(configs)
        for probe in self.probes:
            self.register_probe(probe)

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
    """Planner channels; ``ForecastTablePublisher`` holds the forecast tables' schema and write path."""

    TYPE: str = "soil_predictor"
    CONFIG = PlannerConfig
    CHANNELS = tuple(_WRITE_FAILURE_CHANNELS.values())
    DEFAULT_CONFIGS = _LAST_MEMORY

    _HEADER_TABLE_NAME: str = "agri_field_forecast"
    _HEADER_FORECAST_ID_KEY: str = "forecast_id"
    _HEADER_IS_RECOMMENDED_KEY: str = "is_recommended"
    _HEADER_TOTAL_MIN_KEY: str = "total_min"
    _HEADER_WEATHER_CREATION_KEY: str = "weather_creation"
    _DETAIL_TABLE_NAME: str = "agri_soil_forecast"
    _DETAIL_TIMESTAMP_CREATION_SUFFIX: str = "_timestamp_creation"
    _DETAIL_FORECAST_ID_SUFFIX: str = "_forecast_id"
    _IRRIGATION_TABLE_NAME: str = "agri_field_forecast_irrigation"
    _IRRIGATION_STATE_KEY: str = "irrigation_state"
    _IRRIGATION_TIMESTAMP_CREATION_KEY: str = "irrigation_timestamp_creation"
    _IMAGE_TABLE_NAME: str = "agri_field_forecast_image"
    _IMAGE_KEY: str = "predict_image"
    _IMAGE_COLUMN: str = "image"
    _IMAGE_TIMESTAMP_CREATION_KEY: str = "predict_image_timestamp_creation"

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

    def activate(self) -> None:
        super().activate()
        self.tables().validate_logger_connector()

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
        except KeyError:
            logger.debug("%s: write-failure channel '%s' unavailable; count=%d.", self.name, constant, count)


class ChannelInputs:
    """``Inputs`` over lories connector reads; it makes no decisions."""

    def __init__(self, field: "FieldSimulation") -> None:
        self.field = field
        self._last_invalid_weather_columns: list[str] = []

    def load_state(self) -> Optional[SoilState]:
        soil = self.field.soil_simulation
        if soil is None:
            return None
        try:
            channel = soil.data[SoilSimulation.SIMULATION_STATE]
        except KeyError:
            return None
        if not channel.is_valid():
            return None
        blob = channel.value
        if not blob:
            return None
        return SoilState.from_blob(blob, at=channel.timestamp)

    # --- WEATHER ----------------------------------------------------------

    def weather(self, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
        field = self.field
        frame = field.data.read(field._weather_channels, start=start, end=end)
        frame = self._trim_span(frame, start, end)
        if not self._weather_frame_valid(frame):
            return frame.iloc[0:0]
        return frame

    def _trim_span(self, frame: pd.DataFrame, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
        if frame.empty:
            return frame
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
            if missing != self._last_invalid_weather_columns:
                logger.warning(
                    "%s: weather chunk dropped, required column(s) missing or all-NaN: %s",
                    self.field.name,
                    ", ".join(missing),
                )
            self._last_invalid_weather_columns = missing
            return False
        self._last_invalid_weather_columns = []
        return True

    # --- IRRIGATION ---------------------------------------------------------

    def irrigation(self, start: dt.datetime, end: dt.datetime) -> pd.Series:
        field = self.field
        measured = self._read_measured_flow(start, end)
        if measured is not None:
            return measured
        if field._irrigation_state_channel is not None and field.setup.soil.drip.explicit:
            return self._read_state_span(start, end) * field.setup.soil.drip.design_flow_lpm
        # Unwired: not watering, never a fabricated forcing.
        return pd.Series(dtype=float)

    def _read_measured_flow(self, start: dt.datetime, end: dt.datetime) -> Optional[pd.Series]:
        channel = self.field._irrigation_flow_channel
        if channel is None:
            return None
        samples = self._read_samples(channel, start, end)
        if samples.empty or samples.isna().all():
            return None
        return samples

    def _read_state_span(self, start: dt.datetime, end: dt.datetime) -> pd.Series:
        return self._read_samples(self.field._irrigation_state_channel, start, end).astype(float)

    def _read_samples(self, channel: Any, start: dt.datetime, end: dt.datetime) -> pd.Series:
        frame = self.field.data.read(Channels([channel]), start=start - IRRIGATION_LOOKBACK, end=end, unique=True)
        if frame.empty:
            return pd.Series(dtype=float)
        return frame.iloc[:, 0]

    # --- TENSION --------------------------------------------------------------

    def tension(self, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
        field = self.field
        channels = field._anchor_channels
        if not channels:
            return pd.DataFrame()
        config = field.simulation.assimilator.config
        lookback = max(config.sensor_staleness(key) for key in channels)
        frame = field.data.read(Channels(list(channels.values())), start=start - lookback, end=end, unique=True)

        # One column per sensor even when its read is empty, so an absent
        # reading is NaN once another sensor's rows widen the shared index.
        series_by_key: dict[str, pd.Series] = {}
        for key, channel in channels.items():
            if channel.id not in frame.columns:
                series_by_key[key] = pd.Series(dtype=float)
                continue
            series = frame[channel.id].dropna().sort_index()
            series_by_key[key] = series[~series.index.duplicated(keep="last")]
        return pd.concat(series_by_key, axis=1)

    # --- FORECAST ----------------------------------------------------------

    def forecast(self, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
        weather = self.field.weather
        forecast_sub = weather.forecast if isinstance(weather, WeatherProvider) else None
        if forecast_sub is None or not forecast_sub.is_enabled():
            return pd.DataFrame()
        try:
            frame = forecast_sub.data.to_frame(unique=False)
        except Exception as e:  # noqa: BLE001
            logger.warning("%s: forecast read failed: %s", self.field.name, e)
            return pd.DataFrame()
        if frame.empty:
            return frame
        return frame.loc[(frame.index >= start) & (frame.index <= end)]


def _set_series(channel: Any, series: pd.Series) -> None:
    """One ``set`` carrying every non-NaN row; lories logs a ``Series`` value row by row."""
    series = series.dropna().astype(float)
    if not series.empty:
        channel.set(series.index[0], series)


def _set_strikes(child: ChannelNamespace, ts: pd.Timestamp, strikes: int) -> None:
    """Best effort: the count on ``child`` is the record, the channel only shows it."""
    try:
        child.data[PLOT_STRIKES].set(ts, float(strikes))
    except Exception:  # noqa: BLE001
        logger.debug("%s: plot_strikes channel write failed; count %d kept in memory", child.name, strikes)


class ChannelOutputs:
    """``Outputs`` over channel sets: one ``Series`` per channel per chunk, which the logger writes row by row.
    Progress images render here, one ``set`` per frame; a render failure is counted and never escapes."""

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
            if GroundShading.SHADING_FACTOR in shading_df.columns:
                _set_series(self.shading.data[GroundShading.SHADING_FACTOR], shading_df[GroundShading.SHADING_FACTOR])
            ts = pd.Timestamp(now)
            self._render_progress(
                self.shading,
                GroundShading.SHADING_PROGRESS_IMAGE,
                ts,
                plots.render_shading_png,
                ts,
                result.ground,
                result.pv_rows,
                result.sun_state,
                result.envelope,
            )

            ghi_cols = {c[len("ghi_") :]: c for c in shading_df.columns if c.startswith("ghi_")}
            segment_names = list(self._top_segment_names())
            if ghi_cols and segment_names:
                ts = shading_df.index[-1]
                values = [
                    float(shading_df.loc[ts, ghi_cols[name]]) if name in ghi_cols else 0.0 for name in segment_names
                ]
                self.field.data[FieldSimulation.SEG_GHI].set(ts, values)

        if self.et is not None and not et_df.empty:
            for key in _ET_CHANNEL_KEYS:
                if key in et_df.columns:
                    _set_series(self.et.data[key], et_df[key])
            for constant in _VEGETATION_WRITE_CHANNELS:
                if constant in et_df.columns:
                    _set_series(self.field.data[constant], et_df[constant])

    def steps(self, results: Sequence[StepResult]) -> None:
        if self.soil is None or not results:
            return
        index = pd.DatetimeIndex([r.state.at for r in results])
        diagnostics = pd.DataFrame([dict(r.diagnostics) for r in results], index=index)
        for constant in _SOIL_DIAGNOSTIC_CHANNELS:
            if constant in diagnostics.columns:
                _set_series(self.soil.data[constant], diagnostics[constant])
        tension = pd.DataFrame([dict(r.probe_tension) for r in results], index=index)
        for probe_id in tension.columns:
            if probe_id in self.soil.data:  # sensor-derived probes are sample-only
                _set_series(self.soil.data[probe_id], tension[probe_id])

        if self.soil.plot_config is not None:
            mesh = self.field.simulation.engine.mesh
            geometry = self.field.setup.soil.mesh
            for result in results:
                ts = pd.Timestamp(result.state.at)
                self._render_progress(
                    self.soil,
                    SoilSimulation.SOIL_PROGRESS_IMAGE,
                    ts,
                    plots.render_rel_sat_png,
                    mesh,
                    result.state.se,
                    ts,
                    width_m=geometry.width,
                    height_m=geometry.height,
                )

    def _render_progress(
        self,
        child: ChannelNamespace,
        constant: Constant,
        ts: pd.Timestamp,
        render: Callable[..., bytes],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Render into ``child``'s image channel when its ``[plot]`` interval is due; never raises.
        Each failure is a strike; ``disable_after_failures`` in a row disable plotting, a success resets the count."""
        config = child.plot_config
        if not plots.render_due(child._last_plot_ts, ts, config):
            return
        child._last_plot_ts = ts
        try:
            png = render(*args, tz=self._timezone(), **kwargs)
            child.data[constant].set(ts, png)
        except Exception:  # noqa: BLE001
            child._plot_strikes, disable = plots.count_render_failure(
                logger, f"{child.name} progress image", child._plot_strikes, config.disable_after_failures
            )
            if disable:
                child.plot_config = None
            _set_strikes(child, ts, child._plot_strikes)
            return
        child._plot_strikes = 0
        _set_strikes(child, ts, 0)

    def _timezone(self) -> Any:
        location = self.field.setup.location
        return location.timezone if location is not None else None

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
        self.soil.data[SoilSimulation.SIMULATION_STATE].set(pd.Timestamp(state.at), state.to_blob())

    def _top_segment_names(self) -> Sequence[str]:
        return self.field.simulation.engine.top_segment_names


class FieldSimulation(Component):
    """Configure own and children's sections, bundle, build the simulation, own the thread."""

    TYPE: str = "field_simulation"

    TEMP_GROUND = Constant(float, "temp_ground", "Ground Temperature", "°C")
    LAI = Constant(float, "lai", "Leaf Area Index", "m^2/m^2")
    ROUGHNESS = Constant(float, "roughness", "Roughness", "-")
    PLANT_HEIGHT = Constant(float, "plant_height", "Plant Height", "m")
    NDVI = Constant(float, "ndvi", "Normalized Difference Vegetation Index", "-")
    SEG_GHI = Constant(list, "seg_ghi", "GHI (per segment)", "W/m^2")
    SEG_EVAPOTRANSPIRATION = Constant(list, "seg_evapotranspiration", "Evapotranspiration (per segment)", "kg/(m^2*h)")
    SEG_TEMP_GROUND = Constant(list, "seg_temp_ground", "Ground Temperature (per segment)", "°C")
    VEGETATION_CHANNELS = (TEMP_GROUND, LAI, ROUGHNESS, PLANT_HEIGHT, NDVI)
    SEGMENT_CHANNELS = (SEG_GHI, SEG_EVAPOTRANSPIRATION, SEG_TEMP_GROUND)
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
    _offline: Optional[Simulation] = None

    def configure(self, configs: Configurations) -> None:
        super().configure(configs)
        field = FieldConfig()
        field.configure(configs)  # own keys; child sections and [plot] are allowed, not resolved here
        self.location = get_context(self, System).location

        # Only SoilConfig declares [model]; the other children's strict check would reject the cascade.
        defaults = Component._build_defaults(configs, includes=["plot"], strict=True)
        soil_defaults = Component._build_defaults(configs, includes=["model", "plot"], strict=True)
        self.soil_simulation = self._child(SoilSimulation, configs, soil_defaults)
        if self.soil_simulation is None:
            raise ConfigurationUnavailableError(f"{self.id}: [soil_simulation] block is required")
        soil: SoilConfig = self.soil_simulation.config
        self.ground_shading = self._child(GroundShading, configs, defaults)
        self.evapotranspiration = self._child(Evapotranspiration, configs, defaults)
        self.soil_predictor = self._child(SoilPredictor, configs, {**defaults, "drip": soil.drip.values()})

        soil.mesh.derive(bay_width=field.bay_width)
        if self.ground_shading is not None:
            shading = self.ground_shading.config
        else:
            shading = ShadingConfig.from_dict({"mode": MODE_FREE_FIELD})
        self.setup = FieldSetup(
            field=field,
            soil=soil,
            shading=shading,
            planner=self.soil_predictor.config if self.soil_predictor is not None else None,
            location=self.location,
        )
        self.simulation = Simulation.build(self.setup, name=self.name)
        self.soil_simulation.probes = list(self.simulation.probes)

        for constant in _VEGETATION_CHANNELS:
            self.data.add(constant, **deepcopy(dict(_MEAN_MEMORY)))
        if self._top_segment_names_from_config():
            for constant in _SEGMENT_CHANNELS:
                self.data.add(constant, **deepcopy(dict(_BUNDLE_MEMORY)))

        self.runner: Optional[FieldRunner] = None
        self.ticker: Optional[Ticker] = None

    def _on_configure(self, configs: Configurations) -> None:
        super()._on_configure(configs)
        probes = self.soil_simulation.probes
        # Raises on a duplicate soil_id, with or without a predictor.
        ForecastTablePublisher(self.soil_simulation).resolve_probe_identities(self.soil_simulation.configs, probes)
        if self.soil_predictor is not None:
            self.soil_predictor.register_forecast_tables(self.soil_simulation.configs, probes)

    def activate(self) -> None:
        super().activate()

        system = get_context(self, System)
        self.weather = system.weather if system.has_weather() else None
        if self.weather is None:
            raise ConfigurationUnavailableError(f"{self.name}: no weather component configured to drive the chain")
        field = self.context
        self.irrigation = field.irrigation if field.has_irrigation() else None

        self._weather_channels = Channels(list(self.weather.data.values()))
        self._required_weather_keys = ETModel.REQUIRED_WEATHER_COLUMNS
        self._warn_unwired_weather_channels()

        anchor_enabled = self.simulation.assimilator.config.enabled
        discover_enabled = self.setup.soil.discover_sensor_probes or anchor_enabled
        sensors, self._anchor_channels = self._discover_and_validate_sensors(anchor_enabled, discover_enabled)
        self.simulation.add_sensors(sensors)

        self._irrigation_flow_channel = self._resolve_irrigation_channel(Irrigation.FLOW)
        self._irrigation_state_channel = self._resolve_irrigation_channel(Irrigation.STATE)
        self._validate_irrigation_input()

        self.inputs = ChannelInputs(self)
        self.outputs = ChannelOutputs(
            self, self.ground_shading, self.evapotranspiration, self.soil_simulation, self.soil_predictor
        )
        self.runner = FieldRunner(self.setup, self.simulation, self.inputs, self.outputs)
        self._register_state_listener()
        self.ticker = Ticker(self.setup, self.runner, tz=self.location.timezone, name=self.name)
        self.ticker.start()

    def deactivate(self) -> None:
        if self.ticker is not None:
            self.ticker.stop()
        super().deactivate()

    @property
    def snapshot(self) -> Snapshot:
        """What dash reads. Never a channel round-trip."""
        return self.simulation.snapshot()

    def simulate(
        self,
        weather: pd.DataFrame,
        start: Optional[dt.datetime] = None,
        end: Optional[dt.datetime] = None,
        prior: Optional[pd.DataFrame] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Offline run over a weather frame on its own session, never the current one; with ``prior`` it continues it.
        Without ``prior`` the run starts cold. Returns diagnostics that have a ``soil_simulation`` channel, by id."""
        weather = self._get_range(weather, start, end)
        if weather.empty:
            return pd.DataFrame()
        if prior is None or self._offline is None:
            self._offline = Simulation.build(self.setup, name=self.name)
        results, _ = self._offline.run(weather, pd.Series(0.0, index=weather.index))
        index = pd.DatetimeIndex([r.state.at for r in results])
        frame = pd.DataFrame([dict(r.diagnostics) for r in results], index=index)
        soil = self.soil_simulation.data
        ids = {key: soil[key].id for key in frame.columns if key in soil}
        return frame[list(ids)].rename(columns=ids)

    def _child(self, cls: Type[_C], configs: Configurations, defaults: dict[str, Any]) -> Optional[_C]:
        """Build one channel namespace and its configured section from its ``.d`` member, if present."""
        if not configs.has_member(cls.TYPE, includes=True):
            return None
        member = configs.get_member(cls.TYPE, defaults=defaults)
        section = cls.CONFIG()
        section.configure(member.copy())
        child = cls(self, member)
        child.config = section
        self.components.add(child)
        return child

    def _top_segment_names_from_config(self) -> list[str]:
        """Top-segment names from the resolved mesh config, needed before the
        FiPy engine exists to decide whether the per-segment channels exist."""
        from .core.pde import top_segment_names_from_mesh

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

    # --- warm start --------------------------------------------------------------

    def _register_state_listener(self) -> None:
        soil_data = self.soil_simulation.data
        if not self._check_state_channel_warm_start(soil_data):
            return
        soil_data.register(
            self._on_state,
            soil_data[SoilSimulation.SIMULATION_STATE],
            how="any",
            unique=True,
        )

    def _check_state_channel_warm_start(self, soil_data: Any) -> bool:
        """Warn about a warm-start-breaking state-channel config; return whether
        a read-side connector is present, so the listener is worth registering."""
        state_channel = soil_data[SoilSimulation.SIMULATION_STATE]
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

    # --- irrigation input (flow meter, with state x design-flow fallback) -------

    def _resolve_irrigation_channel(self, constant: Constant) -> Any:
        if self.irrigation is None:
            return None
        return self.irrigation.data[constant]

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

    # --- anchor sensor discovery -------------------------------------------------

    def _discover_sensors(self) -> tuple[list[AnchorSensor], dict[str, Any], list[tuple[str, Exception]]]:
        """One ``AnchorSensor`` per enabled, tension-measured ``SoilMoisture``
        sensor of this component's parent field."""
        from sparcs.components.agriculture.soil.moisture import SoilMoisture

        sensors: list[AnchorSensor] = []
        channels: dict[str, Any] = {}
        failures: list[tuple[str, Exception]] = []
        for sensor in self.context.soil:
            try:
                if not sensor.has_measured_tension:
                    continue
                sensors.append(AnchorSensor(key=sensor.key, x_offset_cm=sensor.x_offset, depth_cm=sensor.depth))
                channels[sensor.key] = sensor.data[SoilMoisture.WATER_TENSION]
            except Exception as e:  # noqa: BLE001
                logger.exception("%s: failed to derive an anchor sensor from %s", self.name, sensor.key)
                failures.append((sensor.key, e))
        return sensors, channels, failures

    def _discover_and_validate_sensors(
        self, anchor_enabled: bool, discover_enabled: bool
    ) -> tuple[list[AnchorSensor], dict[str, Any]]:
        """With ``[anchor]`` enabled, a discovery failure or zero sensors refuses startup.
        With only ``discover_sensor_probes`` on, failures are logged and the sim runs with what was found."""
        if not discover_enabled:
            return [], {}
        strict = anchor_enabled
        try:
            sensors, channels, failures = self._discover_sensors()
        except Exception as e:  # noqa: BLE001
            if strict:
                raise ConfigurationUnavailableError(
                    f"{self.name}: [anchor] is enabled but sensor-probe discovery failed: {e}. "
                    "Verify the field's SoilMoisture sensor wiring before starting."
                ) from e
            logger.exception("%s: sensor-probe discovery failed; continuing without sensor probes", self.name)
            return [], {}
        if not strict:
            return sensors, channels
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
        return sensors, channels


# --------------------------------------------------------------------------- field-level channels

# TEMP_GROUND is registered but not written: only Evapotranspiration.evaluate's
# per-segment publish path derives it, which ETModel does not reproduce.
_VEGETATION_CHANNELS = FieldSimulation.VEGETATION_CHANNELS
_VEGETATION_WRITE_CHANNELS = (
    FieldSimulation.LAI,
    FieldSimulation.ROUGHNESS,
    FieldSimulation.PLANT_HEIGHT,
    FieldSimulation.NDVI,
)

# Only SEG_GHI is derivable from ChainResult; the other two need seg_et.
_SEGMENT_CHANNELS = FieldSimulation.SEGMENT_CHANNELS

_ET_CHANNEL_KEYS = tuple(str(c) for c in Evapotranspiration.CHANNELS)
_SOIL_DIAGNOSTIC_CHANNELS = (
    SoilSimulation.WATER_TOP_IN,
    SoilSimulation.WATER_TOP_OUT,
    SoilSimulation.WATER_BOTTOM,
    SoilSimulation.WATER_TRANSP,
    SoilSimulation.WATER_RUNOFF,
    SoilSimulation.WATER_DEMAND_UNMET,
    SoilSimulation.WATER_BALANCE_RESIDUAL,
    SoilSimulation.WATER_ANCHOR,
    SoilSimulation.WALK_SKIPPED_S,
    SoilSimulation.WALK_RETRIES,
    SoilSimulation.WEATHER_STALL,
    SoilSimulation.TICK_FAILURES,
)
