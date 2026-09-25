# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.fieldsim.components
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The lories layer: the only module in this package that imports lories
components and channels.

``FieldSimulation`` configures the ``FieldConfig`` section from its own
``.conf``, lets each ``.d`` child configure its own section, bundles them
into a frozen ``FieldSetup``, builds the ``Simulation``, the runner and the
scheduler, owns the thread through activate/deactivate, and exposes the
``Snapshot`` for dash.

``GroundShading``, ``Evapotranspiration``, ``SoilSimulation`` and
``SoilPredictor`` remain Components for one reason: channel ids and logger
table names derive from the component id, and those are frozen. Each is a
``ChannelNamespace``: a TYPE, a ``CHANNELS`` spec list, and the ``Config``
class its file is validated against. ``ChannelInputs`` and ``ChannelOutputs``
implement the two runtime ports over those channels.

Every section's keys are declared once, as lories ``Parameter`` descriptors
on the ``Config`` classes in ``core.config``; unknown keys fail the load.
``FieldSimulation.schema()`` returns the whole tree.

Not registered as a component type; the live ``simulation.FieldSimulation``
keeps the ``field_simulation`` type. Class names are reused on purpose so the
skeleton reads against the current diagram.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, replace
from typing import Any, ClassVar, Optional, Sequence, Type, TypeVar

import pandas as pd
from lories import Constant
from lories.components import Component
from lories.components.weather import Weather
from lories.core import Configurations, ConfigurationUnavailableError
from lories.data import Channels
from sparcs.components.agriculture.irrigation import Irrigation

from .core.anchor import AnchorSensor
from .core.assimilator import parse_anchor_config
from .core.config import Config, FieldConfig, FieldSetup, PlannerConfig, PlotConfig, SoilConfig
from .core.evapotranspiration import ETModel
from .core.shading import ShadingConfig
from .core.simulation import Simulation
from .core.state import ChainResult, Plan, Snapshot, SoilState, StepResult
from .runtime.ports import InputKey
from .runtime.runner import FieldRunner
from .runtime.scheduler import TickScheduler

logger = logging.getLogger(__name__)

_C = TypeVar("_C", bound="ChannelNamespace")

# Weather keys the chain fills with a default rather than requiring an
# upstream feed (mirrors core.chain._WEATHER_DEFAULTS); a required-but-absent
# column outside this set makes a chunk invalid.
_OPTIONAL_WEATHER_KEYS: frozenset = frozenset({Weather.CLEAR_SKY_INDEX, Weather.HUMIDITY_REL})


@dataclass(frozen=True)
class ChannelSpec:
    key: str
    type: type
    name: str
    unit: str = ""
    aggregate: str = "mean"
    logged: bool = False  # default; the .d file's [data.channels.<key>.logger] overrides
    column: Optional[str] = None
    table: Optional[str] = None


def _register_channel(data: Any, spec: ChannelSpec) -> None:
    """One ``data.add`` call from a ``ChannelSpec``, shared by every
    ``ChannelNamespace`` and by ``FieldSimulation`` itself (whose vegetation
    and per-segment channels are not owned by any single child)."""
    logger_cfg: dict[str, Any] = {"enabled": spec.logged}
    if spec.table is not None:
        logger_cfg["table"] = spec.table
    if spec.column is not None:
        logger_cfg["column"] = spec.column
    data.add(spec.key, type=spec.type, name=spec.name, unit=spec.unit, aggregate=spec.aggregate, logger=logger_cfg)


class ChannelNamespace(Component):
    """A Component that exists to own channels under its id and to validate
    its ``.d`` file against one ``Config`` class. No logic."""

    CHANNELS: ClassVar[Sequence[ChannelSpec]] = ()
    CONFIG: ClassVar[Optional[Type[Config]]] = None

    config: Optional[Config] = None

    def configure(self, configs: Configurations) -> None:
        super().configure(configs)
        if self.CONFIG is not None:
            self.config = self.CONFIG()
            self.config.configure(configs)  # resolves, applies defaults, rejects unknown keys
        for spec in self.CHANNELS:
            _register_channel(self.data, spec)

    @classmethod
    def schema(cls) -> dict[str, Any]:
        return cls.CONFIG.schema() if cls.CONFIG is not None else {}


class GroundShading(ChannelNamespace):
    TYPE: str = "ground_shading"
    CONFIG = ShadingConfig
    CHANNELS = (
        ChannelSpec("shading_factor", float, "Mean Ground Shading Factor", "-", logged=False),
        ChannelSpec(
            "shading_progress_image",
            bytes,
            "Ground Shading Progress Image",
            "png",
            aggregate="last",
            logged=True,
        ),
        ChannelSpec("plot_strikes", float, "Plot Strikes", "-", aggregate="last", logged=False),
    )


class Evapotranspiration(ChannelNamespace):
    TYPE: str = "evapotranspiration"
    CONFIG = None  # no keys of its own today; its .d file carries only channels
    CHANNELS = (
        ChannelSpec("sat_vapor_pressure", float, "Saturation Vapor Pressure", "kPa"),
        ChannelSpec("ground_vapor_pressure", float, "Vapor Pressure on the Ground Surface", "kPa"),
        ChannelSpec("vaporization_heat", float, "Latent Heat of Vaporization", "J/kg"),
        ChannelSpec("slope_sat_vapor_pressure", float, "Saturation Vapor Pressure Slope", "kPa/K"),
        ChannelSpec("net_irradiance", float, "Net Irradiance", "W/m^2"),
        ChannelSpec("aerodynamic_resistance", float, "Aerodynamic Resistance", "s/m"),
        ChannelSpec("soil_heat_flow", float, "Soil Heat Flow", "W/m^2"),
        ChannelSpec("resistance_surface", float, "Surface Resistance", "s/m"),
        ChannelSpec("radiation_term", float, "Radiation Term", "(kPa*W)/(K*m^2)"),
        ChannelSpec("aerodynamic_term", float, "Aerodynamic Term", "(kPa*J)/(m^2*K*s)"),
        ChannelSpec("evapotranspiration", float, "Evapotranspiration", "kg/(m^2*h)"),
    )


class SoilSimulation(ChannelNamespace):
    TYPE: str = "soil_simulation"
    CONFIG = SoilConfig
    CHANNELS = (
        ChannelSpec(
            "simulation_state", bytes, "Soil Simulation State", "-", aggregate="last", logged=True, column="state"
        ),
        ChannelSpec(
            "soil_progress_image",
            bytes,
            "Soil Simulation Progress Image",
            "png",
            aggregate="last",
            logged=True,
            column="image",
        ),
        ChannelSpec("plot_strikes", float, "Plot Strikes", "-", aggregate="last", logged=False),
        ChannelSpec("top_in", float, "Top Water Input (Irrigation + Rain)", "kg/(m^2*h)", logged=True),
        ChannelSpec("top_out", float, "Top Water Output (Evaporation)", "kg/(m^2*h)", logged=True),
        ChannelSpec("bottom_out", float, "Bottom Water Output (Drainage)", "kg/(m^2*h)", logged=True),
        ChannelSpec("transpiration", float, "Plant Transpiration", "kg/(m^2*h)", logged=True),
        ChannelSpec("runoff", float, "Rejected Top Influx (Runoff)", "kg/(m^2*h)", logged=True),
        ChannelSpec("demand_unmet", float, "Unmet Evap+Transp Demand", "kg/(m^2*h)", logged=True),
        ChannelSpec("balance_residual", float, "Mass-Balance Residual (integral - direct)", "kg/(m^2*h)", logged=True),
        ChannelSpec("anchor", float, "Anchor Assimilation Increment", "kg/m", logged=True),
        ChannelSpec("skipped_s", float, "Skipped Walk Duration (dt_min Unsolvable)", "s", logged=True),
        ChannelSpec("retries", float, "Walk Substep Retries", "-", logged=True),
        ChannelSpec("weather_stall", float, "Consecutive Weather-Stall Ticks", "-", logged=True),
        ChannelSpec("tick_failures", float, "Consecutive Tick Failures", "-", logged=True),
    )

    def register_probe(self, probe: Any) -> None:
        """One float channel per resolved config probe (today's
        ``SoilSimulation._register_probe``). Sensor-derived probes stay
        sample-only -- never called for those (see ``FieldSimulation.activate``)."""
        self.data.add(
            probe.channel_id,
            type=float,
            name=probe.name,
            unit="hPa",
            aggregate="mean",
            logger={"enabled": True, "table": "agri_soil_simulation", "column": "water_tension"},
        )


class SoilPredictor(ChannelNamespace):
    TYPE: str = "soil_predictor"
    CONFIG = PlannerConfig
    CHANNELS = ()  # forecast header / detail / irrigation / image are logger tables, written directly


# --------------------------------------------------------------------------- field-level channels

# Vegetation/ground-surface state, registered on FieldSimulation itself (today's
# FieldSimulation.VEGETATION_CHANNELS): no child owns the field's canopy state.
# TEMP_GROUND is registered (parity with today) but never written by this seam --
# today it is derived per-segment inside Evapotranspiration.evaluate's publish
# path, which this port's WeatherChain/ETModel does not reproduce (see the report).
_VEGETATION_CHANNELS = (
    ChannelSpec("temp_ground", float, "Ground Temperature", "°C", logged=False),
    ChannelSpec("lai", float, "Leaf Area Index", "m^2/m^2", logged=False),
    ChannelSpec("roughness", float, "Roughness", "-", logged=False),
    ChannelSpec("plant_height", float, "Plant Height", "m", logged=False),
    ChannelSpec("ndvi", float, "Normalized Difference Vegetation Index", "-", logged=False),
)
_VEGETATION_WRITE_KEYS = ("lai", "roughness", "plant_height", "ndvi")

# Bundled per-segment channels (today's FieldSimulation.SEGMENT_CHANNELS): each
# holds a list[float] ordered by top_segment_names. Only SEG_GHI is derivable
# from this port's ChainResult (the shading frame's ghi_<segment> columns);
# SEG_EVAPOTRANSPIRATION/SEG_TEMP_GROUND need the per-segment seg_et decomposition,
# which core.state.ChainResult does not carry (see the report).
_SEGMENT_CHANNELS = (
    ChannelSpec("seg_ghi", list, "GHI (per segment)", "W/m^2", aggregate="last", logged=False),
    ChannelSpec(
        "seg_evapotranspiration", list, "Evapotranspiration (per segment)", "kg/(m^2*h)", aggregate="last", logged=False
    ),
    ChannelSpec("seg_temp_ground", list, "Ground Temperature (per segment)", "°C", aggregate="last", logged=False),
)

_ET_CHANNEL_KEYS = tuple(spec.key for spec in Evapotranspiration.CHANNELS)
_SOIL_DIAGNOSTIC_KEYS = (
    "top_in",
    "top_out",
    "bottom_out",
    "transpiration",
    "runoff",
    "demand_unmet",
    "balance_residual",
    "skipped_s",
    "retries",
)


def _walk_components(root: Any) -> list:
    """Flatten a component subtree into a list, root first (port of
    ``simulation._anchor_runtime._walk_components``)."""
    out: list = []
    stack = [root]
    while stack:
        c = stack.pop()
        out.append(c)
        children = getattr(c, "components", None)
        if children:
            try:
                stack.extend(list(children.values()))
            except (AttributeError, TypeError):
                try:
                    stack.extend(list(children))
                except Exception:  # noqa: BLE001
                    logger.warning("could not iterate the children of %s; skipping its subtree.", getattr(c, "key", c))
    return out


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
            channel = soil.data["simulation_state"]
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
        if len(index) == 0:
            return pd.DataFrame(columns=["irrigation_flow_lpm"])
        field = self.field
        measured = self._read_measured_flow(start, end, index)
        if measured is not None:
            series = measured
        elif field._irrigation_state_channel is not None and field.setup.soil.drip.explicit:
            series = self._read_state_span(start, end, index) * field.setup.soil.drip.design_flow_lpm
        else:
            return pd.DataFrame(columns=["irrigation_flow_lpm"])
        return series.to_frame("irrigation_flow_lpm")

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

        # One column per discovered sensor, even when its read comes back empty
        # (an empty pd.Series contributes no rows but still holds the column's
        # place in the concat below) -- absent readings show up as NaN once
        # another sensor's rows widen the shared index, per the port contract.
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
    out per row; writes plan tables best effort, one try/except each."""

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
                self.shading.data["shading_factor"].set(ts, factor)
            if result.image is not None:
                self.shading.data["shading_progress_image"].set(now, result.image)

            ghi_cols = {c[len("ghi_") :]: c for c in shading_df.columns if c.startswith("ghi_")}
            segment_names = list(self._top_segment_names())
            if ghi_cols and segment_names:
                for ts in shading_df.index:
                    values = [
                        float(shading_df.loc[ts, ghi_cols[name]]) if name in ghi_cols else 0.0 for name in segment_names
                    ]
                    self.field.data["seg_ghi"].set(ts, values)

        if self.et is not None and not et_df.empty:
            for key in _ET_CHANNEL_KEYS:
                if key not in et_df.columns:
                    continue
                for ts in et_df.index:
                    self.et.data[key].set(ts, float(et_df.loc[ts, key]))
            for key in _VEGETATION_WRITE_KEYS:
                if key not in et_df.columns:
                    continue
                for ts in et_df.index:
                    self.field.data[key].set(ts, float(et_df.loc[ts, key]))

    def step(self, result: StepResult) -> None:
        if self.soil is None:
            return
        ts = result.state.at
        for key in _SOIL_DIAGNOSTIC_KEYS:
            if key in result.diagnostics:
                self.soil.data[key].set(ts, float(result.diagnostics[key]))
        for probe_id, tension in result.probe_tension.items():
            try:
                self.soil.data[probe_id].set(ts, float(tension))
            except Exception:  # noqa: BLE001
                continue  # sensor-derived probe: sample-only, no registered channel

        assimilator = getattr(getattr(self.field, "simulation", None), "assimilator", None)
        last_result = getattr(assimilator, "last_result", None)
        if last_result is not None and last_result.innovations:
            self.soil.data["anchor"].set(ts, float(sum(last_result.innovations.values())))

    def plan(self, plan: Plan) -> None:
        if self.predictor is None:
            return
        connector = self._resolve_logger_connector()
        if connector is None:
            return
        self._write_table(connector, plan.header, "header table")
        self._write_table(connector, plan.detail, "detail table")
        self._write_table(connector, plan.irrigation, "irrigation table")
        if plan.image is not None:
            self._write_table(connector, plan.image, "image table")

    def save_state(self, state: SoilState) -> None:
        if self.soil is None:
            return
        self.soil.data["simulation_state"].set(state.at, state.to_blob())

    # -- internal -------------------------------------------------------------

    def _top_segment_names(self) -> Sequence[str]:
        simulation = getattr(self.field, "simulation", None)
        engine = getattr(simulation, "engine", None)
        return getattr(engine, "top_segment_names", ()) or ()

    def _resolve_logger_connector(self) -> Any:
        setup = getattr(self.field, "setup", None)
        planner = getattr(setup, "planner", None)
        logger_id = getattr(planner, "logger", None)
        if logger_id is None:
            return None
        connectors = getattr(self.predictor, "connectors", None)
        if connectors is None:
            return None
        connector = getattr(connectors, logger_id, None)
        if connector is None:
            try:
                connector = connectors[logger_id]
            except (KeyError, TypeError):
                connector = None
        return connector

    def _write_table(self, connector: Any, frame: Optional[pd.DataFrame], table_label: str) -> None:
        if frame is None or frame.empty:
            return
        write_frame = frame
        ids = getattr(self.predictor, "_forecast_channel_ids", None)
        if ids:
            write_frame = frame.rename(columns={c: ids[c] for c in frame.columns if c in ids})
        try:
            connector.write(write_frame)
        except Exception:  # noqa: BLE001
            logger.exception(
                "%s: direct write of the %s (%d rows) failed.",
                getattr(self.field, "name", "fieldsim"),
                table_label,
                len(write_frame),
            )


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

        self.ground_shading = self._child(GroundShading, configs)
        self.evapotranspiration = self._child(Evapotranspiration, configs)
        self.soil = self._child(SoilSimulation, configs)
        self.predictor = self._child(SoilPredictor, configs)
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

        for spec in _VEGETATION_CHANNELS:
            _register_channel(self.data, spec)
        if self._top_segment_names_from_config():
            for spec in _SEGMENT_CHANNELS:
                _register_channel(self.data, spec)

        self.simulation: Optional[Simulation] = None
        self.runner: Optional[FieldRunner] = None
        self.scheduler: Optional[TickScheduler] = None

    def activate(self) -> None:
        super().activate()

        system = self.context.context.context
        self.location = getattr(system, "location", None)
        self.weather = getattr(system, "weather", None)
        self.irrigation = getattr(self.context, "irrigation", None)

        if self.evapotranspiration is None:
            return
        if self.weather is None:
            logger.warning("%s: no Weather component resolved; chain will never tick.", self.name)
            return

        self._weather_channels = Channels(list(self.weather.data.values()))
        self._required_weather_keys = ETModel.REQUIRED_WEATHER_COLUMNS
        self._evapo_rename = {c.id: c.key for c in self._weather_channels}

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
        if self.setup.soil.raw is not None:
            probes_block = self.setup.soil.raw.get_member("probes", defaults={}, ensure_exists=True)
            config_probes = self.simulation.engine.probes(probes_block)
        for probe in config_probes:
            self.soil.register_probe(probe)

        self._register_forecast_table_channels()

        self.inputs = ChannelInputs(self)
        self.outputs = ChannelOutputs(self, self.ground_shading, self.evapotranspiration, self.soil, self.predictor)
        self.runner = FieldRunner(self.setup, self.simulation, self.inputs, self.outputs)
        self.scheduler = TickScheduler(self.setup, self.runner, tz=getattr(self.location, "timezone", dt.timezone.utc))
        self.scheduler.start()

    def deactivate(self) -> None:
        if self.scheduler is not None:
            self.scheduler.stop()
        super().deactivate()

    @property
    def snapshot(self) -> Snapshot:
        """What dash reads. Never a channel round-trip."""
        return self.simulation.snapshot()

    @classmethod
    def schema(cls) -> dict[str, Any]:
        """The whole tree of needed and allowed keys: own section, [plot],
        and one entry per child type. What a config editor reads."""
        tree: dict[str, Any] = dict(FieldConfig.schema())
        tree["plot"] = {"type": "group", "children": PlotConfig.schema()}
        for child in cls.CHILDREN:
            tree[child.TYPE] = {"type": "component", "children": child.schema()}
        return tree

    def _child(self, cls: Type[_C], configs: Configurations) -> Optional[_C]:
        """Build one channel namespace from its ``.d`` member, if present."""
        if not configs.has_member(cls.TYPE, includes=True):
            return None
        child = cls(self, configs.get_member(cls.TYPE))
        self.components.add(child)
        return child

    def _top_segment_names_from_config(self) -> list[str]:
        """Pure top-segment-name derivation from the resolved mesh config
        (today's ``top_segment_names_from_mesh``), needed at configure time
        -- before the FiPy engine exists -- to decide whether the per-segment
        channels are registered at all."""
        from sparcs.components.agriculture.simulation._soil import top_segment_names_from_mesh

        mesh = self.setup.soil.mesh
        if mesh.width is None:
            return []
        return top_segment_names_from_mesh(mesh)

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
        sensor in this field (port of ``_anchor_runtime.AnchorRuntime.discover``,
        walked from ``self.context`` -- ``FieldSimulation``'s own parent field --
        one hop shorter than the live ``SoilSimulation``-rooted walk)."""
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
        """Port of ``AnchorRuntime.validate``: with ``[anchor]`` enabled, a
        discovery failure or zero discovered sensors refuses startup; with
        only ``discover_sensor_probes`` on, failures are logged and the sim
        runs with whatever was found."""
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

    # -- forecast-table channel registration (header/detail/irrigation/image) ----

    def _register_forecast_table_channels(self) -> None:
        """Register the four persisted forecast-table channels directly on
        the predictor (port of ``_predictor_tables.ForecastTablePublisher``'s
        registration, simplified: no soil_id/field_id surrogate identity and
        no bound-channel-first connector resolution -- see the report).
        Skipped entirely when there is no predictor or no configured logger,
        same degrade as today."""
        predictor = self.predictor
        if predictor is None:
            return
        planner = self.setup.planner
        if planner is None or planner.logger is None:
            predictor._forecast_channel_ids = {}
            return

        ids: dict[str, str] = {}

        def add(
            key: str,
            type_: type,
            name: str,
            *,
            table: str,
            unit: Optional[str] = None,
            column: Optional[str] = None,
            primary: bool = False,
        ) -> None:
            logger_cfg: dict[str, Any] = {"connector": planner.logger, "table": table, "enabled": True}
            if column is not None:
                logger_cfg["column"] = column
            if primary:
                logger_cfg["primary"] = True
                logger_cfg["nullable"] = False
            kwargs: dict[str, Any] = {"type": type_, "name": name}
            if unit is not None:
                kwargs["unit"] = unit
            predictor.data.add(key, aggregate="last", logger=logger_cfg, **kwargs)
            ids[key] = predictor.data[key].id

        header_table = "agri_field_forecast"
        add("forecast_id", int, "Forecast candidate id", table=header_table, primary=True)
        for i in range(planner.max_windows):
            add(f"w{i}_min", float, f"Window {i} duration", table=header_table, unit="min")
        for i in range(planner.max_windows):
            add(f"w{i}_start", str, f"Window {i} start", table=header_table)
        add("is_recommended", bool, "Is recommended candidate", table=header_table)
        add("total_min", float, "Total watering duration", table=header_table, unit="min")
        add("weather_creation", pd.Timestamp, "Weather forecast issue time", table=header_table)

        detail_table = "agri_soil_forecast"
        for probe in self.setup.soil.probe_specs:
            key = f"traj_{probe.channel_id}"
            add(key, float, f"Trajectory {probe.name}", table=detail_table, unit="hPa", column="water_tension")
            add(
                f"{key}_timestamp_creation",
                pd.Timestamp,
                f"Trajectory {probe.name} run timestamp",
                table=detail_table,
                column="timestamp_creation",
                primary=True,
            )
            add(
                f"{key}_forecast_id",
                int,
                f"Trajectory {probe.name} candidate id",
                table=detail_table,
                column="forecast_id",
                primary=True,
            )

        irrigation_table = "agri_field_forecast_irrigation"
        add("irrigation_state", bool, "Irrigation plan state", table=irrigation_table)
        add(
            "irrigation_timestamp_creation",
            pd.Timestamp,
            "Irrigation plan run timestamp",
            table=irrigation_table,
            column="timestamp_creation",
            primary=True,
        )

        image_table = "agri_field_forecast_image"
        add("predict_image", bytes, "Predicted soil field image", table=image_table, unit="png", column="image")
        add(
            "predict_image_timestamp_creation",
            pd.Timestamp,
            "Predicted image run timestamp",
            table=image_table,
            column="timestamp_creation",
            primary=True,
        )

        predictor._forecast_channel_ids = ids
