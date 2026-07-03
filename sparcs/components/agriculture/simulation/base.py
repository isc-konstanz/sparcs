# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.simulation.base
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Container component that owns the soil-simulation chain:
``GroundShading`` -> ``Evapotranspiration`` -> ``SoilSimulation``.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Type, TypeVar

import pandas as pd
from lories import Component, Constant
from lories.components.weather import Weather
from lories.data import Channels
from lories.typing import Configurations
from sparcs.components.agriculture.irrigation import Irrigation
from sparcs.components.weather import validate_meteo_inputs

from .evapotranspiration import Evapotranspiration, SegmentProperties
from .ground_shading import GroundShading
from .soil import MeshConfig, SoilSimulation, top_segment_names_from_mesh
from .soil_predictor import SoilPredictor

_C = TypeVar("_C", bound=Component)


# Monthly LAI lookup tables keyed by `lai_type` config option.
_LAI_BY_TYPE: dict[str, list[float]] = {
    "fao": [3.0] * 12,
    "grass": [0.2, 0.2, 0.2, 0.3, 0.6, 0.8, 0.9, 1.2, 1.4, 1.2, 0.8, 0.6],
    "apple": [0.2, 0.4, 1.2, 2.5, 3.0, 3.2, 3.0, 2.8, 2.0, 1.0, 0.5, 0.2],
}


class FieldSimulation(Component):
    TYPE: str = "field_simulation"
    INCLUDES = [GroundShading.TYPE, Evapotranspiration.TYPE, SoilSimulation.TYPE, SoilPredictor.TYPE]

    # Vegetation/ground-surface state channels consumed by Evapotranspiration.
    TEMP_GROUND = Constant(float, "temp_ground", "Ground Temperature", "°C")
    LAI = Constant(float, "lai", "Leaf Area Index", "m^2/m^2")
    ROUGHNESS = Constant(float, "roughness", "Roughness", "-")
    PLANT_HEIGHT = Constant(float, "plant_height", "Plant Height", "m")
    NDVI = Constant(float, "ndvi", "Normalized Difference Vegetation Index", "-")

    VEGETATION_CHANNELS = [TEMP_GROUND, LAI, ROUGHNESS, PLANT_HEIGHT, NDVI]

    # Bundled per-segment channels, each holds a ``list[float]`` ordered by ``top_segment_names``.
    SEG_GHI = Constant(list, "seg_ghi", "GHI (per segment)", "W/m^2")
    SEG_EVAPOTRANSPIRATION = Constant(list, "seg_evapotranspiration", "Evapotranspiration (per segment)", "kg/(m^2*h)")
    SEG_TEMP_GROUND = Constant(list, "seg_temp_ground", "Ground Temperature (per segment)", "°C")
    SEGMENT_CHANNELS = [SEG_GHI, SEG_EVAPOTRANSPIRATION, SEG_TEMP_GROUND]

    ground_shading: Optional[GroundShading] = None
    evapotranspiration: Optional[Evapotranspiration] = None
    soil_simulation: Optional[SoilSimulation] = None
    soil_predictor: Optional[SoilPredictor] = None

    location: Any = None
    weather: Optional[Weather] = None
    irrigation: Optional[Irrigation] = None

    _lai_type: str = "grass"
    _evapo_rename: dict[str, str]
    _irrigation_flow_lpm: float = 0.0

    _weather_channels: Optional[Channels] = None
    _required_weather_keys: tuple[str, ...] = ()
    _weather_default_warned: set[str]

    # Weather keys filled with a default when the feed doesn't supply them (one-shot warning per key).
    _WEATHER_DEFAULTS: dict[str, float] = {
        "clear_sky_index": 0.5,
        "humidity_relative": 60.0,  # %
    }
    _OPTIONAL_WEATHER_KEYS: frozenset[str] = frozenset(_WEATHER_DEFAULTS.keys())

    # Bay width (m), shared by soil mesh and ground-shading children.
    _bay_width: float = 3.5

    # Latest per-segment shade factors from GroundShading; {} = no data yet (treated as 1.0).
    _segment_shade: dict[str, float]

    # Bare-soil sentinels for non-canopy top segments. LAI=1.0 keeps PM surface-resistance
    # finite; small height/roughness reflects bare-soil micro-roughness.
    _BARE_LAI: float = 1.0
    _BARE_PLANT_HEIGHT: float = 0.05
    _BARE_NDVI: float = 0.10
    _BARE_ROUGHNESS: float = 0.002
    _CANOPY_SEGMENT_NAMES = ("PlantTopLeftSegment", "PlantTopRightSegment")

    def configure(self, configs: Configurations) -> None:
        super().configure(configs)
        defaults = Component._build_defaults(configs, includes=["model"], strict=True)

        self._lai_type = configs.get("lai_type", default="grass")
        if self._lai_type not in _LAI_BY_TYPE:
            raise ValueError(f"Unsupported lai_type '{self._lai_type}'. Must be one of: {sorted(_LAI_BY_TYPE)}")

        # TODO: remove this later from configs
        self.roughness = configs.get("roughness", default=0.002)
        self.plant_height = configs.get("plant_height", default=0.1)
        self.ndvi = configs.get("ndvi", default=0.25)

        self.bare_lai = configs.get("bare_lai", default=1.0)
        self.bare_roughness = configs.get("bare_roughness", default=0.002)
        self.bare_plant_height = configs.get("bare_plant_height", default=0.1)
        self.bare_ndvi = configs.get("bare_ndvi", default=0.25)

        self._bay_width = float(configs.get("bay_width", default=3.5))

        for c in self.VEGETATION_CHANNELS:
            self.data.add(c, aggregate="mean", logger={"enabled": False})

        self._mesh_config: Optional[MeshConfig] = None
        if configs.has_member(SoilSimulation.TYPE, includes=True):
            soil_block = configs.get_member(SoilSimulation.TYPE, defaults=defaults)
            mesh_block = soil_block.get_member("mesh", defaults={}, ensure_exists=True)
            self._mesh_config = MeshConfig(mesh_block, bay_width=self._bay_width)

        self._register_segment_channels()

        self.ground_shading = self._build_child(GroundShading, configs, defaults)
        self.evapotranspiration = self._build_child(Evapotranspiration, configs, defaults)
        self.soil_simulation = self._build_child(SoilSimulation, configs, defaults)
        # Predictor borrows mesh + probes from soil_simulation; must be built last.
        self.soil_predictor = self._build_child(SoilPredictor, configs, defaults)

        self._evapo_rename = {}
        self._segment_shade = {}
        self._vegetation_placeholder_warned = False
        self._weather_default_warned = set()

    def _register_segment_channels(self) -> None:
        """Register the three bundled ``list[float]`` segment channels.
        No-op when no SoilSimulation is wired."""
        if not self.top_segment_names:
            return
        for c in self.SEGMENT_CHANNELS:
            self.data.add(c, logger={"enabled": False})

    def _build_child(
        self,
        cls: Type[_C],
        configs: Configurations,
        defaults: dict[str, Any],
    ) -> Optional[_C]:
        if not configs.has_member(cls.TYPE, includes=True):
            return None
        child = cls(self, configs.get_member(cls.TYPE, defaults=defaults))
        self.components.add(child)
        return child

    @property
    def bay_width(self) -> float:
        """Bay width / row-spacing in metres."""
        return self._bay_width

    @property
    def mesh_config(self) -> Optional[MeshConfig]:
        # Prefer SoilSimulation's instance once configured; fall back to the eagerly-parsed copy.
        # getattr guards against the predictor reading this before SoilSimulation.configure() runs.
        if self.soil_simulation is not None:
            soil_mesh = getattr(self.soil_simulation, "_mesh_config", None)
            if soil_mesh is not None:
                return soil_mesh
        return self._mesh_config

    @property
    def top_segment_names(self) -> list[str]:
        # Empty when no MeshConfig is known.
        mesh = self.mesh_config
        if mesh is None:
            return []
        return top_segment_names_from_mesh(mesh)

    def set_segment_values(
        self,
        channel: Constant,
        timestamp: pd.Timestamp,
        mapping: dict[str, float],
    ) -> None:
        """Publish per-segment values as ``list[float]`` ordered by ``top_segment_names``.
        Raises if ``mapping`` keys don't exactly match registered segments."""
        names = self.top_segment_names
        if mapping.keys() != set(names):
            raise ValueError(
                f"set_segment_values({channel!s}) keys {sorted(mapping)} != registered segments {sorted(names)}"
            )
        self.data[channel].set(timestamp, [float(mapping[n]) for n in names])

    def get_segment_values(self, channel: Constant) -> dict[str, float]:
        """Latest per-segment values keyed by segment name. Empty dict
        before the first write."""
        value = self.data[channel].value
        if value is None:
            return {}
        return dict(zip(self.top_segment_names, map(float, value)))

    def has_soil_simulation(self) -> bool:
        return self.soil_simulation is not None and self.soil_simulation.is_enabled()

    def activate(self) -> None:
        super().activate()

        system = self.context.context.context
        self.location = system.location
        self.weather = system.weather

        self.irrigation = getattr(self.context, "irrigation", None)

        if self.evapotranspiration is None or self.soil_simulation is None:
            return

        if self.weather is None:
            logging.warning("%s: no Weather component resolved; chain will never tick.", self.name)
            return

        self._weather_channels = Channels(list(self.weather.data.values()))
        self._required_weather_keys = tuple(c.key for c in Evapotranspiration.REQUIRED_WEATHER_CHANNELS)
        self._evapo_rename = {c.id: c.key for c in self._weather_channels}

        soil_data = self.soil_simulation.data
        if not soil_data.simulation_state.has_logger():
            logging.warning(
                "%s: SIMULATION_STATE has no logger configured; soil state will not "
                "persist across restarts. Configure a logger on the channel to enable "
                "warm starts.",
                self.name,
            )
        if soil_data.simulation_state.has_connector():
            self.data.register(
                self._state_callback,
                soil_data.simulation_state,
                how="any",
                unique=True,
            )

        if self.soil_predictor is not None:
            self.data.register(
                self._predict_callback,
                Channels([soil_data.simulation_state]),
                how="any",
                unique=True,
                interval=self.soil_predictor.cooldown,
            )

        if self.irrigation is not None:
            try:
                flow_channel = self.irrigation.data[Irrigation.FLOW]
            except KeyError:
                flow_channel = None
            if flow_channel is not None:
                self.data.register(
                    self._irrigation_callback,
                    Channels([flow_channel]),
                    how="any",
                    unique=True,
                )

        self.data.register(
            self._weather_callback,
            self._weather_channels,
            how="any",
            unique=True,
        )

    def _weather_callback(self, data: pd.DataFrame) -> None:
        if not self._weather_inputs_valid():
            return
        frame = self._weather_channels.to_frame(unique=True)
        if frame.empty:
            return

        now = frame.index[-1]
        last = getattr(self.soil_simulation, "_last_simulated_at", None)
        if last is not None and now <= last:
            return

        et_data, seg_et = self._run_chain(frame.rename(columns=self._evapo_rename))
        now = et_data.index[-1]
        self.soil_simulation.advance(et_data, now, seg_et)

    def _predict_callback(self, data: pd.DataFrame) -> None:
        """Timed predictor trigger, decoupled from the live-sim weather listener.

        Registered on ``SoilSimulation``'s ``SIMULATION_STATE`` channel (published by
        ``advance()`` after every tick) so the heavy roll-out never runs on the
        weather-callback thread; the per-listener ``cooldown`` (``activate()``) plus
        the predictor's own daily self-gate (``SoilPredictor.predict``) keep this cheap.
        """
        if self.soil_predictor is None or data.empty:
            return
        now = data.index[-1]
        self.soil_predictor.predict(
            now,
            forecast_creation=self._read_forecast_epoch(),
        )

    def _read_forecast_epoch(self) -> Optional[pd.Timestamp]:
        forecast_sub = getattr(self.weather, "forecast", None)
        if forecast_sub is None:
            return None
        channel = forecast_sub.data.get("timestamp_creation", None)
        if channel is None or not channel.is_valid():
            return None
        value = channel.value
        # Collapse Series (one epoch repeated per row) to a scalar Timestamp.
        if isinstance(value, pd.Series):
            value = value.dropna()
            if value.empty:
                return None
            value = value.iloc[0]
        try:
            return pd.Timestamp(value)
        except (TypeError, ValueError):
            return None

    def _weather_inputs_valid(self) -> bool:
        if self._weather_channels is None:
            return False
        by_key = {c.key: c for c in self._weather_channels}
        missing = [
            k
            for k in self._required_weather_keys
            if k not in self._OPTIONAL_WEATHER_KEYS and (k not in by_key or not by_key[k].is_valid())
        ]
        if missing:
            logging.debug(
                "%s: skipping advance; weather channels not valid: %s",
                self.name,
                missing,
            )
            return False
        return True

    def _run_chain(
        self,
        weather: pd.DataFrame,
        *,
        publish: bool = True,
    ) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
        """Run weather → vegetation → ground shading → per-segment ET.

        Returns ``(df, seg_et)`` where ``seg_et`` maps segment name to a
        DataFrame of ``("et", "evap", "transp")`` in kg/(m²·s).
        ``publish=False`` suppresses channel writes (used by SoilPredictor
        on forecast input).
        """
        df = self._prepare_weather(weather)
        df = self._populate_vegetation(df, publish=publish)
        if self.ground_shading is not None:
            seg_factors = self.ground_shading.evaluate(df, publish=publish)
            if publish and seg_factors:
                self._segment_shade = dict(seg_factors)
        segments = self._build_segments(df)
        return self.evapotranspiration.evaluate(df, segments, publish=publish)

    def _build_segments(self, df: pd.DataFrame) -> list[SegmentProperties]:
        """Build the per-segment property list for ET evaluation.

        Canopy segments use bulk vegetation state; bare-soil segments use
        ``_BARE_*`` sentinels. Returns a single ``_bulk`` segment when no
        soil mesh is attached.
        """
        canopy_lai = float(df[self.LAI].iloc[-1])
        canopy_plant_height = float(df[self.PLANT_HEIGHT].iloc[-1])
        canopy_ndvi = float(df[self.NDVI].iloc[-1])
        canopy_roughness = float(df[self.ROUGHNESS].iloc[-1])

        if self.soil_simulation is None:
            return [
                SegmentProperties(
                    name="_bulk",
                    lai=canopy_lai,
                    plant_height=canopy_plant_height,
                    ndvi=canopy_ndvi,
                    roughness=canopy_roughness,
                    shade_factor=1.0,
                    face_length=1.0,
                    is_canopy=True,
                )
            ]

        soil = self.soil_simulation
        seg_props: list[SegmentProperties] = []
        for name in soil.top_segment_names():
            is_canopy = name in self._CANOPY_SEGMENT_NAMES
            seg_props.append(
                SegmentProperties(
                    name=name,
                    lai=canopy_lai if is_canopy else self.bare_lai,
                    plant_height=canopy_plant_height if is_canopy else self.bare_plant_height,
                    ndvi=canopy_ndvi if is_canopy else self.bare_ndvi,
                    roughness=canopy_roughness if is_canopy else self.bare_roughness,
                    shade_factor=float(self._segment_shade.get(name, 1.0)),
                    face_length=float(soil.segment_face_length(name)),
                    is_canopy=is_canopy,
                )
            )
        return seg_props

    def _populate_vegetation(self, df: pd.DataFrame, *, publish: bool = True) -> pd.DataFrame:
        # TEMP_GROUND is derived per-segment by Evapotranspiration, not set here.
        if not self._vegetation_placeholder_warned:
            logging.warning(
                "%s: using placeholder vegetation state (LAI from monthly table '%s', "
                "ROUGHNESS=0.002, PLANT_HEIGHT=0.1, NDVI=0.25); "
                "no Crop subcomponent or field-level sensors are publishing these "
                "channels yet. Values are still written for debugging visibility.",
                self.name,
                self._lai_type,
            )
            self._vegetation_placeholder_warned = True
        df[self.LAI] = pd.array(_LAI_BY_TYPE[self._lai_type])[df.index.month - 1].astype(float)
        df[self.ROUGHNESS] = self.roughness
        df[self.PLANT_HEIGHT] = self.plant_height
        df[self.NDVI] = self.ndvi

        for key, default in self._WEATHER_DEFAULTS.items():
            if key not in df.columns or df[key].isna().all():
                if key not in self._weather_default_warned:
                    logging.warning(
                        "%s: weather feed does not supply '%s'; defaulting to %s.",
                        self.name,
                        key,
                        default,
                    )
                    self._weather_default_warned.add(key)
                df[key] = default
            elif df[key].isna().any():
                df[key] = df[key].fillna(default)

        if publish and not df.empty:
            ts = df.index[-1]
            for c in self.VEGETATION_CHANNELS:
                if c is self.TEMP_GROUND:
                    continue  # published by Evapotranspiration as the per-segment bulk mean
                self.data[c].set(ts, float(df[c].iloc[-1]))
        return df

    def _state_callback(self, data: pd.DataFrame) -> None:
        if data.empty or self.soil_simulation is None:
            return
        self.soil_simulation.apply_state_blob(data.iloc[0, 0], data.index[0])

    def _irrigation_callback(self, data: pd.DataFrame) -> None:
        try:
            flow = float(data.iloc[-1, 0])
        except (ValueError, TypeError, IndexError):
            flow = 0.0
        # A NULL flow row means "not watering". ``float(np.nan)`` does not raise,
        # so without this guard a NaN latches into the PDE source (soil.py) and
        # poisons it -- the live analog of the bench asof-latch fix (20431c7).
        self._irrigation_flow_lpm = 0.0 if pd.isna(flow) else flow

    def _prepare_weather(self, data: pd.DataFrame) -> pd.DataFrame:
        return validate_meteo_inputs(data, self.location)

    def simulate(
        self,
        weather: pd.DataFrame,
        prior: Optional[pd.DataFrame] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        if not self.has_soil_simulation() or self.evapotranspiration is None:
            return pd.DataFrame()

        self._restore_prior_state(prior)
        et_data, seg_et = self._run_chain(weather.copy())
        return self.soil_simulation.simulate_loop(et_data, seg_et)

    def _restore_prior_state(self, prior: Optional[pd.DataFrame]) -> None:
        if prior is None or prior.empty:
            return
        state_channel = self.soil_simulation.data[SoilSimulation.SIMULATION_STATE]
        if state_channel.id not in prior.columns:
            return
        blob = prior[state_channel.id].iloc[-1]
        if isinstance(blob, (bytes, bytearray)) and len(blob) > 0:
            self.soil_simulation.apply_state_blob(bytes(blob), prior.index[-1])
