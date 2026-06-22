# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.simulation.base
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Container component that owns the soil-simulation chain:
``GroundShading`` -> ``Evapotranspiration`` -> ``SoilSimulation``.

FieldSimulation owns the cross-system lookups (location, weather,
irrigation), the chain callback, and the offline ``simulate`` driver.
The chain children stay decoupled: they read what they need through
``self.context`` (this component) and never reach across siblings.
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


# Monthly LAI lookup tables. Placeholder until a Crop subcomponent publishes
# LAI from a real growth model — keys must stay in sync with the user-facing
# `lai_type` config option.
_LAI_BY_TYPE: dict[str, list[float]] = {
    "fao": [3.0] * 12,
    "grass": [0.2, 0.2, 0.2, 0.3, 0.6, 0.8, 0.9, 1.2, 1.4, 1.2, 0.8, 0.6],
    "apple": [0.2, 0.4, 1.2, 2.5, 3.0, 3.2, 3.0, 2.8, 2.0, 1.0, 0.5, 0.2],
}


class FieldSimulation(Component):
    TYPE: str = "field_simulation"
    INCLUDES = [GroundShading.TYPE, Evapotranspiration.TYPE, SoilSimulation.TYPE, SoilPredictor.TYPE]

    # Vegetation/ground-surface state channels. They describe the field, not
    # the ET calculation, so they live here. Evapotranspiration consumes them
    # by column-key from the DataFrame the chain hands it.
    TEMP_GROUND = Constant(float, "temp_ground", "Ground Temperature", "°C")
    LAI = Constant(float, "lai", "Leaf Area Index", "m^2/m^2")
    ROUGHNESS = Constant(float, "roughness", "Roughness", "-")
    PLANT_HEIGHT = Constant(float, "plant_height", "Plant Height", "m")
    NDVI = Constant(float, "ndvi", "Normalized Difference Vegetation Index", "-")

    VEGETATION_CHANNELS = [TEMP_GROUND, LAI, ROUGHNESS, PLANT_HEIGHT, NDVI]

    # Bundled per-segment channels — each holds a ``list[float]`` ordered by
    # ``top_segment_names``. Producers write through ``set_segment_values``;
    # consumers read through ``get_segment_values``. Replaces N-per-mesh
    # scalar channels (overwhelming during testing) with one per quantity.
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

    # Weather channels we can synthesize a default for when the connector
    # (or the forecast feed) can't supply them. Filled in ``_populate_vegetation``
    # before the chain consumes the frame, with a one-shot warning per key.
    # ``clear_sky_index`` is a derived quantity that DWD/Brightsky doesn't
    # publish at all; ``humidity_relative`` is missing from some Brightsky
    # forecast responses (the live observation usually carries it).
    _WEATHER_DEFAULTS: dict[str, float] = {
        "clear_sky_index": 0.5,
        "humidity_relative": 60.0,  # %
    }
    _OPTIONAL_WEATHER_KEYS: frozenset[str] = frozenset(_WEATHER_DEFAULTS.keys())

    # Single source of truth for the bay geometry — both the soil mesh
    # ``width`` and the ground-shading row ``distance`` derive from this
    # unless explicitly overridden in their own blocks. Defined here so
    # the two children cannot drift apart silently.
    _bay_width: float = 3.5

    # Latest per-segment shade factors from GroundShading. Owned here so the
    # ET segment list and (later) any other consumer can pick them up without
    # reaching into SoilSimulation. ``{}`` means "no shading data yet";
    # treated as open sky (factor 1.0) per segment.
    _segment_shade: dict[str, float]

    # Bare-soil sentinel values for top segments outside the canopy zone.
    # ``_BARE_LAI=1.0`` keeps Penman-Monteith well-behaved (avoids the
    # ``100/LAI`` surface-resistance blow-up) and gives a realistic
    # bare-soil surface resistance (~100 s/m). Plant-height/roughness
    # stay small so aerodynamic resistance reflects bare-soil micro-roughness.
    # The evap/transp split is gated on ``is_canopy`` (see
    # ``Evapotranspiration.evaluate``) — bare segments dump all their ET
    # to the surface, never to the plant-cell transpiration sink.
    _BARE_LAI: float = 1.0
    _BARE_PLANT_HEIGHT: float = 0.05
    _BARE_NDVI: float = 0.10
    _BARE_ROUGHNESS: float = 0.002
    _CANOPY_SEGMENT_NAMES = ("PlantTopLeftSegment", "PlantTopRightSegment")

    def configure(self, configs: Configurations) -> None:
        super().configure(configs)
        # Propagate a ``[model]`` block (inherited from the field) on down to
        # the chain children so the soil simulation shares the field-level
        # retention ground truth. No-op without a ``[model]`` block.
        defaults = Component._build_defaults(configs, includes=["model"], strict=True)

        self._lai_type = configs.get("lai_type", default="grass")
        if self._lai_type not in _LAI_BY_TYPE:
            raise ValueError(f"Unsupported lai_type '{self._lai_type}'. " f"Must be one of: {sorted(_LAI_BY_TYPE)}")

        self._bay_width = float(configs.get("bay_width", default=3.5))

        for c in self.VEGETATION_CHANNELS:
            self.data.add(c, aggregate="mean", logger={"enabled": False})

        # Soil-mesh top-segment names are derived purely from the MeshConfig
        # parameters. Pre-parse the mesh block here (cheap, no FiPy involved)
        # so per-segment channels can be registered now and the soil sibling
        # (built last in the chain) reuses the same mesh. The chain is
        # constructed in physical order: GroundShading → Evapotranspiration
        # → SoilSimulation.
        self._mesh_config: Optional[MeshConfig] = None
        if configs.has_member(SoilSimulation.TYPE, includes=True):
            soil_block = configs.get_member(SoilSimulation.TYPE, defaults=defaults)
            mesh_block = soil_block.get_member("mesh", defaults={}, ensure_exists=True)
            self._mesh_config = MeshConfig(mesh_block, bay_width=self._bay_width)

        # Per-segment channels live on ``FieldSimulation`` itself: it owns
        # the mesh layout, so segment naming and channel wiring (dummy
        # connectors, loggers, …) are all configured in one place. The
        # children (GroundShading, Evapotranspiration) compute the values
        # and write into these channels via ``set_segment_values``.
        self._register_segment_channels()

        # Each chain child is a singleton — build directly via the same
        # has_member / get_member pattern instead of mixing in load_from_type
        # (which exists for multi-instance siblings like SoilMoisture).
        self.ground_shading = self._build_child(GroundShading, configs, defaults)
        self.evapotranspiration = self._build_child(Evapotranspiration, configs, defaults)
        self.soil_simulation = self._build_child(SoilSimulation, configs, defaults)
        # Predictor is built last because it borrows mesh + probes from
        # ``soil_simulation`` and would crash without it.
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
        """Bay width / row-spacing in metres — the single shared knob that
        sets the soil mesh ``width`` and the ground-shading ``distance``."""
        return self._bay_width

    @property
    def mesh_config(self) -> Optional[MeshConfig]:
        # MeshConfig is parsed eagerly in ``configure()`` so siblings can read
        # it before SoilSimulation itself is built. Once SoilSimulation has
        # been *configured* (not just instantiated) prefer its instance —
        # single source of truth at runtime. Use ``getattr`` rather than
        # plain attribute access because ``soil_predictor`` configures
        # alphabetically before ``soil_simulation`` (lories sorts children
        # by id), so at predictor-configure time the SoilSimulation instance
        # exists but ``_mesh_config`` hasn't been assigned yet.
        if self.soil_simulation is not None:
            soil_mesh = getattr(self.soil_simulation, "_mesh_config", None)
            if soil_mesh is not None:
                return soil_mesh
        return self._mesh_config

    @property
    def top_segment_names(self) -> list[str]:
        # Names of the soil-mesh top segments where evaporation acts.
        # Empty when no MeshConfig is known (live-only field with no PDE).
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
        """Publish per-segment values as a single ``list[float]`` ordered
        by ``top_segment_names``. ``mapping`` must cover exactly the
        registered segments — a missing/extra key raises rather than
        silently misaligning the vector."""
        names = self.top_segment_names
        if mapping.keys() != set(names):
            raise ValueError(
                f"set_segment_values({channel!s}) keys {sorted(mapping)} " f"!= registered segments {sorted(names)}"
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

        # context chain: FieldSimulation -> AgriculturalField -> AgriculturalArea -> System
        system = self.context.context.context
        self.location = system.location
        self.weather = system.weather

        self.irrigation = getattr(self.context, "irrigation", None)

        if self.evapotranspiration is None or self.soil_simulation is None:
            return

        if self.weather is None:
            logging.warning("%s: no Weather component resolved — chain will never tick.", self.name)
            return

        # Trigger is timer-driven, not event-driven on these channels: the
        # vegetation channels are *written* inside the chain, so listening to
        # them caused self-retriggering; and weather cadence (often hours)
        # should not gate PDE cadence. We snapshot the latest validated weather
        # row on each tick instead.
        self._weather_channels = Channels(list(self.weather.data.values()))
        self._required_weather_keys = tuple(c.key for c in Evapotranspiration.REQUIRED_WEATHER_CHANNELS)
        self._evapo_rename = {c.id: c.key for c in self._weather_channels}

        soil_data = self.soil_simulation.data
        if not soil_data.simulation_state.has_logger():
            logging.warning(
                "%s: SIMULATION_STATE has no logger configured — soil state will not "
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

        # Live trigger: advance the soil one step per new observation frame.
        # Listening only on the weather observation channels (not vegetation
        # or ground_shading, which the chain *writes*) avoids self-retrigger.
        # This is also the predictor's trigger: ``_weather_callback`` calls
        # ``soil_predictor.predict`` after each ``soil.advance``. A separate
        # forecast-channel listener used to live here, but it tight-looped
        # on the future-dated ``timestamp_creation`` (and Brightsky returns
        # forecast + observations in one read, so the weather-channel
        # listener covers new forecast issuances anyway).
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

        # Brightsky's "current" source can return a row at the *upcoming*
        # hour boundary, so the weather channels' ``channel.timestamp`` may
        # land slightly in the future and ``Listener.has_update()`` keeps
        # the callback firing for the full notify-timeout window. Dedupe
        # against the live solver's last-simulated time so back-to-back
        # fires for the same observation row become no-ops before the heavy
        # chain replay runs.
        now = frame.index[-1]
        last = getattr(self.soil_simulation, "_last_simulated_at", None)
        if last is not None and now <= last:
            return

        et_data, seg_et = self._run_chain(frame.rename(columns=self._evapo_rename))
        now = et_data.index[-1]
        self.soil_simulation.advance(et_data, now, seg_et)

        # Trigger a fresh prediction on every committed soil advance: the
        # predictor's IC is the live state we just produced, and the
        # output trajectory shifts forward with ``now``. ``predict()`` has
        # its own ``(now, forecast_creation)`` dedup gate so re-fires for
        # an unchanged pair no-op out, and a cold-start advance is the
        # natural moment for the first prediction (no separate stash
        # needed). Brightsky reads forecast and observations together, so
        # this is also where new forecast issuances reach the predictor.
        if self.soil_predictor is not None:
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
        # Brightsky returns a multi-row forecast frame; ``set_frame`` then
        # stores ``timestamp_creation`` as a length-N ``pd.Series`` of
        # identical timestamps (all rows share one issue epoch). Collapse
        # that to a single Timestamp before constructing — ``pd.Timestamp``
        # raises ``TypeError`` on a Series directly.
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
                "%s: skipping advance — weather channels not valid: %s",
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
        """
        Walk the simulation chain up to (and including) Evapotranspiration:

            weather → vegetation → ground shading → per-segment ET

        Returns ``(df, seg_et)``. ``df`` is the ET-augmented frame (bulk
        output columns carry face-length-weighted means across segments).
        ``seg_et`` maps each soil-mesh top segment to a DataFrame of
        ``("et", "evap", "transp")`` time series in kg/(m²·s). The trailing
        soil step (``advance`` for the live callback, ``simulate_loop`` for
        offline) is the only thing that differs between drivers.

        ``publish=False`` (used by ``SoilPredictor`` on forecast input)
        suppresses every channel write and the cached ``_segment_shade``
        update so the predictor can replay the chain on future weather
        without contaminating live dashboards or the live solver's
        per-segment shading dict.
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
        """
        Build the per-segment property list the ET evaluation consumes.

        Canopy segments (``PlantTopLeft/Right``) inherit the field's bulk
        vegetation state (LAI from ``_LAI_BY_TYPE``, plant_height from the
        placeholder vegetation). Bare-soil top strips get LAI=0 + bare
        sentinels so PM applies pure soil evaporation there. ``shade_factor``
        comes from the latest GroundShading dict; absent entries default
        to 1.0 (open sky).

        With no soil mesh attached, returns a single representative segment
        carrying the bulk vegetation values so callers still get a usable
        bulk ET row.
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
                    lai=canopy_lai if is_canopy else self._BARE_LAI,
                    plant_height=canopy_plant_height if is_canopy else self._BARE_PLANT_HEIGHT,
                    ndvi=canopy_ndvi if is_canopy else self._BARE_NDVI,
                    roughness=canopy_roughness if is_canopy else self._BARE_ROUGHNESS,
                    shade_factor=float(self._segment_shade.get(name, 1.0)),
                    face_length=float(soil.segment_face_length(name)),
                    is_canopy=is_canopy,
                )
            )
        return seg_props

    def _populate_vegetation(self, df: pd.DataFrame, *, publish: bool = True) -> pd.DataFrame:
        # Placeholder vegetation state — replace once a Crop subcomponent (or
        # field-level sensors) publishes these channels for real. TEMP_GROUND
        # is intentionally absent: Evapotranspiration derives it per-segment
        # from T_air and shade-scaled GHI and publishes the bulk mean back
        # onto self.data[TEMP_GROUND].
        if not self._vegetation_placeholder_warned:
            logging.warning(
                "%s: using placeholder vegetation state (LAI from monthly table '%s', "
                "ROUGHNESS=0.002, PLANT_HEIGHT=0.1, NDVI=0.25) — "
                "no Crop subcomponent or field-level sensors are publishing these "
                "channels yet. Values are still written for debugging visibility.",
                self.name,
                self._lai_type,
            )
            self._vegetation_placeholder_warned = True
        df[self.LAI] = pd.array(_LAI_BY_TYPE[self._lai_type])[df.index.month - 1].astype(float)
        df[self.ROUGHNESS] = 0.002
        df[self.PLANT_HEIGHT] = 0.1
        df[self.NDVI] = 0.25

        # Required-input channels that the weather connector / forecast feed
        # may not supply (e.g. Brightsky doesn't publish ``clear_sky_index``;
        # the forecast endpoint sometimes lacks ``humidity_relative``). Fill
        # the default only where missing so real values pass through. Warn
        # once per key so partial-coverage providers don't spam the log.
        for key, default in self._WEATHER_DEFAULTS.items():
            if key not in df.columns or df[key].isna().all():
                if key not in self._weather_default_warned:
                    logging.warning(
                        "%s: weather feed does not supply '%s' — defaulting to %s.",
                        self.name,
                        key,
                        default,
                    )
                    self._weather_default_warned.add(key)
                df[key] = default
            elif df[key].isna().any():
                df[key] = df[key].fillna(default)

        # Mirror the latest values onto the actual vegetation channels so
        # dashboards/loggers see them in VALID state (DataFrame writes only
        # populate the ET-evaluation frame, not the channel registry).
        # Skip when the frame is empty — happens at the head of an offline
        # simulate before weather rows are accumulated. Forecast / dry-run
        # callers pass ``publish=False`` so future timestamps don't land on
        # the live channels.
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
            self._irrigation_flow_lpm = float(data.iloc[-1, 0])
        except (ValueError, TypeError, IndexError):
            self._irrigation_flow_lpm = 0.0

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
