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
import re
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


# Per-segment channel key conventions. Kept here because the channels are
# registered on ``FieldSimulation`` (it knows the mesh and so the segment
# list); GroundShading and Evapotranspiration just write into them.
def _segment_key_suffix(seg_name: str) -> str:
    """Camel-case soil-mesh segment names → snake_case channel-id suffix."""
    return re.sub(r"(?<!^)(?=[A-Z])", "_", seg_name).lower()


# All per-segment channels share a common ``seg_<segment>_<quantity>``
# layout so the natural-sort UI groups them by segment (each segment's
# ET / GHI / T_gnd shown in one block) and the whole per-segment
# block sits between the bulk vegetation channels (``lai``, ``ndvi``,
# ``plant_height``, ``roughness``) and the bulk ``temp_ground``.
def segment_ghi_key(seg_name: str) -> str:
    return f"seg_{_segment_key_suffix(seg_name)}_ghi"


def segment_et_key(seg_name: str) -> str:
    return f"seg_{_segment_key_suffix(seg_name)}_evapotranspiration"


def segment_temp_ground_key(seg_name: str) -> str:
    return f"seg_{_segment_key_suffix(seg_name)}_temp_ground"

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
    INCLUDES = [GroundShading.TYPE, Evapotranspiration.TYPE, SoilSimulation.TYPE]

    # Vegetation/ground-surface state channels. They describe the field, not
    # the ET calculation, so they live here. Evapotranspiration consumes them
    # by column-key from the DataFrame the chain hands it.
    TEMP_GROUND = Constant(float, "temp_ground", "Ground Temperature", "°C")
    LAI = Constant(float, "lai", "Leaf Area Index", "m^2/m^2")
    ROUGHNESS = Constant(float, "roughness", "Roughness", "-")
    PLANT_HEIGHT = Constant(float, "plant_height", "Plant Height", "m")
    NDVI = Constant(float, "ndvi", "Normalized Difference Vegetation Index", "-")

    VEGETATION_CHANNELS = [TEMP_GROUND, LAI, ROUGHNESS, PLANT_HEIGHT, NDVI]

    ground_shading: Optional[GroundShading] = None
    evapotranspiration: Optional[Evapotranspiration] = None
    soil_simulation: Optional[SoilSimulation] = None

    location: Any = None
    weather: Optional[Weather] = None
    irrigation: Optional[Irrigation] = None

    _lai_type: str = "grass"
    _evapo_rename: dict[str, str]
    _irrigation_flow_lpm: float = 0.0

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
        defaults = Component._build_defaults(configs, strict=True)

        self._lai_type = configs.get("lai_type", default="grass")
        if self._lai_type not in _LAI_BY_TYPE:
            raise ValueError(
                f"Unsupported lai_type '{self._lai_type}'. "
                f"Must be one of: {sorted(_LAI_BY_TYPE)}"
            )

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
        # and write into these channels via ``self.context.data[...]``.
        self._segment_ghi_keys: dict[str, str] = {}
        self._segment_et_keys: dict[str, str] = {}
        self._segment_temp_ground_keys: dict[str, str] = {}
        self._register_segment_channels()

        # Each chain child is a singleton — build directly via the same
        # has_member / get_member pattern instead of mixing in load_from_type
        # (which exists for multi-instance siblings like SoilMoisture).
        self.ground_shading = self._build_child(GroundShading, configs, defaults)
        self.evapotranspiration = self._build_child(Evapotranspiration, configs, defaults)
        self.soil_simulation = self._build_child(SoilSimulation, configs, defaults)

        self._evapo_rename = {}
        self._segment_shade = {}

    def _register_segment_channels(self) -> None:
        """Add one ``ghi_<seg>``, ``evapotranspiration_<seg>``, and
        ``temp_ground_<seg>`` channel per soil-mesh top segment. No-op
        when no SoilSimulation is wired."""
        for seg_name in self.top_segment_names:
            ghi_key = segment_ghi_key(seg_name)
            et_key = segment_et_key(seg_name)
            tg_key = segment_temp_ground_key(seg_name)
            self._segment_ghi_keys[seg_name] = ghi_key
            self._segment_et_keys[seg_name] = et_key
            self._segment_temp_ground_keys[seg_name] = tg_key
            self.data.add(
                ghi_key,
                type=float,
                name=f"GHI ({seg_name})",
                unit="W/m^2",
                aggregate="mean",
                logger={"enabled": False},
            )
            self.data.add(
                et_key,
                type=float,
                name=f"Evapotranspiration ({seg_name})",
                unit="g/(m^2*s)",
                aggregate="mean",
                logger={"enabled": False},
            )
            self.data.add(
                tg_key,
                type=float,
                name=f"Ground Temperature ({seg_name})",
                unit="°C",
                aggregate="mean",
                logger={"enabled": False},
            )

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
        # constructed, prefer its instance (single source of truth at runtime).
        if self.soil_simulation is not None:
            return self.soil_simulation._mesh_config
        return self._mesh_config

    @property
    def top_segment_names(self) -> list[str]:
        # Names of the soil-mesh top segments where evaporation acts.
        # Empty when no MeshConfig is known (live-only field with no PDE).
        mesh = self.mesh_config
        if mesh is None:
            return []
        return top_segment_names_from_mesh(mesh)

    @property
    def segment_ghi_keys(self) -> dict[str, str]:
        """Read-only view: segment name → ``ghi_<seg>`` channel id on this
        component. GroundShading uses this to publish per-segment GHI."""
        return dict(self._segment_ghi_keys)

    @property
    def segment_et_keys(self) -> dict[str, str]:
        """Read-only view: segment name → ``evapotranspiration_<seg>``
        channel id on this component. Evapotranspiration uses this to
        publish per-segment ET."""
        return dict(self._segment_et_keys)

    @property
    def segment_temp_ground_keys(self) -> dict[str, str]:
        """Read-only view: segment name → ``temp_ground_<seg>`` channel
        id on this component. Evapotranspiration uses this to publish
        per-segment ground temperature."""
        return dict(self._segment_temp_ground_keys)

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

        evapo_input_channels = Channels(
            [
                *self.data[self.VEGETATION_CHANNELS],
                *self.weather.data.values(),
            ]
        )
        if self.ground_shading is not None:
            evapo_input_channels.extend(
                self.ground_shading.data[GroundShading.CHANNELS]
            )
        self._evapo_rename = {c.id: c.key for c in evapo_input_channels}
        self.data.register(
            self._simulation_callback,
            evapo_input_channels,
            how="any",
            unique=True,
        )

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

    def _simulation_callback(self, data: pd.DataFrame) -> None:
        et_data, seg_et = self._run_chain(data.rename(columns=self._evapo_rename))
        self.soil_simulation.advance(et_data, et_data.index[-1], seg_et)

    def _run_chain(
        self, weather: pd.DataFrame
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
        """
        df = self._prepare_weather(weather)
        df = self._populate_vegetation(df)
        if self.ground_shading is not None:
            seg_factors = self.ground_shading.evaluate(df)
            if seg_factors:
                self._segment_shade = dict(seg_factors)
        segments = self._build_segments(df)
        return self.evapotranspiration.evaluate(df, segments)

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

    def _populate_vegetation(self, df: pd.DataFrame) -> pd.DataFrame:
        # Placeholder vegetation state — replace once a Crop subcomponent (or
        # field-level sensors) publishes these channels for real. TEMP_GROUND
        # is intentionally absent: Evapotranspiration derives it per-segment
        # from T_air and shade-scaled GHI and publishes the bulk mean back
        # onto self.data[TEMP_GROUND].
        df[self.LAI] = pd.array(_LAI_BY_TYPE[self._lai_type])[df.index.month - 1].astype(float)
        df[self.ROUGHNESS] = 0.002
        df[self.PLANT_HEIGHT] = 0.1
        df[self.NDVI] = 0.25
        df[Weather.CLEAR_SKY_INDEX] = 0.5

        # Mirror the latest values onto the actual vegetation channels so
        # dashboards/loggers see them in VALID state (DataFrame writes only
        # populate the ET-evaluation frame, not the channel registry).
        # Skip when the frame is empty — happens at the head of an offline
        # simulate before weather rows are accumulated.
        if not df.empty:
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
