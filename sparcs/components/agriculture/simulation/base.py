# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.simulation.base
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Container component that owns the soil-simulation chain:
``GroundShading`` -> ``Evapotranspiration`` -> ``SoilSimulation``.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Optional, Type, TypeVar

import pandas as pd
from lories import Component, Constant
from lories.components.weather import Weather
from lories.data import Channels
from lories.typing import Configurations, Timestamp
from lories.util import floor_date, to_timedelta
from sparcs.components.agriculture.irrigation import Irrigation
from sparcs.components.weather import validate_meteo_inputs

from ._soil import MeshConfig, top_segment_names_from_mesh
from .evapotranspiration import Evapotranspiration, SegmentProperties
from .ground_shading import GroundShading
from .soil import SoilSimulation
from .soil_predictor import SoilPredictor

logger = logging.getLogger(__name__)

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

    # On the orchestrator, not a soil child: clipping at the chain entry delays everything downstream.
    _intake_delay: pd.Timedelta = pd.Timedelta(0)

    # Wall-clock tick cadence ([field_simulation] interval/offset, minutes).
    _interval_min: int = 60
    _offset_min: int = 0

    # Injected clock: the only wall-clock source for the cutoff and the tick
    # loop, replaceable in tests.
    _now: Callable[[], pd.Timestamp] = staticmethod(lambda: pd.Timestamp.now(tz="UTC"))

    # Upper bound on one Event.wait so the loop re-reads the (injectable)
    # clock and deactivate() joins promptly.
    _TICK_WAIT_MAX_S: float = 60.0

    _tick_thread: Optional[threading.Thread] = None
    _tick_interrupt: Optional[threading.Event] = None
    _tick_lock: Optional[threading.Lock] = None

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

        self._intake_delay = self._parse_intake_delay(configs)
        self._interval_min, self._offset_min = self._parse_tick_schedule(configs)

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
            logger.warning("%s: no Weather component resolved; chain will never tick.", self.name)
            return

        self._weather_channels = Channels(list(self.weather.data.values()))
        self._required_weather_keys = tuple(c.key for c in Evapotranspiration.REQUIRED_WEATHER_CHANNELS)
        self._evapo_rename = {c.id: c.key for c in self._weather_channels}

        soil_data = self.soil_simulation.data
        if self._check_state_channel_warm_start(soil_data):
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

        for channel in self._weather_channels:
            key = channel.key
            if (
                key in self._required_weather_keys
                and key not in self._OPTIONAL_WEATHER_KEYS
                and not channel.has_logger()
            ):
                logger.warning(
                    "%s: required weather channel '%s' has no logger configured; "
                    "the wall-clock tick reads logged data and will never see it.",
                    self.name,
                    key,
                )

        self._start_tick_thread()

    def deactivate(self) -> None:
        self._stop_tick_thread()
        super().deactivate()

    # -- replication frontier (intake_delay) ----------------------------------

    @staticmethod
    def _parse_intake_delay(configs: Configurations) -> pd.Timedelta:
        """Parse ``[field_simulation] intake_delay`` (default ``0`` = feature off)."""
        return to_timedelta(configs.get("intake_delay", default="0min"))

    @staticmethod
    def _parse_tick_schedule(configs: Configurations) -> tuple[int, int]:
        """Parse ``[field_simulation] interval``/``offset`` (minutes; 60/0).

        Same vocabulary as ``WeatherForecast``: ticks fire at wall-clock slots
        aligned to ``interval``, shifted by ``offset`` within the interval.
        """
        interval = int(configs.get("interval", default=60))
        offset = int(configs.get("offset", default=0))
        if interval < 1:
            raise ValueError(f"[field_simulation] interval must be >= 1 minute, got {interval}")
        if not 0 <= offset < interval:
            raise ValueError(f"[field_simulation] offset must be in [0, interval), got {offset}")
        return interval, offset

    @staticmethod
    def _clip_to_cutoff(frame: pd.DataFrame, cutoff: Optional[pd.Timestamp]) -> pd.DataFrame:
        """Drop rows stamped after ``cutoff`` (inclusive keep); pure, no wall-clock.

        ``cutoff is None`` returns ``frame`` unchanged. Otherwise returns the rows
        at or before ``cutoff`` (possibly empty), so the caller's
        ``now = frame.index[-1]`` stays a real data timestamp matched to its
        forcing row rather than a synthesized wall-clock value.
        """
        if cutoff is None:
            return frame
        return frame.loc[frame.index <= cutoff]

    # -- wall-clock tick -------------------------------------------------------

    def _next_slot(self, now: pd.Timestamp) -> pd.Timestamp:
        """First aligned slot strictly after ``now``.

        Alignment is absolute (``floor_date`` on the site timezone + offset),
        not relative to activation, so restarts do not shift the schedule.
        """
        timezone = getattr(self.location, "timezone", None)
        slot = floor_date(now, timezone, freq=f"{self._interval_min}min")
        slot += pd.Timedelta(minutes=self._offset_min)
        while slot <= now:
            slot += pd.Timedelta(minutes=self._interval_min)
        return slot

    def _start_tick_thread(self) -> None:
        self._tick_interrupt = threading.Event()
        self._tick_lock = threading.Lock()
        self._tick_thread = threading.Thread(target=self._tick_loop, name=f"{self.name}-tick", daemon=True)
        self._tick_thread.start()

    def _stop_tick_thread(self) -> None:
        if self._tick_interrupt is not None:
            self._tick_interrupt.set()
        if self._tick_thread is not None:
            self._tick_thread.join(timeout=30.0)
            if self._tick_thread.is_alive():
                logger.warning("%s: tick thread did not stop within 30s.", self.name)
            self._tick_thread = None

    def _tick_loop(self) -> None:
        slot = self._next_slot(self._now())
        while not self._tick_interrupt.is_set():
            now = self._now()
            if now < slot:
                wait_s = min((slot - now).total_seconds(), self._TICK_WAIT_MAX_S)
                if self._tick_interrupt.wait(timeout=wait_s):
                    break
                continue
            self._tick()
            slot = self._next_slot(self._now())

    def _tick(self) -> None:
        """Run one slot; skip (never queue) when the previous run still holds the lock."""
        if not self._tick_lock.acquire(blocking=False):
            logger.warning("%s: previous tick still running; skipping slot.", self.name)
            return
        try:
            self._on_tick(self._now())
        except Exception:
            logger.exception("%s: tick failed.", self.name)
        finally:
            self._tick_lock.release()

    def _on_tick(self, now: pd.Timestamp) -> None:
        """Advance the chain over ``(frontier, now - intake_delay]`` from logged data.

        Reads in daily chunks so ``simulation_state`` persists as the frontier
        ratchets and a crash redoes at most one chunk. Advances only as far as
        logged weather reaches; gaps self-heal on later ticks.
        """
        if self.soil_simulation is None or self.evapotranspiration is None or self._weather_channels is None:
            return
        cutoff = now - self._intake_delay
        frontier = getattr(self.soil_simulation, "_last_simulated_at", None)
        start = frontier if frontier is not None else cutoff - pd.Timedelta(minutes=self._interval_min)
        if start >= cutoff:
            return
        for chunk_start, chunk_end in self._iter_day_chunks(start, cutoff):
            weather = self._read_weather_span(chunk_start, chunk_end)
            if weather.empty or not self._weather_frame_valid(weather):
                continue
            et_data, seg_et = self._run_chain(weather)
            if self.soil_simulation._last_simulated_at is None:
                # Cold start: one advance spins the PDE up at the newest row.
                self.soil_simulation.advance(et_data, et_data.index[-1], seg_et)
            else:
                self.soil_simulation.simulate_loop(et_data, seg_et)

    @staticmethod
    def _iter_day_chunks(start: pd.Timestamp, end: pd.Timestamp):
        """Yield ``(start, end]`` split at midnight boundaries, chronological."""
        s = start
        while s < end:
            e = min((s + pd.Timedelta(days=1)).normalize(), end)
            yield s, e
            s = e

    def _read_weather_span(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        frame = self.data.read_logged(self._weather_channels, start=start, end=end, unique=True)
        return self._trim_span(frame, start, end)

    def _trim_span(self, frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """Rename logged columns to weather keys and keep rows in ``(start, end]``."""
        if frame.empty:
            return frame
        frame = frame.rename(columns=self._evapo_rename)
        frame = frame.loc[frame.index > start]
        return self._clip_to_cutoff(frame, end)

    def _weather_frame_valid(self, frame: pd.DataFrame) -> bool:
        missing = [
            k
            for k in self._required_weather_keys
            if k not in self._OPTIONAL_WEATHER_KEYS and (k not in frame.columns or frame[k].isna().all())
        ]
        if missing:
            logger.debug(
                "%s: skipping chunk; weather columns missing or all-NaN: %s",
                self.name,
                missing,
            )
            return False
        return True

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

        Canopy segments use bulk vegetation state; bare-soil segments use the
        ``bare_*`` config values. Returns a single ``_bulk`` segment when no
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
            logger.warning(
                "%s: using placeholder vegetation state (LAI from monthly table '%s', "
                "ROUGHNESS=%s, PLANT_HEIGHT=%s, NDVI=%s); "
                "no Crop subcomponent or field-level sensors are publishing these "
                "channels yet. Values are still written for debugging visibility.",
                self.name,
                self._lai_type,
                self.roughness,
                self.plant_height,
                self.ndvi,
            )
            self._vegetation_placeholder_warned = True
        df[self.LAI] = pd.array(_LAI_BY_TYPE[self._lai_type])[df.index.month - 1].astype(float)
        df[self.ROUGHNESS] = self.roughness
        df[self.PLANT_HEIGHT] = self.plant_height
        df[self.NDVI] = self.ndvi

        for key, default in self._WEATHER_DEFAULTS.items():
            if key not in df.columns or df[key].isna().all():
                if key not in self._weather_default_warned:
                    logger.warning(
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

    def _check_state_channel_warm_start(self, soil_data: Any) -> bool:
        """Warn about a warm-start-breaking SIMULATION_STATE config; return
        whether the restore listener should be registered (a read-side
        connector is present)."""
        state_channel = soil_data.simulation_state
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
        start: Optional[Timestamp] = None,
        end: Optional[Timestamp] = None,
        prior: Optional[pd.DataFrame] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        if not self.has_soil_simulation() or self.evapotranspiration is None:
            return pd.DataFrame()

        if start is not None:
            weather = weather.loc[weather.index >= start]
        if end is not None:
            weather = weather.loc[weather.index <= end]
        if weather.empty:
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
