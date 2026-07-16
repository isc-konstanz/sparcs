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
from lories.core import ConfigurationUnavailableError
from lories.data import Channels
from lories.typing import Configurations, Timestamp
from lories.util import to_timedelta
from sparcs.components.agriculture.irrigation import Irrigation
from sparcs.components.weather import validate_meteo_inputs

from ._schedule import parse_tick_schedule, slot_ceil
from ._soil import (
    _DEFAULT_BAY_WIDTH,
    FIELD_SIMULATION_ALLOWED_KEYS,
    DripConfig,
    MeshConfig,
    PDEConfig,
    ProbeSpec,
    resolve_pde_config,
    top_segment_names_from_mesh,
    warn_unknown_keys,
)
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

    _irrigation_flow_channel: Any = None
    # On/off state feed: multiplied by the drip design flow when the meter is
    # silent/absent (broken meter). None = no state channel wired.
    _irrigation_state_channel: Any = None
    # Whole-field design flow [l/min] from [soil_simulation.drip]; 0.0 = no drip block.
    _design_flow_lpm: float = 0.0
    # True only when [soil_simulation.drip] was declared explicitly (not defaulted).
    _drip_explicit: bool = False
    # Metres of total drip-line the flow meter feeds (n_rows * row_length); see
    # SoilSimulation.configure's total_drip_line_length_m docstring. 1.0 = no
    # [soil_simulation] block (reads the metered value as already per-metre).
    _total_drip_line_length_m: float = 1.0
    # Read-back so watering that started before a chunk still forces its first timesteps.
    _FLOW_LOOKBACK: pd.Timedelta = pd.Timedelta(days=1)

    # On the orchestrator, not a soil child: clipping at the chain entry delays everything downstream.
    _intake_delay: pd.Timedelta = pd.Timedelta(0)

    # Tick cadence in minutes ([field_simulation] interval/offset).
    _interval_min: int = 60
    _offset_min: int = 0

    # The only wall-clock source; inject in tests.
    _now: Callable[[], pd.Timestamp] = staticmethod(lambda: pd.Timestamp.now(tz="UTC"))

    # Cap one Event.wait so the loop re-reads the clock and deactivate() joins promptly.
    _TICK_WAIT_MAX_S: float = 60.0

    _tick_thread: Optional[threading.Thread] = None
    _tick_interrupt: Optional[threading.Event] = None
    _tick_lock: Optional[threading.Lock] = None
    _watchdog_timer: Optional[threading.Timer] = None

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
    _bay_width: float = _DEFAULT_BAY_WIDTH

    # Latest per-segment shade factors from GroundShading; {} = no data yet (treated as 1.0).
    _segment_shade: dict[str, float]

    _CANOPY_SEGMENT_NAMES = ("PlantTopLeftSegment", "PlantTopRightSegment")

    def configure(self, configs: Configurations) -> None:
        super().configure(configs)
        warn_unknown_keys(configs, FIELD_SIMULATION_ALLOWED_KEYS, "field_simulation")
        # "plot" cascades a field-level [plot] block (enabled + interval) to every
        # subcomponent as its default, overridable by the child's own [<type>.plot]
        # -- the same mechanism [model] already uses.
        defaults = Component._build_defaults(configs, includes=["model", "plot"], strict=True)

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

        self._bay_width = float(configs.get("bay_width", default=_DEFAULT_BAY_WIDTH))

        self._intake_delay = self._parse_intake_delay(configs)
        self._interval_min, self._offset_min = self._parse_tick_schedule(configs)

        for c in self.VEGETATION_CHANNELS:
            self.data.add(c, aggregate="mean", logger={"enabled": False})

        self._mesh_config: Optional[MeshConfig] = None
        self._soil_pde_config: Optional[PDEConfig] = None
        self._drip_config: Optional[DripConfig] = None
        if configs.has_member(SoilSimulation.TYPE, includes=True):
            soil_block = configs.get_member(SoilSimulation.TYPE, defaults=defaults)
            mesh_block = soil_block.get_member("mesh", defaults={}, ensure_exists=True)
            self._mesh_config = MeshConfig(mesh_block, bay_width=self._bay_width)

            # Eager PDEConfig parse (mirrors the mesh seam above): ONE canonical
            # resolution site for the [pde]/[model]/forcing cascade, so the
            # predictor and the sim can never resolve it differently. Children
            # configure in alphanumeric id order (soil_predictor BEFORE
            # soil_simulation), so this eager copy -- not a live _ode_config --
            # is what soil_pde_config serves at predictor-configure time (see
            # that property's docstring). Reusing SoilPredictor's pinned
            # _resolve_model_block makes the [soil_simulation.model]-over-[model]
            # merge equivalent by construction; it returns the SAME stored member
            # object as the soil_block fetch above (get_member mutates in place).
            soil_block, model_block = SoilPredictor._resolve_model_block(configs)
            self._soil_pde_config = resolve_pde_config(soil_block, model_block)

            # Drip design flow -- ONE canonical parse of [soil_simulation.drip],
            # so the predictor's own [soil_predictor.drip] override and the
            # sim's state-driven fallback feed can never disagree on defaults.
            self._drip_config = DripConfig(soil_block)
            self._drip_explicit = self._drip_config.explicit
            self._design_flow_lpm = self._drip_config.design_flow_lpm

            # total_drip_line_length_m: eager fallback parse (mirrors the mesh
            # seam above) from the SAME soil_block, same default as
            # SoilSimulation.configure's own parse -- no duplicate positivity
            # check here; the sim itself raises moments later, at
            # SoilSimulation.configure() time (soil.py).
            self._total_drip_line_length_m = float(soil_block.get("total_drip_line_length_m", default=1.0))

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
    def soil_pde_config(self) -> Optional[PDEConfig]:
        """The resolved PDEConfig -- ONE canonical resolution site for the
        [pde]/[model]/forcing cascade, so the predictor and SoilSimulation can
        never resolve it differently.

        Same guard shape as ``mesh_config``: prefers SoilSimulation's live
        ``_ode_config`` once configured, falling back to the eagerly-parsed copy
        built in ``configure()``. Children configure in alphanumeric id order, so
        ``soil_predictor`` configures BEFORE ``soil_simulation`` -- at
        predictor-configure time this ALWAYS serves the eager fallback, never the
        live object (the live ``_ode_config`` is only ever seen by a caller that
        reads this property AFTER SoilSimulation.configure() has run).

        Handing either object across this seam is safe only because ``PDEConfig``
        carries no numpy arrays: every field is a plain str/float/Optional[float]/
        pd.Timedelta scalar, and its ``.ponding``/``.feddes`` are config objects
        built from plain scalars too (_soil.py) -- so no shared mutable array
        state leaks between predictor and sim through this property.
        """
        if self.soil_simulation is not None:
            soil_pde = getattr(self.soil_simulation, "_ode_config", None)
            if soil_pde is not None:
                return soil_pde
        return self._soil_pde_config

    @property
    def drip_config(self) -> Optional[DripConfig]:
        """The parsed [soil_simulation.drip] layout -- ONE canonical resolution,
        so the predictor's own [soil_predictor.drip] per-key override
        (SoilPredictor.configure) and the sim's state-driven fallback feed can
        never disagree on defaults.

        No sim-side preference needed (unlike mesh_config/soil_pde_config): the
        PARENT parses [soil_simulation.drip] itself in configure(), so this
        always serves self._drip_config directly. getattr guards object.__new__
        test instances that bypass configure() entirely.
        """
        return getattr(self, "_drip_config", None)

    @property
    def total_drip_line_length_m(self) -> float:
        """Metres of total drip-line the flow meter feeds (n_rows * row_length);
        see SoilSimulation.configure's total_drip_line_length_m docstring.

        Same guard shape as mesh_config/soil_pde_config: prefers
        SoilSimulation's live _total_drip_line_length_m once configured,
        falling back to the eagerly-parsed copy built in configure(). Never
        raises -- the sim's own positivity ValueError (soil.py) fires moments
        later at SoilSimulation.configure() time; this fallback needs no
        duplicate check.
        """
        if self.soil_simulation is not None:
            length = getattr(self.soil_simulation, "_total_drip_line_length_m", None)
            if length is not None:
                return length
        return self._total_drip_line_length_m

    def get_probes(self) -> list[ProbeSpec]:
        """Resolved probe sampling recipes (point + area), delegated from
        ``SoilSimulation.get_probes()``.

        Method (not ``@property``): ``ProbeSpec`` carries numpy arrays
        (``@dataclass(eq=False)``, _soil.py) and lories' ``get_members``
        reflection executes every property and value-compares the results --
        the ambiguous-truth trap ``SoilSimulation.get_probes()`` already
        dodges for the same reason (see its docstring, soil.py). Cross-ref
        ``.scratch/lories-frictions/issues/01-reflection-truth-test-trap.md``
        ("Reflection trap: get_members executes every property and
        value-compares results") -- that issue's soil.py:348-360 refs predate
        this refactor (now soil.py:421-433).

        getattr-guards like ``mesh_config``: returns ``[]`` (never raises)
        when ``soil_simulation`` is absent, or before
        ``SoilSimulation._probes`` is assigned -- a bare class annotation
        until ``_configure_probes`` runs, so a raw
        ``soil_simulation.get_probes()`` call at that point would raise
        ``AttributeError``.

        Shared-object semantics: the returned list is a FRESH container (per
        ``SoilSimulation.get_probes()``'s own contract), but its
        ``ProbeSpec`` elements are the SAME objects the sim owns -- no
        defensive copy, by design (the parallel worker path already reuses
        parent ProbeSpecs verbatim); callers must not mutate them.
        """
        if self.soil_simulation is None:
            return []
        if getattr(self.soil_simulation, "_probes", None) is None:
            return []
        return self.soil_simulation.get_probes()

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

        self._irrigation_flow_channel = self._resolve_irrigation_channel(Irrigation.FLOW)
        self._irrigation_state_channel = self._resolve_irrigation_channel(Irrigation.STATE)
        self._validate_irrigation_input()

        for channel in self._weather_channels:
            key = channel.key
            if (
                key in self._required_weather_keys
                and key not in self._OPTIONAL_WEATHER_KEYS
                and not channel.has_connector()
            ):
                logger.warning(
                    "%s: required weather channel '%s' has no connector configured; "
                    "the wall-clock tick reads its span from the connector and will never see it.",
                    self.name,
                    key,
                )

        self._start_tick_thread()

    def deactivate(self) -> None:
        self._stop_tick_thread()
        super().deactivate()

    # -- irrigation input (flow meter, with state x design-flow fallback) ------

    def _resolve_irrigation_channel(self, constant: Constant) -> Any:
        """Resolve one irrigation channel (FLOW or STATE) from the sibling
        Irrigation component, or None when irrigation or that channel is absent."""
        if self.irrigation is None:
            return None
        try:
            return self.irrigation.data[constant]
        except KeyError:
            return None

    def _validate_irrigation_input(self) -> None:
        """Refuse to start (before any tick) a field whose irrigation is configured
        but has no usable input: neither a connected flow meter nor a connected
        on/off state channel backed by an explicit [soil_simulation.drip] block. A
        rain-fed field (no [irrigation] block at all) is left alone -- 0 l/min is
        the deliberate answer there, not a masked misconfiguration.
        """
        if self.irrigation is None:
            return
        flow_wired = self._irrigation_flow_channel is not None and self._irrigation_flow_channel.has_connector()
        state_wired = (
            self._irrigation_state_channel is not None
            and self._irrigation_state_channel.has_connector()
            and self._drip_explicit
        )
        if not (flow_wired or state_wired):
            raise ConfigurationUnavailableError(
                f"{self.name}: irrigation is configured but no usable input is wired. "
                "Wire the metered feed ([irrigation.data.channels.flow] with a connector), "
                "or the on/off state feed ([irrigation.data.channels.state] with a connector "
                "PLUS a [soil_simulation.drip] block giving nozzle_count and nozzle_flow_lph). "
                "Refusing to start on a silent 0 l/min fallback."
            )

    # -- replication frontier (intake_delay) ----------------------------------

    @staticmethod
    def _parse_intake_delay(configs: Configurations) -> pd.Timedelta:
        """Parse ``[field_simulation] intake_delay`` (default ``0`` = read up to now)."""
        return to_timedelta(configs.get("intake_delay", default="0min"))

    @staticmethod
    def _parse_tick_schedule(configs: Configurations) -> tuple[int, int]:
        """Parse ``[field_simulation] interval``/``offset`` (minutes; 60/0).

        Same vocabulary as ``WeatherForecast``: ticks fire at wall-clock slots
        aligned to ``interval``, shifted by ``offset`` within the interval.

        Thin wrapper over ``_schedule.parse_tick_schedule`` -- see that
        module's docstring for why the validation has one home.
        """
        return parse_tick_schedule(
            configs,
            default_interval=60,
            default_offset=0,
            section_name="field_simulation",
        )

    # -- wall-clock tick -------------------------------------------------------

    def _next_slot(self, now: pd.Timestamp) -> pd.Timestamp:
        """First aligned slot strictly after ``now``.

        Alignment is absolute (``floor_date`` on the site timezone + offset),
        not relative to activation, so restarts do not shift the schedule.

        Thin wrapper over ``_schedule.slot_ceil`` -- see that module's
        docstring for why the slot math has one home.
        """
        timezone = getattr(self.location, "timezone", None)
        return slot_ceil(now, timezone, self._interval_min, self._offset_min)

    def _start_tick_thread(self) -> None:
        self._tick_interrupt = threading.Event()
        self._tick_lock = threading.Lock()
        self._watchdog_timer = None
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
        self._cancel_watchdog()

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
        start = self._now()
        watchdog_done = threading.Event()
        try:
            # Visibility must never suppress the tick itself: an arming failure
            # (e.g. thread-creation limits) is logged and the tick proceeds.
            try:
                self._arm_watchdog(start, watchdog_done)
            except Exception:
                logger.exception("%s: failed to arm the slot watchdog; tick continues.", self.name)
            self._on_tick(start)
        except Exception:
            logger.exception("%s: tick failed.", self.name)
        finally:
            watchdog_done.set()
            self._cancel_watchdog()
            self._tick_lock.release()
        self._log_tick_overrun(start)

    # -- slot-boundary watchdog (log-only, lock-free) --------------------------
    #
    # NO deadline, NO cancel: an unbounded tick (e.g. a non-converged dt_min
    # grind) stays unbounded by design (Q3) -- this unit only makes an overrun
    # visible. The Timer callback reads nothing but the immutable ``boundary``
    # snapshot captured at arm time and a completion ``Event``; it never takes
    # the tick lock and never touches PDE/tick state, so it is safe to run from
    # a second thread while a tick is in flight.

    def _arm_watchdog(self, reference: pd.Timestamp, done: threading.Event) -> None:
        """Arm a Timer for the slot boundary following ``reference``. While
        ``done`` is unset, the callback logs and re-arms for the following
        boundary in turn, so a multi-slot overrun is reported once per slot
        crossed, not just once."""
        boundary = self._next_slot(reference)
        delay_s = max((boundary - reference).total_seconds(), 0.0)
        timer = threading.Timer(delay_s, self._on_watchdog_boundary, args=(boundary, done))
        timer.daemon = True
        self._watchdog_timer = timer
        timer.start()

    def _on_watchdog_boundary(self, boundary: pd.Timestamp, done: threading.Event) -> None:
        if done.is_set():
            return
        logger.warning("%s: tick still running; slot skipped (boundary %s).", self.name, boundary)
        if not done.is_set():
            self._arm_watchdog(boundary, done)

    def _cancel_watchdog(self) -> None:
        timer = self._watchdog_timer
        if timer is not None:
            timer.cancel()
        self._watchdog_timer = None

    def _log_tick_overrun(self, start: pd.Timestamp) -> None:
        """Log-only summary when the tick's wall-clock duration crossed one or
        more slot boundaries; no-op for a tick that finished within its slot.
        ``slots_skipped`` is duration // interval -- an approximation that can
        undercount by one when the tick started off-boundary; fine for a
        log-only signal (the per-boundary watchdog lines are the exact record)."""
        duration = self._now() - start
        interval = pd.Timedelta(minutes=self._interval_min)
        skipped = int(duration // interval)
        if skipped < 1:
            return
        logger.warning(
            "%s: tick overran its slot (duration=%s, slots_skipped=%d).",
            self.name,
            duration,
            skipped,
        )

    def _on_tick(self, now: pd.Timestamp) -> None:
        """Advance the chain over ``(frontier, now - intake_delay]`` from connector reads.

        Reads in daily chunks so ``simulation_state`` persists as the frontier
        ratchets and a crash redoes at most one chunk. Advances only as far as
        the weather feed reaches; gaps self-heal on later ticks.
        """
        if self.soil_simulation is None or self.evapotranspiration is None or self._weather_channels is None:
            return
        cutoff = now - self._intake_delay
        frontier = getattr(self.soil_simulation, "_last_simulated_at", None)
        # The tick clock is UTC (self._now); a frontier inherited from the weather feed
        # can carry a site-local offset (Brightsky indexes its rows in location.timezone),
        # which would pair a non-UTC start with the UTC cutoff -- Brightsky's ranged read
        # slices source_data.loc[start:end] and pandas rejects two bounds at different UTC
        # offsets. Align the frontier to the cutoff's zone so every read window built from
        # (start, cutoff] carries a single offset.
        if frontier is not None:
            frontier = frontier.tz_convert(cutoff.tz)
        start = frontier if frontier is not None else cutoff - pd.Timedelta(minutes=self._interval_min)
        if start >= cutoff:
            return
        # Shutdown-only cancel (B8): threaded into walk_window's existing cancel
        # param so a grinding walk exits promptly instead of blocking
        # _stop_tick_thread's 30s join. None outside a running tick thread (e.g.
        # object.__new__ test fixtures that never start it), so those are unaffected.
        cancel = self._tick_interrupt.is_set if self._tick_interrupt is not None else None
        for chunk_start, chunk_end in self._iter_day_chunks(start, cutoff):
            weather = self._read_weather_span(chunk_start, chunk_end)
            if weather.empty or not self._weather_frame_valid(weather):
                continue
            et_data, seg_et = self._run_chain(weather)
            et_data[SoilSimulation.IRRIGATION_FLOW_LPM] = self._irrigation_flow_lpm(
                chunk_start, chunk_end, et_data.index
            )
            # Range-read the tensiometers over this chunk so the anchor assimilates
            # each reading at its own timestamp (no-op when anchoring is off).
            self.soil_simulation.load_anchor_history(chunk_start, chunk_end)
            if self.soil_simulation._last_simulated_at is None:
                # Cold start: one advance spins the PDE up at the newest row.
                self.soil_simulation.advance(et_data, et_data.index[-1], seg_et, cancel=cancel)
            else:
                self.soil_simulation.simulate_loop(et_data, seg_et, cancel=cancel)

        # The predictor's own interval/offset gate decides whether a roll-out runs.
        new_frontier = self.soil_simulation._last_simulated_at
        if self.soil_predictor is not None and new_frontier is not None and new_frontier != frontier:
            self.soil_predictor.predict(
                new_frontier,
                forecast_creation=self._read_forecast_epoch(),
            )

    @staticmethod
    def _iter_day_chunks(start: pd.Timestamp, end: pd.Timestamp):
        """Yield ``(start, end]`` split at midnight boundaries, chronological."""
        s = start
        while s < end:
            e = min((s + pd.Timedelta(days=1)).normalize(), end)
            yield s, e
            s = e

    def _read_weather_span(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        # Read the span straight from the weather connector (the kob_tracker
        # station table, or the Brightsky observation API -- both honour the
        # ranged read), not a logger: the source already holds the observed
        # history, so there is nothing to re-log. read() issues a ranged SELECT.
        frame = self.data.read(self._weather_channels, start=start, end=end, unique=True)
        return self._trim_span(frame, start, end)

    def _irrigation_flow_lpm(self, start: pd.Timestamp, end: pd.Timestamp, index: pd.DatetimeIndex) -> pd.Series:
        """Irrigation flow [l/min] on ``index`` via the fallback chain: the metered
        flow when the meter is reporting over the span, else the on/off state x the
        drip design flow (broken/absent meter), else 0.0 (a rain-fed field with no
        irrigation input -- the only unwired case ``activate()`` lets start).

        A meter that reports 0 counts as "alive, not watering" and wins over the
        state feed; only a meter that reported NO rows at all falls back to state.

        The state feed requires an EXPLICIT [soil_simulation.drip] block (same
        invariant _validate_irrigation_input enforces at startup): without it the
        design flow would be the placeholder default (~0.017 l/min), so a bare
        meter gap on a flow-primary field must read as 0 (not watering), never a
        fabricated forcing.
        """
        measured = self._read_measured_flow(start, end, index)
        if measured is not None:
            return measured
        if self._irrigation_state_channel is not None and self._drip_explicit:
            return self._read_state_span(start, end, index) * self._design_flow_lpm
        return pd.Series(0.0, index=index)

    def _read_measured_flow(
        self, start: pd.Timestamp, end: pd.Timestamp, index: pd.DatetimeIndex
    ) -> Optional[pd.Series]:
        """Metered flow [l/min] aligned onto ``index``, or None when the flow
        channel is unwired or the meter reported no usable rows over the span (a
        dead/absent meter -- the caller then falls back to the state feed)."""
        if self._irrigation_flow_channel is None:
            return None
        frame = self.data.read(
            Channels([self._irrigation_flow_channel]),
            start=start - self._FLOW_LOOKBACK,
            end=end,
            unique=True,
        )
        if frame.empty or frame.iloc[:, 0].isna().all():
            return None
        return self._align_flow(frame, index)

    def _read_flow_span(self, start: pd.Timestamp, end: pd.Timestamp, index: pd.DatetimeIndex) -> pd.Series:
        """Irrigation flow [l/min] aligned onto the weather timesteps, or an all-0.0
        series when the meter channel is unwired or silent. Thin wrapper over
        ``_read_measured_flow``; the fallback chain uses ``_irrigation_flow_lpm``."""
        measured = self._read_measured_flow(start, end, index)
        return measured if measured is not None else pd.Series(0.0, index=index)

    def _read_state_span(self, start: pd.Timestamp, end: pd.Timestamp, index: pd.DatetimeIndex) -> pd.Series:
        """Irrigation on/off state read from its connector, aligned onto the weather
        timesteps as 0.0/1.0. Backward-filled; NULL rows and leading gaps read as
        0.0 (off), same guard as ``_align_flow`` -- NaN must never reach the PDE."""
        if self._irrigation_state_channel is None:
            return pd.Series(0.0, index=index)
        frame = self.data.read(
            Channels([self._irrigation_state_channel]),
            start=start - self._FLOW_LOOKBACK,
            end=end,
            unique=True,
        )
        return self._align_flow(frame, index)

    @staticmethod
    def _align_flow(frame: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
        """Backward-fill flow rows onto ``index``: each timestep gets the most
        recent flow at or before it. NULL rows and leading gaps read as 0.0
        (not watering) -- NaN must never reach the PDE source (cf. 20431c7)."""
        if frame.empty:
            return pd.Series(0.0, index=index)
        series = frame.iloc[:, 0].sort_index()
        series = series[~series.index.duplicated(keep="last")]
        aligned = series.reindex(index, method="ffill")
        return aligned.fillna(0.0).astype(float)

    def _trim_span(self, frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """Rename connector columns to weather keys and keep rows in ``(start, end]``."""
        if frame.empty:
            return frame
        frame = frame.rename(columns=self._evapo_rename)
        return frame.loc[(frame.index > start) & (frame.index <= end)]

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
        # Fresh factors feed this run either way; only published runs may update
        # the stored field state (a publish=False forecast replay must not leak
        # its shading into the live tick, nor run on the live tick's factors).
        segment_shade = self._segment_shade
        if self.ground_shading is not None:
            seg_factors = self.ground_shading.evaluate(df, publish=publish)
            if seg_factors:
                segment_shade = dict(seg_factors)
                if publish:
                    self._segment_shade = segment_shade
        segments = self._build_segments(df, segment_shade)
        return self.evapotranspiration.evaluate(df, segments, publish=publish)

    def _build_segments(self, df: pd.DataFrame, segment_shade: dict[str, float]) -> list[SegmentProperties]:
        """Build the per-segment property list for ET evaluation.

        Canopy segments use bulk vegetation state; bare-soil segments use the
        ``bare_*`` config values. Returns a single ``_bulk`` segment when no
        soil mesh is attached. ``segment_shade`` carries the shade factors for
        the run being evaluated (a forecast replay passes its own, not the
        stored live-tick state).
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
                    shade_factor=float(segment_shade.get(name, 1.0)),
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
