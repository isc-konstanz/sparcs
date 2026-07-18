# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.simulation.soil
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Live Richards-equation soil solver: advances on weather ticks, publishes
flux diagnostics and probe tensions, and anchors to tensiometer readings.
"""

from __future__ import annotations

import logging
import os
import threading
from contextlib import nullcontext
from typing import Any, Callable, Optional

import matplotlib.pyplot as plt
import meshio
from matplotlib.collections import LineCollection, PolyCollection

import numpy as np
import pandas as pd
from lories import Constant
from lories.components.weather import Weather
from lories.core import ConfigurationError
from lories.data import Channels
from lories.typing import Configurations

from . import _anchor_runtime, plot_render, plot_style
from ._anchor import AnchorSensor
from ._soil import (
    SOIL_SIMULATION_ALLOWED_KEYS,
    ClipDiagnostics,
    FluxRates,
    MeshConfig,
    ProbeSpec,
    SoilBase,
    WalkResult,
    ensure_mesh,
    flow_m3s_per_m,
    resolve_pde_config,
    resolve_probes,
    warn_unknown_keys,
)

logger = logging.getLogger(__name__)

# Default cadence for [plot] progress-image snapshots, absent a [plot] interval override.
_DEFAULT_PLOT_INTERVAL: str = "5min"


# The anchor lifecycle (config parse, sensor discovery, per-tick history read,
# assimilation apply) lives in _anchor_runtime; re-export the two module-level
# names that tests and configure() reach via this module's path.
_walk_components = _anchor_runtime._walk_components
_parse_anchor_config = _anchor_runtime._parse_anchor_config


def _resolve_mesh_config(context: Any, configs: Configurations) -> MeshConfig:
    """Resolve this component's ``MeshConfig``.

    Reuses ``context.mesh_config`` (the parent ``FieldSimulation``'s
    already-parsed instance) when present -- the parent parses the SAME
    ``[soil_simulation.mesh]`` block eagerly in its own ``configure()``,
    strictly before any child (including this one) configures, so reusing it
    here never reads a different block. Falls back to parsing this
    component's own ``[mesh]`` block for standalone construction (no
    ``FieldSimulation`` parent in ``context``), using ``context.bay_width`` --
    or ``MeshConfig``'s own default when that, too, is absent.
    """
    parent_mesh = getattr(context, "mesh_config", None)
    if parent_mesh is not None:
        return parent_mesh
    return MeshConfig(
        configs.get_member("mesh", defaults={}, ensure_exists=True),
        bay_width=getattr(context, "bay_width", None),
    )


class SoilSimulation(SoilBase):
    TYPE: str = "soil_simulation"
    INCLUDES = ["mesh", "pde", "plot", "probes", "anchor"]

    SIMULATION_STATE = Constant(bytes, "simulation_state", "Soil Simulation State", "-")
    SOIL_PROGRESS_IMAGE = Constant(bytes, "soil_progress_image", "Soil Simulation Progress Image", "png")

    # Internal math in kg/(m²·s); channels publish in kg/(m²·h). Short keys with
    # context="water" (house pattern, cf. context="pv" in solar/system.py): the
    # registry id stays "water_*"-unique while the bare key becomes the channel
    # key / agri_field_simulation SQL column.
    WATER_TOP_IN = Constant(float, "top_in", "Top Water Input (Irrigation + Rain)", "kg/(m^2*h)", context="water")
    WATER_TOP_OUT = Constant(float, "top_out", "Top Water Output (Evaporation)", "kg/(m^2*h)", context="water")
    WATER_BOTTOM = Constant(float, "bottom_out", "Bottom Water Output (Drainage)", "kg/(m^2*h)", context="water")
    WATER_TRANSP = Constant(float, "transpiration", "Plant Transpiration", "kg/(m^2*h)", context="water")
    # Per-step clipper residuals, area-normalised [kg/(m²·h)].
    WATER_RUNOFF = Constant(float, "runoff", "Rejected Top Influx (Runoff)", "kg/(m^2*h)", context="water")
    WATER_DEMAND_UNMET = Constant(float, "demand_unmet", "Unmet Evap+Transp Demand", "kg/(m^2*h)", context="water")
    # Gap between integral closure and independent bottom-face drainage estimate; non-zero flags solver drift.
    WATER_BALANCE_RESIDUAL = Constant(
        float,
        "balance_residual",
        "Mass-Balance Residual (integral - direct)",
        "kg/(m^2*h)",
        context="water",
    )
    # Per-step assimilation increment from anchoring [kg per out-of-plane metre,
    # matching total_water()]. A diagnostic of how hard the correction works; not
    # a flux, so it is excluded from the mass-balance residual.
    WATER_ANCHOR = Constant(float, "anchor", "Anchor Assimilation Increment", "kg/m", context="water")

    # Seconds of the advance window where a substep could not converge even at
    # dt_min and was skipped (state held through the gap; accept+mark per
    # issue 10/B7 -- no hold, no retry). Set every tick, 0.0 when nothing was
    # skipped: a column that only appears on failure cannot be dashboarded.
    WALK_SKIPPED_S = Constant(float, "skipped_s", "Skipped Walk Duration (dt_min Unsolvable)", "s", context="water")

    # Adaptive-walk substep rollbacks for the window (walk_window shrinks dt and
    # re-solves); persisted every tick (0 when clean) so a solver grinding
    # through retries is visible in the data -- the log stays DEBUG (issue 22/W2.4).
    WALK_RETRIES = Constant(float, "retries", "Walk Substep Retries", "-", context="water")

    # Count of consecutive fully-stalled ticks (every weather chunk in the
    # window empty/invalid) that PRECEDED the tick committing this row; 0.0 in
    # steady state, mirrored down from FieldSimulation._on_tick because
    # _record_diagnostics never runs during a stall itself (issue 19/W2.1).
    WEATHER_STALL = Constant(float, "weather_stall", "Consecutive Weather-Stall Ticks", "-", context="water")

    # Count of consecutive tick failures (_on_tick raised) that PRECEDED the
    # tick committing this row; 0.0 in steady state, mirrored down from
    # FieldSimulation._on_tick because a failed tick commits no row itself
    # (issue 20/W2.2).
    TICK_FAILURES = Constant(float, "tick_failures", "Consecutive Tick Failures", "-", context="water")

    _plot_config: Optional[plot_style.PlotConfig] = None

    # Irrigation strip fluxes above this are almost certainly a unit or
    # normalization error (mis-scaled irrigation_flow values, or a wrong
    # total_drip_line_length_m), not physics: drip irrigation sits around
    # 20 mm/h over the strip and typical k_s near 180 mm/h.
    STRIP_FLUX_WARN_MM_H: float = 500.0

    # et_data column, whole-field flow [l/min] per timestep; written by FieldSimulation._on_tick.
    IRRIGATION_FLOW_LPM: str = "irrigation_flow_lpm"

    _last_simulated_at: Optional[pd.Timestamp] = None
    _simulating: bool = False
    _strip_flux_warned: bool = False
    # Mirrored down from FieldSimulation._on_tick before any chunk in a tick can
    # commit a row (object.__new__ safety: never set -> 0.0, steady state).
    _weather_stall_ticks: float = 0.0
    # Mirrored down from FieldSimulation._on_tick, same object.__new__ safety
    # as _weather_stall_ticks above (issue 20/W2.2).
    _tick_failures: float = 0.0

    # Staleness watch state (issue 23/W2.5): None on bare object.__new__
    # instances (the _pde_lock idiom); _configure_probes assigns fresh
    # per-instance containers, load_anchor_history lazy-inits the rest.
    _anchor_last_read: Optional[dict[str, pd.Timestamp]] = None
    _anchor_stale_warned: Optional[set[str]] = None

    # Q11: dedicated PDE-state lock serializing advance()'s PDE-mutating span
    # against apply_state_blob (NOT FieldSimulation._tick_lock -- that guards
    # tick re-entry on a different object; the race here is inside
    # SoilSimulation itself). Class-level ``None`` default: ~20 test files
    # build instances via ``object.__new__`` (no ``__init__``/``configure()``),
    # so every acquire site below falls back to an unlocked no-op
    # (``nullcontext()``) for those instances -- unchanged from pre-lock
    # behavior. A plain ``Lock`` is safe (not ``RLock``): apply_state_blob is
    # only ever invoked from the application's polled listener dispatch (a
    # separate thread/executor from the simulation's own tick thread) or
    # sequentially before simulate_loop starts (offline path) -- never
    # synchronously from within advance()'s call stack -- so this lock is
    # never re-acquired by the thread already holding it.
    _pde_lock: Optional[threading.Lock] = None

    # Progress-plot state; _plot_config is None when plotting is disabled.
    _plot_session: Any = None
    _last_plot_simtime: Optional[pd.Timestamp] = None
    # Consecutive render failures toward plot_style.PLOT_DISABLE_AFTER (W2.9);
    # class default for object.__new__ fixtures driving _render_progress_safe.
    _plot_strikes: int = 0

    _probes: list[ProbeSpec]

    def configure(self, configs: Configurations) -> None:
        super().configure(configs)
        warn_unknown_keys(configs, SOIL_SIMULATION_ALLOWED_KEYS, "soil_simulation")

        # Q11: created per configured instance; object.__new__ instances
        # (bypassing configure()) keep the class-level None default and run
        # advance()/apply_state_blob() unlocked (see _pde_lock).
        self._pde_lock = threading.Lock()

        self._mesh_config = _resolve_mesh_config(self.context, configs)
        ensure_mesh(self._mesh_config)
        # The production flow meter measures the whole-field total [l/min], but the
        # 2D mesh is one bay cross-section representing 1 m of a single row, so the
        # metered flow is divided by the TOTAL drip-line length the meter feeds
        # (n_rows * row_length, not a single field dimension; dividing by one row's
        # length when there are N rows leaves the input N times too strong). The
        # default of 1.0 reads the metered value as already per out-of-plane metre.
        self._total_drip_line_length_m = float(configs.get("total_drip_line_length_m", default=1.0))
        if self._total_drip_line_length_m <= 0:
            raise ValueError("Invalid parameters: total_drip_line_length_m must be positive")
        # Retention params from [model]; [pde] carries only solver/IC/timestep knobs.
        model_block = configs.get_member("model", defaults={})
        if not any(k in model_block for k in ("theta_r", "theta_s", "alpha", "n", "k_s")):
            logger.warning(
                "%s: no [model] retention params found (field-level [model] not "
                "inherited?). Using built-in van Genuchten defaults; any retention "
                "params under [pde] are ignored; move them into a [model] block.",
                self.name,
            )
        # ponding + feddes are sibling blocks of [pde], not nested under it.
        self._ode_config = resolve_pde_config(configs, model_block)

        self._register_state_channel()
        for c in (
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
        ):
            self.data.add(c, aggregate="mean", logger={"enabled": True})

        self._anchor_cfg = _parse_anchor_config(configs.get_member("anchor", defaults={}))

        if configs.get_bool("plot_structure", default=False):
            structure_dir = configs.get(
                "plot.dir",
                default=str(configs.dirs.data.joinpath("soil_simulation")),
            )
            os.makedirs(structure_dir, exist_ok=True)
            self._plot_mesh(os.path.join(structure_dir, "mesh.png"))

        self._plot_config = plot_style.load_plot_config(configs, default_interval=_DEFAULT_PLOT_INTERVAL)
        # In-memory strike counter (W2.9): registered or it surfaces nowhere.
        self.data.add("plot_strikes", type=float, name="Plot Strikes", aggregate="last", logger={"enabled": False})
        if self._plot_config is not None:
            self._last_plot_simtime = None
            self._plot_session = None
            self._register_progress_image_channel()

        self._pde = self._build_pde()
        logger.info(
            "%s: soil model = %s (k_s=%.3e m/s, theta_r=%.3f, theta_s=%.3f)",
            self.name,
            self._soil_model.__class__.__name__,
            self._ode_config.k_s,
            self._ode_config.theta_r,
            self._ode_config.theta_s,
        )

        self._configure_probes(configs)

    def advance(
        self,
        et_data: pd.DataFrame,
        now: pd.Timestamp,
        seg_et: dict[str, pd.DataFrame],
        cancel: Optional[Callable[[], bool]] = None,
    ) -> dict[str, float]:
        if self._simulating:
            logger.warning("%s: solve still running, skipping interval at %s", self.name, now)
            return {}

        self._simulating = True
        try:
            # Cold start: spin up with current weather to approximate steady state.
            if self._last_simulated_at is None:
                elapsed = self._ode_config.cold_start
                logger.info(
                    "%s: cold start spin-up: %s with weather at %s",
                    self.name,
                    elapsed,
                    now,
                )
            else:
                elapsed = now - self._last_simulated_at

            if not elapsed:
                return {}

            logger.debug("%s: advance dt=%s now=%s", self.name, elapsed, now)
            elapsed_s = float(elapsed.total_seconds())

            rates = self._compute_flux_rates(et_data, seg_et, elapsed_s)
            sim_t0 = now - pd.Timedelta(seconds=elapsed_s)
            # Q11: PDE-mutating span (through _save_state) serialized against
            # apply_state_blob so a restore cannot land mid-solve. This is the
            # widest correct scope: a narrower one reintroduces the race this
            # lock exists to close; blocking callers for the walk's duration is
            # accepted (correctness over latency). object.__new__ instances
            # have _pde_lock=None and run this block unlocked, as before.
            with self._pde_lock or nullcontext():
                # Include the surface ponds so pond build-up/drain-down does not
                # masquerade as bottom drainage in the balance diagnostics.
                storage_before = self._total_water() + self._pde.surface_water()
                interval = self._plot_config.interval if self._plot_config is not None else None
                clip_total = ClipDiagnostics()
                walk_result = self._walk(
                    rates=rates,
                    window_s=elapsed_s,
                    clip_total=clip_total,
                    sim_t0=sim_t0,
                    plot_interval=interval,
                    cancel=cancel,
                )
                if walk_result.cancelled:
                    # Shutdown-only escape hatch (see walk_window): the walk exited
                    # mid-window, so nothing about this tick is committed -- no
                    # diagnostics, no state save, no anchor, no frontier ratchet.
                    # The in-memory PDE field may already hold partial substep
                    # progress through the window; that's fine, the process is
                    # about to exit and a restart warm-starts from the last
                    # persisted simulation_state, not this abandoned field.
                    logger.info(
                        "%s: walk cancelled at %s (shutdown in progress); discarding the "
                        "partial window, frontier holds.",
                        self.name,
                        now,
                    )
                    return {}

                # End-of-window render, interval-gated: simulate_loop calls advance
                # once per (minute-resolution) weather row, so an unconditional render
                # here would emit one frame per minute no matter what [plot] interval
                # says. Gate on the shared _last_plot_simtime (see _render_progress_if_due).
                self._render_progress_if_due(now)

                # Snapshot the PDE-only storage change before anchoring so the
                # mass-balance residual sees the solver alone, not the correction.
                delta_storage = self._total_water() + self._pde.surface_water() - storage_before
                if self._anchor_cfg.enabled and self._anchor_sensors:
                    self._apply_anchor(now, water_after_walk=self._total_water())
                diagnostics = self._record_diagnostics(
                    rates,
                    now,
                    delta_storage,
                    elapsed_s,
                    clip_total,
                    walk_result.skipped_s,
                    walk_result.retries,
                )
                self._save_state(now)
                return diagnostics
        finally:
            self._simulating = False

    def simulate_loop(
        self,
        et_data: pd.DataFrame,
        seg_et: dict[str, pd.DataFrame],
        cancel: Optional[Callable[[], bool]] = None,
    ) -> pd.DataFrame:
        """Step the soil PDE through ``et_data`` deterministically (offline mode).

        Returns a DataFrame indexed by timestamp with diagnostic flux-density channels.
        """
        if et_data.empty:
            return pd.DataFrame()

        if self._last_simulated_at is None:
            self._last_simulated_at = et_data.index[0]

        rows = {}
        for ts in et_data.index:
            if ts <= self._last_simulated_at:
                continue
            seg_et_step = {name: frame.loc[[ts]] for name, frame in seg_et.items()}
            diagnostics = self.advance(et_data.loc[[ts]], ts, seg_et_step, cancel=cancel)
            if diagnostics:
                rows[ts] = diagnostics

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame.from_dict(rows, orient="index")

    def top_segment_names(self) -> list[str]:
        """Names of soil-mesh top segments where evaporation acts."""
        return list(self._top_segment_names)

    def segment_face_length(self, name: str) -> float:
        """Top-boundary face length [m] for a segment."""
        return float(self._segment_face_len.get(name, 0.0))

    def get_probes(self) -> list[ProbeSpec]:
        """Resolved probe sampling recipes (point + area).

        Method (not ``@property``) to avoid numpy ambiguous-truth in lories' ``get_members`` reflection.
        """
        return list(self._probes)

    def get_rel_sat_snapshot(self) -> np.ndarray:
        """Copy of the live saturation field for ``SoilPredictor`` forecast roll-outs.

        Method (not ``@property``) for the same reason as :meth:`get_probes`.
        """
        return self._pde.snapshot()

    def apply_state_blob(self, raw: bytes, timestamp: pd.Timestamp) -> None:
        if raw is None or len(raw) == 0:
            return
        if self._last_simulated_at is not None and timestamp <= self._last_simulated_at:
            # Restrict the restore to the initial read: once wired, the read-side
            # connector's listener also fires on every tick's own _save_state()
            # self-notification, and a stale connector read arriving after ticking
            # starts must not rewind the sim clock backwards on the next advance().
            logger.debug(
                "%s: ignoring soil state blob from %s (already simulated through %s)",
                self.name,
                timestamp,
                self._last_simulated_at,
            )
            return
        # Q11: serialized against advance()'s PDE-mutating span via the same
        # lock, so a restore landing mid-advance blocks here until the tick's
        # critical section (through _save_state) has finished, then applies.
        with self._pde_lock or nullcontext():
            # Re-check after acquiring: the advance() this restore blocked on
            # may have ratcheted _last_simulated_at past this blob's timestamp
            # (double-checked locking -- the unlocked guard above is only the
            # cheap early-out).
            if self._last_simulated_at is not None and timestamp <= self._last_simulated_at:
                logger.debug(
                    "%s: ignoring soil state blob from %s (superseded while waiting; simulated through %s)",
                    self.name,
                    timestamp,
                    self._last_simulated_at,
                )
                return
            try:
                self._pde.load_state_blob(raw)
            except Exception:  # noqa: BLE001
                # A blob from a different mesh (config changed between runs) must not
                # kill the state-restore callback; cold-start instead.
                logger.exception("%s: ignoring incompatible soil state blob from %s", self.name, timestamp)
                return
            self._last_simulated_at = timestamp
            logger.info("%s: restored soil state from %s", self.name, timestamp)

    def _save_state(self, timestamp: pd.Timestamp) -> None:
        self.data[SoilSimulation.SIMULATION_STATE].set(timestamp, self._pde.save_state_blob())
        self._last_simulated_at = timestamp

    def _compute_flux_rates(
        self,
        et_data: pd.DataFrame,
        seg_et: dict[str, pd.DataFrame],
        elapsed_s: float,
    ) -> FluxRates:
        """Per-zone mass fluxes [kg/(m²·s)] constant over the advance window.

        Negative ET (radiative cooling) is clipped to zero.
        ``rain_flux = precip_mm / elapsed_s`` distributes precipitation mass-conservatively over substeps.
        """
        seg_evap: dict[str, float] = {}
        seg_transp: dict[str, float] = {}
        for name, frame in seg_et.items():
            evap = max(0.0, float(frame["evap"].iloc[-1]))
            transp = max(0.0, float(frame["transp"].iloc[-1]))
            if evap > 0.0:
                seg_evap[name] = evap
            if transp > 0.0:
                seg_transp[name] = transp

        # Whole-field meter [l/min] → m³/s per out-of-plane metre (see configure()).
        # NULL/absent flow means "not watering"; NaN must never reach the source.
        flow_lpm = 0.0
        if SoilSimulation.IRRIGATION_FLOW_LPM in et_data.columns:
            value = et_data[SoilSimulation.IRRIGATION_FLOW_LPM].iloc[-1]
            if pd.notna(value):
                flow_lpm = float(value)
        flow_m3s = flow_m3s_per_m(flow_lpm, self._total_drip_line_length_m)
        self._warn_absurd_strip_flux(flow_m3s)

        rain_flux = 0.0
        if elapsed_s > 0 and Weather.PRECIPITATION in et_data.columns:
            precip_mm = et_data[Weather.PRECIPITATION].iloc[-1]
            if pd.notna(precip_mm) and precip_mm > 0:
                rain_flux = float(precip_mm) / elapsed_s  # mm/s == kg/(m²·s)

        return FluxRates(
            seg_evap=seg_evap,
            seg_transp=seg_transp,
            flow_m3s=flow_m3s,
            rain_flux=rain_flux,
        )

    def _warn_absurd_strip_flux(self, flow_m3s: float) -> None:
        """Warn once if the watering-strip flux is beyond plausible irrigation rates."""
        if self._strip_flux_warned or flow_m3s <= 0.0:
            return
        watering_width = self._mesh_config.watering_width
        if watering_width <= 0.0:
            return
        strip_mm_h = flow_m3s / watering_width * 3.6e6
        if strip_mm_h > SoilSimulation.STRIP_FLUX_WARN_MM_H:
            self._strip_flux_warned = True
            logger.warning(
                "%s: irrigation strip flux %.0f mm/h exceeds %.0f mm/h. Check the "
                "irrigation_flow values (must be true l/min, whole-field total) and "
                "total_drip_line_length_m (%.1f m; must be n_rows * row_length).",
                self.name,
                strip_mm_h,
                SoilSimulation.STRIP_FLUX_WARN_MM_H,
                self._total_drip_line_length_m,
            )

    def _walk(
        self,
        *,
        rates: FluxRates,
        window_s: float,
        clip_total: ClipDiagnostics,
        sim_t0: pd.Timestamp,
        plot_interval: Optional[pd.Timedelta],
        cancel: Optional[Callable[[], bool]] = None,
    ) -> WalkResult:
        """Adaptive-dt walk over ``window_s``.

        Under-converged substeps at ``dt_min`` are accepted; non-finite ones are skipped.
        Returns the underlying ``WalkResult`` so the caller can persist
        ``skipped_s`` and react to ``cancelled`` (the shutdown-only escape
        hatch; see ``walk_window``).
        """

        def on_step(t_offset: float) -> None:
            if plot_interval is None:
                return
            sim_t = sim_t0 + pd.Timedelta(seconds=t_offset)
            if plot_style.render_due(self._last_plot_simtime, sim_t, plot_interval):
                self._render_progress_safe(sim_t)

        result = self._pde.walk_window(
            rates=rates,
            window_s=window_s,
            accept_at_dt_min=True,
            cancel=cancel,
            on_step=on_step,
            log_name=self.name,
        )
        clip_total.add(result.clip)

        if result.skipped_s > 0:
            logger.error(
                "%s: held state through %.1fs of a %.1fs window (substeps "
                "unsolvable at dt_min); flux diagnostics assume forcing over "
                "the full window, so drainage is misstated for the skipped slices.",
                self.name,
                result.skipped_s,
                window_s,
            )
        if result.retries:
            logger.debug(
                "%s: adaptive walk completed window=%.1fs with %d retry(s)",
                self.name,
                window_s,
                result.retries,
            )
        return result

    def _record_diagnostics(
        self,
        rates: FluxRates,
        now: pd.Timestamp,
        delta_storage: float,
        elapsed_s: float,
        clip: ClipDiagnostics,
        skipped_s: float,
        retries: int = 0,
    ) -> dict[str, float]:
        """Write the seven per-callback flux-density channels [kg/(m²·h)], the
        skipped-at-dt_min, walk-retries, weather-stall, and tick-failure
        diagnostics channels, and sample probes."""
        diagnostics = self._compute_diagnostics(rates, delta_storage, elapsed_s, clip)
        diagnostics[SoilSimulation.WALK_SKIPPED_S.key] = skipped_s
        diagnostics[SoilSimulation.WALK_RETRIES.key] = float(retries)
        diagnostics[SoilSimulation.WEATHER_STALL.key] = float(getattr(self, "_weather_stall_ticks", 0.0))
        diagnostics[SoilSimulation.TICK_FAILURES.key] = float(getattr(self, "_tick_failures", 0.0))
        for key, value in diagnostics.items():
            self.data[key].set(now, value)
        self._sample_probes(now)
        return {self.data[key].id: value for key, value in diagnostics.items()}

    def _configure_probes(self, configs: Configurations) -> None:
        """Resolve probe specs from ``[probes]`` and register one float channel per probe."""
        self._probes = []
        # Sensor-derived probes are discovered once at activation, via
        # validate_sensor_probes() (separate from config probes: sample-only, no
        # logged channel). Opt-in: the only consumer is the anchor work, so
        # discovery stays off unless a config explicitly enables it.
        self._sensor_probes = []
        # Anchor sensors + their tension channels + per-sensor data context (for the
        # per-tick ranged read) + "last anchored at" map, populated by
        # _discover_sensor_probes at activation. ``_anchor_history`` holds the
        # per-sensor tension series load_anchor_history() range-reads each tick, so
        # each reading is assimilated at its own timestamp (see _read_history_tension).
        self._anchor_sensors: list[AnchorSensor] = []
        self._anchor_channels: dict[str, Any] = {}
        self._anchor_data: dict[str, Any] = {}
        self._anchor_history: dict[str, pd.Series] = {}
        self._last_anchored: dict[str, pd.Timestamp] = {}
        # Wall-clock staleness watch (issue 23/W2.5): lories converts connector
        # errors into EMPTY frames, so a dead tensiometer reads as benign no-data
        # forever. Track the last non-empty read per sensor (wall clock, never the
        # chunk bounds -- catch-up replays historical windows) and warn once per
        # dry spell, latched until a non-empty read recovers the sensor.
        self._anchor_last_read: dict[str, pd.Timestamp] = {}
        self._anchor_stale_warned: set[str] = set()
        # Enabling [anchor] drives discovery on, so the operator sets one switch.
        self._discover_sensor_probes_enabled = (
            configs.get_bool("discover_sensor_probes", default=False) or self._anchor_cfg.enabled
        )
        if not configs.has_member("probes"):
            return
        probes_cfg = configs.get_member("probes", defaults={})
        for probe in resolve_probes(probes_cfg, self._mesh_fipy, self._mesh_config, log_name=self.name):
            self._register_probe(probe)
            self._probes.append(probe)

        if self._probes:
            self._validate_probe_soil_ids()
            logger.info(
                "%s: registered %d tension probe(s)",
                self.name,
                len(self._probes),
            )

    def _register_probe(self, probe: ProbeSpec) -> None:
        self.data.add(
            probe.channel_id,
            type=float,
            name=probe.name,
            unit="hPa",
            aggregate="mean",
            logger={"enabled": True, "table": "agri_soil_simulation", "column": "water_tension"},
        )

    def _register_state_channel(self) -> None:
        self.data.add(
            SoilSimulation.SIMULATION_STATE,
            aggregate="last",
            logger={"enabled": True, "column": "state"},
        )

    def _register_progress_image_channel(self) -> None:
        self.data.add(
            SoilSimulation.SOIL_PROGRESS_IMAGE,
            aggregate="last",
            logger={"enabled": True, "column": "image"},
        )

    def _validate_probe_soil_ids(self) -> None:
        """soil_id identity (R6): per-probe ``[data.channels.<key>] soil_id = N``.

        A duplicate soil_id within the field raises; a probe missing soil_id
        only warns -- fixtures gain soil_ids in a later issue, so a
        configure-time raise here would red the suite until then.
        """
        channels_cfg = self.data.configs.get_member(Channels.TYPE, defaults={})
        seen: dict[Any, str] = {}
        for probe in self._probes:
            probe_cfg = channels_cfg.get_member(probe.channel_id, defaults={})
            soil_id = probe_cfg.get("soil_id", default=None)
            if soil_id is None:
                logger.warning(
                    "%s: probe '%s' has no soil_id configured; its "
                    "agri_soil_simulation rows cannot be attributed to a probe. "
                    "Set [data.channels.%s] soil_id = N (mirrored probes reuse "
                    "the twin sensor's id; model-only probes use ids >= 100).",
                    self.name,
                    probe.channel_id,
                    probe.channel_id,
                )
                continue
            if soil_id in seen:
                raise ConfigurationError(
                    f"{self.name}: duplicate soil_id {soil_id!r} on probes '{seen[soil_id]}' and '{probe.channel_id}'"
                )
            seen[soil_id] = probe.channel_id

    def _sample_probes(self, now: pd.Timestamp) -> None:
        if not self._probes:
            return
        for probe in self._probes:
            se = self._pde.sample(probe)
            self.data[probe.channel_id].set(now, self._tension_from_se(se))

    def get_sensor_probes(self) -> list[ProbeSpec]:
        """Probes derived from tension-measured SoilMoisture sensors, for
        model-vs-sensor comparison (the live anchor). Populated at activation by
        validate_sensor_probes(); empty before then. Method (not property) for
        the same reason as get_probes()."""
        return list(getattr(self, "_sensor_probes", []))

    # Anchor lifecycle -- bodies live in _anchor_runtime.AnchorRuntime, a
    # per-call duck-typed view over this sim; the delegates below keep the
    # pinned names/signatures, and anchor STATE stays resident on the sim.
    # The view is constructed INLINE per call (never a shared helper, never
    # cached): the history-read pins call these unbound on SimpleNamespace
    # fakes, which carry only the ``_anchor_*`` attributes themselves.

    def _discover_sensor_probes(self) -> list[tuple[str, Exception]]:
        """Sensor-probe/anchor discovery; see ``AnchorRuntime.discover``."""
        return _anchor_runtime.AnchorRuntime(self).discover()

    def validate_sensor_probes(self) -> None:
        """Run sensor-probe/anchor discovery once, at activation (called by
        FieldSimulation.activate before the tick thread starts); with [anchor]
        enabled a wiring failure refuses startup. See
        ``_anchor_runtime.AnchorRuntime.validate`` for the full semantics."""
        _anchor_runtime.AnchorRuntime(self).validate()

    def load_anchor_history(self, start: pd.Timestamp, end: pd.Timestamp) -> None:
        """Range-read each anchor sensor's tension for this tick (called by
        FieldSimulation._on_tick). See ``_anchor_runtime.AnchorRuntime.load_history``."""
        _anchor_runtime.AnchorRuntime(self).load_history(start, end)

    def _read_history_tension(self, sensor: AnchorSensor, now: pd.Timestamp) -> tuple[Optional[pd.Timestamp], float]:
        """Assimilation backend; see ``AnchorRuntime.read_history_tension``."""
        return _anchor_runtime.AnchorRuntime(self).read_history_tension(sensor, now)

    def _apply_anchor(self, now: pd.Timestamp, water_after_walk: float) -> None:
        """Nudge the post-walk saturation field toward fresh tensiometer readings.

        Runs only on the live path (advance()); the SoilPredictor forecast never
        anchors. Called after the walk and after the PDE-only mass-balance snapshot
        is taken, so the correction is excluded from the residual. The state
        update happens in ``AnchorRuntime.apply``; the WATER_ANCHOR publish and
        the per-sensor innovation log stay here (they need ``self.data`` /
        ``self._total_water``).
        """
        result = _anchor_runtime.AnchorRuntime(self).apply(now, water_after_walk)
        if result is None:
            return
        self.data[SoilSimulation.WATER_ANCHOR].set(now, self._total_water() - water_after_walk)
        for key, innovation in result.innovations.items():
            logger.debug("%s: anchor innovation %s = %+.4f Se", self.name, key, innovation)

    def _plot_mesh(self, save_path: Optional[str] = None):
        mesh = meshio.read(self._mesh_config.filename)

        points = mesh.points[:, :2]  # 2D coordinates

        lines = []
        line_colors = []
        triangles = []
        triangles_colors = []

        physical_tags = mesh.field_data
        physical_tags = sorted(physical_tags.keys(), key=lambda x: physical_tags[x][0])

        for cell_block in mesh.cells:
            tags = mesh.cell_data_dict["gmsh:physical"][cell_block.type]
            if cell_block.type == "line":
                for line, tag in zip(cell_block.data, tags):
                    lines.append([points[i] for i in line])
                    line_colors.append(tag - 1)

            if cell_block.type == "triangle":
                for tri, tag in zip(cell_block.data, tags):
                    triangles.append([points[i] for i in tri])
                    triangles_colors.append(tag - 1)

        lines = np.array(lines)
        line_colors = np.array(line_colors)
        triangles = np.array(triangles)
        triangles_colors = np.array(triangles_colors)

        fig, ax = plt.subplots(figsize=(8, 3), dpi=200)
        tab20 = plt.get_cmap("tab20")

        lc = LineCollection(lines, colors=tab20(line_colors), linewidths=3)
        ax.add_collection(lc)

        pc = PolyCollection(triangles, facecolors=tab20(triangles_colors), edgecolors="k", alpha=0.5)
        ax.add_collection(pc)

        ax.autoscale()
        ax.set_xlim(np.min(points[:, 0] - 1), np.max(points[:, 0]) + 1)
        ax.set_ylim(np.min(points[:, 1] - 1), np.max(points[:, 1]) + 1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        ax.legend(
            handles=[plt.Line2D([0], [0], color=tab20(i), lw=3) for i in range(len(physical_tags))],
            labels=physical_tags,
            loc="upper left",
            bbox_to_anchor=(1, 1),
            prop={"size": 6},
            ncol=2,
        )

        ax.set_title("Soil Mesh")
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=200)
            logger.info("%s: wrote mesh structure to %s", self.name, save_path)
        plt.close(fig)

    def _render_progress_if_due(self, now: pd.Timestamp) -> None:
        """Render the end-of-window frame only if a plot interval has elapsed.

        ``_last_plot_simtime`` is shared with the in-walk throttle and advanced
        by ``_render_progress_safe``, so gating here makes ``[plot] interval``
        govern the per-row ``simulate_loop`` path too, not just the substeps of a
        single long ``advance``. The first render (``_last_plot_simtime is None``)
        always fires so a fresh run isn't blank until the first interval passes.
        """
        if self._plot_config is None:
            return
        if plot_style.render_due(self._last_plot_simtime, now, self._plot_config.interval):
            self._render_progress_safe(now)

    def _render_progress_safe(self, sim_t: pd.Timestamp) -> None:
        """Render at sim_t with error containment: a failure skips this tick's
        rendering, and plot_style.PLOT_DISABLE_AFTER consecutive failures
        disable plotting for the rest of the run (W2.9 N-strikes policy) rather
        than crashing the solver."""
        self._last_plot_simtime = sim_t
        try:
            self._render_progress(sim_t)
        except Exception:  # noqa: BLE001
            self._plot_strikes, disable = plot_style.count_render_failure(logger, self.name, self._plot_strikes)
            plot_style.set_strike_channel(self, "plot_strikes", self._plot_strikes)
            if disable:
                self._plot_config = None
            return
        if self._plot_strikes:
            self._plot_strikes = 0
            plot_style.set_strike_channel(self, "plot_strikes", 0)

    def _render_progress(self, sim_t: pd.Timestamp) -> None:
        # plot_render._ensure_safe_backend (inside the session's lazy figure
        # init) forces Agg when off the main thread / headless -- the solver
        # runs on the field tick's worker thread.
        if self._plot_session is None:
            self._plot_session = plot_render.RenderSession(
                self._mesh_config.width,
                self._mesh_config.height,
            )
        timezone = getattr(getattr(self.context, "location", None), "timezone", None)
        png_bytes = self._plot_session.render(
            self._pde.mesh,
            self._pde.rel_sat.value,
            sim_t,
            tz=timezone,
        )
        self.data[SoilSimulation.SOIL_PROGRESS_IMAGE].set(sim_t, png_bytes)
