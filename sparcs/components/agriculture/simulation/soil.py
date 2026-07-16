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
from typing import Any, Optional

import matplotlib
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
from lories.util import to_timedelta

from . import plot_render, plot_style
from ._anchor import AnchorConfig, AnchorSensor, SensorOverrides, anchor_update, latest_reading_at
from ._soil import (
    SE_MAX,
    SE_MIN,
    ClipDiagnostics,
    FluxRates,
    MeshConfig,
    PDEConfig,
    ProbeSpec,
    SoilBase,
    apply_surface_forcing,
    ensure_mesh,
    resolve_probe_from_sensor,
    resolve_probes,
)

logger = logging.getLogger(__name__)


def _walk_components(root) -> list:
    """Flatten a component subtree into a list (root first)."""
    out: list = []
    stack = [root]
    while stack:
        c = stack.pop()
        out.append(c)
        children = getattr(c, "components", None)
        if children:
            try:
                stack.extend(list(children.values()))
            except Exception:
                # Some contexts expose iteration differently, so be liberal.
                try:
                    stack.extend(list(children))
                except Exception:
                    pass
    return out


def _opt_float(spec: Configurations, key: str) -> Optional[float]:
    """Read an optional float override from a per-sensor sub-block (``None`` if absent)."""
    value = spec.get(key, default=None)
    return None if value is None else float(value)


def _parse_anchor_config(configs: Configurations) -> AnchorConfig:
    """Parse an ``[anchor]`` block into the FiPy-free ``AnchorConfig``.

    Off by default, so existing configs and calibration runs are unaffected. Two
    mutually exclusive ways to name the allowlist:

    - a bare ``sensors`` value (list of keys, or a comma string) -- every sensor
      inherits the ``[anchor]`` globals; or
    - per-sensor ``[anchor.sensors.<key>]`` sub-blocks (mirroring the
      ``[probes.points.<name>]`` precedent), each optionally overriding
      ``sigma_meas_pf``/``staleness``/``r_horizontal``/``r_vertical``; any omitted
      key inherits the global. ``sigma_sys`` stays global (see ``SensorOverrides``).
    """
    if configs.has_member("sensors"):
        sensors: dict[str, SensorOverrides | None] = {
            str(key): SensorOverrides(
                sigma_meas_pf=_opt_float(spec, "sigma_meas_pf"),
                staleness=(
                    None if spec.get("staleness", default=None) is None else to_timedelta(spec.get("staleness"))
                ),
                r_horizontal=_opt_float(spec, "r_horizontal"),
                r_vertical=_opt_float(spec, "r_vertical"),
            )
            for key, spec in configs.get_member("sensors").items()
        }
    else:
        raw = configs.get("sensors", default=[]) or []
        if isinstance(raw, str):
            raw = [s.strip() for s in raw.split(",") if s.strip()]
        sensors = {str(key): None for key in raw}
    return AnchorConfig(
        enabled=configs.get_bool("enabled", default=False),
        sigma_sys=float(configs.get("sigma_sys", default=0.05)),
        sigma_meas_pf=float(configs.get("sigma_meas_pf", default=0.15)),
        r_horizontal=float(configs.get("r_horizontal", default=0.5)),
        r_vertical=float(configs.get("r_vertical", default=0.2)),
        staleness=to_timedelta(configs.get("staleness", default="6h")),
        sensors=sensors,
        min_tension_hpa=float(configs.get("min_tension_hpa", default=1.0)),
    )


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

    # Progress-plot state; _plot_config is None when plotting is disabled.
    _plot_fig: Any = None
    _plot_ax: Any = None
    _last_plot_simtime: Optional[pd.Timestamp] = None

    _probes: list[ProbeSpec]

    def configure(self, configs: Configurations) -> None:
        super().configure(configs)

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
        self._ode_config = PDEConfig(configs.get_member("pde", defaults={}), model_configs=model_block)
        # ponding + feddes are sibling blocks of [pde], not nested under it.
        apply_surface_forcing(self._ode_config, configs)

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

        self._plot_config = plot_style.load_plot_config(configs, default_interval="5min")
        if self._plot_config is not None:
            self._last_plot_simtime = None
            self._plot_fig = None
            self._plot_ax = None
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
    ) -> dict[str, float]:
        if self._simulating:
            logger.warning("%s: solve still running, skipping interval at %s", self.name, now)
            return {}

        self._simulating = True
        try:
            self._ensure_sensor_probes()
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
            # Include the surface ponds so pond build-up/drain-down does not
            # masquerade as bottom drainage in the balance diagnostics.
            storage_before = self._total_water() + self._pde.surface_water()
            interval = self._plot_config.interval if self._plot_config is not None else None
            clip_total = ClipDiagnostics()
            self._walk(
                rates=rates,
                window_s=elapsed_s,
                clip_total=clip_total,
                sim_t0=sim_t0,
                plot_interval=interval,
            )

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
            )
            self._save_state(now)
            return diagnostics
        finally:
            self._simulating = False

    def simulate_loop(
        self,
        et_data: pd.DataFrame,
        seg_et: dict[str, pd.DataFrame],
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
            diagnostics = self.advance(et_data.loc[[ts]], ts, seg_et_step)
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
        flow_m3s = flow_lpm / (60_000.0 * self._total_drip_line_length_m)
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
    ) -> None:
        """Adaptive-dt walk over ``window_s``.

        Under-converged substeps at ``dt_min`` are accepted; non-finite ones are skipped.
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
            on_step=on_step,
            log_name=self.name,
        )
        clip_total.add(result.clip)

        if result.skipped_s > 0:
            logger.warning(
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

    def _record_diagnostics(
        self,
        rates: FluxRates,
        now: pd.Timestamp,
        delta_storage: float,
        elapsed_s: float,
        clip: ClipDiagnostics,
    ) -> dict[str, float]:
        """Write the seven per-callback flux-density channels [kg/(m²·h)] and sample probes."""
        diagnostics = self._compute_diagnostics(rates, delta_storage, elapsed_s, clip)
        for key, value in diagnostics.items():
            self.data[key].set(now, value)
        self._sample_probes(now)
        return {self.data[key].id: value for key, value in diagnostics.items()}

    def _configure_probes(self, configs: Configurations) -> None:
        """Resolve probe specs from ``[probes]`` and register one float channel per probe."""
        self._probes = []
        # Sensor-derived probes are discovered lazily on first advance() (separate
        # from config probes: sample-only, no logged channel). See _discover_sensor_probes.
        # Opt-in: the only consumer is the (separate) anchor work, so discovery
        # stays off unless a config explicitly enables it.
        self._sensor_probes = []
        self._sensor_probes_ready = False
        # Anchor sensors + their tension channels + per-sensor data context (for the
        # per-tick ranged read) + "last anchored at" map, populated by
        # _discover_sensor_probes on first advance(). ``_anchor_history`` holds the
        # per-sensor tension series load_anchor_history() range-reads each tick, so
        # each reading is assimilated at its own timestamp (see _read_history_tension).
        self._anchor_sensors: list[AnchorSensor] = []
        self._anchor_channels: dict[str, Any] = {}
        self._anchor_data: dict[str, Any] = {}
        self._anchor_history: dict[str, pd.Series] = {}
        self._last_anchored: dict[str, pd.Timestamp] = {}
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
            # psi_from_se is the signed matric potential (negative hPa; drier ->
            # more negative), the tensiometer / DB convention, so publish it
            # directly. The PDE core (anchoring, total_water) stays in Se.
            self.data[probe.channel_id].set(now, float(self._soil_model.psi_from_se(se)))

    def get_sensor_probes(self) -> list[ProbeSpec]:
        """Probes derived from tension-measured SoilMoisture sensors, for
        model-vs-sensor comparison (the live anchor). Populated lazily on the
        first advance(); empty before then. Method (not property) for the same
        reason as get_probes()."""
        return list(getattr(self, "_sensor_probes", []))

    def _discover_sensor_probes(self) -> None:
        """Derive one probe per enabled, tension-measured SoilMoisture sensor in
        this field. A sensor is a probe that also carries measured data; here we
        resolve only the model-side sampling recipe (sample-only, no logged
        channel). Runs once, on the first advance(), when the whole component tree
        is configured. This sim's field is reached via SoilSimulation.context
        (the FieldSimulation) -> .context (the AgriculturalField that owns the
        sensors)."""
        # Set the flag first so a failure here doesn't retry on every step.
        self._sensor_probes_ready = True
        from sparcs.components.agriculture.soil.moisture import SoilMoisture

        field = getattr(getattr(self, "context", None), "context", None)
        if field is None:
            return
        found: list[ProbeSpec] = []
        anchor_sensors: list[AnchorSensor] = []
        anchor_channels: dict[str, Any] = {}
        anchor_data: dict[str, Any] = {}
        for comp in _walk_components(field):
            if not isinstance(comp, SoilMoisture):
                continue
            try:
                if not comp.has_measured_tension:
                    continue
                found.append(resolve_probe_from_sensor(comp, self._mesh_fipy, self._mesh_config))
                anchor_sensors.append(AnchorSensor(key=comp.key, x_offset_cm=comp.x_offset, depth_cm=comp.depth))
                anchor_channels[comp.key] = comp.data[SoilMoisture.WATER_TENSION]
                # Keep the sensor's data context so load_anchor_history can issue a
                # ranged tension read through its connector each tick.
                anchor_data[comp.key] = comp.data
            except Exception:
                logger.exception(
                    "%s: failed to derive probe from sensor %s",
                    self.name,
                    getattr(comp, "key", "?"),
                )
        self._sensor_probes = found
        self._anchor_sensors = anchor_sensors
        self._anchor_channels = anchor_channels
        self._anchor_data = anchor_data
        if found:
            logger.info("%s: discovered %d sensor probe(s) for anchoring", self.name, len(found))

    def _ensure_sensor_probes(self) -> None:
        """Run anchor sensor-probe discovery once, when enabled. Idempotent (guarded
        by ``_sensor_probes_ready``); a failure marks ready so it never retries or
        breaks the solve tick."""
        if not self._discover_sensor_probes_enabled or self._sensor_probes_ready:
            return
        try:
            self._discover_sensor_probes()
        except Exception:
            self._sensor_probes_ready = True
            logger.exception("%s: sensor-probe discovery failed; continuing", self.name)

    def load_anchor_history(self, start: pd.Timestamp, end: pd.Timestamp) -> None:
        """Range-read each anchor sensor's tension over ``(start - lookback, end]`` for
        this tick, so the anchor assimilates each reading at its own timestamp rather
        than smearing the latest value across the window. ``lookback`` is the widest
        sensor staleness, so a reading just before ``start`` still covers the opening
        rows; ``anchor_update``'s staleness gate drops anything too old. No-op when
        anchoring is off or no sensor was discovered; a per-sensor read failure leaves
        that sensor unanchored (predict-only) rather than breaking the tick."""
        self._anchor_history = {}
        self._ensure_sensor_probes()
        if not (self._anchor_cfg.enabled and self._anchor_sensors):
            return
        lookback = max(
            (self._anchor_cfg.sensor_staleness(s.key) for s in self._anchor_sensors),
            default=pd.Timedelta(0),
        )
        read_start = start - lookback
        for sensor in self._anchor_sensors:
            data = self._anchor_data.get(sensor.key)
            channel = self._anchor_channels.get(sensor.key)
            if data is None or channel is None:
                continue
            try:
                frame = data.read(Channels([channel]), start=read_start, end=end, unique=True)
            except Exception:
                logger.exception("%s: anchor history read failed for %s", self.name, sensor.key)
                continue
            if frame is None or frame.empty:
                continue
            series = frame.iloc[:, 0].dropna().sort_index()
            series = series[~series.index.duplicated(keep="last")]
            if not series.empty:
                self._anchor_history[sensor.key] = series

    def _read_history_tension(self, sensor: AnchorSensor, now: pd.Timestamp) -> tuple[Optional[pd.Timestamp], float]:
        """Assimilation backend: the tensiometer reading contemporaneous with sim step
        ``now`` -- the latest at or before it from this tick's ranged read
        (:meth:`load_anchor_history`), or ``(None, nan)`` when none. Replaces the old
        single-latest live read so each reading anchors at its own time."""
        return latest_reading_at(self._anchor_history.get(sensor.key), now)

    def _apply_anchor(self, now: pd.Timestamp, water_after_walk: float) -> None:
        """Nudge the post-walk saturation field toward fresh tensiometer readings.

        Runs only on the live path (advance()); the SoilPredictor forecast never
        anchors. Called after the walk and after the PDE-only mass-balance snapshot
        is taken, so the correction is excluded from the residual. Publishes the
        assimilation increment and logs each sensor's innovation.
        """
        sensors = [s for s in self._anchor_sensors if s.key in self._anchor_cfg.sensors]
        if not sensors:
            return
        result = anchor_update(
            np.asarray(self._pde.rel_sat.value),
            np.asarray(self._pde.mesh.cellCenters),
            sensors,
            lambda sensor: self._read_history_tension(sensor, now),
            now,
            self._anchor_cfg,
            self._pde.soil_model,
            self._mesh_config.width,
            self._last_anchored,
            SE_MIN,
            SE_MAX,
        )
        if result is None:
            return
        self._pde.set_state(result.se_new, update_old=True)
        self._last_anchored.update(result.anchored_at)
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
        """Render at sim_t with error containment; a single render failure
        disables progress plotting for the rest of the run (``_plot_config`` set
        to None) rather than crashing the solver."""
        self._last_plot_simtime = sim_t
        try:
            self._render_progress(sim_t)
        except Exception:  # noqa: BLE001
            logger.exception("%s: progress-plot render failed; disabling.", self.name)
            self._plot_config = None

    def _init_progress_figure(self) -> None:
        # The solver runs on the field tick's worker thread, so render headless.
        if matplotlib.get_backend().lower() not in (
            "agg",
            "module://matplotlib_inline.backend_inline",
        ):
            matplotlib.use("Agg", force=True)
        fig, ax, norm = plot_render.init_rel_sat_figure(
            self._mesh_config.width,
            self._mesh_config.height,
        )
        self._plot_fig = fig
        self._plot_ax = ax
        self._plot_norm = norm

    def _render_progress(self, sim_t: pd.Timestamp) -> None:
        if self._plot_fig is None:
            self._init_progress_figure()

        timezone = getattr(getattr(self.context, "location", None), "timezone", None)
        png_bytes = plot_render.render_rel_sat_png(
            self._plot_fig,
            self._plot_ax,
            self._plot_norm,
            self._pde.mesh,
            self._pde.rel_sat.value,
            sim_t,
            tz=timezone,
        )
        self.data[SoilSimulation.SOIL_PROGRESS_IMAGE].set(sim_t, png_bytes)
