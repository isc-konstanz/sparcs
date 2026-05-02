# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.soil_simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""

from __future__ import annotations

import io
import logging
import os
import threading
from dataclasses import dataclass
from typing import Any, Optional

import gmsh
import matplotlib
import matplotlib.pyplot as plt
import meshio
from fipy import CellVariable, DiffusionTerm, FaceVariable, TransientTerm
from fipy.meshes import Gmsh2D
from fipy.tools import serialComm
from matplotlib.collections import LineCollection, PolyCollection
from scipy.interpolate import griddata

import numpy as np
import pandas as pd
from lories import Component, Constant
from lories.components.weather import Weather
from lories.typing import Configurations
from sparcs.components.agriculture.soil import Genuchten, SoilModel

logging.getLogger("fipy").setLevel(logging.WARNING)
np.seterr(all="ignore")

RHO_W: float = 1000.0       # kg/m³
SE_MIN: float = 1e-6        # effective-saturation floor for source clipping
SE_MAX: float = 0.999       # effective-saturation ceiling for source clipping
PLOT_HISTORY_WINDOW: pd.Timedelta = pd.Timedelta(days=1)


@dataclass
class FluxRates:
    """Per-callback fluxes consumed by ``_apply_source`` and diagnostics.

    All flux densities are kg/(m²·s); ``flow_m3s`` is volumetric m³/s.
    ``seg_evap`` and ``seg_transp`` are keyed by mesh segment name and only
    contain entries that participate in their respective sink (canopy
    segments for transpiration, all top segments for evaporation).
    """

    seg_evap: dict[str, float]
    seg_transp: dict[str, float]
    flow_m3s: float
    rain_flux: float

# TODO: Remove — exploratory Parameters() draft from before MeshConfig/PDEConfig
#   landed. Re-introduce as a real Parameters schema if/when lories.Parameter
#   adds the missing converter / Select / List / Component support.
# MESH_PARAMS = Parameters(
#     "mesh",
#     [
#         Parameter(name="filename", default="soil.msh", required=True, desc="Mesh filename"),
#         Parameter(name="dl", default=0.1, required=True, desc="Mesh element size"),
#         Parameter(name="width", default=10.0, required=True, desc="Width of the agricultural field (m)"),
#         Parameter(name="height", default=5.0, required=True, desc="Height of the agricultural field (m)"),
#         Parameter(name="plant_width", default=2.0, required=True, desc="Width of the plant (m)"),
#         Parameter(name="plant_height", default=2.0, required=True, desc="Height of the plant (m)"),
#         Parameter(name="watering_width", default=1.0, required=True, desc="Width of the watering area (m)"),
#         Parameter(name="d_x", default=0.5, required=True, desc="Width of ground segments (m)"),
#     ],
# )

# PDE_PARAMS = Parameters(
#     "pde",
#     [
#         Parameter(name="theta_r", default=0.05, required=True, desc="Residual water content"),
#         Parameter(name="theta_s", default=0.43, required=True, desc="Saturated water content"),
#         Parameter(name="alpha", default=0.08, required=True, desc="Van Genuchten parameter alpha"),
#         Parameter(name="n", default=1.6, required=True, desc="Van Genuchten parameter n"),
#         Parameter(name="k_s", default=1.0e-4, required=True, desc="Saturated hydraulic conductivity (m/s)"),
#         Parameter(name="dt", default=50.0, required=True, desc="Time step for the simulation (s)"),
#     ],
# )


_DEFAULT_BAY_WIDTH: float = 10.0


@dataclass
class MeshConfig:
    def __init__(self, configs: Configurations, bay_width: Optional[float] = None):
        # ``bay_width`` is the single source of truth for the soil-mesh /
        # PV-row-spacing width (set on ``FieldSimulation``). When passed in
        # we use it as the default for ``width`` so the mesh and ground-
        # shading bay always agree by construction. A ``width`` explicitly
        # set in the [mesh] block still wins (back-compat).
        default_width = _DEFAULT_BAY_WIDTH if bay_width is None else bay_width
        self.filename: str = configs.get("filename", default="soil.msh")
        self.dl: float = configs.get("dl", default=0.1)
        self.width: float = configs.get("width", default=default_width)
        self.height: float = configs.get("height", default=5.0)
        self.plant_width: float = configs.get("plant_width", default=2.0)
        self.plant_height: float = configs.get("plant_height", default=2.0)
        self.watering_width: float = configs.get("watering_width", default=1.0)
        self.dx: float = configs.get("d_x", default=0.5)


def top_segment_names_from_mesh(mesh: "MeshConfig") -> list[str]:
    """Soil-mesh top-segment names derived purely from MeshConfig.

    Mirrors the naming used in ``SoilSimulation._build_segment_index``; kept
    standalone so GroundShading and Evapotranspiration can register one
    channel per segment at configure time, before SoilSimulation itself is
    built. Order matches the mesh: left bare strips, plant tops, right bare
    strips.
    """
    n_pv_segments = int((mesh.width - mesh.plant_width) / (2 * mesh.dx))
    open_sky = [
        f"{side}TopSegment_{i}"
        for i in range(n_pv_segments)
        for side in ("Left", "Right")
    ]
    return [*open_sky, "PlantTopLeftSegment", "PlantTopRightSegment"]


@dataclass
class PDEConfig:
    def __init__(self, configs: Configurations):
        self.theta_r: float = configs.get("theta_r", default=0.05)
        self.theta_s: float = configs.get("theta_s", default=0.43)
        self.alpha: float = configs.get("alpha", default=0.08)
        self.n: float = configs.get("n", default=1.6)
        self.k_s: float = configs.get("k_s", default=1.0e-4)
        self.dt: float = configs.get("dt", default=50.0)


@dataclass
class PlotConfig:
    def __init__(self, configs: Configurations, default_dir: str):
        interval = configs.get("interval", default="5min")
        if isinstance(interval, (int, float)):
            self.interval: pd.Timedelta = pd.Timedelta(seconds=float(interval))
        else:
            self.interval: pd.Timedelta = pd.Timedelta(interval)
        # `live`: overwrite a single file `progress.png` each interval — pair it
        # with the auto-generated `progress.html` for a live browser view.
        # `save`: also archive a timestamped PNG per frame.
        # `show`: pop a matplotlib window (only works when the solver runs on
        # the main thread; auto-disabled otherwise).
        self.live: bool = configs.get_bool("live", default=True)
        self.save: bool = configs.get_bool("save", default=False)
        self.show: bool = configs.get_bool("show", default=False)
        self.dir: str = configs.get("dir", default=default_dir)


class SoilSimulation(Component):
    TYPE: str = "soil_simulation"
    INCLUDES = ["mesh", "pde", "plot"]

    SIMULATION_STATE = Constant(bytes, "simulation_state", "Soil Simulation State", "-")
    SOIL_PROGRESS_IMAGE = Constant(bytes, "soil_progress_image", "Soil Simulation Progress Image", "png")

    # Diagnostic flux densities reported per callback. Internal flux
    # math runs in kg/(m^2*s); these channels publish in g/(m^2*s) so
    # typical 1e-4 mass-flux values display readably (~0.1) in the UI.
    WATER_TOP_IN = Constant(float, "water_top_in", "Top Water Input (Irrigation)", "g/(m^2*s)")
    WATER_TOP_OUT = Constant(float, "water_top_out", "Top Water Output (Evaporation)", "g/(m^2*s)")
    WATER_BOTTOM = Constant(float, "water_bottom", "Bottom Water Output (Drainage)", "g/(m^2*s)")
    WATER_TRANSP = Constant(float, "water_transpiration", "Plant Transpiration", "g/(m^2*s)")

    _mesh_filename: str
    _mesh_fipy: Gmsh2D
    _variables: dict[str, Any]
    _equations: dict[str, Any]
    _soil_model: SoilModel

    _mesh_config: MeshConfig
    _ode_config: PDEConfig
    _plot_config: Optional[PlotConfig] = None

    _last_simulated_at: Optional[pd.Timestamp] = None
    _simulating: bool = False

    # Progress-plot state
    _plot_progress: bool = False
    _plot_fig: Any = None
    _plot_axes: Any = None
    _plot_history: list = None
    _last_plot_simtime: Optional[pd.Timestamp] = None

    # Boundary / bulk index caches built once after mesh load
    _segment_cells: dict[str, np.ndarray]
    _segment_face_len: dict[str, float]
    _segment_cell_volume: dict[str, float]
    _top_segment_names: list[str]           # segments where soil evaporation acts
    _open_sky_segment_names: list[str]      # bare-soil strips not under the PV roof — rain falls here
    _plant_cells: np.ndarray
    _plant_volume: float

    # Pre-computed θ-rate conversion factors (multiplied by mass fluxes per callback)
    _irrigation_factor: float               # 1 / vol                         [1/m^3]
    _theta_diff: float                      # θ_s - θ_r
    _rain_face_len: float                   # Σ face_len over open-sky segments

    def configure(self, configs: Configurations) -> None:
        super().configure(configs)

        self._mesh_config = MeshConfig(
            configs.get_member("mesh", defaults={}, ensure_exists=True),
            bay_width=getattr(self.context, "bay_width", None),
        )
        self._ode_config = PDEConfig(configs.get_member("pde", defaults={}))

        # Evapotranspiration's input + output channels live on the
        # Evapotranspiration sibling now; SoilSimulation only owns its own
        # state + diagnostic flux channels and subscribes to the sibling.

        self.data.add(SoilSimulation.SIMULATION_STATE, aggregate="last", logger={"enabled": True})
        for c in (
            SoilSimulation.WATER_TOP_IN,
            SoilSimulation.WATER_TOP_OUT,
            SoilSimulation.WATER_BOTTOM,
            SoilSimulation.WATER_TRANSP,
        ):
            self.data.add(c, aggregate="mean", logger={"enabled": False})

        self._soil_model = Genuchten(
            theta_r=self._ode_config.theta_r,
            theta_s=self._ode_config.theta_s,
            alpha=self._ode_config.alpha,
            n=self._ode_config.n,
            k_s=self._ode_config.k_s,
        )

        if not os.path.exists(self._mesh_config.filename):
            self._create_mesh()

        # `plot_structure`: dump a static mesh.png next to the progress plots
        # (or into <data_dir>/soil_simulation if no [plot] block is configured).
        # NEVER opens a GUI window — _plot_mesh used to call a blocking
        # plt.show() that froze startup on threaded / headless setups.
        if configs.get_bool("plot_structure", default=False):
            structure_dir = configs.get(
                "plot.dir",
                default=str(configs.dirs.data.joinpath("soil_simulation")),
            )
            os.makedirs(structure_dir, exist_ok=True)
            self._plot_mesh(os.path.join(structure_dir, "mesh.png"))

        # TODO: Fix not working
        self._plot_progress = configs.get_bool("plot_progress", default=True)
        logging.info(
            "%s: plot_progress=%s (configs keys: %s)",
            self.name, self._plot_progress, sorted(k for k in configs),
        )
        if self._plot_progress:
            default_plot_dir = str(configs.dirs.data.joinpath("soil_simulation"))
            self._plot_config = PlotConfig(
                configs.get_member("plot", defaults={}, ensure_exists=True),
                default_dir=default_plot_dir,
            )
            self._plot_history = []
            self._last_plot_simtime = None
            self._plot_fig = None
            self._plot_axes = None
            if self._plot_config.save or self._plot_config.live:
                os.makedirs(self._plot_config.dir, exist_ok=True)
            self.data.add(SoilSimulation.SOIL_PROGRESS_IMAGE, aggregate="last", logger={"enabled": True})

        self._mesh_fipy = Gmsh2D(self._mesh_config.filename, communicator=serialComm)

        self._build_eq()
        self._constrain_eq()
        self._build_segment_index()

    def advance(
        self,
        et_data: pd.DataFrame,
        now: pd.Timestamp,
        seg_et: dict[str, pd.DataFrame],
    ) -> dict[str, float]:
        if self._simulating:
            logging.warning("%s: solve still running, skipping interval at %s", self.name, now)
            return {}

        self._simulating = True
        try:
            # Cold start (no logger restore): spin up one hour with current
            # weather to reach an approximate steady state instead of using the
            # static IC from `_constrain_eq`.
            if self._last_simulated_at is None:
                elapsed = pd.Timedelta(hours=1)
                logging.info("%s: cold start spin-up — 1h with weather at %s", self.name, now)
            else:
                elapsed = now - self._last_simulated_at

            dt = self._ode_config.dt
            n_steps = max(1, int(elapsed.total_seconds() / dt))
            elapsed_s = n_steps * dt

            rates = self._compute_flux_rates(et_data, seg_et, elapsed_s)
            sim_t0 = now - pd.Timedelta(seconds=elapsed_s)
            storage_before = self._total_water()
            step_storage_prev = storage_before
            for i in range(n_steps):
                self._apply_source(rates, dt)
                self._solve(dt)
                if self._plot_progress:
                    step_storage_now = self._total_water()
                    step_drainage = self._balance_drainage_flux(
                        rates, step_storage_now - step_storage_prev, dt
                    )
                    step_storage_prev = step_storage_now
                    sim_t = sim_t0 + pd.Timedelta(seconds=(i + 1) * dt)
                    self._capture_progress(sim_t, rates, step_drainage)

            delta_storage = self._total_water() - storage_before
            diagnostics = self._record_diagnostics(rates, now, delta_storage, elapsed_s)
            self._save_state(now)
            return diagnostics
        finally:
            self._simulating = False

    def simulate_loop(
        self,
        et_data: pd.DataFrame,
        seg_et: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Step the soil PDE through ``et_data`` deterministically (offline mode).

        ``seg_et`` carries the same time index as ``et_data`` (one DataFrame
        per segment, columns ``et``/``evap``/``transp``); each timestep is
        sliced out and handed to ``advance``.

        Returns a DataFrame indexed by timestamp with the four diagnostic
        flux-density channels keyed by channel id.
        """
        if et_data.empty:
            return pd.DataFrame()

        # Anchor the simulation clock to the first input timestamp so the
        # initial advance integrates t[0] -> t[1] with t[1]'s weather instead
        # of doing a fictional 1h cold-start spin-up.
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

    def apply_state_blob(self, raw: bytes, timestamp: pd.Timestamp) -> None:
        if raw is None or len(raw) == 0:
            return
        buf = io.BytesIO(raw)
        arrays = np.load(buf)
        self._variables["rel_sat"].setValue(arrays["rel_sat"])
        self._variables["rel_sat"]._old.setValue(arrays["rel_sat_old"])
        self._last_simulated_at = timestamp
        logging.info("%s: restored soil state from %s", self.name, timestamp)

    def _save_state(self, timestamp: pd.Timestamp) -> None:
        rel_sat = self._variables["rel_sat"]
        buf = io.BytesIO()
        np.savez(buf, rel_sat=rel_sat.value.copy(), rel_sat_old=rel_sat._old.value.copy())
        self.data[SoilSimulation.SIMULATION_STATE].set(timestamp, buf.getvalue())
        self._last_simulated_at = timestamp

    def _compute_flux_rates(
        self,
        et_data: pd.DataFrame,
        seg_et: dict[str, pd.DataFrame],
        elapsed_s: float,
    ) -> FluxRates:
        """
        Per-zone mass fluxes that are constant over a callback's elapsed
        window. Evap and transp are read straight from the per-segment ET
        decomposition that ``Evapotranspiration.evaluate`` produced — the
        Beer-Lambert split was already applied there using the segment's
        local LAI. Negative ET (radiative cooling case) is clipped to zero
        so we never inject water through the boundaries by accident.

        ``rain_flux`` comes from ``Weather.PRECIPITATION`` (mm accumulated over
        the data interval) divided by ``elapsed_s`` (1 mm == 1 kg/m²).
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

        flow_m3s = self.context._irrigation_flow_lpm / 60_000.0   # l/min → m³/s

        rain_flux = 0.0
        if elapsed_s > 0 and Weather.PRECIPITATION in et_data.columns:
            precip_mm = et_data[Weather.PRECIPITATION].iloc[-1]
            if pd.notna(precip_mm) and precip_mm > 0:
                rain_flux = float(precip_mm) / elapsed_s   # mm/s == kg/(m²·s)

        return FluxRates(
            seg_evap=seg_evap,
            seg_transp=seg_transp,
            flow_m3s=flow_m3s,
            rain_flux=rain_flux,
        )

    def _apply_source(self, rates: FluxRates, dt: float) -> None:
        """
        Rebuild the ``source`` CellVariable for the next ``dt``-second step.

        Source value is θ-rate (1/s). All contributions are summed per cell,
        then clipped together so Sₑ stays within ``[SE_MIN, SE_MAX]`` after
        one step.
        """
        se = self._variables["rel_sat"].value
        coeff = self._theta_diff
        theta_rate = np.zeros_like(se)

        # === Rain on the bare-soil strips left and right of the plant
        # block — these are open to the sky. The plant zone (plant tops +
        # watering strip) sits under the PV roof and gets no rain;
        # irrigation reaches the roots through the volumetric drip source
        # on WateringTopSegment below.
        if rates.rain_flux != 0.0:
            for name in self._open_sky_segment_names:
                cells = self._segment_cells.get(name)
                if cells is None or cells.size == 0:
                    continue
                vol = self._segment_cell_volume.get(name, 0.0)
                if vol <= 0:
                    continue
                factor = self._segment_face_len[name] / (RHO_W * vol)
                theta_rate[cells] += rates.rain_flux * factor

        # === Soil evaporation per segment. The shading factor is already
        # baked into each ``rates.seg_evap[name]`` value (applied at ET
        # evaluation time), so this loop just maps the segment flux density
        # onto its boundary cells. Watering segment is excluded — its
        # surface state is governed by the irrigation source.
        for name, evap in rates.seg_evap.items():
            cells = self._segment_cells.get(name)
            if cells is None or cells.size == 0:
                continue
            vol = self._segment_cell_volume.get(name, 0.0)
            if vol <= 0:
                continue
            factor = self._segment_face_len[name] / (RHO_W * vol)
            theta_rate[cells] -= evap * factor

        # === Drip irrigation on WateringTopSegment (volumetric, additive) ===
        if rates.flow_m3s > 0.0 and self._irrigation_factor > 0:
            cells = self._segment_cells["WateringTopSegment"]
            if cells.size:
                theta_rate[cells] += rates.flow_m3s * self._irrigation_factor

        # === Canopy transpiration over PlantSurface bulk cells. Per-segment
        # transp values share the same canopy bulk as a sink: total mass per
        # unit out-of-plane = Σ transp_seg · face_len_seg [kg/(m·s)], then
        # divided by RHO_W · plant_volume to land at θ-rate (1/s).
        if rates.seg_transp and self._plant_volume > 0:
            transp_mass = sum(
                value * self._segment_face_len.get(name, 0.0)
                for name, value in rates.seg_transp.items()
            )
            if transp_mass > 0:
                theta_rate[self._plant_cells] -= transp_mass / (RHO_W * self._plant_volume)

        # Per-cell clip so Sₑ stays within bounds after one dt step.
        src = self._clip_theta_rate(se, theta_rate, dt, coeff)
        self._variables["source"].setValue(src)

    @staticmethod
    def _clip_theta_rate(se_now: np.ndarray, theta_rate: float, dt: float, coeff: float) -> np.ndarray:
        """
        Clip a uniform θ-rate so ``Sₑ_now + (theta_rate/coeff)·dt`` stays in
        ``[SE_MIN, SE_MAX]`` per cell. Mirrors check_top/check_bottom from the
        validation notebook.
        """
        max_pos = np.maximum((SE_MAX - se_now) / dt, 0.0) * coeff
        max_neg = np.minimum((SE_MIN - se_now) / dt, 0.0) * coeff
        return np.clip(theta_rate, max_neg, max_pos)

    def _total_water(self) -> float:
        # Σ θ · cellVolume · ρ_w. In 2D this is mass per unit out-of-plane
        # depth, in the same units as the time-integrated face-flux balance
        # used by ``_balance_drainage_flux`` (so storage Δ and inflow Δ
        # cancel exactly when no water leaves through the bottom).
        se = self._variables["rel_sat"].value
        theta = self._ode_config.theta_r + self._theta_diff * se
        return float(np.sum(theta * np.asarray(self._mesh_fipy.cellVolumes))) * RHO_W

    def _balance_drainage_flux(
        self,
        rates: FluxRates,
        delta_storage: float,
        duration_s: float,
    ) -> float:
        # Bottom drainage from the integral mass balance:
        #   drainage_mass = (rain_in + irr_in - evap_out - transp_out)·dt - Δstorage
        # then divided by (bottom_face_len · dt) to land back in kg/(m²·s).
        # The face-flux estimator that lived here previously missed the
        # contribution of the Dirichlet BC at the bottom, so it always
        # under-reported drainage during a wet front passing through.
        bottom_len = self._segment_face_len.get("GroundBottomSegment", 0.0)
        if bottom_len <= 0 or duration_s <= 0:
            return 0.0
        evap_mass = sum(
            value * self._segment_face_len.get(name, 0.0)
            for name, value in rates.seg_evap.items()
        )
        transp_mass = sum(
            value * self._segment_face_len.get(name, 0.0)
            for name, value in rates.seg_transp.items()
        )
        in_rate = rates.rain_flux * self._rain_face_len + rates.flow_m3s * RHO_W
        out_rate = evap_mass + transp_mass
        drainage_mass = (in_rate - out_rate) * duration_s - delta_storage
        return drainage_mass / (bottom_len * duration_s)

    def _record_diagnostics(
        self,
        rates: FluxRates,
        now: pd.Timestamp,
        delta_storage: float,
        elapsed_s: float,
    ) -> dict[str, float]:
        """Write the four per-callback flux-density channels in g/(m²·s).

        WATER_TOP_OUT and WATER_TRANSP are reported as face-length-weighted
        spatial means so a single number stays meaningful while the underlying
        physics is per-segment. Internal flux math runs in kg/(m²·s); we
        convert at this boundary so channels and the returned diagnostics
        DataFrame agree with the unit declared on each Constant.
        """
        watering_len = self._segment_face_len.get("WateringTopSegment", 0.0)
        irr_flux = rates.flow_m3s * RHO_W / watering_len if watering_len > 0 else 0.0
        # WATER_TOP_IN reports total top input (irrigation drip + rain).
        top_in = irr_flux + rates.rain_flux

        e_flux_mean = self._face_weighted_mean(rates.seg_evap, self._top_segment_names)
        t_flux_mean = self._face_weighted_mean(rates.seg_transp, self._top_segment_names)
        bottom = self._balance_drainage_flux(rates, delta_storage, elapsed_s)

        kg_to_g = 1000.0
        diagnostics = {
            self.data[SoilSimulation.WATER_TOP_OUT]: e_flux_mean * kg_to_g,
            self.data[SoilSimulation.WATER_TRANSP]: t_flux_mean * kg_to_g,
            self.data[SoilSimulation.WATER_TOP_IN]: top_in * kg_to_g,
            self.data[SoilSimulation.WATER_BOTTOM]: bottom * kg_to_g,
        }
        for channel, value in diagnostics.items():
            channel.set(now, value)
        return {channel.id: value for channel, value in diagnostics.items()}

    def _face_weighted_mean(
        self,
        per_segment: dict[str, float],
        names: list[str],
    ) -> float:
        """Face-length-weighted mean over ``names``; missing entries count as 0."""
        total_len = 0.0
        weighted = 0.0
        for name in names:
            face_len = self._segment_face_len.get(name, 0.0)
            if face_len <= 0:
                continue
            total_len += face_len
            weighted += per_segment.get(name, 0.0) * face_len
        if total_len <= 0:
            return 0.0
        return weighted / total_len

    def _create_mesh(self) -> None:
        dl = self._mesh_config.dl
        width = self._mesh_config.width
        height = self._mesh_config.height
        plant_width = self._mesh_config.plant_width
        plant_height = self._mesh_config.plant_height
        watering_width = self._mesh_config.watering_width
        d_x = self._mesh_config.dx

        gmsh.initialize()
        gmsh.model.add("soil")

        # =============================
        # check parameters validity
        # width >= plant_width + 2 * d_x
        if width < plant_width + 2 * d_x:
            raise ValueError("Invalid parameters: width must be at least plant_width + 2 * d_x")

        # height > 0
        if height <= 0:
            raise ValueError("Invalid parameters: height must be positive")

        # height > plant_height
        if height <= plant_height:
            raise ValueError("Invalid parameters: height must be greater than plant_height")

        # the width - plant_width is a multiple of d_x
        if ((width - plant_width) / 2) % d_x != 0 and (width - plant_width) / (2 * d_x) > 0:
            raise ValueError("Invalid parameters: (width - plant_width) must be a multiple of 2 * d_x")
        # =============================

        surface_count = int((width - plant_width) / (2 * d_x))

        lines_tl = []
        lines_tr = []

        # =============================
        # Top left
        # =============================
        point_sim_tl = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, dl)
        point_prev = point_sim_tl
        offset = d_x
        for i in range(surface_count):
            point = gmsh.model.geo.addPoint(offset + d_x * i, 0.0, 0.0, dl)
            line = gmsh.model.geo.addLine(point_prev, point)
            lines_tl.append(line)
            gmsh.model.geo.synchronize()
            gmsh.model.setPhysicalName(1, gmsh.model.addPhysicalGroup(1, [line]), f"LeftTopSegment_{i}")
            point_prev = point

        # =============================
        # Plant
        # =============================
        point_plant_tl = point_prev
        point_watering_tl = gmsh.model.geo.addPoint(
            d_x * surface_count + plant_width / 2 - watering_width / 2, 0.0, 0.0, dl
        )
        point_watering_tr = gmsh.model.geo.addPoint(
            d_x * surface_count + plant_width / 2 + watering_width / 2, 0.0, 0.0, dl
        )
        point_plant_tr = gmsh.model.geo.addPoint(d_x * surface_count + plant_width, 0.0, 0.0, dl)
        point_plant_bl = gmsh.model.geo.addPoint(d_x * surface_count, -plant_height, 0.0, dl)
        point_plant_br = gmsh.model.geo.addPoint(d_x * surface_count + plant_width, -plant_height, 0.0, dl)

        line_plant_top_1 = gmsh.model.geo.addLine(point_plant_tl, point_watering_tl)
        line_plant_top_2 = gmsh.model.geo.addLine(point_watering_tl, point_watering_tr)
        line_plant_top_3 = gmsh.model.geo.addLine(point_watering_tr, point_plant_tr)
        line_plant_right = gmsh.model.geo.addLine(point_plant_tr, point_plant_br)
        line_plant_bottom = gmsh.model.geo.addLine(point_plant_br, point_plant_bl)
        line_plant_left = gmsh.model.geo.addLine(point_plant_bl, point_plant_tl)

        loop_plant = gmsh.model.geo.addCurveLoop(
            [line_plant_top_1, line_plant_top_2, line_plant_top_3, line_plant_right, line_plant_bottom, line_plant_left]
        )
        surface_plant = gmsh.model.geo.addPlaneSurface([loop_plant])
        gmsh.model.geo.synchronize()
        gmsh.model.setPhysicalName(1, gmsh.model.addPhysicalGroup(1, [line_plant_top_1]), "PlantTopLeftSegment")
        gmsh.model.setPhysicalName(1, gmsh.model.addPhysicalGroup(1, [line_plant_top_2]), "WateringTopSegment")
        gmsh.model.setPhysicalName(1, gmsh.model.addPhysicalGroup(1, [line_plant_top_3]), "PlantTopRightSegment")
        gmsh.model.setPhysicalName(2, gmsh.model.addPhysicalGroup(2, [surface_plant]), "PlantSurface")

        # =============================
        # Top right
        # =============================
        point_prev = point_plant_tr
        offset = d_x * surface_count + plant_width + d_x
        for i in range(surface_count):
            point = gmsh.model.geo.addPoint(offset + d_x * i, 0.0, 0.0, dl)
            line = gmsh.model.geo.addLine(point_prev, point)
            lines_tr.append(line)
            gmsh.model.geo.synchronize()
            gmsh.model.setPhysicalName(1, gmsh.model.addPhysicalGroup(1, [line]), f"RightTopSegment_{i}")
            point_prev = point
        upper_right_point = point_prev

        # =============================
        # Ground layer
        # =============================
        point_sim_bl = gmsh.model.geo.addPoint(0.0, -height, 0.0, dl)
        point_sim_br = gmsh.model.geo.addPoint(width, -height, 0.0, dl)

        line_sim_right = gmsh.model.geo.addLine(upper_right_point, point_sim_br)
        line_sim_bottom = gmsh.model.geo.addLine(point_sim_br, point_sim_bl)
        line_sim_left = gmsh.model.geo.addLine(point_sim_bl, point_sim_tl)

        loop_sim = gmsh.model.geo.addCurveLoop(
            [
                *lines_tl,
                -line_plant_left,
                -line_plant_bottom,
                -line_plant_right,
                *lines_tr,
                line_sim_right,
                line_sim_bottom,
                line_sim_left,
            ]
        )
        surface_sim = gmsh.model.geo.addPlaneSurface([loop_sim])
        gmsh.model.geo.synchronize()
        gmsh.model.setPhysicalName(1, gmsh.model.addPhysicalGroup(1, [line_sim_bottom]), "GroundBottomSegment")
        gmsh.model.setPhysicalName(2, gmsh.model.addPhysicalGroup(2, [surface_sim]), "GroundSurface")

        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(2)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)

        gmsh.write(self._mesh_config.filename)

    def _plot_mesh(self, save_path: Optional[str] = None):
        mesh = meshio.read(self._mesh_config.filename)

        points = mesh.points[:, :2]  # 2D coordinates

        lines = []
        line_colors = []
        triangles = []
        triangles_colors = []

        physical_tags = mesh.field_data
        physical_tags = sorted(physical_tags.keys(), key=lambda x: physical_tags[x][0])

        # Assuming 'physical' refers to a specific key in cell_data_dict
        # Check for physical tags in lines
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

        # -----------------------------
        # Extract triangles and physical tags
        # -----------------------------

        lines = np.array(lines)
        line_colors = np.array(line_colors)
        triangles = np.array(triangles)
        triangles_colors = np.array(triangles_colors)

        # -----------------------------
        # Plot
        # -----------------------------
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
            # Force the Agg-style headless save path so we don't depend on a
            # display being available — mesh dumps run during configure(),
            # well before the matplotlib GUI loop is up.
            fig.savefig(save_path, dpi=200)
            logging.info("%s: wrote mesh structure to %s", self.name, save_path)
        plt.close(fig)

    # TODO: Remove — only called by _solve_period (also dead). Superseded by
    #   _render_progress, which writes progress.png with the saturation field.
    def _plot_mesh_2(self):
        xi = np.linspace(min(self._mesh_fipy.cellCenters[0]), max(self._mesh_fipy.cellCenters[0]), 100)
        yi = np.linspace(min(self._mesh_fipy.cellCenters[1]), max(self._mesh_fipy.cellCenters[1]), 100)

        #
        # mapping the unstructured grid to a structured
        #
        x, y = self._mesh_fipy.cellCenters
        zi = griddata((x, y), self._variables["rel_sat"].value, (xi[None, :], yi[:, None]), method="cubic")

        jet = plt.get_cmap("jet")
        # using matplotlib for plotting
        #
        # clar plot
        plt.clf()
        plt.contour(xi, yi, zi, 15, linewidths=1.0, colors="k")
        plt.contourf(xi, yi, zi, 15, cmap=jet)
        plt.colorbar()

        # value max and min
        plt.clim(0, 1)

        plt.grid()
        plt.title("Relative Saturation")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.pause(0.01)

    def _capture_progress(
        self,
        sim_t: pd.Timestamp,
        rates: FluxRates,
        drainage: float,
    ) -> None:
        if (
            self._last_plot_simtime is not None
            and (sim_t - self._last_plot_simtime) < self._plot_config.interval
        ):
            return
        self._last_plot_simtime = sim_t

        watering_len = self._segment_face_len.get("WateringTopSegment", 0.0)
        irr_flux = rates.flow_m3s * RHO_W / watering_len if watering_len > 0 else 0.0
        e_flux = self._face_weighted_mean(rates.seg_evap, self._top_segment_names)
        t_flux = self._face_weighted_mean(rates.seg_transp, self._top_segment_names)

        self._plot_history.append({
            "timestamp": sim_t,
            "rain": rates.rain_flux,
            "irrigation": irr_flux,
            "evaporation": e_flux,
            "transpiration": t_flux,
            "drainage": drainage,
        })
        cutoff = sim_t - PLOT_HISTORY_WINDOW
        self._plot_history = [h for h in self._plot_history if h["timestamp"] >= cutoff]
        self._render_progress(sim_t)

    def _init_progress_figure(self) -> None:
        on_main_thread = threading.current_thread() is threading.main_thread()
        if self._plot_config.show and not on_main_thread:
            logging.warning(
                "%s: progress plot 'show' disabled — solver runs on a worker thread "
                "(matplotlib GUI requires the main thread). Use 'live = true' and "
                "open progress.html in a browser for a live view.",
                self.name,
            )
            self._plot_config.show = False
            if matplotlib.get_backend().lower() not in ("agg", "module://matplotlib_inline.backend_inline"):
                matplotlib.use("Agg", force=True)

        if self._plot_config.show:
            plt.ion()
        fig, axes = plt.subplots(
            2, 1,
            figsize=(8, 6),
            dpi=120,
            gridspec_kw={"height_ratios": [3, 2]},
        )
        sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(vmin=0.0, vmax=1.0))
        sm.set_array([])
        fig.colorbar(sm, ax=axes[0], shrink=0.8, label=r"$S_e$")
        self._plot_fig = fig
        self._plot_axes = axes

        if self._plot_config.live:
            self._write_progress_html()
            logging.info(
                "%s: live progress at file://%s/progress.html (refreshes every 2s)",
                self.name, self._plot_config.dir,
            )

    def _write_progress_html(self) -> None:
        html = (
            "<!DOCTYPE html>\n<html><head><meta charset='utf-8'>"
            "<title>Soil simulation progress</title><style>"
            "body{background:#111;margin:0;padding:1em;font-family:sans-serif;color:#ccc;}"
            "img{max-width:100%;height:auto;display:block;margin:0 auto;}"
            ".meta{text-align:center;padding:0.5em;font-size:0.9em;}"
            "</style></head><body>"
            "<div class='meta'>auto-refresh every 2s — last reload: <span id='t'></span></div>"
            "<img id='plot' src='progress.png'>"
            "<script>"
            "function r(){const i=document.getElementById('plot');"
            "i.src='progress.png?t='+Date.now();"
            "document.getElementById('t').textContent=new Date().toLocaleTimeString();}"
            "setInterval(r,2000);r();"
            "</script></body></html>\n"
        )
        path = os.path.join(self._plot_config.dir, "progress.html")
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)

    def _render_progress(self, sim_t: pd.Timestamp) -> None:
        if self._plot_fig is None:
            self._init_progress_figure()
        ax_sat, ax_ts = self._plot_axes

        x, y = self._mesh_fipy.cellCenters
        xi = np.linspace(np.min(x), np.max(x), 100)
        yi = np.linspace(np.min(y), np.max(y), 100)
        zi = griddata(
            (np.asarray(x), np.asarray(y)),
            self._variables["rel_sat"].value,
            (xi[None, :], yi[:, None]),
            method="cubic",
        )

        ax_sat.clear()
        ax_sat.contourf(xi, yi, zi, levels=15, cmap="jet", vmin=0.0, vmax=1.0)
        ax_sat.contour(xi, yi, zi, levels=15, linewidths=0.5, colors="k")
        ax_sat.set_aspect("equal", adjustable="box")
        ax_sat.set_title(f"Relative Saturation @ {sim_t.isoformat()}")
        ax_sat.set_xlabel("x (m)")
        ax_sat.set_ylabel("y (m)")

        ax_ts.clear()
        if self._plot_history:
            df = pd.DataFrame(self._plot_history).set_index("timestamp")
            ax_ts.plot(df.index, df["rain"], label="rain in", color="tab:cyan")
            ax_ts.plot(df.index, df["irrigation"], label="irrigation in", color="tab:blue")
            ax_ts.plot(df.index, df["evaporation"], label="evaporation out", color="tab:orange")
            ax_ts.plot(df.index, df["transpiration"], label="transpiration", color="tab:green")
            ax_ts.plot(df.index, df["drainage"], label="drainage out", color="tab:red")
            ax_ts.set_ylabel(r"kg/(m$^2\cdot$s)")
            ax_ts.legend(loc="upper right", fontsize=7)
            ax_ts.grid(True, alpha=0.3)
            for label in ax_ts.get_xticklabels():
                label.set_rotation(20)
                label.set_ha("right")

        self._plot_fig.tight_layout()

        if self._plot_config.show:
            try:
                self._plot_fig.canvas.draw_idle()
                plt.pause(0.001)
            except Exception:  # noqa: BLE001
                pass

        # Render the figure to PNG once, then fan out to all sinks (channel
        # for database persistence, live overwrite, archived per-frame file).
        buf = io.BytesIO()
        self._plot_fig.savefig(buf, dpi=120, format="png")
        png_bytes = buf.getvalue()

        self.data[SoilSimulation.SOIL_PROGRESS_IMAGE].set(sim_t, png_bytes)

        # Atomic-ish overwrite via tmp + replace so the browser never grabs a
        # half-written PNG.
        if self._plot_config.live:
            target = os.path.join(self._plot_config.dir, "progress.png")
            tmp = target + ".tmp"
            with open(tmp, "wb") as f:
                f.write(png_bytes)
            os.replace(tmp, target)

        if self._plot_config.save:
            fname = sim_t.strftime("%Y%m%dT%H%M%S") + ".png"
            with open(os.path.join(self._plot_config.dir, fname), "wb") as f:
                f.write(png_bytes)

    def _build_eq(self):
        mesh = self._mesh_fipy

        rel_sat = CellVariable(mesh=mesh, name="relative saturation", hasOld=True)
        g_faces = FaceVariable(mesh=mesh, name="gravity faces", value=(0, 1.0))
        source = CellVariable(mesh=mesh, name="source", value=0.0)
        self._variables = {
            "rel_sat": rel_sat,
            "g_faces": g_faces,
            "source": source,
        }

        kf = self._soil_model.k_from_se(rel_sat)
        psi = self._soil_model.psi_from_se(rel_sat)
        d_psi = self._soil_model.dpsi_dse(rel_sat)

        # Richards' equation in Se form, expressed against Se directly:
        #   (θs-θr) ∂Se/∂t = ∇·[K · D(Se) ∇Se] + ∂K/∂y + source
        # where D(Se) = d|ψ|/dSe. Previously the diffusion coefficient was
        # K·ψ (missing the chain rule) and gravity was a VanLeer convection
        # on Se that leaked mass out the top boundary; both replaced.
        gravity_flux = g_faces * kf.faceValue          # K·ẑ at faces (ẑ up)
        gravity_div = gravity_flux.divergence          # cell-centered ∂K/∂y
        richards = TransientTerm(coeff=self._ode_config.theta_s - self._ode_config.theta_r) == (
            DiffusionTerm(coeff=(kf * d_psi))
            + gravity_div
            + source
        )
        self._equations = {
            "kf": kf,
            "psi": psi,
            "d_psi": d_psi,
            "gravity_div": gravity_div,
            "richards": richards,
        }

    def _constrain_eq(self):
        # Top-boundary water exchange (evap/irrigation) is applied as a source
        # term in _update_source. Only the bottom drainage is held by a
        # Dirichlet constraint here.
        mesh = self._mesh_fipy
        rel_sat = self._variables["rel_sat"]

        rel_sat.setValue(0.35, where=mesh.physicalCells["GroundSurface"])
        rel_sat.setValue(0.35, where=mesh.physicalCells["PlantSurface"])
        rel_sat.constrain(0.35, where=mesh.physicalFaces["GroundBottomSegment"])
        rel_sat.updateOld()

    def _build_segment_index(self) -> None:
        mesh = self._mesh_fipy
        width = self._mesh_config.width
        plant_width = self._mesh_config.plant_width
        dx = self._mesh_config.dx
        n_pv_segments = int((width - plant_width) / (2 * dx))

        # Bare-soil top strips on the left and right of the plant block.
        # The PV roof covers the plant zone (plant tops + watering strip);
        # these flanking strips are open to the sky and rain falls here.
        self._open_sky_segment_names = [
            f"{side}TopSegment_{i}"
            for i in range(n_pv_segments)
            for side in ("Left", "Right")
        ]
        # Top segments where soil evaporation is applied (open-sky strips +
        # plant top edges). The watering strip is excluded — drip irrigation
        # there governs the surface state. PV-driven shading is now baked
        # into the per-segment ET upstream (see ``Evapotranspiration`` and
        # ``FieldSimulation._build_segments``), not handled here.
        self._top_segment_names = [
            *self._open_sky_segment_names,
            "PlantTopLeftSegment",
            "PlantTopRightSegment",
        ]

        face_areas = np.asarray(mesh._faceAreas)
        cell_volumes = np.asarray(mesh.cellVolumes)

        self._segment_cells = {}
        self._segment_face_len = {}
        self._segment_cell_volume = {}
        for name in [*self._top_segment_names, "WateringTopSegment", "GroundBottomSegment"]:
            face_mask = mesh.physicalFaces[name]
            cell_ids = mesh.faceCellIDs[:, face_mask]
            cell_ids = np.unique(np.asarray(cell_ids[cell_ids >= 0]).ravel())
            self._segment_cells[name] = cell_ids
            self._segment_face_len[name] = float(face_areas[face_mask].sum())
            self._segment_cell_volume[name] = float(cell_volumes[cell_ids].sum())

        plant_mask = np.asarray(mesh.physicalCells["PlantSurface"], dtype=bool)
        self._plant_cells = np.where(plant_mask)[0]
        self._plant_volume = float(cell_volumes[self._plant_cells].sum())

        # Pre-compute θ-rate conversion factors. theta_rate = flux * factor.
        self._theta_diff = self._ode_config.theta_s - self._ode_config.theta_r
        watering_vol = self._segment_cell_volume.get("WateringTopSegment", 0.0)
        self._irrigation_factor = 1.0 / watering_vol if watering_vol > 0 else 0.0

        # Open-sky face length for the integral mass balance in
        # ``_balance_drainage_flux`` (rain only falls on these strips).
        self._rain_face_len = sum(
            self._segment_face_len[n] for n in self._open_sky_segment_names
        )

    def _solve(self, dt: float):
        eq = self._equations["richards"]
        rel_sat = self._variables["rel_sat"]
        res = 1e6
        for _ in range(10):
            res = eq.sweep(dt=dt, var=rel_sat)
            if res <= 0.5:
                break
        rel_sat.updateOld()
