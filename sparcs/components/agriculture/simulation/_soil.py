# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.simulation._soil
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Shared base for the soil simulation chain. Holds:

* the mesh / PDE / plot config dataclasses (``MeshConfig``,
  ``PDEConfig``, ``PondingConfig``, ``FeddesConfig``, ``PlotConfig``);
* the FiPy mesh + Richards-equation core (``SoilPDECore``);
* the per-window diagnostic math (``_compute_diagnostics`` etc.) shared
  via the :class:`SoilBase` parent component;
* the mesh-generation helpers (``create_mesh`` / ``ensure_mesh``) and
  probe parsing (``resolve_probes`` / ``ProbeSpec``).

Both :class:`SoilSimulation` (live solver) and :class:`SoilPredictor`
(forecast roll-outs) inherit from :class:`SoilBase` here.
"""

from __future__ import annotations

import io
import logging
import os
import sys
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

from . import plot_style
from lories import Component, Constant
from lories.components.weather import Weather
from lories.typing import Configurations
from lories.util import to_timedelta
from sparcs.components.agriculture.soil import (
    DEFAULT_SOIL_MODEL,
    SoilModel,
    create_soil_model,
)

logging.getLogger("fipy").setLevel(logging.WARNING)
np.seterr(all="ignore")

RHO_W: float = 1000.0       # kg/m³
SE_MIN: float = 1e-6        # effective-saturation floor for source clipping
SE_MAX: float = 0.999       # effective-saturation ceiling for source clipping


@dataclass
class SolveResult:
    """One Picard sweep loop's outcome.

    ``converged`` is the physical criterion: ``max|Δθ_per_sweep| ≤ tol_th``.
    ``residual`` is the FiPy linear-system residual at the final sweep
    (kept for logging — not a direct convergence criterion). ``sweeps``
    is how many `eq.sweep` calls were performed. The wall-clock walk in
    :meth:`SoilSimulation.advance` uses these to decide whether to halve
    ``dt`` and retry the current substep.
    """

    residual: float
    converged: bool
    sweeps: int


@dataclass
class ClipDiagnostics:
    """Mass that the per-step ``[SE_MIN, SE_MAX]`` clipper *would* have
    moved but couldn't, for a single ``apply_source`` call.

    Values are in **kg per metre of out-of-plane row depth**, integrated
    over the substep's ``dt``. Two sides:

    - ``top_rejected``: positive θ-rate that got clipped down because a
      cell was at or near saturation. Physically this is **runoff** /
      rejected infiltration — water we tried to add but the top cell
      couldn't hold.
    - ``bottom_rejected``: negative θ-rate that got clipped up because
      a cell was at or near the residual floor. Physically this is
      **unmet evaporative or root-uptake demand** — the soil was too
      dry to satisfy the requested sink.

    Callers accumulate these across the per-substep loop and convert
    to per-area-per-hour rates for the diagnostic channels.
    """

    top_rejected: float = 0.0
    bottom_rejected: float = 0.0
    # Deliberate overflow when the ponding bucket fills past
    # ``PondingConfig.h_max_mm``. With ponding disabled this stays 0
    # and runoff still surfaces via ``top_rejected``; with ponding
    # enabled the clipper top side should be ~0 (the bucket absorbs
    # the excess first) and ``ponding_overflow`` carries the runoff.
    ponding_overflow: float = 0.0

    def add(self, other: "ClipDiagnostics") -> None:
        self.top_rejected += other.top_rejected
        self.bottom_rejected += other.bottom_rejected
        self.ponding_overflow += other.ponding_overflow


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


def resolve_probes(
    probes_cfg: Configurations,
    mesh_fipy: Gmsh2D,
    mesh_config: "MeshConfig",
    log_name: Optional[str] = None,
) -> list[ProbeSpec]:
    """Resolve ``[probes.points.<name>]`` / ``[probes.areas.<name>]`` blocks
    against a FiPy mesh into a list of ``ProbeSpec`` sampling recipes.

    Caller is responsible for channel registration. Shared by ``SoilSimulation``
    (registers ``<key>`` probe channels) and ``SoilPredictor`` (registers
    ``predict_<key>`` channels) — both end up with identical cell-index /
    weight recipes because they reuse the same .msh file, so a single
    parser keeps them in lock-step.

    Coordinates: ``x`` bay-centered, ``y`` positive depth in metres (see
    ``soil_simulation.conf`` header for the full convention).
    """
    probes: list[ProbeSpec] = []
    cell_centers = np.asarray(mesh_fipy.cellCenters)
    cell_x, cell_y = cell_centers[0], cell_centers[1]
    cell_volumes = np.asarray(mesh_fipy.cellVolumes)
    x_offset = mesh_config.width / 2.0

    if probes_cfg.has_member("points"):
        for key, spec in probes_cfg.get_member("points").items():
            x = float(spec["x"])
            y = float(spec["y"])
            idx = int(np.argmin((cell_x - (x + x_offset)) ** 2 + (cell_y - (-y)) ** 2))
            probes.append(ProbeSpec(
                name=f"Probe point (x={x:.3f}, depth={y:.3f})",
                channel_id=key,
                cell_indices=np.array([idx], dtype=int),
                weights=np.array([1.0]),
            ))

    if probes_cfg.has_member("areas"):
        for key, spec in probes_cfg.get_member("areas").items():
            x_min = float(spec["x_min"])
            x_max = float(spec["x_max"])
            y_min = float(spec["y_min"])
            y_max = float(spec["y_max"])
            mask = (
                (cell_x >= x_min + x_offset) & (cell_x <= x_max + x_offset)
                & (cell_y >= -y_max) & (cell_y <= -y_min)
            )
            indices = np.flatnonzero(mask)
            if indices.size == 0:
                if log_name:
                    logging.warning(
                        "%s: probe area '%s' (x=[%.3f,%.3f], depth=[%.3f,%.3f]) "
                        "covers no mesh cells — skipping.",
                        log_name, key, x_min, x_max, y_min, y_max,
                    )
                continue
            probes.append(ProbeSpec(
                name=(
                    f"Probe area x=[{x_min:.3f},{x_max:.3f}] "
                    f"depth=[{y_min:.3f},{y_max:.3f}], {indices.size} cells"
                ),
                channel_id=key,
                cell_indices=indices,
                weights=cell_volumes[indices].copy(),
            ))
    return probes


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
class FeddesConfig:
    """Feddes (1978) piecewise-linear root-water-uptake stress reduction.

    Four pF thresholds bracket a stress curve α(h) ∈ [0, 1] on the
    canopy root zone:

    ::

        |h| < |P0|              α = 0    (anaerobic stress; too wet)
        |P0| ≤ |h| < |P1|       α : 0 → 1 (anaerobic ramp; less stress as drier)
        |P1| ≤ |h| ≤ |P2|       α = 1    (optimal range)
        |P2| <  |h| < |P3|      α : 1 → 0 (dry ramp; more stress as drier)
        |h| ≥ |P3|              α = 0    (wilting point)

    pF is ``log10(|h|)`` with ``|h|`` in cm of water column. Larger pF
    means drier soil. Defaults match the commonly cited HYDRUS-1D
    "field crop" template (pF 1 / 3 / 4.2). Set ``anaerobic = True`` to
    enable the wet-side P0/P1 ramp; off by default because many crops
    tolerate brief waterlogging and the wet-side numbers are highly
    crop-specific.

    Translated to Se at construction time (SoilPDECore precomputes the
    threshold saturations from the configured retention model), so the
    runtime hot path just does a piecewise-linear interpolation on Se
    per plant cell.
    """

    enabled: bool = False
    anaerobic: bool = False
    p0_pf: float = 0.0    # anaerobic upper limit (only used when anaerobic=True)
    p1_pf: float = 1.0    # optimal lower (drier side of the anaerobic ramp)
    p2_pf: float = 3.0    # optimal upper (wetter side of the dry ramp)
    p3_pf: float = 4.2    # wilting point

    # Root distribution β(z) on the plant cells. ``"uniform"`` weights
    # every plant cell by its volume; ``"linear"`` decays linearly from
    # the surface (max at top, zero at ``plant_height``); ``"exponential"``
    # decays as ``exp(−z / root_decay_length)`` where z is depth from the
    # surface in metres. β is normalised so ``Σ β · cell_vol = 1`` —
    # downstream code is invariant to the unnormalised shape.
    root_distribution: str = "uniform"
    root_decay_length: float = 0.3   # m, only used for "exponential"

    # Šimůnek compensation threshold. Without compensation
    # (``omega_c = 1.0``, the default), realised transpiration falls to
    # ``ω · T_pot`` whenever some plant cells are stressed (``ω = Σ α·β``).
    # With ``omega_c < 1``, the demand is redistributed to less-stressed
    # cells so total uptake recovers to ``T_pot`` as long as ``ω ≥ ω_c``;
    # below ``ω_c`` we still can't fully meet demand and total uptake is
    # ``T_pot · ω / ω_c``. HYDRUS-1D's typical value is ``0.5``.
    omega_c: float = 1.0

    def __init__(self, configs: Optional[Configurations] = None):
        if configs is None:
            self.enabled = False
            self.anaerobic = False
            self.p0_pf = 0.0
            self.p1_pf = 1.0
            self.p2_pf = 3.0
            self.p3_pf = 4.2
            self.root_distribution = "uniform"
            self.root_decay_length = 0.3
            self.omega_c = 1.0
            return
        self.enabled = configs.get_bool("enabled", default=False)
        self.anaerobic = configs.get_bool("anaerobic", default=False)
        self.p0_pf = float(configs.get("p0_pf", default=0.0))
        self.p1_pf = float(configs.get("p1_pf", default=1.0))
        self.p2_pf = float(configs.get("p2_pf", default=3.0))
        self.p3_pf = float(configs.get("p3_pf", default=4.2))
        self.root_distribution = str(
            configs.get("root_distribution", default="uniform")
        ).strip().lower()
        self.root_decay_length = float(
            configs.get("root_decay_length", default=0.3)
        )
        self.omega_c = float(configs.get("omega_c", default=1.0))


def _alpha_feddes_per_cell(
    se: np.ndarray,
    *,
    se_p2: float,
    se_p3: float,
    se_p0: Optional[float] = None,
    se_p1: Optional[float] = None,
) -> np.ndarray:
    """Piecewise-linear α factor per cell from Se thresholds.

    Drier ⇒ smaller Se, so Se thresholds are ordered ``se_p0 ≥ se_p1 ≥
    se_p2 ≥ se_p3``. When ``se_p0`` / ``se_p1`` are None, the anaerobic
    branch is disabled and α stays at 1 for any Se ≥ se_p2.
    """
    se = np.asarray(se, dtype=float)
    alpha = np.zeros_like(se)
    anaerobic = se_p0 is not None and se_p1 is not None

    # Optimal plateau.
    if anaerobic:
        plateau = (se < se_p1) & (se >= se_p2)
    else:
        plateau = se >= se_p2
    alpha[plateau] = 1.0

    # Anaerobic ramp (wet side, α: 0 → 1 as soil dries from saturation toward P1).
    if anaerobic and se_p0 > se_p1:
        anaerobic_ramp = (se < se_p0) & (se >= se_p1)
        alpha[anaerobic_ramp] = (se_p0 - se[anaerobic_ramp]) / (se_p0 - se_p1)

    # Dry ramp (α: 1 → 0 as soil dries from P2 to P3 / wilting).
    if se_p2 > se_p3:
        dry_ramp = (se < se_p2) & (se > se_p3)
        alpha[dry_ramp] = (se[dry_ramp] - se_p3) / (se_p2 - se_p3)

    return alpha


@dataclass
class PondingConfig:
    """Per-top-segment surface-ponding state knobs.

    When enabled, incoming rain accumulates in a per-segment "bucket"
    (``surface_h`` in metres of water column). Each substep the bucket
    drains into the underlying soil up to that segment's infiltration
    capacity (set by SE_MAX headroom on the adjacent cells); any
    ponding above ``h_max_mm`` overflows immediately as runoff. When
    disabled (the default), rain is applied straight to the soil
    cells as before, and the clipper-rejected mass on the top side
    surfaces as runoff via ``ClipDiagnostics.top_rejected``.

    Scope (v1): only rain on the open-sky segments uses the ponding
    bucket. Irrigation drip on ``WateringTopSegment`` keeps its direct
    volumetric path because the flow rate is operator-controlled and
    typically well below saturation; revisit if drip + storm events
    co-occur and the watering strip starts to pond.
    """

    enabled: bool = False
    h_max_mm: float = 5.0   # max ponding depth before overflow [mm]

    def __init__(self, configs: Optional[Configurations] = None):
        if configs is None:
            self.enabled = False
            self.h_max_mm = 5.0
            return
        self.enabled = configs.get_bool("enabled", default=False)
        self.h_max_mm = float(configs.get("h_max_mm", default=5.0))


@dataclass
class PDEConfig:
    def __init__(self, configs: Configurations):
        # Retention/conductivity model selector. ``"van_genuchten"`` (alias
        # ``"vg"``) is the historical default; ``"brooks_corey"`` (``"bc"``)
        # is also available. Extra model-specific parameters (e.g. Mualem
        # ``bpar``) are read from the same block and filtered by the
        # ``create_soil_model`` factory against the chosen class's signature.
        self.model: str = str(configs.get("model", default=DEFAULT_SOIL_MODEL))
        self.theta_r: float = configs.get("theta_r", default=0.05)
        self.theta_s: float = configs.get("theta_s", default=0.43)
        self.alpha: float = configs.get("alpha", default=0.08)
        self.n: float = configs.get("n", default=1.6)
        self.k_s: float = configs.get("k_s", default=1.0e-4)
        # Mualem pore-interaction exponent (L in Mualem 1976, ``BPar`` in
        # HYDRUS-1D). Class defaults differ (0.5 for VG, 2.0 for BC); we
        # only forward this when the user sets it explicitly so each
        # model keeps its own canonical default.
        self.bpar: Optional[float] = (
            float(configs.get("bpar")) if "bpar" in configs else None
        )
        # PDE timestep as a lories freq string ("50s", "1min", "5min").
        # ``dt`` is the *target* substep size; the wall-clock walk in
        # ``SoilSimulation.advance`` can halve it when the Picard sweep
        # loop fails to converge, down to ``dt_min``. Set ``dt_min`` to
        # the same value as ``dt`` to disable adaptation entirely.
        self.dt: float = to_timedelta(configs.get("dt", default="50s")).total_seconds()
        self.dt_min: float = to_timedelta(configs.get("dt_min", default="1s")).total_seconds()

        # Initial condition. Two modes:
        # - **Uniform** (default): every cell starts at ``ic_se`` (effective
        #   saturation). 0.35 is a generic "moderate moisture" value
        #   matching the historical hard-coded IC.
        # - **Hydrostatic equilibrium**: when ``ic_water_table_depth`` is
        #   set (metres below the soil surface, positive), the IC is the
        #   gravity-matric balance Se(z) profile derived from the
        #   retention curve — fully saturated at and below the water
        #   table, decreasing upward as |h(z)| = (z above WT). This is the
        #   physically meaningful IC HYDRUS uses by default.
        self.ic_se: float = float(configs.get("ic_se", default=0.35))
        self.ic_water_table_depth: Optional[float] = (
            float(configs.get("ic_water_table_depth"))
            if "ic_water_table_depth" in configs
            else None
        )

        # Wall-clock-equivalent duration the cold-start spin-up integrates
        # before the first real callback. 3h gives a flat constant IC time
        # to relax under the current weather so the first reported flux
        # isn't dominated by the initial transient. With a hydrostatic IC
        # the equilibrium itself is the relaxed state, so spin-up would
        # only re-perturb it — we default ``cold_start`` to zero in that
        # case (unless the user has explicitly set it).
        if "cold_start" in configs:
            self.cold_start: pd.Timedelta = to_timedelta(configs.get("cold_start"))
        elif self.ic_water_table_depth is not None:
            self.cold_start = pd.Timedelta(0)
        else:
            self.cold_start = to_timedelta("3h")
        # Feddes (1978) root-water-uptake stress reduction. Off by default
        # to preserve historical behaviour; enable via ``[pde.feddes] enabled
        # = true``. When enabled, plant cells with Se below the dry ramp
        # contribute less to transpiration, and below the wilting point
        # they contribute nothing — matching HYDRUS-1D's SINK.FOR α(h).
        feddes_block: Optional[Configurations] = None
        if hasattr(configs, "has_member") and configs.has_member("feddes"):
            feddes_block = configs.get_member("feddes", defaults={})
        self.feddes: FeddesConfig = FeddesConfig(feddes_block)

        # Surface ponding state. Off by default — preserves the
        # historical "rain straight onto cells" path; opt in via
        # ``[pde.ponding] enabled = true`` to track the bucket.
        ponding_block: Optional[Configurations] = None
        if hasattr(configs, "has_member") and configs.has_member("ponding"):
            ponding_block = configs.get_member("ponding", defaults={})
        self.ponding: PondingConfig = PondingConfig(ponding_block)

    def build_model(self) -> SoilModel:
        """Instantiate the configured :class:`SoilModel` via the factory."""
        kwargs: dict[str, Any] = {
            "theta_r": self.theta_r,
            "theta_s": self.theta_s,
            "alpha": self.alpha,
            "n": self.n,
            "k_s": self.k_s,
        }
        if self.bpar is not None:
            kwargs["bpar"] = self.bpar
        return create_soil_model(self.model, **kwargs)


@dataclass
class ProbeSpec:
    """Resolved sampling recipe for one probe (point or area).

    ``cell_indices`` selects rows of the FiPy ``rel_sat`` cell array;
    ``weights`` is broadcastable against them — length-1 ``[1.0]`` for a
    point (nearest-cell pick), per-cell volumes for an area (yields a
    volume-weighted mean over the box).
    """

    name: str
    channel_id: str
    cell_indices: np.ndarray
    weights: np.ndarray


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


class SoilPDECore:
    """Shared FiPy / Richards-equation core used by both
    :class:`SoilSimulation` (live solver) and
    :class:`SoilPredictor` (forecast roll-outs).

    Owns the mesh, the assembled Richards PDE, the segment index
    bookkeeping, and the pure integration primitives (apply_source,
    solve, sample, total_water, state I/O). Channels, plotting, weather
    parsing, and mass-balance accounting live on the calling Component.

    Constructed once both ``MeshConfig`` and ``PDEConfig`` are available
    and the .msh file exists on disk.
    """

    soil_model: SoilModel

    # FiPy state — exposed for the few places (plotting, state-blob I/O)
    # that need raw access; most callers go through the methods below.
    mesh: Gmsh2D
    rel_sat: CellVariable
    source_var: CellVariable
    richards: Any

    # Segment index — populated by ``_build_segment_index``.
    segment_cells: dict[str, np.ndarray]
    segment_face_len: dict[str, float]
    segment_cell_volume: dict[str, float]
    top_segment_names: list[str]
    open_sky_segment_names: list[str]
    plant_cells: np.ndarray
    plant_volume: float

    # Pre-computed conversion factors.
    theta_diff: float                # θ_s - θ_r
    irrigation_factor: float         # 1 / vol_watering          [1/m^3]
    rain_face_len: float             # Σ face_len over open-sky  [m]

    # Feddes Se-threshold cache (only populated when ode_config.feddes.enabled).
    _feddes_se_p2: Optional[float]
    _feddes_se_p3: Optional[float]
    _feddes_se_p0: Optional[float]
    _feddes_se_p1: Optional[float]

    # Normalised root density β̂(z) on plant cells (Σ β̂_i · V_i = 1, units
    # 1/m² for the 2-D mesh). Always populated; defaults to a uniform
    # weighting when the user doesn't pick a distribution shape.
    _root_beta_normalized: np.ndarray
    _root_cell_volumes: np.ndarray

    def __init__(
        self,
        mesh_config: MeshConfig,
        ode_config: PDEConfig,
        *,
        rel_sat_name: str = "relative saturation",
    ) -> None:
        self.mesh_config = mesh_config
        self.ode_config = ode_config
        self.soil_model = ode_config.build_model()
        self.mesh = Gmsh2D(mesh_config.filename, communicator=serialComm)
        self._build_eq(rel_sat_name)
        self._build_segment_index()
        self._build_feddes_thresholds()
        self._build_root_beta()
        # Ponding bucket — one entry per open-sky segment, in metres of
        # water column. Initialised dry; persists across substeps via the
        # state-blob round-trip. Only used when ode_config.ponding.enabled.
        self.surface_h: dict[str, float] = {
            name: 0.0 for name in self.open_sky_segment_names
        }

    # -- PDE assembly ----------------------------------------------------------

    def _build_eq(self, rel_sat_name: str) -> None:
        mesh = self.mesh
        rel_sat = CellVariable(mesh=mesh, name=rel_sat_name, hasOld=True)
        g_faces = FaceVariable(mesh=mesh, name="gravity faces", value=(0, 1.0))
        source = CellVariable(mesh=mesh, name="source", value=0.0)

        kf = self.soil_model.k_from_se(rel_sat)
        d_h = self.soil_model.dh_dse(rel_sat)

        # Richards' equation in Se form (see SOIL.md §1).
        #   (θs-θr) ∂Se/∂t = ∇·[K · |dh/dSe| ∇Se] + ∂K/∂y + source
        # K is in m/s and |dh/dSe| in m/Se, so K·|dh/dSe| lands in m²/s
        # — the right units on a metre-scaled mesh. Free drainage at the
        # bottom emerges naturally: the DiffusionTerm picks up FiPy's
        # default zero-gradient Neumann (no matric gradient) and the
        # gravity divergence term carries water out through the bottom.
        gravity_flux = g_faces * kf.faceValue
        gravity_div = gravity_flux.divergence
        richards = TransientTerm(
            coeff=self.ode_config.theta_s - self.ode_config.theta_r
        ) == (
            DiffusionTerm(coeff=(kf * d_h)) + gravity_div + source
        )

        # Initial condition. Hydrostatic equilibrium when a water-table
        # depth is configured (zero net flux at t=0), uniform Se(=ic_se)
        # otherwise.
        ic_wt = self.ode_config.ic_water_table_depth
        if ic_wt is not None:
            rel_sat.setValue(self._hydrostatic_ic_array(ic_wt))
            logging.info(
                "SoilPDECore: hydrostatic IC (water table at %.2f m below surface) "
                "Se min=%.3f max=%.3f",
                ic_wt,
                float(np.min(rel_sat.value)),
                float(np.max(rel_sat.value)),
            )
        else:
            rel_sat.setValue(float(self.ode_config.ic_se))
        rel_sat.updateOld()

        self.rel_sat = rel_sat
        self.source_var = source
        self._g_faces = g_faces
        self.richards = richards
        self._kf = kf  # kept for diagnostics / future use

    def _hydrostatic_ic_array(self, water_table_depth_m: float) -> np.ndarray:
        """Build a hydrostatic-equilibrium Se array for the current mesh.

        Mesh y is positive upward with the soil surface at ``y = 0`` and
        the bottom at ``y = -height``. A water table at depth ``h_wt``
        below the surface sits at ``y_wt = −h_wt``.

        At and below the water table the soil is saturated (``ψ = 0`` →
        ``Se = 1``). Above, gravity balances matric suction so the
        pressure head magnitude equals the elevation above the water
        table: ``|h(z)|_m = max(0, y_cell − y_wt)``. Converting to hPa
        (``1 m water ≈ 98.0665 hPa``) and feeding through
        :meth:`SoilModel.se_from_psi` yields the corresponding Se field,
        clipped to ``[SE_MIN, SE_MAX]`` for safety.
        """
        y_centers = np.asarray(self.mesh.cellCenters[1], dtype=float)
        y_wt = -float(water_table_depth_m)
        h_above_wt_m = np.maximum(y_centers - y_wt, 0.0)
        psi_hpa_signed = -h_above_wt_m * 100.0 * 0.980665
        se = np.asarray(self.soil_model.se_from_psi(psi_hpa_signed), dtype=float)
        return np.clip(se, SE_MIN, SE_MAX)

    def _build_segment_index(self) -> None:
        mesh = self.mesh
        names = top_segment_names_from_mesh(self.mesh_config)
        # Bare-soil strips on the left/right of the plant block. The PV
        # roof covers the plant zone; rain reaches these strips only.
        self.open_sky_segment_names = [
            n for n in names
            if n not in ("PlantTopLeftSegment", "PlantTopRightSegment")
        ]
        # Every top segment where soil evaporation acts (open-sky strips
        # + plant top edges). The watering strip is governed by the
        # irrigation source and excluded here.
        self.top_segment_names = list(names)

        face_areas = np.asarray(mesh._faceAreas)
        cell_volumes = np.asarray(mesh.cellVolumes)
        self.segment_cells = {}
        self.segment_face_len = {}
        self.segment_cell_volume = {}
        for seg_name in [*names, "WateringTopSegment", "GroundBottomSegment"]:
            face_mask = mesh.physicalFaces[seg_name]
            cell_ids = mesh.faceCellIDs[:, face_mask]
            cell_ids = np.unique(np.asarray(cell_ids[cell_ids >= 0]).ravel())
            self.segment_cells[seg_name] = cell_ids
            self.segment_face_len[seg_name] = float(face_areas[face_mask].sum())
            self.segment_cell_volume[seg_name] = float(cell_volumes[cell_ids].sum())

        plant_mask = np.asarray(mesh.physicalCells["PlantSurface"], dtype=bool)
        self.plant_cells = np.where(plant_mask)[0]
        self.plant_volume = float(cell_volumes[self.plant_cells].sum())

        self.theta_diff = self.ode_config.theta_s - self.ode_config.theta_r
        watering_vol = self.segment_cell_volume.get("WateringTopSegment", 0.0)
        self.irrigation_factor = 1.0 / watering_vol if watering_vol > 0 else 0.0
        self.rain_face_len = sum(
            self.segment_face_len[n] for n in self.open_sky_segment_names
        )

    def _build_feddes_thresholds(self) -> None:
        """Translate the Feddes pF thresholds into effective-saturation
        thresholds using the configured retention curve. Done once at
        build time so the runtime path stays cheap; if the user disables
        Feddes (the default), all four are left at ``None`` and
        :meth:`apply_source` short-circuits."""
        f = self.ode_config.feddes
        if not f.enabled:
            self._feddes_se_p2 = None
            self._feddes_se_p3 = None
            self._feddes_se_p0 = None
            self._feddes_se_p1 = None
            return

        def se_from_pf(pf: float) -> float:
            psi = self.soil_model.psi_from_pf(pf)
            return float(np.clip(self.soil_model.se_from_psi(psi), 0.0, 1.0))

        self._feddes_se_p2 = se_from_pf(f.p2_pf)
        self._feddes_se_p3 = se_from_pf(f.p3_pf)
        if f.anaerobic:
            self._feddes_se_p0 = se_from_pf(f.p0_pf)
            self._feddes_se_p1 = se_from_pf(f.p1_pf)
        else:
            self._feddes_se_p0 = None
            self._feddes_se_p1 = None

        if not (self._feddes_se_p2 > self._feddes_se_p3):
            logging.warning(
                "Feddes pF thresholds map to non-monotone Se (P2 → Se=%.3f, "
                "P3 → Se=%.3f). Check pF ordering — dry ramp will be disabled.",
                self._feddes_se_p2, self._feddes_se_p3,
            )

    def _build_root_beta(self) -> None:
        """Precompute the per-plant-cell normalised root density β̂(z).

        Three shapes are supported by ``FeddesConfig.root_distribution``:

        - ``"uniform"`` (default): every plant cell weighted equally by
          its volume — equivalent to the historical "uniform transp
          over PlantSurface" behaviour.
        - ``"linear"``: β decays linearly from the surface (max at the
          top of the plant block, zero at depth = ``plant_height``).
        - ``"exponential"``: β decays as ``exp(−z / root_decay_length)``
          where z is depth from the surface in metres.

        The raw shape is divided by ``Σ β · cell_vol`` so the
        downstream Šimůnek formula is invariant to the unnormalised
        magnitude. The constant-1 (uniform) case reproduces the prior
        ``uniform_rate * alpha`` distribution exactly.
        """
        cell_vols = np.asarray(self.mesh.cellVolumes)[self.plant_cells]
        if cell_vols.size == 0:
            self._root_beta_normalized = np.zeros(0)
            self._root_cell_volumes = np.zeros(0)
            return
        # Mesh y is positive upward, surface at y=0, plant block from
        # y=-plant_height to y=0. Depth from surface = -y.
        y_centers = np.asarray(self.mesh.cellCenters[1])[self.plant_cells]
        depth_m = np.maximum(-y_centers, 0.0)

        shape = self.ode_config.feddes.root_distribution
        if shape == "linear":
            ph = self.mesh_config.plant_height
            raw = np.clip(1.0 - depth_m / max(ph, 1.0e-9), 0.0, 1.0)
        elif shape == "exponential":
            L = max(self.ode_config.feddes.root_decay_length, 1.0e-6)
            raw = np.exp(-depth_m / L)
        else:
            if shape != "uniform":
                logging.warning(
                    "FeddesConfig.root_distribution=%r unknown; falling back "
                    "to 'uniform'.", shape,
                )
            raw = np.ones_like(cell_vols)

        norm = float(np.sum(raw * cell_vols))
        self._root_cell_volumes = cell_vols
        self._root_beta_normalized = (raw / norm) if norm > 0 else np.zeros_like(raw)

    def feddes_alpha(self, se: np.ndarray) -> np.ndarray:
        """Per-cell Feddes α(Se) ∈ [0, 1]. Returns a constant-1 array when
        Feddes is disabled — kept callable in both regimes so callers
        don't need to branch.
        """
        if self._feddes_se_p2 is None:
            return np.ones_like(se)
        return _alpha_feddes_per_cell(
            se,
            se_p2=self._feddes_se_p2,
            se_p3=self._feddes_se_p3,
            se_p0=self._feddes_se_p0,
            se_p1=self._feddes_se_p1,
        )

    def _infiltration_capacity_m(self, seg_name: str, dt: float) -> float:
        """Max metres of water column the cells beneath ``seg_name`` can
        absorb in one ``dt`` step before the per-cell clip kicks in.

        Mirrors the ``max_pos = (SE_MAX − Se)/dt · θ_diff`` per-cell
        clipper limit in :meth:`apply_source`. Sums the per-cell
        headroom volume and divides by the segment's top face length
        to land at a water-column depth (m) over the segment.
        """
        cells = self.segment_cells.get(seg_name)
        if cells is None or cells.size == 0:
            return 0.0
        face_len = self.segment_face_len.get(seg_name, 0.0)
        if face_len <= 0:
            return 0.0
        se_cells = np.asarray(self.rel_sat.value)[cells]
        cell_vols = np.asarray(self.mesh.cellVolumes)[cells]
        headroom_m2 = float(np.sum(
            np.maximum(SE_MAX - se_cells, 0.0) * self.theta_diff * cell_vols
        ))
        return headroom_m2 / face_len

    def _route_rain_through_ponding(
        self,
        rain_flux: float,
        dt: float,
    ) -> tuple[dict[str, float], float]:
        """Run the per-segment ponding bucket on incoming rain.

        Returns ``(effective_seg_flux, overflow_mass)``. ``effective_seg_flux``
        maps open-sky segment → an effective rain flux density
        (kg/(m²·s)) to apply to the soil this substep; ``overflow_mass``
        is the total deliberate overflow (kg per metre out-of-plane row
        depth) that runs off above ``h_max_mm``.

        Mass-balance accounting: ``incoming = infiltrated + Δsurface_h +
        overflow`` per segment. The caller logs ``overflow`` as
        ponding-overflow in :class:`ClipDiagnostics`; the bucket state
        persists across substeps.
        """
        h_max_m = self.ode_config.ponding.h_max_mm / 1000.0
        effective: dict[str, float] = {}
        overflow_mass = 0.0
        for name in self.open_sky_segment_names:
            face_len = self.segment_face_len.get(name, 0.0)
            if face_len <= 0:
                continue
            incoming_m = (rain_flux * dt) / RHO_W if rain_flux > 0 else 0.0
            self.surface_h[name] = self.surface_h.get(name, 0.0) + incoming_m
            capacity_m = self._infiltration_capacity_m(name, dt)
            infiltrated_m = min(self.surface_h[name], capacity_m)
            self.surface_h[name] -= infiltrated_m
            if self.surface_h[name] > h_max_m:
                seg_overflow_m = self.surface_h[name] - h_max_m
                self.surface_h[name] = h_max_m
                overflow_mass += seg_overflow_m * RHO_W * face_len
            effective[name] = infiltrated_m * RHO_W / dt if dt > 0 else 0.0
        return effective, overflow_mass

    # -- integration primitives -----------------------------------------------

    def apply_source(
        self,
        *,
        seg_evap: dict[str, float],
        seg_transp: dict[str, float],
        rain_flux: float,
        flow_m3s: float,
        dt: float,
    ) -> ClipDiagnostics:
        """Rebuild the ``source`` CellVariable for the next ``dt`` step.

        All contributions are summed per cell as a θ-rate (1/s), then
        clipped together so Se stays in ``[SE_MIN, SE_MAX]`` after one
        step. The rain / evap / irrigation / transp accounting matches
        the documented boundary-condition table in SOIL.md §6.2.

        Returns a :class:`ClipDiagnostics` carrying the mass that the
        clipper *removed* from the requested sources — top side (runoff)
        and bottom side (unmet sink demand). Callers accumulate these
        across an ``advance`` to drive the ``WATER_RUNOFF`` and
        ``WATER_DEMAND_UNMET`` channels.
        """
        se = self.rel_sat.value
        coeff = self.theta_diff
        theta_rate = np.zeros_like(se)

        # Rain on the bare-soil strips (PV roof covers the plant zone;
        # irrigation drips reach the roots through the volumetric source
        # on WateringTopSegment instead). When ponding is enabled, route
        # incoming rain through the per-segment bucket: it accumulates,
        # drains into the soil at no more than the segment's
        # infiltration capacity, and overflows past ``h_max`` as
        # runoff. When ponding is off (the default), the original
        # straight-onto-cells path runs and the clipper handles excess.
        ponding_overflow_mass = 0.0
        if self.ode_config.ponding.enabled:
            effective_flux, ponding_overflow_mass = self._route_rain_through_ponding(
                rain_flux, dt,
            )
            for name, flux in effective_flux.items():
                if flux <= 0:
                    continue
                cells = self.segment_cells.get(name)
                vol = self.segment_cell_volume.get(name, 0.0)
                if cells is None or cells.size == 0 or vol <= 0:
                    continue
                factor = self.segment_face_len[name] / (RHO_W * vol)
                theta_rate[cells] += flux * factor
        elif rain_flux != 0.0:
            for name in self.open_sky_segment_names:
                cells = self.segment_cells.get(name)
                vol = self.segment_cell_volume.get(name, 0.0)
                if cells is None or cells.size == 0 or vol <= 0:
                    continue
                factor = self.segment_face_len[name] / (RHO_W * vol)
                theta_rate[cells] += rain_flux * factor

        # Per-segment soil evaporation. Shading is baked in upstream
        # by ``Evapotranspiration.evaluate``, so this loop just maps the
        # segment-local flux density onto its boundary cells.
        for name, evap in seg_evap.items():
            cells = self.segment_cells.get(name)
            vol = self.segment_cell_volume.get(name, 0.0)
            if cells is None or cells.size == 0 or vol <= 0:
                continue
            factor = self.segment_face_len[name] / (RHO_W * vol)
            theta_rate[cells] -= evap * factor

        # Drip irrigation on WateringTopSegment — volumetric, additive.
        if flow_m3s > 0.0 and self.irrigation_factor > 0:
            cells = self.segment_cells.get("WateringTopSegment")
            if cells is not None and cells.size:
                theta_rate[cells] += flow_m3s * self.irrigation_factor

        # Canopy transpiration distributed over PlantSurface cells. Per-
        # segment transp values share the same canopy bulk as one sink:
        # total kg/(m·s) = Σ transp_seg · face_len_seg. The Šimůnek &
        # Hopmans (2009) compensated-uptake formula then gives
        #
        #   S_i = T_pot · α_i · β̂_i / max(ω, ω_c)        (kg/m³/s)
        #   theta_rate_i = S_i / ρ_w                       (1/s)
        #
        # where ω = Σ α_j · β̂_j · V_j is the volume-weighted mean α.
        # When ``omega_c = 1.0`` (the default) compensation is off and
        # the formula reduces to ``α · β̂ · T_pot`` per cell, giving
        # total realised uptake ``= T_pot · ω`` (uncompensated Feddes).
        # When ``omega_c < 1`` and ``ω ≥ ω_c``, demand is redistributed
        # to less-stressed cells and total uptake recovers to ``T_pot``;
        # below ``ω_c`` we can't fully compensate and total uptake is
        # ``T_pot · ω / ω_c``.
        if seg_transp and self.plant_volume > 0 and self._root_beta_normalized.size > 0:
            transp_mass = sum(
                v * self.segment_face_len.get(name, 0.0)
                for name, v in seg_transp.items()
            )
            if transp_mass > 0:
                alpha = self.feddes_alpha(se[self.plant_cells])
                beta_hat = self._root_beta_normalized
                omega = float(np.sum(alpha * beta_hat * self._root_cell_volumes))
                divisor = max(omega, self.ode_config.feddes.omega_c)
                if divisor > 0:
                    theta_rate[self.plant_cells] -= (
                        transp_mass * alpha * beta_hat / (RHO_W * divisor)
                    )

        # Per-cell clip — Sₑ ∈ [SE_MIN, SE_MAX] after one dt step.
        max_pos = np.maximum((SE_MAX - se) / dt, 0.0) * coeff
        max_neg = np.minimum((SE_MIN - se) / dt, 0.0) * coeff
        clipped = np.clip(theta_rate, max_neg, max_pos)
        self.source_var.setValue(clipped)

        # Mass that the clipper threw away on each side of the band, in
        # kg per unit out-of-plane depth, integrated over this dt:
        #   excess [1/s] · cellVolume [m²/m] · ρ_w [kg/m³] · dt [s]
        # cellVolume is the FiPy 2-D cell area per unit row metre.
        excess = theta_rate - clipped
        cell_vol = np.asarray(self.mesh.cellVolumes)
        top_excess = np.maximum(excess, 0.0)       # wanted to add, couldn't
        bot_excess = np.maximum(-excess, 0.0)      # wanted to remove, couldn't
        return ClipDiagnostics(
            top_rejected=float(np.sum(top_excess * cell_vol)) * RHO_W * dt,
            bottom_rejected=float(np.sum(bot_excess * cell_vol)) * RHO_W * dt,
            ponding_overflow=ponding_overflow_mass,
        )

    # HYDRUS-1D's WATFLOW.FOR uses a water-content tolerance ``TolTh``
    # (typical 1e-3) as the Picard convergence criterion. We mirror that
    # here: the FiPy `sweep` residual is a vector-norm of the linear
    # residual that's hard to interpret physically; what we actually
    # care about is whether θ changed by less than a soil-moisture-
    # meaningful amount between sweeps.
    DEFAULT_TOL_TH: float = 1.0e-3
    DEFAULT_MAX_SWEEPS: int = 25

    def solve(
        self,
        dt: float,
        *,
        max_sweeps: int = DEFAULT_MAX_SWEEPS,
        tol_th: float = DEFAULT_TOL_TH,
        log_name: Optional[str] = None,
    ) -> SolveResult:
        """Picard sweep loop. Converges on ``max|Δθ_per_sweep| ≤ tol_th``.

        Always commits the post-sweep state (``rel_sat.updateOld()``). The
        caller decides whether to keep the result or roll back (via
        :meth:`snapshot` / :meth:`set_state`) and retry with a smaller
        ``dt``. ``log_name`` is only consulted on non-convergence and
        only used for the warning message — the wall-clock walk that
        wraps this logs at a higher level too when no further retry is
        possible.
        """
        eq = self.richards
        rel_sat = self.rel_sat
        coeff = self.theta_diff
        res = float("inf")
        prev_se = np.asarray(rel_sat.value).copy()
        converged = False
        dtheta_max = float("inf")
        sweeps = 0
        for k in range(max_sweeps):
            res = eq.sweep(dt=dt, var=rel_sat)
            cur_se = np.asarray(rel_sat.value)
            dtheta_max = float(np.max(np.abs(coeff * (cur_se - prev_se))))
            sweeps = k + 1
            if dtheta_max <= tol_th:
                converged = True
                break
            prev_se = cur_se.copy()
        if not converged and log_name is not None:
            logging.warning(
                "%s: PDE non-converged at dt=%.2fs in %d sweeps (final "
                "|Δθ|=%.2e, residual=%.2e, tol_th=%.0e).",
                log_name, float(dt), sweeps, dtheta_max, float(res), tol_th,
            )
        rel_sat.updateOld()
        return SolveResult(residual=float(res), converged=converged, sweeps=sweeps)

    # -- diagnostics & state ---------------------------------------------------

    def sample(self, probe: ProbeSpec) -> float:
        rel_sat = np.asarray(self.rel_sat.value)
        values = rel_sat[probe.cell_indices]
        return float(np.dot(probe.weights, values) / probe.weights.sum())

    def total_water(self) -> float:
        """Σ θ · cellVolume · ρ_w (kg per unit out-of-plane depth)."""
        se = self.rel_sat.value
        theta = self.ode_config.theta_r + self.theta_diff * se
        return float(np.sum(theta * np.asarray(self.mesh.cellVolumes))) * RHO_W

    def bottom_drainage_estimate(self) -> float:
        """Independent gravity-drainage flux estimate at the bottom face.

        With the free-drainage BC the bottom-face Darcy flux is
        ``q_bot = −K(h_bot)·cos(α)`` (HYDRUS-1D ``WATFLOW.FOR:213``). In
        our coordinate convention that's a downward, out-of-domain
        volumetric flux. We approximate the face Se by the mean of the
        bottom-adjacent cell Se values and return ``K · ρ_w`` so the
        result lives in the same kg/(m²·s) flux-density units as the
        per-segment evap/transp values.

        Independent of the integral-balance estimate in
        :meth:`SoilSimulation._balance_drainage_flux` (which forces
        closure by construction); compare the two for a global mass-
        balance check.
        """
        bot_cells = self.segment_cells.get("GroundBottomSegment")
        if bot_cells is None or bot_cells.size == 0:
            return 0.0
        se_bot_mean = float(np.mean(np.asarray(self.rel_sat.value)[bot_cells]))
        k_bot = float(np.asarray(self.soil_model.k_from_se(se_bot_mean)))
        return k_bot * RHO_W

    def snapshot(self) -> np.ndarray:
        """Copy of the live saturation field."""
        return np.asarray(self.rel_sat.value).copy()

    def set_state(self, arr: np.ndarray, *, update_old: bool = True) -> None:
        self.rel_sat.setValue(arr)
        if update_old:
            self.rel_sat._old.setValue(arr)

    def save_state_blob(self) -> bytes:
        buf = io.BytesIO()
        # The ponding bucket is also part of the solver state — without
        # it a restored simulation forgets standing surface water and
        # mass balance breaks across the restart boundary.
        surface_names = np.array(list(self.surface_h.keys()), dtype=object)
        surface_values = np.array(
            [self.surface_h[k] for k in surface_names], dtype=float
        )
        np.savez(
            buf,
            rel_sat=self.rel_sat.value.copy(),
            rel_sat_old=self.rel_sat._old.value.copy(),
            surface_names=surface_names,
            surface_h=surface_values,
        )
        return buf.getvalue()

    def load_state_blob(self, raw: bytes) -> None:
        buf = io.BytesIO(raw)
        arrays = np.load(buf, allow_pickle=True)
        self.rel_sat.setValue(arrays["rel_sat"])
        self.rel_sat._old.setValue(arrays["rel_sat_old"])
        # Older blobs predate ponding state; treat them as dry surfaces.
        if "surface_names" in arrays.files and "surface_h" in arrays.files:
            names = arrays["surface_names"]
            values = arrays["surface_h"]
            self.surface_h = {str(n): float(v) for n, v in zip(names, values)}


class SoilBase(Component):
    """Base for soil-PDE components.

    Owns the ``MeshConfig`` / ``PDEConfig`` / ``SoilPDECore`` triad that
    :class:`SoilSimulation` (live solver) and :class:`SoilPredictor`
    (forecast roll-outs) share. Subclasses populate ``self._mesh_config``
    and ``self._ode_config`` from their own config layout (the live
    solver reads its own ``[mesh]`` / ``[pde]`` blocks; the predictor
    pulls the mesh from its parent ``FieldSimulation`` and falls back to
    the live solver's PDE block) and then call :meth:`_build_pde` to
    instantiate an independent FiPy mesh + equation + ``CellVariable``.

    ``REL_SAT_NAME`` labels the FiPy cell variable so the live solver's
    state and the predictor's state are distinguishable in debug logs.
    """

    # Subclass overrides to label its CellVariable in the PDECore.
    REL_SAT_NAME: str = "relative saturation"

    _mesh_config: MeshConfig
    _ode_config: PDEConfig
    _pde: SoilPDECore

    def _build_pde(self) -> SoilPDECore:
        """Idempotently generate the .msh file (if missing) and build a
        fresh ``SoilPDECore``. Call once during ``configure()`` after
        ``self._mesh_config`` and ``self._ode_config`` are populated."""
        ensure_mesh(self._mesh_config)
        return SoilPDECore(
            self._mesh_config,
            self._ode_config,
            rel_sat_name=self.REL_SAT_NAME,
        )

    # -- thin accessors onto SoilPDECore --------------------------------------

    @property
    def _mesh_fipy(self) -> Gmsh2D:
        return self._pde.mesh

    @property
    def _soil_model(self) -> SoilModel:
        return self._pde.soil_model

    @property
    def _segment_cells(self) -> dict[str, np.ndarray]:
        return self._pde.segment_cells

    @property
    def _segment_face_len(self) -> dict[str, float]:
        return self._pde.segment_face_len

    @property
    def _segment_cell_volume(self) -> dict[str, float]:
        return self._pde.segment_cell_volume

    @property
    def _top_segment_names(self) -> list[str]:
        return self._pde.top_segment_names

    @property
    def _open_sky_segment_names(self) -> list[str]:
        return self._pde.open_sky_segment_names

    @property
    def _plant_cells(self) -> np.ndarray:
        return self._pde.plant_cells

    @property
    def _plant_volume(self) -> float:
        return self._pde.plant_volume

    @property
    def _theta_diff(self) -> float:
        return self._pde.theta_diff

    @property
    def _irrigation_factor(self) -> float:
        return self._pde.irrigation_factor

    @property
    def _rain_face_len(self) -> float:
        return self._pde.rain_face_len

    def _total_water(self) -> float:
        return self._pde.total_water()

    # -- shared diagnostic math -----------------------------------------------

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

    def _balance_drainage_flux(
        self,
        rates: FluxRates,
        delta_storage: float,
        duration_s: float,
    ) -> float:
        # Bottom drainage from the integral mass balance:
        #   drainage_mass = (rain_in + irr_in - evap_out - transp_out)·dt - Δstorage
        # then divided by (bottom_face_len · dt) to land back in kg/(m²·s).
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

    def _compute_diagnostics(
        self,
        rates: FluxRates,
        delta_storage: float,
        elapsed_s: float,
        clip: ClipDiagnostics,
    ) -> dict[str, float]:
        """Compute the 7 per-window flux-density diagnostics in kg/(m²·h).

        Pure math — no channel writes. Used by :class:`SoilSimulation` (one
        call per live ``advance``) and :class:`SoilPredictor` (one call per
        forecast interval) so the two solvers report directly comparable
        numbers. Returns a dict keyed by channel-key strings — callers look
        up the matching channel on ``self.data``.
        """
        watering_len = self._segment_face_len.get("WateringTopSegment", 0.0)
        irr_flux = rates.flow_m3s * RHO_W / watering_len if watering_len > 0 else 0.0
        top_in = irr_flux + rates.rain_flux

        e_flux_mean = self._face_weighted_mean(rates.seg_evap, self._top_segment_names)
        t_flux_mean = self._face_weighted_mean(rates.seg_transp, self._top_segment_names)
        bottom = self._balance_drainage_flux(rates, delta_storage, elapsed_s)
        direct_bottom = self._pde.bottom_drainage_estimate()  # kg/(m²·s)
        balance_residual = bottom - direct_bottom              # kg/(m²·s)

        top_face_len = float(self._rain_face_len) + sum(
            self._segment_face_len.get(n, 0.0)
            for n in ("PlantTopLeftSegment", "PlantTopRightSegment", "WateringTopSegment")
        )
        evap_face_len = sum(
            self._segment_face_len.get(n, 0.0) for n in self._top_segment_names
        )
        runoff_mass = clip.top_rejected + clip.ponding_overflow
        runoff_rate = (
            runoff_mass / (top_face_len * elapsed_s)
            if top_face_len > 0 and elapsed_s > 0 else 0.0
        )
        unmet_rate = (
            clip.bottom_rejected / (evap_face_len * elapsed_s)
            if evap_face_len > 0 and elapsed_s > 0 else 0.0
        )

        kg_per_s_to_kg_per_h = 3600.0
        return {
            "water_top_out": e_flux_mean * kg_per_s_to_kg_per_h,
            "water_transpiration": t_flux_mean * kg_per_s_to_kg_per_h,
            "water_top_in": top_in * kg_per_s_to_kg_per_h,
            "water_bottom": bottom * kg_per_s_to_kg_per_h,
            "water_runoff": runoff_rate * kg_per_s_to_kg_per_h,
            "water_demand_unmet": unmet_rate * kg_per_s_to_kg_per_h,
            "water_balance_residual": balance_residual * kg_per_s_to_kg_per_h,
        }


def create_mesh(mesh_config: MeshConfig) -> None:
    """Build the soil cross-section .msh file at ``mesh_config.filename``.

    Heavy gmsh call — callers should gate on file existence via
    :func:`ensure_mesh` rather than invoking this directly.
    """
    dl = mesh_config.dl
    width = mesh_config.width
    height = mesh_config.height
    plant_width = mesh_config.plant_width
    plant_height = mesh_config.plant_height
    watering_width = mesh_config.watering_width
    d_x = mesh_config.dx

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

    gmsh.write(mesh_config.filename)


def ensure_mesh(mesh_config: MeshConfig) -> None:
    """No-op if the mesh file already exists; otherwise generate it via
    :func:`create_mesh`. Idempotent — safe for both siblings to call."""
    if not os.path.exists(mesh_config.filename):
        create_mesh(mesh_config)
