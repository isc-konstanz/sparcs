# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.simulation.core.pde
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mesh / PDE / drip config dataclasses, the FiPy Richards-equation core
(``SoilPDECore``), and the mesh-generation / probe helpers.
"""

from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass, field, fields
from typing import Any, Callable, Optional

import gmsh

# FiPy 4.0.2 imports the numpy-2-deprecated `numpy.core`; silence that DeprecationWarning
# before importing fipy (E402 ignored per file). No fixed FiPy release exists yet.
warnings.filterwarnings(
    "ignore",
    message=r"numpy\.core is deprecated",
    category=DeprecationWarning,
)

from fipy import CellVariable, DiffusionTerm, FaceVariable, ImplicitSourceTerm, TransientTerm
from fipy.meshes import Gmsh2D
from fipy.solvers import LinearGMRESSolver
from fipy.tools import serialComm

import numpy as np
import pandas as pd
from lories.components.weather import Weather
from lories.typing import Configurations
from lories.util import to_timedelta
from sparcs.components.agriculture.soil import (
    DEFAULT_SOIL_MODEL,
    SoilModel,
    create_soil_model,
)

from .state import decode_state_blob, encode_state_blob

logging.getLogger("fipy").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

RHO_W: float = 1000.0  # kg/m³
SE_MIN: float = 1e-6  # effective-saturation floor for source clipping
SE_MAX: float = 0.999  # effective-saturation ceiling for source clipping
# Floor on (SE_MAX - Se) when linearizing the implicit irrigation intake;
# bounds the penalty coefficient B when the strip is already at saturation.
IRR_HEADROOM_EPS: float = 1e-4


def design_flow_lpm(nozzle_count: int, nozzle_flow_lph: float) -> float:
    """Whole-field design flow [l/min] from the drip layout: nozzle output x count.
    Same l/min unit the physical flow meter reports."""
    return nozzle_count * nozzle_flow_lph / 60.0


def flow_m3s_per_m(flow_lpm: float, total_drip_line_length_m: float) -> float:
    """Whole-field flow [l/min] normalized to m³/s per out-of-plane metre of row,
    spread over the total drip-line length."""
    return flow_lpm / (60_000.0 * total_drip_line_length_m)


@dataclass
class SolveResult:
    """One Picard sweep loop's outcome; ``converged`` means ``max|Δθ_per_sweep| ≤ tol_th``.
    ``finite`` is False on NaN/Inf after the sweep: state not committed, the caller must roll back."""

    residual: float
    converged: bool
    sweeps: int
    finite: bool = True
    error: Optional[str] = None

    @property
    def failed(self) -> bool:
        """True when the substep must be rolled back or retried."""
        return not self.converged or not self.finite or self.error is not None


@dataclass
class WalkResult:
    """Outcome of one :meth:`SoilPDECore.walk_window` call; ``ok`` is False in strict mode on a failure at ``dt_min``
    or a ``cancel()``. In accept mode non-finite substeps are skipped and their seconds add up in ``skipped_s``."""

    ok: bool = True
    reason: Optional[str] = None
    clip: "ClipDiagnostics" = field(default_factory=lambda: ClipDiagnostics())
    retries: int = 0
    skipped_s: float = 0.0
    cancelled: bool = False


@dataclass
class ClipDiagnostics:
    """Mass [kg per metre of row, over dt] that did not move as requested: ``top_rejected`` rain at saturated cells
    (ponding off), ``bottom_rejected`` unmet ET demand, ``ponding_overflow`` runoff past ``h_max_mm``."""

    top_rejected: float = 0.0
    bottom_rejected: float = 0.0
    ponding_overflow: float = 0.0

    def add(self, other: "ClipDiagnostics") -> None:
        self.top_rejected += other.top_rejected
        self.bottom_rejected += other.bottom_rejected
        self.ponding_overflow += other.ponding_overflow


@dataclass
class PondingPlan:
    """Pond updates planned by ``apply_source`` and applied by :meth:`SoilPDECore.commit_ponding` only after the
    substep's solve commits, so rollbacks cannot double-count inflow. Depths in m of water column, ``irr_b`` in 1/s."""

    dt: float
    rain_bucket_m: dict[str, float] = field(default_factory=dict)
    irr_available_m: float = 0.0
    irr_cells: Optional[np.ndarray] = None
    irr_b: Optional[np.ndarray] = None


@dataclass
class FluxRates:
    """Per-callback fluxes keyed by mesh segment name, in kg/(m²·s); ``flow_m3s`` in m³/s per out-of-plane metre
    of row (whole-field flow divided by the total drip-line length)."""

    seg_evap: dict[str, float]
    seg_transp: dict[str, float]
    flow_m3s: float
    rain_flux: float


def segment_flux_dicts(
    seg_et: dict[str, pd.DataFrame],
    ts: pd.Timestamp,
) -> tuple[dict[str, float], dict[str, float]]:
    """Per-segment ET flux dicts at ``ts``; negative ET (radiative cooling) clips to zero and zero-flux segments
    are skipped."""
    seg_evap: dict[str, float] = {}
    seg_transp: dict[str, float] = {}
    for name, frame in seg_et.items():
        if ts not in frame.index:
            continue
        evap = max(0.0, float(frame.loc[ts, "evap"]))
        transp = max(0.0, float(frame.loc[ts, "transp"]))
        if evap > 0.0:
            seg_evap[name] = evap
        if transp > 0.0:
            seg_transp[name] = transp
    return seg_evap, seg_transp


def rain_flux(et_data: pd.DataFrame, ts: pd.Timestamp, elapsed_s: float) -> float:
    """Rain flux density [kg/(m²·s)] for the interval ending at ``ts``, spread evenly over ``elapsed_s``.
    A missing column or row, NaN or non-positive precipitation means no rain."""
    col = Weather.PRECIPITATION
    if elapsed_s <= 0 or col not in et_data.columns or ts not in et_data.index:
        return 0.0
    precip = et_data.loc[ts, col]
    if pd.isna(precip) or precip <= 0:
        return 0.0
    return float(precip) / elapsed_s  # mm/s == kg/(m²·s)


# Real rig bay width [m]; matches the ``bay_width`` parameter default.
_DEFAULT_BAY_WIDTH: float = 3.5


@dataclass
class MeshConfig:
    def __init__(self, configs: Configurations, bay_width: Optional[float] = None):
        default_width = _DEFAULT_BAY_WIDTH if bay_width is None else bay_width
        self.filename: str = configs.get("filename", default="soil.msh")
        self.dl: float = configs.get("dl", default=0.1)
        self.width: float = configs.get("width", default=default_width)
        self.height: float = configs.get("height", default=5.0)
        self.plant_width: float = configs.get("plant_width", default=2.0)
        self.plant_height: float = configs.get("plant_height", default=2.0)
        self.watering_width: float = configs.get("watering_width", default=1.0)
        self.dx: float = configs.get("d_x", default=0.5)


def _nearest_cell_m(
    cell_x: np.ndarray,
    cell_y: np.ndarray,
    x_m: float,
    depth_m: float,
    x_offset: float,
) -> int:
    """Index of the cell nearest to bay-centered ``x_m`` and positive-downward ``depth_m`` (mesh y is ``-depth_m``).
    ``x_offset`` is the bay-center shift, ``mesh_config.width / 2.0``."""
    return int(np.argmin((cell_x - (x_m + x_offset)) ** 2 + (cell_y - (-depth_m)) ** 2))


def _coords_to_cell(
    mesh_fipy: "Gmsh2D",
    mesh_config: "MeshConfig",
    x_offset_cm: float,
    depth_cm: float,
) -> int:
    """Nearest FiPy cell index for bay-centered sensor coordinates in cm."""
    x_m = x_offset_cm * 0.01
    depth_m = depth_cm * 0.01
    cell_centers = np.asarray(mesh_fipy.cellCenters)
    cell_x, cell_y = cell_centers[0], cell_centers[1]
    x_offset = mesh_config.width / 2.0
    return _nearest_cell_m(cell_x, cell_y, x_m, depth_m, x_offset)


def resolve_sensor_probe(
    key: str,
    x_offset_cm: float,
    depth_cm: float,
    mesh_fipy: "Gmsh2D",
    mesh_config: "MeshConfig",
) -> ProbeSpec:
    """Point :class:`ProbeSpec` at a sensor's bay-centered cm location; ``channel_id`` is the sensor key."""
    idx = _coords_to_cell(mesh_fipy, mesh_config, x_offset_cm, depth_cm)
    return ProbeSpec(
        name=f"Sensor {key} (x_offset={x_offset_cm:.1f}cm, depth={depth_cm:.1f}cm)",
        channel_id=key,
        cell_indices=np.array([idx], dtype=int),
        weights=np.array([1.0]),
    )


def resolve_probe_from_sensor(
    sensor: Any,
    mesh_fipy: "Gmsh2D",
    mesh_config: "MeshConfig",
) -> ProbeSpec:
    """:func:`resolve_sensor_probe` for a ``SoilMoisture`` sensor (``key``, ``x_offset``, ``depth``)."""
    return resolve_sensor_probe(sensor.key, sensor.x_offset, sensor.depth, mesh_fipy, mesh_config)


def resolve_probes(
    probes_cfg: Configurations,
    mesh_fipy: Gmsh2D,
    mesh_config: "MeshConfig",
    log_name: Optional[str] = None,
) -> list[ProbeSpec]:
    """Point ``ProbeSpec`` list from ``[probes.points.<name>]`` blocks, in centimetres like a SoilMoisture sensor:
    ``x_offset`` bay-centered (left negative), ``depth`` positive-downward. Missing keys raise ValueError."""
    probes: list[ProbeSpec] = []
    if not probes_cfg.has_member("points"):
        return probes
    for key, spec in probes_cfg.get_member("points").items():
        try:
            x_offset_cm = float(spec["x_offset"])
            depth_cm = float(spec["depth"])
        except (KeyError, TypeError):
            raise ValueError(
                f"probe point '{key}' must use centimetre coordinates "
                f"'x_offset'/'depth'; the old metre keys 'x'/'y' were removed. "
                f"Convert with x_offset = x * 100, depth = y * 100 "
                f"(see docs/adr/0001-soil-coordinate-units-cm.md)."
            ) from None
        idx = _coords_to_cell(mesh_fipy, mesh_config, x_offset_cm, depth_cm)
        probes.append(
            ProbeSpec(
                name=f"Probe point (x_offset={x_offset_cm:.1f}cm, depth={depth_cm:.1f}cm)",
                channel_id=key,
                cell_indices=np.array([idx], dtype=int),
                weights=np.array([1.0]),
            )
        )
    return probes


def top_segment_count(mesh: "MeshConfig") -> int:
    """Bare top segments per side of the root zone, rounded the way ``create_mesh`` lays them out."""
    return round((mesh.width - mesh.plant_width) / (2 * mesh.dx))


def top_segment_names_from_mesh(mesh: "MeshConfig") -> list[str]:
    """Top-segment names derived from MeshConfig (left bare strips, plant tops, right bare strips)."""
    open_sky = [f"{side}TopSegment_{i}" for i in range(top_segment_count(mesh)) for side in ("Left", "Right")]
    return [*open_sky, "PlantTopLeftSegment", "PlantTopRightSegment"]


@dataclass
class FeddesConfig:
    """Feddes (1978) piecewise-linear root-water-uptake stress factor α(h) ∈ [0, 1], off by default.
    pF = log10(|h| in cm): α is 0 wetter than P0 (if anaerobic), 1 from P1 to P2, 0 at P3, linear ramps between."""

    enabled: bool = False
    anaerobic: bool = False
    p0_pf: float = 0.0  # anaerobic upper limit
    p1_pf: float = 1.0  # optimal lower bound
    p2_pf: float = 3.0  # optimal upper bound
    p3_pf: float = 4.2  # wilting point

    # Root distribution β(z), normalised so Σ β · cell_vol = 1: "uniform", "linear" (0 at plant_height)
    # or "exponential" (exp(-z / root_decay_length)).
    root_distribution: str = "uniform"
    root_decay_length: float = 0.3  # m; only used for "exponential"

    # Šimůnek compensation threshold ω_c: below 1, demand moves to less-stressed cells so uptake is T_pot
    # while ω ≥ ω_c and T_pot · ω / ω_c below.
    omega_c: float = 1.0

    @classmethod
    def from_configs(cls, configs: Configurations, base: Optional[FeddesConfig] = None) -> FeddesConfig:
        """Parse every field from ``configs``; a missing key falls back to ``base.<field>`` when given,
        else the field default, so an explicit key always wins over ``base``."""
        self = cls()
        for f in fields(cls):
            default = f.default if base is None else getattr(base, f.name)
            if f.type == "bool":
                value = configs.get_bool(f.name, default=default)
            elif f.type == "float":
                value = float(configs.get(f.name, default=default))
            elif f.type == "str":
                value = str(configs.get(f.name, default=default))
            else:
                raise TypeError(f"FeddesConfig.from_configs: unmapped field type {f.type!r} for {f.name!r}")
            if f.name == "root_distribution":
                value = value.strip().lower()
            setattr(self, f.name, value)
        return self

    def __init__(self, configs: Optional[Configurations] = None, base: Optional[FeddesConfig] = None):
        # configs=None ignores base and keeps the plain field defaults; otherwise copy from_configs' fields.
        if configs is None:
            for f in fields(self):
                setattr(self, f.name, f.default)
            return
        resolved = FeddesConfig.from_configs(configs, base=base)
        for f in fields(self):
            setattr(self, f.name, getattr(resolved, f.name))


def _alpha_feddes_per_cell(
    se: np.ndarray,
    *,
    se_p2: float,
    se_p3: float,
    se_p0: Optional[float] = None,
    se_p1: Optional[float] = None,
) -> np.ndarray:
    """Piecewise-linear α ∈ [0, 1] per cell from Se thresholds ordered se_p0 ≥ se_p1 ≥ se_p2 ≥ se_p3.
    With se_p0 or se_p1 None the anaerobic branch is skipped."""
    se = np.asarray(se, dtype=float)
    alpha = np.zeros_like(se)
    anaerobic = se_p0 is not None and se_p1 is not None

    if anaerobic:
        plateau = (se < se_p1) & (se >= se_p2)
    else:
        plateau = se >= se_p2
    alpha[plateau] = 1.0

    if anaerobic and se_p0 > se_p1:
        anaerobic_ramp = (se < se_p0) & (se >= se_p1)
        alpha[anaerobic_ramp] = (se_p0 - se[anaerobic_ramp]) / (se_p0 - se_p1)

    if se_p2 > se_p3:
        dry_ramp = (se < se_p2) & (se > se_p3)
        alpha[dry_ramp] = (se[dry_ramp] - se_p3) / (se_p2 - se_p3)

    return alpha


@dataclass
class PondingConfig:
    """Per-top-segment rain pond, off by default; overflow above ``h_max_mm`` is runoff. Irrigation always ponds
    on the watering strip, capped by ``watering_h_max_mm`` (defaults to ``h_max_mm``, or to ``base``'s when given)."""

    enabled: bool = False
    h_max_mm: float = 5.0  # max rain-ponding depth before overflow [mm]
    watering_h_max_mm: float = 5.0  # max emitter-pond depth on the watering strip [mm]

    @classmethod
    def from_configs(cls, configs: Configurations, base: Optional[PondingConfig] = None) -> PondingConfig:
        """Parse every field from ``configs``; a missing key falls back to ``base.<field>`` when given, else the field
        default; without ``base``, ``watering_h_max_mm`` defaults to the parsed ``h_max_mm`` (field order matters)."""
        self = cls()
        for f in fields(cls):
            if base is not None:
                default = getattr(base, f.name)
            elif f.name == "watering_h_max_mm":
                default = self.h_max_mm
            else:
                default = f.default
            if f.type == "bool":
                value = configs.get_bool(f.name, default=default)
            elif f.type == "float":
                value = float(configs.get(f.name, default=default))
            elif f.type == "str":
                value = str(configs.get(f.name, default=default))
            else:
                raise TypeError(f"PondingConfig.from_configs: unmapped field type {f.type!r} for {f.name!r}")
            setattr(self, f.name, value)
        return self

    def __init__(self, configs: Optional[Configurations] = None, base: Optional[PondingConfig] = None):
        # configs=None ignores base and keeps the plain field defaults; otherwise copy from_configs' fields.
        if configs is None:
            for f in fields(self):
                setattr(self, f.name, f.default)
            return
        resolved = PondingConfig.from_configs(configs, base=base)
        for f in fields(self):
            setattr(self, f.name, getattr(resolved, f.name))


@dataclass
class PDEConfig:
    def __init__(self, configs: Configurations, model_configs: Optional[Configurations] = None):
        # Hydraulic params from [model] block; fall back to configs for direct construction.
        m = model_configs if model_configs is not None else configs
        self.model: str = str(m.get("type", default=m.get("model", default=DEFAULT_SOIL_MODEL)))
        self.theta_r: float = m.get("theta_r", default=0.05)
        self.theta_s: float = m.get("theta_s", default=0.43)
        self.alpha: float = m.get("alpha", default=0.08)
        self.n: float = m.get("n", default=1.6)
        self.k_s: float = m.get("k_s", default=1.0e-4)
        # Mualem pore-interaction exponent; only forwarded when set explicitly.
        self.bpar: Optional[float] = float(m.get("bpar")) if "bpar" in m else None
        # dt: target substep size; dt_min: floor for adaptive refinement.
        self.dt: float = to_timedelta(configs.get("dt", default="50s")).total_seconds()
        self.dt_min: float = to_timedelta(configs.get("dt_min", default="1s")).total_seconds()

        # Width [m] of the PV vertical footprint where rain is blocked; 0 = no shadow.
        self.rain_shadow_width: float = float(configs.get("rain_shadow_width", default=0.0))

        # Fraction of intercepted rain that runs off onto the open soil (1.0 = all).
        self.rain_runoff_fraction: float = float(configs.get("rain_runoff_fraction", default=1.0))

        # Fraction of rain passing through the PV shadow onto shaded soil (leaky roof, edge drip),
        # clamped to [0, 1]; 0 = fully blocked, the default.
        self.rain_shadow_passthrough: float = min(
            1.0, max(0.0, float(configs.get("rain_shadow_passthrough", default=0.0)))
        )

        # Initial condition: uniform Se (ic_se) or hydrostatic equilibrium
        # (ic_water_table_depth metres below surface).
        self.ic_se: float = float(configs.get("ic_se", default=0.35))
        self.ic_water_table_depth: Optional[float] = (
            float(configs.get("ic_water_table_depth")) if "ic_water_table_depth" in configs else None
        )

        # Cold-start spin-up duration; defaults to 0 with a hydrostatic IC.
        if "cold_start" in configs:
            self.cold_start: pd.Timedelta = to_timedelta(configs.get("cold_start"))
        elif self.ic_water_table_depth is not None:
            self.cold_start = pd.Timedelta(0)
        else:
            self.cold_start = to_timedelta("3h")

        # [ponding]/[feddes] are siblings of [pde], not nested, so a whole-block [pde] override cannot drop them;
        # apply_surface_forcing fills these defaults.
        self.feddes: FeddesConfig = FeddesConfig(None)
        self.ponding: PondingConfig = PondingConfig(None)

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


def apply_surface_forcing(
    ode_config: PDEConfig,
    configs: Optional[Configurations],
    ponding_base: Optional[PondingConfig] = None,
    feddes_base: Optional[FeddesConfig] = None,
) -> PDEConfig:
    """Set ``ode_config.ponding``/``.feddes`` from sibling ``[ponding]``/``[feddes]`` blocks; an absent block keeps
    the current value. ``ponding_base``/``feddes_base`` are per-key defaults, so a present block merges by key."""
    if configs is not None and hasattr(configs, "has_member"):
        if configs.has_member("ponding"):
            ode_config.ponding = PondingConfig(configs.get_member("ponding", defaults={}), base=ponding_base)
        if configs.has_member("feddes"):
            ode_config.feddes = FeddesConfig(configs.get_member("feddes", defaults={}), base=feddes_base)
    return ode_config


def resolve_pde_config(
    component_block: Configurations,
    model_block: Configurations,
    inherit_forcing_from: Optional[PDEConfig] = None,
) -> PDEConfig:
    """Build a component's ``PDEConfig`` with surface forcing. ``inherit_forcing_from``'s ponding/feddes objects are
    seeded first and kept when the block is absent; a component's own block key-merges over them and wins."""
    cfg = PDEConfig(component_block.get_member("pde", defaults={}, ensure_exists=True), model_configs=model_block)
    if inherit_forcing_from is not None:
        cfg.ponding = inherit_forcing_from.ponding
        cfg.feddes = inherit_forcing_from.feddes
        apply_surface_forcing(
            cfg,
            component_block,
            ponding_base=inherit_forcing_from.ponding,
            feddes_base=inherit_forcing_from.feddes,
        )
    else:
        apply_surface_forcing(cfg, component_block)
    return cfg


# eq=False: identity equality avoids ambiguous numpy array comparisons.
@dataclass(eq=False)
class ProbeSpec:
    """Resolved sampling recipe for one probe: ``rel_sat`` cell indices with weight 1.0 for a point
    or per-cell volumes for an area (volume-weighted mean)."""

    name: str
    channel_id: str
    cell_indices: np.ndarray
    weights: np.ndarray


class SoilPDECore:
    """FiPy Richards-equation core: mesh, PDE, segment index, integration primitives and state I/O."""

    soil_model: SoilModel

    mesh: Gmsh2D
    rel_sat: CellVariable
    source_var: CellVariable
    irr_source_var: CellVariable  # explicit part A of the implicit irrigation source
    irr_impl_var: CellVariable  # implicit coefficient (holds -B ≤ 0)
    richards: Any

    segment_cells: dict[str, np.ndarray]
    segment_face_len: dict[str, float]
    segment_cell_volume: dict[str, float]
    top_segment_names: list[str]
    open_sky_segment_names: list[str]
    rain_open_fraction: dict[str, float]  # per top segment: fraction reached by rain
    rain_runoff_amplification: float
    plant_cells: np.ndarray
    plant_volume: float

    theta_diff: float  # θ_s - θ_r
    rain_face_len: float  # Σ face_len over open-sky [m]

    _feddes_se_p2: Optional[float]
    _feddes_se_p3: Optional[float]
    _feddes_se_p0: Optional[float]
    _feddes_se_p1: Optional[float]

    # Normalised root density β̂(z) on plant cells, Σ β̂_i · V_i = 1 [1/m²].
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
        # GMRES + ILU: avoids scipy LU hard-crash on near-singular matrices.
        self._solver = LinearGMRESSolver(
            tolerance=1.0e-8,
            iterations=1000,
            precon="default",
        )
        self.mesh = Gmsh2D(mesh_config.filename, communicator=serialComm)
        self._build_eq(rel_sat_name)
        self._build_segment_index()
        self._build_feddes_thresholds()
        self._build_root_beta()
        # Per-segment surface-pond depth [m of water column], kept in the state blob. Rain ponds on open-sky
        # segments when PondingConfig.enabled; irrigation always ponds on the watering strip.
        self.surface_h: dict[str, float] = {name: 0.0 for name in [*self.open_sky_segment_names, "WateringTopSegment"]}

    # PDE assembly

    def _build_eq(self, rel_sat_name: str) -> None:
        mesh = self.mesh
        rel_sat = CellVariable(mesh=mesh, name=rel_sat_name, hasOld=True)
        g_faces = FaceVariable(mesh=mesh, name="gravity faces", value=(0, 1.0))
        source = CellVariable(mesh=mesh, name="source", value=0.0)
        # Irrigation intake as a linearized implicit source r(Se) = B·(SE_MAX - Se), B frozen per substep,
        # so intake throttles to zero as the strip saturates. Both stay 0 outside irrigation substeps.
        irr_source = CellVariable(mesh=mesh, name="irrigation source", value=0.0)
        irr_impl = CellVariable(mesh=mesh, name="irrigation intake coeff", value=0.0)

        kf = self.soil_model.k_from_se(rel_sat)
        d_h = self.soil_model.dh_dse(rel_sat)

        # Richards' equation in Se form; free drainage comes from FiPy's zero-gradient Neumann BC plus gravity.
        # Its divergence also sums exterior faces and would feed K(Se_top) in at the top, so keep the bottom face only.
        g_values = np.array(g_faces.value, dtype=float)
        g_values[:, np.asarray(mesh.exteriorFaces) & ~np.asarray(mesh.physicalFaces["GroundBottomSegment"])] = 0.0
        g_faces.setValue(g_values)
        gravity_flux = g_faces * kf.faceValue
        gravity_div = gravity_flux.divergence
        richards = TransientTerm(coeff=self.ode_config.theta_s - self.ode_config.theta_r) == (
            DiffusionTerm(coeff=(kf * d_h)) + gravity_div + source + irr_source + ImplicitSourceTerm(coeff=irr_impl)
        )

        ic_wt = self.ode_config.ic_water_table_depth
        if ic_wt is not None:
            rel_sat.setValue(self._hydrostatic_ic_array(ic_wt))
            logger.info(
                "SoilPDECore: hydrostatic IC (water table at %.2f m below surface) Se min=%.3f max=%.3f",
                ic_wt,
                float(np.min(rel_sat.value)),
                float(np.max(rel_sat.value)),
            )
        else:
            # Clamp the unvalidated config value: an ic_se of exactly 0/1 sits on
            # the retention curve's singularities before any post-sweep clipping.
            rel_sat.setValue(float(np.clip(self.ode_config.ic_se, SE_MIN, SE_MAX)))
        rel_sat.updateOld()

        self.rel_sat = rel_sat
        self.source_var = source
        self.irr_source_var = irr_source
        self.irr_impl_var = irr_impl
        self.richards = richards

    def _hydrostatic_ic_array(self, water_table_depth_m: float) -> np.ndarray:
        """Hydrostatic-equilibrium Se field: saturated at and below the water table, gravity-matric balance above it.
        y is positive upward with the surface at 0."""
        y_centers = np.asarray(self.mesh.cellCenters[1], dtype=float)
        y_wt = -float(water_table_depth_m)
        h_above_wt_m = np.maximum(y_centers - y_wt, 0.0)
        psi_hpa_signed = -h_above_wt_m * 100.0 * 0.980665
        se = np.asarray(self.soil_model.se_from_psi(psi_hpa_signed), dtype=float)
        return np.clip(se, SE_MIN, SE_MAX)

    def _build_segment_index(self) -> None:
        mesh = self.mesh
        names = top_segment_names_from_mesh(self.mesh_config)
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

        # Per-segment fraction of rain reaching the soil (1 = fully open, 0 = under modules).
        self.rain_open_fraction = self._compute_rain_open_fractions(names)
        self.open_sky_segment_names = [n for n in names if self.rain_open_fraction.get(n, 1.0) > 0.0]
        open_face_len = sum(self.segment_face_len[n] * self.rain_open_fraction.get(n, 1.0) for n in names)
        blocked_face_len = sum(self.segment_face_len[n] * (1.0 - self.rain_open_fraction.get(n, 1.0)) for n in names)
        # Amplification factor: PV-intercepted rain redistributed to open soil.
        runoff = min(1.0, max(0.0, self.ode_config.rain_runoff_fraction))
        self.rain_runoff_amplification = 1.0 + runoff * blocked_face_len / open_face_len if open_face_len > 0.0 else 1.0
        self.rain_face_len = open_face_len * self.rain_runoff_amplification

    def _compute_rain_open_fractions(self, names: list) -> dict:
        """Fraction of each top segment reached by rain: 1 outside the centered PV shadow (``rain_shadow_width`` [m]),
        ``rain_shadow_passthrough`` inside it."""
        mc = self.mesh_config
        shadow = max(0.0, float(getattr(self.ode_config, "rain_shadow_width", 0.0)))
        passthrough = min(1.0, max(0.0, float(getattr(self.ode_config, "rain_shadow_passthrough", 0.0))))
        center = mc.width / 2.0
        lo, hi = center - shadow / 2.0, center + shadow / 2.0

        dx = mc.dx
        n_pv = top_segment_count(mc)
        plant_left = n_pv * dx
        plant_right = plant_left + mc.plant_width
        watering_left = plant_left + (mc.plant_width - mc.watering_width) / 2.0
        watering_right = watering_left + mc.watering_width
        extents = {
            "PlantTopLeftSegment": (plant_left, watering_left),
            "PlantTopRightSegment": (watering_right, plant_right),
        }
        for i in range(n_pv):
            extents[f"LeftTopSegment_{i}"] = (i * dx, (i + 1) * dx)
            x0 = plant_right + i * dx
            extents[f"RightTopSegment_{i}"] = (x0, x0 + dx)

        fractions: dict = {}
        for name in names:
            a, b = extents.get(name, (0.0, 0.0))
            seg_len = b - a
            if seg_len <= 0.0:
                fractions[name] = 1.0
                continue
            covered = max(0.0, min(b, hi) - max(a, lo))
            shadow_frac = covered / seg_len
            # the shaded portion still admits `passthrough` of the rain
            fractions[name] = max(0.0, 1.0 - shadow_frac * (1.0 - passthrough))
        return fractions

    def _build_feddes_thresholds(self) -> None:
        """Convert Feddes pF thresholds to Se using the retention curve (once at build time)."""
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
            logger.warning(
                "Feddes pF thresholds map to non-monotone Se (P2 → Se=%.3f, "
                "P3 → Se=%.3f). Check pF ordering; dry ramp will be disabled.",
                self._feddes_se_p2,
                self._feddes_se_p3,
            )

    def _build_root_beta(self) -> None:
        """Precompute normalised root density β̂(z) on plant cells (Σ β̂_i · V_i = 1); unknown shapes fall back to
        "uniform" with a warning."""
        cell_vols = np.asarray(self.mesh.cellVolumes)[self.plant_cells]
        if cell_vols.size == 0:
            self._root_beta_normalized = np.zeros(0)
            self._root_cell_volumes = np.zeros(0)
            return
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
                logger.warning(
                    "FeddesConfig.root_distribution=%r unknown; falling back to 'uniform'.",
                    shape,
                )
            raw = np.ones_like(cell_vols)

        norm = float(np.sum(raw * cell_vols))
        self._root_cell_volumes = cell_vols
        self._root_beta_normalized = (raw / norm) if norm > 0 else np.zeros_like(raw)

    def feddes_alpha(self, se: np.ndarray) -> np.ndarray:
        """Per-cell Feddes α(Se) ∈ [0, 1]; all ones when Feddes is disabled."""
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
        """Max water-column depth [m] the cells beneath ``seg_name`` can absorb in ``dt`` [s]."""
        cells = self.segment_cells.get(seg_name)
        if cells is None or cells.size == 0:
            return 0.0
        face_len = self.segment_face_len.get(seg_name, 0.0)
        if face_len <= 0:
            return 0.0
        se_cells = np.asarray(self.rel_sat.value)[cells]
        cell_vols = np.asarray(self.mesh.cellVolumes)[cells]
        headroom_m2 = float(np.sum(np.maximum(SE_MAX - se_cells, 0.0) * self.theta_diff * cell_vols))
        return headroom_m2 / face_len

    def _plan_rain_ponding(
        self,
        rain_flux: float,
        dt: float,
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Plan one dt of the rain ponding buckets without mutating state: effective flux [kg/(m²·s)] and
        bucket depth [m] per segment, before the ``h_max_mm`` trim in :meth:`commit_ponding`."""
        effective: dict[str, float] = {}
        bucket_after: dict[str, float] = {}
        for name in self.open_sky_segment_names:
            face_len = self.segment_face_len.get(name, 0.0)
            if face_len <= 0:
                continue
            weight = self.rain_open_fraction.get(name, 1.0) * self.rain_runoff_amplification
            incoming_m = (rain_flux * dt) / RHO_W * weight if rain_flux > 0 else 0.0
            bucket_m = self.surface_h.get(name, 0.0) + incoming_m
            capacity_m = self._infiltration_capacity_m(name, dt)
            infiltrated_m = min(bucket_m, capacity_m)
            bucket_after[name] = bucket_m - infiltrated_m
            effective[name] = infiltrated_m * RHO_W / dt if dt > 0 else 0.0
        return effective, bucket_after

    # Integration primitives

    def apply_source(
        self,
        *,
        seg_evap: dict[str, float],
        seg_transp: dict[str, float],
        rain_flux: float,
        flow_m3s: float,
        dt: float,
    ) -> tuple[ClipDiagnostics, PondingPlan]:
        """Rebuild the sources for the next ``dt``: rain and ET as clipped θ-rate [1/s], irrigation as implicit source.
        Pass the returned :class:`PondingPlan` to :meth:`commit_ponding` once the substep's solve is committed."""
        se = self.rel_sat.value
        coeff = self.theta_diff
        theta_rate = np.zeros_like(se)
        plan = PondingPlan(dt=dt)

        if self.ode_config.ponding.enabled:
            effective_flux, plan.rain_bucket_m = self._plan_rain_ponding(
                rain_flux,
                dt,
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
                weight = self.rain_open_fraction.get(name, 1.0) * self.rain_runoff_amplification
                theta_rate[cells] += rain_flux * factor * weight

        for name, evap in seg_evap.items():
            cells = self.segment_cells.get(name)
            vol = self.segment_cell_volume.get(name, 0.0)
            if cells is None or cells.size == 0 or vol <= 0:
                continue
            factor = self.segment_face_len[name] / (RHO_W * vol)
            theta_rate[cells] -= evap * factor

        self.irr_source_var.setValue(0.0)
        self.irr_impl_var.setValue(0.0)
        watering_len = self.segment_face_len.get("WateringTopSegment", 0.0)
        watering_vol = self.segment_cell_volume.get("WateringTopSegment", 0.0)
        cells = self.segment_cells.get("WateringTopSegment")
        if watering_len > 0 and watering_vol > 0 and cells is not None and cells.size:
            plan.irr_available_m = self.surface_h.get("WateringTopSegment", 0.0)
            if flow_m3s > 0.0 and dt > 0:
                plan.irr_available_m += flow_m3s * dt / watering_len
            if plan.irr_available_m > 0 and dt > 0:
                # Offer the whole pond this substep as r(Se) = B·(SE_MAX - Se),
                # scaled so r(Se_now) empties the pond in dt where headroom allows.
                r_offer = plan.irr_available_m * watering_len / (dt * watering_vol)
                headroom = np.maximum(SE_MAX - np.asarray(se)[cells], IRR_HEADROOM_EPS)
                b = r_offer / headroom
                a_arr = np.zeros_like(theta_rate)
                b_arr = np.zeros_like(theta_rate)
                a_arr[cells] = b * SE_MAX
                b_arr[cells] = -b
                self.irr_source_var.setValue(a_arr)
                self.irr_impl_var.setValue(b_arr)
                plan.irr_cells = cells
                plan.irr_b = b

        # Šimůnek & Hopmans (2009) compensated uptake; ω is the volume-weighted mean stress factor.
        if seg_transp and self.plant_volume > 0 and self._root_beta_normalized.size > 0:
            transp_mass = sum(v * self.segment_face_len.get(name, 0.0) for name, v in seg_transp.items())
            if transp_mass > 0:
                alpha = self.feddes_alpha(se[self.plant_cells])
                beta_hat = self._root_beta_normalized
                omega = float(np.sum(alpha * beta_hat * self._root_cell_volumes))
                divisor = max(omega, self.ode_config.feddes.omega_c)
                if divisor > 0:
                    theta_rate[self.plant_cells] -= transp_mass * alpha * beta_hat / (RHO_W * divisor)

        # Clip Se to [SE_MIN, SE_MAX] after one dt step.
        max_pos = np.maximum((SE_MAX - se) / dt, 0.0) * coeff
        max_neg = np.minimum((SE_MIN - se) / dt, 0.0) * coeff
        clipped = np.clip(theta_rate, max_neg, max_pos)
        self.source_var.setValue(clipped)

        excess = theta_rate - clipped
        cell_vol = np.asarray(self.mesh.cellVolumes)
        top_excess = np.maximum(excess, 0.0)
        bot_excess = np.maximum(-excess, 0.0)
        clip = ClipDiagnostics(
            top_rejected=float(np.sum(top_excess * cell_vol)) * RHO_W * dt,
            bottom_rejected=float(np.sum(bot_excess * cell_vol)) * RHO_W * dt,
        )
        return clip, plan

    def commit_ponding(self, plan: PondingPlan) -> float:
        """Apply a substep's planned pond updates after its solve committed; watering intake is read back from the
        committed field. Returns the overflow (runoff) mass [kg per metre of row]."""
        h_max_m = self.ode_config.ponding.h_max_mm / 1000.0
        overflow_mass = 0.0

        for name, bucket_m in plan.rain_bucket_m.items():
            face_len = self.segment_face_len.get(name, 0.0)
            if bucket_m > h_max_m:
                overflow_mass += (bucket_m - h_max_m) * RHO_W * face_len
                bucket_m = h_max_m
            self.surface_h[name] = bucket_m

        if plan.irr_cells is not None and plan.irr_b is not None:
            face_len = self.segment_face_len.get("WateringTopSegment", 0.0)
            if face_len > 0:
                watering_h_max_m = self.ode_config.ponding.watering_h_max_mm / 1000.0
                se_new = np.asarray(self.rel_sat.value)[plan.irr_cells]
                intake_rate = np.maximum(plan.irr_b * (SE_MAX - se_new), 0.0)
                cell_vols = np.asarray(self.mesh.cellVolumes)[plan.irr_cells]
                intake_m = float(np.sum(intake_rate * cell_vols)) * plan.dt / face_len
                bucket_m = plan.irr_available_m - min(intake_m, plan.irr_available_m)
                if bucket_m > watering_h_max_m:
                    overflow_mass += (bucket_m - watering_h_max_m) * RHO_W * face_len
                    bucket_m = watering_h_max_m
                self.surface_h["WateringTopSegment"] = bucket_m

        return overflow_mass

    # Picard convergence: max|Δθ| per sweep ≤ tol_th (water-content tolerance).
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
        """Picard sweep loop until ``max|Δθ_per_sweep| ≤ tol_th``; a raise or non-finite field is reported without
        committing state. A finite field is clipped to ``[SE_MIN, SE_MAX]`` before ``updateOld()``."""
        eq = self.richards
        rel_sat = self.rel_sat
        coeff = self.theta_diff
        res = float("inf")
        prev_se = np.asarray(rel_sat.value).copy()
        converged = False
        dtheta_max = float("inf")
        sweeps = 0
        for k in range(max_sweeps):
            try:
                # Scope FP-warning suppression to the solve: GMRES matmul on
                # near-singular saturation matrices is the known noise source.
                with np.errstate(all="ignore"):
                    res = eq.sweep(dt=dt, var=rel_sat, solver=self._solver)
            except Exception as e:  # noqa: BLE001  (scipy/FiPy raise a zoo of types)
                if log_name is not None:
                    logger.warning(
                        "%s: PDE sweep raised at dt=%.2fs (sweep %d): %s: %s",
                        log_name,
                        float(dt),
                        k + 1,
                        type(e).__name__,
                        e,
                    )
                return SolveResult(
                    residual=float("inf"),
                    converged=False,
                    sweeps=k,
                    finite=False,
                    error=f"{type(e).__name__}: {e}",
                )
            cur_se = np.asarray(rel_sat.value)
            dtheta_max = float(np.max(np.abs(coeff * (cur_se - prev_se))))
            sweeps = k + 1
            if dtheta_max <= tol_th:
                converged = True
                break
            prev_se = cur_se.copy()

        se = np.asarray(rel_sat.value)
        finite = bool(np.all(np.isfinite(se)))
        if not finite:
            if log_name is not None:
                logger.warning(
                    "%s: PDE produced non-finite Se at dt=%.2fs after %d sweeps; state not committed.",
                    log_name,
                    float(dt),
                    sweeps,
                )
            return SolveResult(
                residual=float(res),
                converged=False,
                sweeps=sweeps,
                finite=False,
            )

        if np.any(se > SE_MAX) or np.any(se < SE_MIN):
            rel_sat.setValue(np.clip(se, SE_MIN, SE_MAX))

        if not converged and log_name is not None:
            logger.warning(
                "%s: PDE non-converged at dt=%.2fs in %d sweeps (final |Δθ|=%.2e, residual=%.2e, tol_th=%.0e).",
                log_name,
                float(dt),
                sweeps,
                dtheta_max,
                float(res),
                tol_th,
            )
        rel_sat.updateOld()
        return SolveResult(residual=float(res), converged=converged, sweeps=sweeps)

    def walk_window(
        self,
        *,
        rates: FluxRates,
        window_s: float,
        accept_at_dt_min: bool = True,
        cancel: Optional[Callable[[], bool]] = None,
        on_step: Optional[Callable[[float], None]] = None,
        log_name: Optional[str] = None,
    ) -> WalkResult:
        """Adaptive-dt walk over ``window_s``: a failed substep rolls back and retries at a third, down to ``dt_min``.
        At ``dt_min``, ``accept_at_dt_min`` keeps finite unconverged states and skips non-finite ones, else aborts."""
        dt_max = self.ode_config.dt
        dt_min = max(self.ode_config.dt_min, 1.0e-6)
        sub_dt = dt_max
        t_offset = 0.0
        out = WalkResult()

        while t_offset < window_s - 1.0e-9:
            if cancel is not None and cancel():
                out.ok = False
                out.cancelled = True
                out.reason = "cancelled"
                return out

            attempted = min(sub_dt, window_s - t_offset)
            snap = self.snapshot()
            clip, plan = self.apply_source(
                seg_evap=rates.seg_evap,
                seg_transp=rates.seg_transp,
                rain_flux=rates.rain_flux,
                flow_m3s=rates.flow_m3s,
                dt=attempted,
            )
            result = self.solve(attempted, log_name=log_name)

            if result.failed and attempted > dt_min:
                self.set_state(snap)
                sub_dt = max(dt_min, attempted / 3.0)
                out.retries += 1
                continue

            if result.failed:
                failure = result.error or (
                    "non-finite Se field"
                    if not result.finite
                    else f"non-convergent after {result.sweeps} sweeps (res={result.residual:.3g})"
                )
                if not accept_at_dt_min:
                    self.set_state(snap)
                    out.ok = False
                    out.reason = f"{failure} at dt_min={dt_min:g}s"
                    return out
                if not result.finite or result.error is not None:
                    self.set_state(snap)
                    out.skipped_s += attempted
                    logger.warning(
                        "%s: substep skipped at dt_min=%gs (%s); state held for %.1fs of the window.",
                        log_name or "SoilPDECore",
                        dt_min,
                        failure,
                        attempted,
                    )
                    t_offset += attempted
                    if on_step is not None:
                        on_step(t_offset)
                    continue
            t_offset += attempted
            clip.ponding_overflow += self.commit_ponding(plan)
            out.clip.add(clip)
            if result.converged and result.sweeps <= 3 and sub_dt < dt_max:
                sub_dt = min(dt_max, sub_dt * 1.5)
            if on_step is not None:
                on_step(t_offset)

        return out

    # Diagnostics and state

    def sample(self, probe: ProbeSpec) -> float:
        rel_sat = np.asarray(self.rel_sat.value)
        values = rel_sat[probe.cell_indices]
        return float(np.dot(probe.weights, values) / probe.weights.sum())

    def total_water(self) -> float:
        """Σ θ · cellVolume · ρ_w (kg per unit out-of-plane depth)."""
        se = self.rel_sat.value
        theta = self.ode_config.theta_r + self.theta_diff * se
        return float(np.sum(theta * np.asarray(self.mesh.cellVolumes))) * RHO_W

    def surface_water(self) -> float:
        """Water held in the surface ponds (kg per unit out-of-plane depth), on top of :meth:`total_water`."""
        return float(sum(h * self.segment_face_len.get(name, 0.0) for name, h in self.surface_h.items())) * RHO_W

    def bottom_drainage_estimate(self) -> float:
        """Gravity-drainage flux at the bottom face [kg/(m²·s)], from K(Se_bottom) · ρ_w."""
        bot_cells = self.segment_cells.get("GroundBottomSegment")
        if bot_cells is None or bot_cells.size == 0:
            return 0.0
        se_bot_mean = float(np.mean(np.asarray(self.rel_sat.value)[bot_cells]))
        k_bot = float(np.asarray(self.soil_model.k_from_se(se_bot_mean)))
        return k_bot * RHO_W

    def snapshot(self) -> np.ndarray:
        """Copy of the current saturation field."""
        return np.asarray(self.rel_sat.value).copy()

    def set_state(self, arr: np.ndarray, *, update_old: bool = True) -> None:
        self.rel_sat.setValue(arr)
        if update_old:
            self.rel_sat._old.setValue(arr)

    def save_state_blob(self) -> bytes:
        return encode_state_blob(self.rel_sat.value, self.rel_sat._old.value, self.surface_h)

    def load_state_blob(self, raw: bytes) -> None:
        """Load an ``encode_state_blob`` blob; ``ValueError`` when it needs pickle or
        carries a different cell count than this mesh."""
        rel_sat, rel_sat_old, surface_h = decode_state_blob(raw)
        expected = np.asarray(self.rel_sat.value).shape
        if rel_sat.shape != expected:
            raise ValueError(
                f"soil state blob carries {rel_sat.shape} cells but the mesh has {expected}; "
                "stale blob from a different mesh configuration"
            )
        self.rel_sat.setValue(rel_sat)
        self.rel_sat._old.setValue(rel_sat_old)
        # Merge over zeroed defaults: blobs from before the watering pond
        # existed lack its key, and every known bucket must stay addressable.
        self.surface_h = {
            **{name: 0.0 for name in [*self.open_sky_segment_names, "WateringTopSegment"]},
            **surface_h,
        }


def create_mesh(mesh_config: MeshConfig) -> None:
    """Build the soil cross-section .msh file at ``mesh_config.filename``; slow, so call :func:`ensure_mesh`."""
    dl = mesh_config.dl
    width = mesh_config.width
    height = mesh_config.height
    plant_width = mesh_config.plant_width
    plant_height = mesh_config.plant_height
    watering_width = mesh_config.watering_width
    d_x = mesh_config.dx

    # check parameters validity (before gmsh.initialize, so a bad config never
    # leaves the gmsh library initialized)
    if width < plant_width + 2 * d_x:
        raise ValueError("Invalid parameters: width must be at least plant_width + 2 * d_x")
    if height <= 0:
        raise ValueError("Invalid parameters: height must be positive")
    if height <= plant_height:
        raise ValueError("Invalid parameters: height must be greater than plant_height")
    # Tolerance-based multiple check: a float modulo (`% d_x`) spuriously rejects
    # valid widths (e.g. 0.3 % 0.1 != 0 in binary floats).
    half_width = (width - plant_width) / 2
    surface_count = round(half_width / d_x)
    if surface_count < 1 or abs(half_width - surface_count * d_x) > 1e-9 * max(1.0, half_width):
        raise ValueError("Invalid parameters: (width - plant_width) must be a multiple of 2 * d_x")

    gmsh.initialize()
    try:
        gmsh.model.add("soil")

        lines_tl = []
        lines_tr = []

        # Top left
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

        # Plant
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
            [
                line_plant_top_1,
                line_plant_top_2,
                line_plant_top_3,
                line_plant_right,
                line_plant_bottom,
                line_plant_left,
            ]
        )
        surface_plant = gmsh.model.geo.addPlaneSurface([loop_plant])
        gmsh.model.geo.synchronize()
        gmsh.model.setPhysicalName(1, gmsh.model.addPhysicalGroup(1, [line_plant_top_1]), "PlantTopLeftSegment")
        gmsh.model.setPhysicalName(1, gmsh.model.addPhysicalGroup(1, [line_plant_top_2]), "WateringTopSegment")
        gmsh.model.setPhysicalName(1, gmsh.model.addPhysicalGroup(1, [line_plant_top_3]), "PlantTopRightSegment")
        gmsh.model.setPhysicalName(2, gmsh.model.addPhysicalGroup(2, [surface_plant]), "PlantSurface")

        # Top right
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

        # Ground layer
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
    finally:
        gmsh.finalize()


def ensure_mesh(mesh_config: MeshConfig) -> None:
    """No-op if the mesh file already exists; otherwise generate it via
    :func:`create_mesh`. Idempotent, safe for both siblings to call."""
    if not os.path.exists(mesh_config.filename):
        create_mesh(mesh_config)
