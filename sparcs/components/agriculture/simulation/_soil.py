# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.simulation._soil
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Shared base for the soil simulation chain: mesh / PDE / plot config
dataclasses, the FiPy Richards-equation core (``SoilPDECore``), shared
diagnostic math (``SoilBase``), and mesh-generation / probe helpers.

Both :class:`SoilSimulation` (live solver) and :class:`SoilPredictor`
(forecast roll-outs) inherit from :class:`SoilBase`.
"""

from __future__ import annotations

import io
import logging
import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import gmsh

# FiPy 4.0.2 imports the numpy-2-deprecated `numpy.core` in its numerix module;
# silence that import-time DeprecationWarning before importing fipy (E402-ignored
# per file). No fixed FiPy release exists yet.
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
from lories import Component
from lories.typing import Configurations
from lories.util import to_timedelta
from sparcs.components.agriculture.soil import (
    DEFAULT_SOIL_MODEL,
    SoilModel,
    create_soil_model,
)

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

    The single source for the drip-derived flow, shared by the live sim (which
    feeds it when the physical meter is unavailable) and the predictor's
    ``_derive_flow_m3s`` (which normalizes it per out-of-plane metre). Same
    l/min unit the physical flow meter reports.
    """
    return nozzle_count * nozzle_flow_lph / 60.0


@dataclass
class SolveResult:
    """One Picard sweep loop's outcome.

    ``converged``: ``max|Δθ_per_sweep| ≤ tol_th``. ``finite``: False when the
    post-sweep field contains NaN/Inf; state not committed, caller must roll back.
    ``error``: exception message if the sweep raised.
    """

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
    """Outcome of one :meth:`SoilPDECore.walk_window` call.

    ``ok`` is False in strict mode when a substep failed at ``dt_min`` or
    ``cancel()`` fired. In accept mode, non-finite substeps are skipped (state
    held) and their seconds accumulate in ``skipped_s``.
    """

    ok: bool = True
    reason: Optional[str] = None
    clip: "ClipDiagnostics" = field(default_factory=lambda: ClipDiagnostics())
    retries: int = 0
    skipped_s: float = 0.0
    cancelled: bool = False


@dataclass
class ClipDiagnostics:
    """Mass (kg per metre of out-of-plane row depth, integrated over dt) that
    did not enter or leave the soil as requested.

    - ``top_rejected``: rain clipped at a saturated cell (only with ponding
      disabled; irrigation never contributes — its excess ponds instead).
    - ``bottom_rejected``: unmet evaporation or root-uptake demand (cell too dry).
    - ``ponding_overflow``: true runoff — pond overflow past ``PondingConfig.h_max_mm``.
    """

    top_rejected: float = 0.0
    bottom_rejected: float = 0.0
    ponding_overflow: float = 0.0

    def add(self, other: "ClipDiagnostics") -> None:
        self.top_rejected += other.top_rejected
        self.bottom_rejected += other.bottom_rejected
        self.ponding_overflow += other.ponding_overflow


@dataclass
class PondingPlan:
    """Deferred surface-pond bookkeeping for one substep.

    ``apply_source`` plans pond updates without mutating ``surface_h``;
    :meth:`SoilPDECore.commit_ponding` applies them only after the substep's
    solve is committed, so adaptive-walk rollbacks and skips cannot
    double-count inflow into the buckets.

    - ``rain_bucket_m``: per open-sky segment bucket depth after this substep's
      inflow and planned infiltration, before the ``h_max_mm`` overflow trim.
    - ``irr_available_m``: watering-strip pond plus this substep's irrigation
      inflow (water-column metres over the strip); the actual intake is read
      back from the implicit source at commit time.
    - ``irr_cells`` / ``irr_b``: strip cells and the frozen per-cell intake
      coefficient B [1/s]; ``None`` when no irrigation water is on offer.
    """

    dt: float
    rain_bucket_m: dict[str, float] = field(default_factory=dict)
    irr_available_m: float = 0.0
    irr_cells: Optional[np.ndarray] = None
    irr_b: Optional[np.ndarray] = None


@dataclass
class FluxRates:
    """Per-callback fluxes for ``_apply_source`` and diagnostics.

    Flux densities in kg/(m²·s); ``flow_m3s`` in m³/s per out-of-plane metre of
    row (the whole-field metered flow divided by the total drip-line length).
    ``seg_evap`` and ``seg_transp`` are keyed by mesh segment name.
    """

    seg_evap: dict[str, float]
    seg_transp: dict[str, float]
    flow_m3s: float
    rain_flux: float


# Shared with FieldSimulation._bay_width's default (base.py) so the standalone
# and in-context mesh parses agree on one fallback: 3.5 m is real rig bay
# geometry, not an invented standalone default (was 10.0; see B6).
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
    """Return the index of the cell nearest to (x_m, depth_m) in mesh coordinates.

    ``x_offset`` is ``mesh_config.width / 2.0``, the bay-center shift that maps
    bay-centered x to absolute mesh x. ``depth_m`` is positive-downward depth; the
    mesh uses negative-y for depth, so cell_y target is ``-depth_m``.
    """
    return int(np.argmin((cell_x - (x_m + x_offset)) ** 2 + (cell_y - (-depth_m)) ** 2))


def _coords_to_cell(
    mesh_fipy: "Gmsh2D",
    mesh_config: "MeshConfig",
    x_offset_cm: float,
    depth_cm: float,
) -> int:
    """Convert sensor coordinates (cm, bay-centered) to the nearest FiPy cell index.

    Converts cm to metres (×0.01), then applies the same nearest-cell mapping used
    by :func:`resolve_probes` for point probes.
    """
    x_m = x_offset_cm * 0.01
    depth_m = depth_cm * 0.01
    cell_centers = np.asarray(mesh_fipy.cellCenters)
    cell_x, cell_y = cell_centers[0], cell_centers[1]
    x_offset = mesh_config.width / 2.0
    return _nearest_cell_m(cell_x, cell_y, x_m, depth_m, x_offset)


def resolve_probe_from_sensor(
    sensor: Any,
    mesh_fipy: "Gmsh2D",
    mesh_config: "MeshConfig",
) -> ProbeSpec:
    """Build a point :class:`ProbeSpec` at a sensor's location.

    A sensor is a probe that also carries measured data: its ``x_offset`` and
    ``depth`` (both cm, bay-centered) resolve to a single mesh cell exactly as a
    configured point probe would. ``channel_id`` is the sensor key.
    """
    idx = _coords_to_cell(mesh_fipy, mesh_config, sensor.x_offset, sensor.depth)
    return ProbeSpec(
        name=f"Sensor {sensor.key} (x_offset={sensor.x_offset:.1f}cm, depth={sensor.depth:.1f}cm)",
        channel_id=sensor.key,
        cell_indices=np.array([idx], dtype=int),
        weights=np.array([1.0]),
    )


def resolve_probes(
    probes_cfg: Configurations,
    mesh_fipy: Gmsh2D,
    mesh_config: "MeshConfig",
    log_name: Optional[str] = None,
) -> list[ProbeSpec]:
    """Resolve ``[probes.points.<name>]`` config blocks against a FiPy mesh into
    a list of point ``ProbeSpec`` sampling recipes.

    Coordinates use the same vocabulary and unit as a SoilMoisture sensor
    (a sensor is-a probe): ``x_offset`` bay-centered and signed (left negative),
    ``depth`` positive-downward, both in centimetres. See
    ``docs/adr/0001-soil-coordinate-units-cm.md``.
    """
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


def top_segment_names_from_mesh(mesh: "MeshConfig") -> list[str]:
    """Top-segment names derived from MeshConfig (left bare strips, plant tops, right bare strips)."""
    n_pv_segments = int((mesh.width - mesh.plant_width) / (2 * mesh.dx))
    open_sky = [f"{side}TopSegment_{i}" for i in range(n_pv_segments) for side in ("Left", "Right")]
    return [*open_sky, "PlantTopLeftSegment", "PlantTopRightSegment"]


@dataclass
class FeddesConfig:
    """Feddes (1978) piecewise-linear root-water-uptake stress factor α(h) ∈ [0, 1].

    Four pF thresholds (pF = log10(|h|), |h| in cm water) bracket the stress curve::

        |h| < |P0|          α = 0    (anaerobic; too wet)
        |P0|–|P1|           α : 0 → 1 (anaerobic ramp)
        |P1|–|P2|           α = 1    (optimal)
        |P2|–|P3|           α : 1 → 0 (dry ramp)
        |h| ≥ |P3|          α = 0    (wilting point)

    Thresholds are converted to Se at build time; the runtime path does a
    piecewise-linear interpolation on Se per plant cell.
    """

    enabled: bool = False
    anaerobic: bool = False
    p0_pf: float = 0.0  # anaerobic upper limit
    p1_pf: float = 1.0  # optimal lower bound
    p2_pf: float = 3.0  # optimal upper bound
    p3_pf: float = 4.2  # wilting point

    # Root distribution β(z): "uniform" (volume-weighted), "linear" (decays to 0
    # at plant_height), or "exponential" (exp(-z / root_decay_length)).
    # Normalised so Σ β · cell_vol = 1.
    root_distribution: str = "uniform"
    root_decay_length: float = 0.3  # m; only used for "exponential"

    # Šimůnek compensation threshold ω_c. At ω_c < 1 demand is redistributed
    # to less-stressed cells so total uptake = T_pot when ω ≥ ω_c;
    # below ω_c total uptake = T_pot · ω / ω_c.
    omega_c: float = 1.0

    def __init__(self, configs: Optional[Configurations] = None, base: Optional[FeddesConfig] = None):
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
        # base, when given, supplies the per-key parse default instead of the
        # hardcoded defaults above -- a key-level merge for a predictor overriding
        # only some of the sim's fields (see apply_surface_forcing).
        self.enabled = configs.get_bool("enabled", default=False if base is None else base.enabled)
        self.anaerobic = configs.get_bool("anaerobic", default=False if base is None else base.anaerobic)
        self.p0_pf = float(configs.get("p0_pf", default=0.0 if base is None else base.p0_pf))
        self.p1_pf = float(configs.get("p1_pf", default=1.0 if base is None else base.p1_pf))
        self.p2_pf = float(configs.get("p2_pf", default=3.0 if base is None else base.p2_pf))
        self.p3_pf = float(configs.get("p3_pf", default=4.2 if base is None else base.p3_pf))
        self.root_distribution = (
            str(configs.get("root_distribution", default="uniform" if base is None else base.root_distribution))
            .strip()
            .lower()
        )
        self.root_decay_length = float(
            configs.get("root_decay_length", default=0.3 if base is None else base.root_decay_length)
        )
        self.omega_c = float(configs.get("omega_c", default=1.0 if base is None else base.omega_c))


def _alpha_feddes_per_cell(
    se: np.ndarray,
    *,
    se_p2: float,
    se_p3: float,
    se_p0: Optional[float] = None,
    se_p1: Optional[float] = None,
) -> np.ndarray:
    """Piecewise-linear α ∈ [0, 1] per cell from Se thresholds.

    Se thresholds ordered se_p0 ≥ se_p1 ≥ se_p2 ≥ se_p3 (drier ⇒ smaller Se).
    When se_p0 / se_p1 are None the anaerobic branch is skipped.
    """
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
    """Per-top-segment surface-ponding bucket (metres of water column).

    When enabled, rain accumulates per segment and drains into the soil up to
    the segment's infiltration capacity; excess above ``h_max_mm`` overflows
    as runoff. When disabled, rain goes straight to soil cells and the
    clipper-rejected mass surfaces via ``ClipDiagnostics.top_rejected``.

    Irrigation always ponds on the watering strip regardless of ``enabled``
    (the drip emitter physically ponds; discarding the excess was a numerics
    bug). ``watering_h_max_mm`` bounds that pond separately — an emitter
    basin holds far more water column over its narrow strip than sheet
    ponding does on open ground — and defaults to ``h_max_mm`` when unset
    (with no ``base``; see ``base`` below).

    ``base``, when supplied, becomes the per-key parse default instead of the
    hardcoded defaults (a key-level merge rather than whole-block replacement)
    and ``watering_h_max_mm`` then defaults to ``base.watering_h_max_mm``
    instead of the just-parsed ``h_max_mm``.
    """

    enabled: bool = False
    h_max_mm: float = 5.0  # max rain-ponding depth before overflow [mm]
    watering_h_max_mm: float = 5.0  # max emitter-pond depth on the watering strip [mm]

    def __init__(self, configs: Optional[Configurations] = None, base: Optional[PondingConfig] = None):
        if configs is None:
            self.enabled = False
            self.h_max_mm = 5.0
            self.watering_h_max_mm = 5.0
            return
        self.enabled = configs.get_bool("enabled", default=False if base is None else base.enabled)
        self.h_max_mm = float(configs.get("h_max_mm", default=5.0 if base is None else base.h_max_mm))
        if base is None:
            # Dynamic default: an unset watering_h_max_mm follows the just-parsed
            # h_max_mm (pinned by tests/test_soil_strip_ponding.py).
            self.watering_h_max_mm = float(configs.get("watering_h_max_mm", default=self.h_max_mm))
        else:
            # base is fully resolved; the follow-h_max coupling above does not
            # apply -- an unset key falls back to base's OWN resolved value.
            self.watering_h_max_mm = float(configs.get("watering_h_max_mm", default=base.watering_h_max_mm))


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

        # Fraction of rain that passes THROUGH the PV shadow onto the shaded soil
        # (0 = fully blocked, the default; 1 = the shadow admits all rain). Models
        # an imperfectly sealing PV roof / edge drip reaching the bay-center column,
        # so a physical k_s can respond to rain there without over-draining. Clamped
        # to [0, 1].
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

        # Surface-forcing configs (ponding, feddes) are sibling blocks of [pde] at
        # the soil-component level, NOT nested under [pde] -- so a whole-block [pde]
        # override cannot silently drop them. They default here and are populated
        # from the component's [ponding]/[feddes] blocks via apply_surface_forcing.
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
    """Populate ``ode_config.ponding`` / ``.feddes`` from a soil component's
    sibling ``[ponding]`` / ``[feddes]`` blocks (peers of ``[pde]``, not nested).

    A block that is absent leaves the current value untouched, so the caller can
    seed inherited defaults first (e.g. a predictor seeding the live sim's forcing)
    and let the component's own block override. Keeping ponding and feddes out of
    ``[pde]`` means a whole-block ``[pde]`` override can never silently drop them.

    ``ponding_base`` / ``feddes_base``, when given, become the per-key parse
    defaults for a present block -- a key-level merge instead of a whole-block
    replacement against hardcoded defaults, so a predictor overriding only some
    keys does not silently reset the rest. Sim-block parses pass neither (fresh
    parse against the hardcoded defaults, unchanged).
    """
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
    """Build a component's ``PDEConfig`` and populate its surface forcing in one place.

    Collapses the construct-then-``apply_surface_forcing`` sequence duplicated
    across ``SoilSimulation.configure`` (site a), ``FieldSimulation.configure``'s
    eager ``_soil_pde_config`` parse (site b), and
    ``SoilPredictor._resolve_ode_config``'s own-``[pde]`` branch (site c) into
    one canonical resolution, so the three sites can never resolve
    ``[pde]``/``[model]``/forcing differently.

    ``component_block.get_member("pde", defaults={}, ensure_exists=True)`` parses
    against the component's own ``[pde]`` block -- a no-op merge when the block
    already exists; harmlessly materializes an empty one when absent.

    When ``inherit_forcing_from`` is given (site c inheriting the live sim's
    forcing), ``cfg.ponding``/``cfg.feddes`` are seeded to
    ``inherit_forcing_from``'s SAME objects (``is`` identity) BEFORE
    ``apply_surface_forcing`` runs, and that call passes
    ``ponding_base``/``feddes_base=inherit_forcing_from.ponding``/``.feddes`` so a
    present sibling block key-merges against the sim's resolved values instead
    of the hardcoded ``PondingConfig``/``FeddesConfig`` defaults.

    The seed-before-apply ORDER is mandatory, not cosmetic: reversing it --
    seeding AFTER calling ``apply_surface_forcing`` -- would let the plain
    identity assignment clobber ``apply_surface_forcing``'s merge result
    whenever the component states its OWN explicit ``[ponding]``/``[feddes]``
    override, silently discarding that override in favour of the sim's object.
    Seeding first means ``apply_surface_forcing``'s own-block branch (when
    present) always has the last word; the seed only supplies the correct
    fallback ``cfg.ponding``/``.feddes`` for the absent-block case, where
    ``apply_surface_forcing`` is a no-op.

    Sites with no ``inherit_forcing_from`` (sites a/b, fresh-parse semantics)
    call plain ``apply_surface_forcing(cfg, component_block)``.
    """
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
    """Resolved sampling recipe for one probe (point or area).

    ``cell_indices`` selects FiPy ``rel_sat`` cells; ``weights`` are 1.0 for
    a point probe or per-cell volumes for an area (volume-weighted mean).
    """

    name: str
    channel_id: str
    cell_indices: np.ndarray
    weights: np.ndarray


class SoilPDECore:
    """Shared FiPy / Richards-equation core for both the live solver and predictor.

    Owns the mesh, Richards PDE, segment index, and integration primitives
    (apply_source, solve, sample, total_water, state I/O).
    """

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
        # Per-segment surface-pond depth [m of water column]; persists via state blob.
        # Rain ponds on the open-sky segments (gated on PondingConfig.enabled);
        # irrigation always ponds on the watering strip.
        self.surface_h: dict[str, float] = {name: 0.0 for name in [*self.open_sky_segment_names, "WateringTopSegment"]}

    # -- PDE assembly ----------------------------------------------------------

    def _build_eq(self, rel_sat_name: str) -> None:
        mesh = self.mesh
        rel_sat = CellVariable(mesh=mesh, name=rel_sat_name, hasOld=True)
        g_faces = FaceVariable(mesh=mesh, name="gravity faces", value=(0, 1.0))
        source = CellVariable(mesh=mesh, name="source", value=0.0)
        # Irrigation intake as a linearized implicit source r(Se) = A - B·Se
        # (A = B·SE_MAX, B ≥ 0, frozen per substep): the solver throttles intake
        # to zero as the strip cell saturates, instead of inject-then-clip.
        # Both variables stay 0 outside irrigation substeps (no-op in the matrix).
        irr_source = CellVariable(mesh=mesh, name="irrigation source", value=0.0)
        irr_impl = CellVariable(mesh=mesh, name="irrigation intake coeff", value=0.0)

        kf = self.soil_model.k_from_se(rel_sat)
        d_h = self.soil_model.dh_dse(rel_sat)

        # Richards' equation in Se form:
        #   (θs-θr) ∂Se/∂t = ∇·[K · |dh/dSe| ∇Se] + ∂K/∂y + source
        # Free-drainage BC emerges from FiPy's zero-gradient Neumann + gravity divergence.
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
        """Hydrostatic-equilibrium Se field: saturated at/below the water table,
        Se(z) from gravity-matric balance above it. y positive upward, surface at 0.
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
        """Fraction of each top segment reached by rain: fully open outside the PV
        shadow (rain_shadow_width [m], centered), and ``rain_shadow_passthrough``
        of the rain inside it (0 = fully blocked, the default)."""
        mc = self.mesh_config
        shadow = max(0.0, float(getattr(self.ode_config, "rain_shadow_width", 0.0)))
        passthrough = min(1.0, max(0.0, float(getattr(self.ode_config, "rain_shadow_passthrough", 0.0))))
        center = mc.width / 2.0
        lo, hi = center - shadow / 2.0, center + shadow / 2.0

        dx = mc.dx
        n_pv = int((mc.width - mc.plant_width) / (2 * dx))
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
        """Precompute normalised root density β̂(z) on plant cells (Σ β̂_i · V_i = 1).

        Shapes: "uniform" (volume-weighted), "linear" (decays to 0 at plant_height),
        "exponential" (exp(-z / root_decay_length)).
        """
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
        """Per-cell Feddes α(Se) ∈ [0, 1]. Returns a constant-1 array when
        Feddes is disabled, kept callable in both regimes so callers
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
        """Plan one dt of the per-segment rain ponding buckets (no state mutation).

        Returns (effective_seg_flux [kg/(m²·s)], bucket_after_m per segment —
        pre-overflow; the ``h_max_mm`` trim happens in :meth:`commit_ponding`).
        """
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

    # -- integration primitives -----------------------------------------------

    def apply_source(
        self,
        *,
        seg_evap: dict[str, float],
        seg_transp: dict[str, float],
        rain_flux: float,
        flow_m3s: float,
        dt: float,
    ) -> tuple[ClipDiagnostics, PondingPlan]:
        """Rebuild the source variables for the next ``dt`` step.

        Sums rain / evap / transpiration as θ-rate [1/s] per cell and clips to
        keep Se ∈ [SE_MIN, SE_MAX]; irrigation (plus any ponded strip water) is
        offered through the linearized implicit source instead, so the solver
        throttles intake near saturation. Returns the clipped mass as
        :class:`ClipDiagnostics` plus a :class:`PondingPlan` the caller must
        pass to :meth:`commit_ponding` once the substep's solve is committed.
        """
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

        # Šimůnek & Hopmans (2009) compensated uptake:
        #   S_i = T_pot · α_i · β̂_i / max(ω, ω_c)  [kg/m³/s]
        # where ω = Σ α_j · β̂_j · V_j (volume-weighted mean stress factor).
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
        """Apply a substep's deferred pond updates after its solve committed.

        Decrements the watering pond by the intake the implicit source actually
        delivered (read back from the committed field), stores the planned rain
        buckets, and trims every bucket to ``h_max_mm``. Returns the trimmed
        (true-runoff) mass [kg per metre of row].
        """
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
        """Picard sweep loop; converges on ``max|Δθ_per_sweep| ≤ tol_th``.

        Raised sweeps and non-finite fields are reported via :class:`SolveResult`
        without committing state. Finite fields are safety-clipped to
        ``[SE_MIN, SE_MAX]`` before ``updateOld()``.
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
        """Adaptive-dt walk over ``window_s`` seconds.

        On failure, rolls back and retries at ``sub_dt / 3`` down to ``dt_min``.
        After fast convergence (≤ 3 sweeps), ``sub_dt`` grows back toward ``dt`` (×1.5).

        At ``dt_min``: ``accept_at_dt_min=True`` accepts finite under-converged states
        and skips non-finite ones (``WalkResult.skipped_s``);
        ``accept_at_dt_min=False`` aborts with ``WalkResult(ok=False)``.
        """
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

    def surface_water(self) -> float:
        """Σ pond depth · face_len · ρ_w — water held in the surface ponds
        (kg per unit out-of-plane depth), on top of :meth:`total_water`."""
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
        """Copy of the live saturation field."""
        return np.asarray(self.rel_sat.value).copy()

    def set_state(self, arr: np.ndarray, *, update_old: bool = True) -> None:
        self.rel_sat.setValue(arr)
        if update_old:
            self.rel_sat._old.setValue(arr)

    def save_state_blob(self) -> bytes:
        buf = io.BytesIO()
        # Fixed-width unicode (not object) dtype, so the blob needs no pickle.
        surface_names = np.array(list(self.surface_h.keys()), dtype=np.str_)
        surface_values = np.array([self.surface_h[k] for k in surface_names], dtype=float)
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
        # allow_pickle stays for legacy blobs whose surface_names were saved
        # with object dtype; new blobs are pickle-free.
        arrays = np.load(buf, allow_pickle=True)
        rel_sat = np.asarray(arrays["rel_sat"])
        expected = np.asarray(self.rel_sat.value).shape
        if rel_sat.shape != expected:
            raise ValueError(
                f"soil state blob carries {rel_sat.shape} cells but the mesh has {expected}; "
                "stale blob from a different mesh configuration"
            )
        self.rel_sat.setValue(rel_sat)
        # Legacy blobs (pre soil-refactor B3) wrote ONLY `rel_sat` via
        # `np.savez(buf, rel_sat=rel_sat)` -- no `rel_sat_old`, no surface fields.
        # Fall back to `rel_sat` itself so those pre-fix debug blobs stay loadable.
        rel_sat_old = arrays["rel_sat_old"] if "rel_sat_old" in arrays.files else rel_sat
        self.rel_sat._old.setValue(rel_sat_old)
        if "surface_names" in arrays.files and "surface_h" in arrays.files:
            names = arrays["surface_names"]
            values = arrays["surface_h"]
            # Merge over zeroed defaults: blobs from before the watering pond
            # existed lack its key, and every known bucket must stay addressable.
            self.surface_h = {
                **{name: 0.0 for name in [*self.open_sky_segment_names, "WateringTopSegment"]},
                **{str(n): float(v) for n, v in zip(names, values)},
            }


class SoilBase(Component):
    """Base for soil-PDE components (live solver and predictor).

    Subclasses populate ``_mesh_config`` / ``_ode_config`` then call
    :meth:`_build_pde`. ``REL_SAT_NAME`` labels the FiPy CellVariable.
    """

    REL_SAT_NAME: str = "relative saturation"

    _mesh_config: MeshConfig
    _ode_config: PDEConfig
    _pde: SoilPDECore

    def _build_pde(self) -> SoilPDECore:
        """Generate .msh if missing and build a fresh ``SoilPDECore``."""
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
    def _segment_face_len(self) -> dict[str, float]:
        return self._pde.segment_face_len

    @property
    def _top_segment_names(self) -> list[str]:
        return self._pde.top_segment_names

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
        # drainage = (in - out)·dt - Δstorage, then / (bottom_face_len · dt) → kg/(m²·s)
        bottom_len = self._segment_face_len.get("GroundBottomSegment", 0.0)
        if bottom_len <= 0 or duration_s <= 0:
            return 0.0
        evap_mass = sum(value * self._segment_face_len.get(name, 0.0) for name, value in rates.seg_evap.items())
        transp_mass = sum(value * self._segment_face_len.get(name, 0.0) for name, value in rates.seg_transp.items())
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
        """Compute the 7 per-window flux-density diagnostics [kg/(m²·h)].

        Returns a dict keyed by channel-key strings; no channel writes.
        """
        watering_len = self._segment_face_len.get("WateringTopSegment", 0.0)
        irr_flux = rates.flow_m3s * RHO_W / watering_len if watering_len > 0 else 0.0
        top_in = irr_flux + rates.rain_flux

        e_flux_mean = self._face_weighted_mean(rates.seg_evap, self._top_segment_names)
        t_flux_mean = self._face_weighted_mean(rates.seg_transp, self._top_segment_names)
        bottom = self._balance_drainage_flux(rates, delta_storage, elapsed_s)
        direct_bottom = self._pde.bottom_drainage_estimate()  # kg/(m²·s)
        balance_residual = bottom - direct_bottom  # kg/(m²·s)

        # Geometric top face (all segments + watering strip, not amplified like rain_face_len).
        top_face_len = self._segment_face_len.get("WateringTopSegment", 0.0) + sum(
            self._segment_face_len.get(n, 0.0) for n in self._top_segment_names
        )
        evap_face_len = sum(self._segment_face_len.get(n, 0.0) for n in self._top_segment_names)
        runoff_mass = clip.top_rejected + clip.ponding_overflow
        runoff_rate = runoff_mass / (top_face_len * elapsed_s) if top_face_len > 0 and elapsed_s > 0 else 0.0
        unmet_rate = clip.bottom_rejected / (evap_face_len * elapsed_s) if evap_face_len > 0 and elapsed_s > 0 else 0.0

        kg_per_s_to_kg_per_h = 3600.0
        return {
            "top_out": e_flux_mean * kg_per_s_to_kg_per_h,
            "transpiration": t_flux_mean * kg_per_s_to_kg_per_h,
            "top_in": top_in * kg_per_s_to_kg_per_h,
            "bottom_out": bottom * kg_per_s_to_kg_per_h,
            "runoff": runoff_rate * kg_per_s_to_kg_per_h,
            "demand_unmet": unmet_rate * kg_per_s_to_kg_per_h,
            "balance_residual": balance_residual * kg_per_s_to_kg_per_h,
        }


def create_mesh(mesh_config: MeshConfig) -> None:
    """Build the soil cross-section .msh file at ``mesh_config.filename``.

    Heavy gmsh call; callers should gate on file existence via
    :func:`ensure_mesh` rather than invoking this directly.
    """
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
