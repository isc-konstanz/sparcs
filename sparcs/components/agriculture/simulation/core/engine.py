# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.simulation.core.engine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The one soil engine over a single ``SoilPDECore`` instance: state in, state out.
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Any, Callable, Mapping, Optional

import numpy as np

from .anchor import AnchorSensor
from .config import SoilConfig
from .pde import (
    RHO_W,
    SE_MAX,
    SE_MIN,
    ClipDiagnostics,
    FluxRates,
    PDEConfig,
    SoilPDECore,
    ensure_mesh,
    resolve_pde_config,
    resolve_probes,
    resolve_sensor_probe,
)
from .state import Forcing, SoilState, StepResult

logger = logging.getLogger(__name__)

Cancel = Optional[Callable[[], bool]]


class SoilEngine:
    def __init__(self, config: SoilConfig, ode: PDEConfig, pde: SoilPDECore) -> None:
        self.config = config
        self.ode = ode
        self.pde = pde
        self.model = pde.soil_model
        self.mesh_config = config.mesh
        self._current: Optional[SoilState] = None

    @classmethod
    def build(
        cls,
        soil: SoilConfig,
        model: Optional[Any] = None,
        *,
        rel_sat_name: str = "relative saturation",
    ) -> "SoilEngine":
        """Create the mesh (if missing) and the ``SoilPDECore``."""
        if soil.mesh.width is None:
            raise ValueError("soil.mesh.width is not set; call SoilConfig.mesh.derive(bay_width=...) first")
        ensure_mesh(soil.mesh)
        model_block = model if model is not None else soil.configs.get_member("model", defaults={}, ensure_exists=True)
        ode = resolve_pde_config(soil.configs, model_block)
        pde = SoilPDECore(soil.mesh, ode, rel_sat_name=rel_sat_name)
        return cls(soil, ode, pde)

    @property
    def mesh(self) -> Any:
        """The FiPy mesh the state arrays are laid out on."""
        return self.pde.mesh

    @property
    def cell_centers(self) -> np.ndarray:
        return np.asarray(self.mesh.cellCenters)

    @property
    def se_bounds(self) -> tuple[float, float]:
        return SE_MIN, SE_MAX

    @property
    def cold_start_s(self) -> float:
        """Spin-up window for the first advance, from ``PDEConfig.cold_start``."""
        return float(self.ode.cold_start.total_seconds())

    @property
    def top_segment_names(self) -> list[str]:
        return list(self.pde.top_segment_names)

    @property
    def segment_face_length(self) -> dict[str, float]:
        return {name: self.pde.segment_face_len.get(name, 0.0) for name in self.pde.top_segment_names}

    def initial_state(self, at: dt.datetime) -> SoilState:
        """Hydrostatic or uniform initial condition; does not touch the PDE core."""
        water_table_depth = self.ode.ic_water_table_depth
        if water_table_depth is not None:
            se = self.pde._hydrostatic_ic_array(water_table_depth)
        else:
            n = int(self.pde.rel_sat.value.shape[0])
            se = np.full(n, self.ode.ic_se, dtype=float)
        surface_h = {name: 0.0 for name in self.pde.surface_h}
        return SoilState(se=se, se_old=se.copy(), surface_h=surface_h, at=at)

    def advance(self, state: SoilState, forcing: Forcing, *, cancel: Cancel = None) -> StepResult:
        """Walk one forcing window; a cancelled result carries the input state."""
        if state is not self._current:
            self.load(state)
        storage_before = self.pde.total_water() + self.pde.surface_water()
        rates = FluxRates(
            seg_evap=dict(forcing.seg_evap),
            seg_transp=dict(forcing.seg_transp),
            flow_m3s=forcing.flow_m3s,
            rain_flux=forcing.rain_flux,
        )
        walk = self.pde.walk_window(rates=rates, window_s=forcing.dt_s, accept_at_dt_min=True, cancel=cancel)
        if walk.cancelled:
            self._current = None
            return StepResult(state=state, diagnostics={}, cancelled=True)

        if walk.skipped_s > 0:
            logger.error(
                "SoilEngine: held state through %.1fs of a %.1fs window (substeps "
                "unsolvable at dt_min); flux diagnostics assume forcing over "
                "the full window, so drainage is misstated for the skipped slices.",
                walk.skipped_s,
                forcing.dt_s,
            )

        new_state = self._read_back(at=forcing.end)
        delta_storage = (self.pde.total_water() + self.pde.surface_water()) - storage_before
        diagnostics = dict(self._compute_diagnostics(rates, delta_storage, forcing.dt_s, walk.clip))
        diagnostics.update(
            water_total=self.pde.total_water(),
            surface_water=self.pde.surface_water(),
            delta_storage=delta_storage,
            skipped_s=walk.skipped_s,
            retries=float(walk.retries),
            walk_ok=1.0 if walk.ok else 0.0,
        )
        self._current = new_state
        return StepResult(state=new_state, diagnostics=diagnostics)

    def tension_at(self, state: SoilState, probe: Any) -> float:
        """Sample Se at a probe and convert to signed negative hPa."""
        if state is not self._current:
            self.load(state)
        se = self.pde.sample(probe)
        return float(self.model.psi_from_se(se))

    def invalidate(self) -> None:
        """Drop the identity cache after the PDE core was mutated outside ``advance``."""
        self._current = None

    def diagnostics(self, state: SoilState) -> Mapping[str, float]:
        """The state-only summary: total water and surface water."""
        if state is not self._current:
            self.load(state)
        return {"water_total": self.pde.total_water(), "surface_water": self.pde.surface_water()}

    def probes(self, probes_block: Any) -> list:
        if not probes_block:
            return []
        return resolve_probes(probes_block, self.pde.mesh, self.mesh_config)

    def probe_from_sensor(self, sensor: AnchorSensor) -> Any:
        return resolve_sensor_probe(sensor.key, sensor.x_offset_cm, sensor.depth_cm, self.pde.mesh, self.mesh_config)

    def load(self, state: SoilState) -> None:
        """Put ``state`` into the PDE core; ``ValueError`` when its cell count does not fit the mesh."""
        self.pde.load_state_blob(state.to_blob())
        self._current = state

    def _read_back(self, at: dt.datetime) -> SoilState:
        return SoilState(
            se=np.asarray(self.pde.rel_sat.value).copy(),
            se_old=np.asarray(self.pde.rel_sat._old.value).copy(),
            surface_h=dict(self.pde.surface_h),
            at=at,
        )

    def _face_weighted_mean(self, per_segment: dict[str, float], names: list[str]) -> float:
        total_len = 0.0
        weighted = 0.0
        for name in names:
            face_len = self.pde.segment_face_len.get(name, 0.0)
            if face_len <= 0:
                continue
            total_len += face_len
            weighted += per_segment.get(name, 0.0) * face_len
        if total_len <= 0:
            return 0.0
        return weighted / total_len

    def _balance_drainage_flux(self, rates: FluxRates, delta_storage: float, duration_s: float) -> float:
        bottom_len = self.pde.segment_face_len.get("GroundBottomSegment", 0.0)
        if bottom_len <= 0 or duration_s <= 0:
            return 0.0
        evap_mass = sum(value * self.pde.segment_face_len.get(name, 0.0) for name, value in rates.seg_evap.items())
        transp_mass = sum(value * self.pde.segment_face_len.get(name, 0.0) for name, value in rates.seg_transp.items())
        in_rate = rates.rain_flux * self.pde.rain_face_len + rates.flow_m3s * RHO_W
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
        watering_len = self.pde.segment_face_len.get("WateringTopSegment", 0.0)
        irr_flux = rates.flow_m3s * RHO_W / watering_len if watering_len > 0 else 0.0
        top_in = irr_flux + rates.rain_flux

        e_flux_mean = self._face_weighted_mean(rates.seg_evap, self.pde.top_segment_names)
        t_flux_mean = self._face_weighted_mean(rates.seg_transp, self.pde.top_segment_names)
        bottom = self._balance_drainage_flux(rates, delta_storage, elapsed_s)
        direct_bottom = self.pde.bottom_drainage_estimate()
        balance_residual = bottom - direct_bottom

        top_face_len = self.pde.segment_face_len.get("WateringTopSegment", 0.0) + sum(
            self.pde.segment_face_len.get(n, 0.0) for n in self.pde.top_segment_names
        )
        evap_face_len = sum(self.pde.segment_face_len.get(n, 0.0) for n in self.pde.top_segment_names)
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
