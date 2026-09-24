# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.fieldsim.core.engine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The one soil engine, built once from ``SoilConfig`` and shared by the live
tick and the planner. Replaces the ``SoilBase`` inheritance that today makes
``SoilSimulation`` and ``SoilPredictor`` each own PDE plumbing.

Owns the FiPy mesh, the soil model and the equation (today's ``SoilPDECore``
plus the thin accessors on ``SoilBase``). Stateless with respect to time:
every call takes a ``SoilState`` and returns a new one, so the planner can
roll candidates off a snapshot without the engine knowing.
"""

from __future__ import annotations

from typing import Any, Mapping

from .config import SoilConfig
from .state import Forcing, SoilState, StepResult


class SoilEngine:
    def __init__(self, config: SoilConfig, model: Any, mesh: Any, pde: Any) -> None:
        self.config = config
        self.model = model  # soil.models.SoilModel (van Genuchten)
        self.mesh = mesh  # FiPy mesh from simulation._soil.create_mesh
        self.pde = pde  # simulation._soil.SoilPDECore, the numerics

    @classmethod
    def build(cls, config: SoilConfig) -> SoilEngine:
        """Create mesh, model and equation. The only place FiPy objects are
        constructed for the live chain; planner workers call it too, from the
        pickled config."""
        raise NotImplementedError

    def initial_state(self, at: Any) -> SoilState:
        """Hydrostatic initial condition from ``config.mesh`` / ``config.pde``
        (today ``_hydrostatic_ic_array``)."""
        raise NotImplementedError

    def advance(self, state: SoilState, forcing: Forcing, *, cancel: Any = None) -> StepResult:
        """Walk one forcing window: apply source, solve with sub-stepping and
        rollback (today ``SoilPDECore.walk_window``), commit ponding, return
        the new state plus mass-balance diagnostics. Never publishes."""
        raise NotImplementedError

    def tension_at(self, state: SoilState, probe: Any) -> float:
        """Sample Se at a probe and convert to signed negative hPa."""
        raise NotImplementedError

    def diagnostics(self, state: SoilState) -> Mapping[str, float]:
        """Total water, surface water, per-layer summaries
        (today ``SoilBase._compute_diagnostics``)."""
        raise NotImplementedError
