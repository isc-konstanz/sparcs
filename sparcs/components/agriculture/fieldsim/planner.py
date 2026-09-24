# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.fieldsim.planner
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Irrigation planning over a forecast horizon. Receives the shared engine and a
state snapshot; never reaches into the live simulation or replays the chain
through a parent component.

Candidate enumeration and scoring stay module functions
(``simulation._predictor_candidates``). Ladder versus parallel rollout is a
flag on ``PlannerConfig``, not a strategy hierarchy; both produce
``{candidate: trajectory}`` and the rest of ``plan`` does not care which ran.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .config import PlannerConfig
from .engine import SoilEngine
from .state import Forcing, Plan, SoilState

Candidates = Sequence[Any]
Trajectories = Mapping[Any, Any]


class IrrigationPlanner:
    def __init__(self, config: PlannerConfig, engine: SoilEngine) -> None:
        self.config = config
        self.engine = engine

    def plan(self, state: SoilState, horizon: Sequence[Forcing]) -> Plan:
        """
        1. zero-flow baseline: ``rollout([no-irrigation])``
        2. candidates from ``config.windows`` x ``config.durations_min``
        3. ``rollout(candidates)`` ladder or parallel
        4. score each trajectory against ``config.threshold_hpa`` at the
           decision probes, argmin, tie-break on total minutes
        5. build header / detail / irrigation frames for the sinks
        Any failure after step 1 degrades to the zero-flow plan.
        """
        raise NotImplementedError

    def rollout(self, state: SoilState, horizon: Sequence[Forcing], candidates: Candidates) -> Trajectories:
        """Ladder (shared-prefix caterpillar) or parallel (process pool,
        workers rebuild an engine from the pickled ``SoilConfig``)."""
        if self.config.parallel:
            return self._rollout_parallel(state, horizon, candidates)
        return self._rollout_ladder(state, horizon, candidates)

    def _rollout_ladder(self, state: SoilState, horizon: Sequence[Forcing], candidates: Candidates) -> Trajectories:
        raise NotImplementedError

    def _rollout_parallel(self, state: SoilState, horizon: Sequence[Forcing], candidates: Candidates) -> Trajectories:
        raise NotImplementedError
