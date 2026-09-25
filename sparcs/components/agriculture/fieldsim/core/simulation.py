# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.fieldsim.core.simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The simulation session: the scenario and test API. Owns the current state
and the latest results, composes engine, chain, assimilator and planner, and
runs the per-chunk sequence. No I/O, no clock, no gating; those belong to
``runtime.runner``. Notebooks, tuning campaigns and dt benches call ``run``
directly with frames.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Mapping, Optional, Sequence

import pandas as pd

from .assimilator import Assimilator
from .chain import WeatherChain
from .config import FieldSetup
from .engine import Cancel, SoilEngine
from .planner import IrrigationPlanner
from .state import ChainResult, Plan, Snapshot, SoilState, StepResult

logger = logging.getLogger(__name__)


class Simulation:
    def __init__(
        self,
        setup: FieldSetup,
        engine: SoilEngine,
        chain: WeatherChain,
        assimilator: Assimilator,
        planner: Optional[IrrigationPlanner] = None,
    ) -> None:
        self.setup = setup
        self.engine = engine
        self.chain = chain
        self.assimilator = assimilator
        self.planner = planner
        self.state: Optional[SoilState] = None
        self._last_chain: Optional[ChainResult] = None
        self._last_step: Optional[StepResult] = None
        self._last_plan: Optional[Plan] = None

    def resume(self, state: SoilState) -> None:
        self.state = state

    def run(
        self,
        weather: pd.DataFrame,
        irrigation_lpm: pd.Series,
        tension_history: Optional[Mapping[str, pd.Series]] = None,
        cancel: Cancel = None,
    ) -> tuple[Sequence[StepResult], ChainResult]:
        """Advance over one weather chunk: chain once, then per row advance,
        assimilate, sample probes. Stops early on cancel; committed rows stay
        committed. Returns the completed steps and the chain outputs."""
        forcing, chain = self.chain.forcing_series(weather, irrigation_lpm)
        if tension_history and self.assimilator.enabled:
            self.assimilator.ingest(tension_history)
        probes = self.setup.soil.probe_specs
        results: list[StepResult] = []
        for step in forcing:
            if self.state is None:
                self.state = self.engine.initial_state(step.start)
            result = self.engine.advance(self.state, step, cancel=cancel)
            if result.cancelled:
                logger.info("advance cancelled at %s, %d rows committed", step.start, len(results))
                break
            state = result.state
            if self.assimilator.enabled:
                state = self.assimilator.update(state, step.end)
            tension = {p.key: self.engine.tension_at(state, p) for p in probes}
            result = replace(result, state=state, probe_tension=tension)
            self.state = state
            results.append(result)
        self._last_chain = chain
        if results:
            self._last_step = results[-1]
        return results, chain

    def plan(self, forecast: pd.DataFrame) -> Optional[Plan]:
        """Roll the planner over a forecast horizon from the current state.
        Returns None when there is no planner or no state yet."""
        if self.planner is None or self.state is None or forecast.empty:
            return None
        horizon, _ = self.chain.forcing_series(forecast, pd.Series(0.0, index=forecast.index))
        plan = self.planner.plan(self.state, horizon)
        self._last_plan = plan
        return plan

    def snapshot(self) -> Snapshot:
        return Snapshot(
            state=self.state,
            last_chain=self._last_chain,
            last_step=self._last_step,
            last_plan=self._last_plan,
        )
