# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.fieldsim.runtime.runner
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Live policy only: frontier and intake delay, day-sized catch-up chunks,
reads through ``Inputs``, writes through ``Outputs``, the daily planner
gate. The per-row sequence lives in ``core.simulation``. No thread, no
lories. ``ScenarioRunner`` drives the identical ``run_tick`` with a synthetic
clock, so catch-up and chunking are exercised by every scenario run.
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Iterator, Mapping

import pandas as pd

from ..core.config import FieldConfig
from ..core.engine import Cancel
from ..core.simulation import Simulation
from .ports import InputKey, Inputs, Outputs

logger = logging.getLogger(__name__)


class FieldRunner:
    def __init__(self, config: FieldConfig, simulation: Simulation, inputs: Inputs, outputs: Outputs) -> None:
        self.config = config
        self.simulation = simulation
        self.inputs = inputs
        self.outputs = outputs
        self._resumed = False
        self._last_planned: dt.date | None = None

    def run_tick(self, now: dt.datetime, cancel: Cancel = None) -> bool:
        """Advance the simulation from its frontier up to ``now - intake_delay``.

        Returns True when at least one row was processed. Stall accounting
        is the scheduler's job; it only needs this bool.
        """
        cutoff = now - self.config.intake_delay
        if not self._resumed:
            state = self.inputs.load_state()
            if state is not None:
                self.simulation.resume(state)
            self._resumed = True
        frontier = self.simulation.state.at if self.simulation.state is not None else cutoff - self.config.interval
        if frontier >= cutoff:
            return False

        processed = False
        for chunk_start, chunk_end in self._day_chunks(frontier, cutoff):
            weather = self.inputs.read(InputKey.WEATHER, chunk_start, chunk_end)
            if weather.empty:
                logger.warning("no valid weather for %s..%s, skipping chunk", chunk_start, chunk_end)
                continue
            irrigation = self._irrigation_lpm(chunk_start, chunk_end, weather.index)
            tension = self._tension_history(chunk_start, chunk_end)
            results, chain = self.simulation.run(weather, irrigation, tension, cancel=cancel)
            self.outputs.chain(chunk_end, chain)
            for result in results:
                self.outputs.step(result)
                self.outputs.save_state(result.state)
                processed = True
            if cancel is not None and cancel():
                return processed

        if processed and self._planner_due(now):
            forecast = self.inputs.read(InputKey.FORECAST, now, now + self.config.planner.horizon)
            plan = self.simulation.plan(forecast)
            if plan is None:
                logger.warning("planner skipped: empty forecast or no state")
            else:
                self.outputs.plan(plan)
                self._last_planned = now.date()
        return processed

    def _irrigation_lpm(self, start: dt.datetime, end: dt.datetime, index: pd.Index) -> pd.Series:
        frame = self.inputs.read(InputKey.IRRIGATION, start, end)
        if frame.empty:
            return pd.Series(0.0, index=index)  # rain-fed field, or no usable input
        return frame.iloc[:, 0]

    def _tension_history(self, start: dt.datetime, end: dt.datetime) -> Mapping[str, pd.Series]:
        if not self.simulation.assimilator.enabled:
            return {}
        frame = self.inputs.read(InputKey.TENSION, start, end)
        return {str(c): frame[c].dropna() for c in frame.columns}

    def _planner_due(self, now: dt.datetime) -> bool:
        """Daily boundary gate with dedup (today ``SoilPredictor._gate_boundary``)."""
        return self.config.planner is not None and self._last_planned != now.date()

    @staticmethod
    def _day_chunks(start: dt.datetime, end: dt.datetime) -> Iterator[tuple[dt.datetime, dt.datetime]]:
        """Split a catch-up span into day-sized reads (today ``_iter_day_chunks``)."""
        cursor = start
        while cursor < end:
            nxt = min(cursor + dt.timedelta(days=1), end)
            yield cursor, nxt
            cursor = nxt
