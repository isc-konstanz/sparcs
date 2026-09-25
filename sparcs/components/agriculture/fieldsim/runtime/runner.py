# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.fieldsim.runtime.runner
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Live tick policy: frontier and intake delay, midnight-aligned catch-up
chunks, reads through ``Inputs``, writes through ``Outputs``, the daily
planner gate. The per-row sequence lives in ``core.simulation``.
"""

from __future__ import annotations

import datetime as dt
import logging
import threading
from typing import Iterator, Mapping

import pandas as pd

from ..core.config import FieldSetup
from ..core.engine import Cancel
from ..core.simulation import Simulation
from ..core.state import SoilState, StepResult
from .ports import InputKey, Inputs, Outputs

logger = logging.getLogger(__name__)

WEATHER_STALL_ERROR_TICKS = 3


class FieldRunner:
    def __init__(self, setup: FieldSetup, simulation: Simulation, inputs: Inputs, outputs: Outputs) -> None:
        self.setup = setup
        self.simulation = simulation
        self.inputs = inputs
        self.outputs = outputs
        self.weather_stall_ticks: float = 0.0
        self.tick_failures: float = 0.0
        self._resumed = False
        self._last_planned: dt.date | None = None
        self._last_plan_warned: dt.date | None = None
        self._pending_lock = threading.Lock()
        self._pending_state: SoilState | None = None

    def restore(self, state: SoilState) -> None:
        """Queue a state for the next tick; called from the listener thread."""
        with self._pending_lock:
            self._pending_state = state

    def run_tick(self, now: dt.datetime, cancel: Cancel = None) -> bool:
        """Advance the simulation from its frontier up to ``now - intake_delay``.

        Returns True when at least one row was processed.
        """
        cutoff = now - self.setup.field.intake_delay
        if not self._resumed:
            state = self.inputs.load_state()
            if state is not None:
                self.simulation.resume(state)
            self._resumed = True
        self._apply_pending()

        frontier = self.simulation.state.at if self.simulation.state is not None else None
        if frontier is not None and frontier.tzinfo is not None:
            frontier = pd.Timestamp(frontier).tz_convert(cutoff.tzinfo)
        start = frontier if frontier is not None else cutoff - self.setup.field.interval_td
        if start >= cutoff:
            return False

        extra = {"weather_stall": float(self.weather_stall_ticks), "tick_failures": float(self.tick_failures)}
        processed = False
        for chunk_start, chunk_end in self._day_chunks(start, cutoff):
            weather = self.inputs.read(InputKey.WEATHER, chunk_start, chunk_end)
            if weather.empty:
                continue
            irrigation = self._irrigation_lpm(chunk_start, chunk_end, weather.index)
            tension = self._tension_history(chunk_start, chunk_end)
            results, chain = self.simulation.run(weather, irrigation, tension, cancel, extra_diagnostics=extra)
            self.outputs.chain(chunk_end, chain)
            for result in results:
                self._write(result)
                processed = True
            if cancel is not None and cancel():
                break

        if processed:
            self.weather_stall_ticks = 0.0
        else:
            self.weather_stall_ticks += 1.0
            logger.warning(
                "weather stall %s..%s: every chunk empty, the frontier has not advanced (consecutive stalls=%d)",
                start,
                cutoff,
                int(self.weather_stall_ticks),
            )
            if int(self.weather_stall_ticks) == WEATHER_STALL_ERROR_TICKS:
                logger.error("weather feed stalled for %d consecutive ticks", WEATHER_STALL_ERROR_TICKS)
            return False

        if self._planner_due(now):
            forecast = self.inputs.read(InputKey.FORECAST, now, now + self.setup.planner.horizon)
            plan = self.simulation.plan(forecast)
            if plan is None:
                if self._last_plan_warned != now.date():
                    logger.warning("planner skipped: empty forecast or no state")
                    self._last_plan_warned = now.date()
            else:
                self.outputs.plan(plan)
                self._last_planned = now.date()
        return processed

    def _apply_pending(self) -> None:
        with self._pending_lock:
            state, self._pending_state = self._pending_state, None
        if state is None:
            return
        current = self.simulation.state
        if current is None or state.at > current.at:
            self.simulation.resume(state)
        else:
            logger.info("restored state at %s is not newer than the frontier %s; dropped", state.at, current.at)

    def _write(self, result: StepResult) -> None:
        try:
            self.outputs.step(result)
            self.outputs.save_state(result.state)
        except Exception:
            logger.exception("outputs failed for the row at %s; continuing", result.state.at)

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
        return self.setup.planner is not None and self._last_planned != now.date()

    @staticmethod
    def _day_chunks(start: dt.datetime, end: dt.datetime) -> Iterator[tuple[pd.Timestamp, pd.Timestamp]]:
        """Split a catch-up span into ``(start, end]`` chunks broken at midnight."""
        cursor = pd.Timestamp(start)
        stop = pd.Timestamp(end)
        while cursor < stop:
            nxt = min((cursor + pd.Timedelta(days=1)).normalize(), stop)
            yield cursor, nxt
            cursor = nxt
