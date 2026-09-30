# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.simulation.runtime.runner
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Live tick policy: frontier and intake delay, midnight-aligned catch-up
chunks, reads through ``Inputs``, one write per output per chunk through
``Outputs``, the planner gate on the ``[soil_predictor]`` interval/offset.
The per-row sequence lives in ``core.simulation``.
"""

from __future__ import annotations

import datetime as dt
import logging
import threading
from typing import Iterator, Mapping, Optional, Sequence

import pandas as pd
import pytz

from ..core.config import FieldSetup
from ..core.engine import Cancel
from ..core.schedule import slot_floor
from ..core.simulation import Simulation
from ..core.state import ChainResult, SoilState, StepResult
from .ports import Inputs, Outputs

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
        self._last_planned: Optional[pd.Timestamp] = None
        self._last_plan_warned: Optional[pd.Timestamp] = None
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
            self._resumed = True
            self._resume_persisted()
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
            weather = self.inputs.weather(chunk_start, chunk_end)
            if not weather.empty:
                irrigation = self._align_flow(self.inputs.irrigation(chunk_start, chunk_end), weather.index)
                tension = self._tension_history(chunk_start, chunk_end)
                results, chain = self.simulation.run(weather, irrigation, tension, cancel, extra_diagnostics=extra)
                self._publish(chunk_end, chain, results)
                if results:
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

        if self.setup.planner is not None:
            self._plan(now)
        return processed

    def _resume_persisted(self) -> None:
        try:
            state = self.inputs.load_state()
            if state is not None:
                self.simulation.resume(state)
        except ValueError as e:
            logger.warning("persisted soil state is incompatible, cold start instead: %s", e)

    def _apply_pending(self) -> None:
        with self._pending_lock:
            state, self._pending_state = self._pending_state, None
        if state is None:
            return
        current = self.simulation.state
        if current is not None and state.at <= current.at:
            logger.info("restored state at %s is not newer than the frontier %s; dropped", state.at, current.at)
            return
        try:
            self.simulation.resume(state)
        except ValueError as e:
            logger.warning("restored state at %s is incompatible; dropped: %s", state.at, e)

    def _publish(self, end: pd.Timestamp, chain: ChainResult, results: Sequence[StepResult]) -> None:
        """Hand one chunk to the outputs; a failing output never stops the frontier."""
        writes = [(self.outputs.chain, (end, chain))]
        if results:
            writes += [(self.outputs.steps, (results,)), (self.outputs.save_state, (results[-1].state,))]
        for write, args in writes:
            try:
                write(*args)
            except Exception:
                logger.exception("%s output for the chunk up to %s failed; continuing", write.__name__, end)

    @staticmethod
    def _align_flow(flow: pd.Series, index: pd.Index) -> pd.Series:
        """The flow in force at each weather row: the last sample at or before it, 0.0
        where there is none or it is NaN."""
        if flow.empty:
            return pd.Series(0.0, index=index)
        flow = flow.sort_index()
        flow = flow[~flow.index.duplicated(keep="last")]
        return flow.reindex(index, method="ffill").fillna(0.0).astype(float)

    def _tension_history(self, start: dt.datetime, end: dt.datetime) -> Mapping[str, pd.Series]:
        if not self.simulation.assimilator.enabled:
            return {}
        frame = self.inputs.tension(start, end)
        return {str(c): frame[c].dropna() for c in frame.columns}

    def _plan(self, now: dt.datetime) -> None:
        """Plan once per ``[soil_predictor]`` interval/offset slot the frontier is in,
        aligned in the site timezone."""
        planner = self.setup.planner
        location = self.setup.location
        boundary = slot_floor(
            pd.Timestamp(self.simulation.state.at),
            location.timezone if location is not None else pytz.UTC,
            planner.interval,
            planner.offset,
        )
        if boundary == self._last_planned:
            return
        plan = self.simulation.plan(self.inputs.forecast(now, now + planner.horizon))
        if plan is None:
            if self._last_plan_warned != boundary:
                logger.warning("planner skipped: empty forecast or no state")
                self._last_plan_warned = boundary
            return
        self.outputs.plan(plan)
        self._last_planned = boundary

    @staticmethod
    def _day_chunks(start: dt.datetime, end: dt.datetime) -> Iterator[tuple[pd.Timestamp, pd.Timestamp]]:
        """Split a catch-up span into ``(start, end]`` chunks broken at midnight in the span's zone."""
        cursor = pd.Timestamp(start)
        stop = pd.Timestamp(end)
        while cursor < stop:
            nxt = min(cursor.normalize() + pd.DateOffset(days=1), stop)
            yield cursor, nxt
            cursor = nxt
