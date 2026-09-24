# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.fieldsim.base.runner
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The whole tick sequence in one place, driven by a clock value and talking to
one ``FieldIO``. No thread, no lories, nothing to monkeypatch: tests build a
``FieldRunner`` with a fake ``FieldIO`` and call ``run_tick``.

This is the only module in the skeleton whose body is real control flow, so
the sequence can be read as code next to ``sequences.md`` diagram 2.
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Iterator

import pandas as pd

from ..core.assimilator import Assimilator
from ..core.chain import WeatherChain
from ..core.config import FieldConfig
from ..core.engine import SoilEngine
from ..core.planner import IrrigationPlanner
from ..core.state import SoilState
from .io import FieldIO

logger = logging.getLogger(__name__)


class FieldRunner:
    def __init__(
        self,
        config: FieldConfig,
        io: FieldIO,
        engine: SoilEngine,
        chain: WeatherChain,
        assimilator: Assimilator,
        planner: IrrigationPlanner | None = None,
    ) -> None:
        self.config = config
        self.io = io
        self.engine = engine
        self.chain = chain
        self.assimilator = assimilator
        self.planner = planner
        self.state: SoilState | None = None
        self._last_planned: dt.date | None = None

    def run_tick(self, now: dt.datetime) -> bool:
        """Advance the simulation from its frontier up to ``now - intake_delay``.

        Returns True when at least one row was processed. Stall accounting
        (warn per stalled tick, error at the third) is the scheduler's job;
        it only needs this bool.
        """
        cutoff = now - self.config.intake_delay
        if self.state is None:
            self.state = self.io.load_state()
        frontier = self.state.at if self.state is not None else cutoff - self.config.interval
        if frontier >= cutoff:
            return False

        processed = False
        for chunk_start, chunk_end in self._day_chunks(frontier, cutoff):
            weather = self.io.read_weather(chunk_start, chunk_end)
            if weather.empty:
                logger.warning("no valid weather for %s..%s, skipping chunk", chunk_start, chunk_end)
                continue
            irrigation = self.io.read_irrigation_lpm(chunk_start, chunk_end)
            forcing = self.chain.forcing_series(weather, irrigation)
            if self.assimilator.enabled:
                self.assimilator.ingest(self.io.read_tension_history(chunk_start, chunk_end))
            for step in forcing:
                if self.state is None:
                    self.state = self.engine.initial_state(step.start)
                result = self.engine.advance(self.state, step)
                if result.cancelled:
                    return processed
                state = result.state
                if self.assimilator.enabled:
                    state = self.assimilator.update(state, step.end)
                tension = {p.key: self.engine.tension_at(state, p) for p in self.config.soil.probes}
                self.io.publish(step.end, result, tension)
                self.io.save_state(state)
                self.state = state
                processed = True

        if processed and self.planner is not None and self._planner_due(now):
            self._plan(now)
        return processed

    def _planner_due(self, now: dt.datetime) -> bool:
        """Daily boundary gate with dedup (today ``SoilPredictor._gate_boundary``)."""
        return self._last_planned != now.date()

    def _plan(self, now: dt.datetime) -> None:
        assert self.planner is not None and self.state is not None
        forecast = self.io.read_forecast(None)
        if forecast.empty:
            logger.warning("empty forecast, planner skipped")
            return
        horizon = self.chain.forcing_series(forecast, pd.Series(0.0, index=forecast.index))
        plan = self.planner.plan(self.state, horizon)
        self.io.write_plan(plan)
        self._last_planned = now.date()

    @staticmethod
    def _day_chunks(start: dt.datetime, end: dt.datetime) -> Iterator[tuple[dt.datetime, dt.datetime]]:
        """Split a catch-up span into day-sized reads (today ``_iter_day_chunks``)."""
        cursor = start
        while cursor < end:
            nxt = min(cursor + dt.timedelta(days=1), end)
            yield cursor, nxt
            cursor = nxt
