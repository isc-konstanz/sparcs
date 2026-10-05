# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.simulation.runtime.ports
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Protocols between the runner and the outside world: ``Inputs`` is four ranged reads plus the persisted state.
``Outputs`` has one method per result: ``chain``, ``steps`` and ``save_state`` per weather chunk, ``plan`` per slot.
"""

from __future__ import annotations

import datetime as dt
from typing import Protocol, Sequence

import pandas as pd

from ..core.state import ChainResult, Plan, SoilState, StepResult

# How far before a chunk the irrigation read reaches, so the sample in force at
# the chunk start is part of the series.
IRRIGATION_LOOKBACK = dt.timedelta(days=1)


class Inputs(Protocol):
    def weather(self, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
        """Weather rows in ``(start, end]``; empty when nothing usable is available."""
        ...

    def irrigation(self, start: dt.datetime, end: dt.datetime) -> pd.Series:
        """Raw flow samples [l/min] in ``(start - IRRIGATION_LOOKBACK, end]``, not aligned to the weather.
        Metered flow, else valve state times design flow, else an empty series (not watering)."""
        ...

    def tension(self, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
        """Measured tension [hPa], one column per anchor sensor."""
        ...

    def forecast(self, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
        """Weather forecast rows in ``[start, end]``."""
        ...

    def load_state(self) -> SoilState | None:
        """Last persisted state, or None on a cold start; ``ValueError`` when it cannot be decoded."""
        ...


class Outputs(Protocol):
    def chain(self, now: dt.datetime, result: ChainResult) -> None:
        """Shading and ET outputs for one weather chunk."""
        ...

    def steps(self, results: Sequence[StepResult]) -> None:
        """Mass-balance, walk and probe outputs for the rows one chunk advanced."""
        ...

    def plan(self, plan: Plan) -> None:
        """Forecast header / detail / irrigation / image tables."""
        ...

    def save_state(self, state: SoilState) -> None:
        """The state after the last row of a chunk."""
        ...
