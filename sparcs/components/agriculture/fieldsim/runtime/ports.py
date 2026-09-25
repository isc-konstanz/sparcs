# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.fieldsim.runtime.ports
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The two protocols between the runner and the outside world: ``Inputs`` is
one keyed ranged read plus the persisted state, ``Outputs`` one method per
result type.
"""

from __future__ import annotations

import datetime as dt
import enum
from typing import Protocol

import pandas as pd

from ..core.state import ChainResult, Plan, SoilState, StepResult


class InputKey(enum.Enum):
    WEATHER = "weather"  # ranged connector read of the weather channels
    IRRIGATION = "irrigation"  # flow l/min: meter, else valve state x design flow, else empty
    TENSION = "tension"  # one column per anchor sensor, measured tension in hPa
    FORECAST = "forecast"  # weather forecast frame for the planner horizon


class Inputs(Protocol):
    def read(self, key: InputKey, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
        """Rows in ``(start, end]``; empty frame when nothing is available."""
        ...

    def load_state(self) -> SoilState | None:
        """Last persisted state, or None on a cold start."""
        ...


class Outputs(Protocol):
    def chain(self, now: dt.datetime, result: ChainResult) -> None:
        """Shading and ET outputs for one weather chunk."""
        ...

    def step(self, result: StepResult) -> None:
        """Mass-balance, walk and probe outputs for one advanced row."""
        ...

    def plan(self, plan: Plan) -> None:
        """Forecast header / detail / irrigation / image tables."""
        ...

    def save_state(self, state: SoilState) -> None: ...
