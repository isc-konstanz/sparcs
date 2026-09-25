# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.fieldsim.runtime.ports
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The two protocols between the live runner and the outside world, split by
direction. ``Inputs`` is one keyed ranged read plus the persisted state;
``Outputs`` is one method per result type. Adding an input is a new key, not
a new method. Dash does not read through ``Outputs``; it reads the
``Snapshot`` the ``Simulation`` keeps.
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
        """Rows in ``[start, end)``; empty frame when nothing is available."""
        ...

    def load_state(self) -> SoilState | None:
        """Last persisted state, or None on a cold start. Called once, on
        the first tick; there is no listener."""
        ...


class Outputs(Protocol):
    def chain(self, now: dt.datetime, result: ChainResult) -> None:
        """Shading and ET outputs for one weather chunk. The sink owns the
        cadence: channels fan out per row, a recorder keeps the frame."""
        ...

    def step(self, result: StepResult) -> None:
        """Mass-balance, walk and probe outputs for one advanced row."""
        ...

    def plan(self, plan: Plan) -> None:
        """Forecast header / detail / irrigation / image tables, best effort."""
        ...

    def save_state(self, state: SoilState) -> None: ...
