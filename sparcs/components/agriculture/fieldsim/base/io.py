# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.fieldsim.base.io
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The one protocol between the tick sequence and the outside world. Eight
methods, one channel-backed implementation in ``components.ChannelIO``, and a
thirty-line fake in tests.
"""

from __future__ import annotations

import datetime as dt
from typing import Mapping, Protocol

import pandas as pd

from ..core.state import Plan, SoilState, StepResult


class FieldIO(Protocol):
    def read_weather(self, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
        """Ranged connector read (not a logger read), trimmed to the span."""
        ...

    def read_irrigation_lpm(self, start: dt.datetime, end: dt.datetime) -> pd.Series:
        """Measured meter flow, else valve state times design flow, else zeros."""
        ...

    def read_tension_history(self, start: dt.datetime, end: dt.datetime) -> Mapping[str, pd.Series]:
        """Per anchor sensor, the measured tension series in the span."""
        ...

    def read_forecast(self, creation: dt.datetime | None) -> pd.DataFrame:
        """Weather forecast frame the planner rolls against."""
        ...

    def load_state(self) -> SoilState | None:
        """Last persisted state, or None on a cold start."""
        ...

    def save_state(self, state: SoilState) -> None: ...

    def publish(self, now: dt.datetime, result: StepResult, probe_tension: Mapping[str, float]) -> None:
        """Mass-balance, walk and probe channels for one advanced row."""
        ...

    def write_plan(self, plan: Plan) -> None:
        """Forecast header / detail / irrigation / image tables, best effort."""
        ...
