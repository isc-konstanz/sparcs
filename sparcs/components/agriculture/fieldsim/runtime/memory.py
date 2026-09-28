# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.fieldsim.runtime.memory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In-memory implementations of the two ports, used by ``ScenarioRunner``,
tests, notebooks and tuning campaigns.
"""

from __future__ import annotations

import datetime as dt
from typing import Optional, Sequence

import pandas as pd

from ..core.state import ChainResult, Plan, SoilState, StepResult
from .ports import IRRIGATION_LOOKBACK


class FrameInputs:
    """``Inputs`` over in-memory frames indexed by timestamp, sliced with the
    bounds the ``Inputs`` protocol documents. A missing frame reads as empty."""

    def __init__(
        self,
        *,
        weather: Optional[pd.DataFrame] = None,
        irrigation: Optional[pd.Series] = None,
        tension: Optional[pd.DataFrame] = None,
        forecast: Optional[pd.DataFrame] = None,
        state: Optional[SoilState] = None,
    ) -> None:
        self._weather = weather
        self._irrigation = irrigation
        self._tension = tension
        self._forecast = forecast
        self.state = state

    def weather(self, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
        if self._weather is None or self._weather.empty:
            return pd.DataFrame()
        return self._weather.loc[(self._weather.index > start) & (self._weather.index <= end)]

    def irrigation(self, start: dt.datetime, end: dt.datetime) -> pd.Series:
        if self._irrigation is None or self._irrigation.empty:
            return pd.Series(dtype=float)
        index = self._irrigation.index
        return self._irrigation.loc[(index > start - IRRIGATION_LOOKBACK) & (index <= end)]

    def tension(self, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
        if self._tension is None or self._tension.empty:
            return pd.DataFrame()
        return self._tension.loc[(self._tension.index > start) & (self._tension.index <= end)]

    def forecast(self, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
        if self._forecast is None or self._forecast.empty:
            return pd.DataFrame()
        return self._forecast.loc[(self._forecast.index >= start) & (self._forecast.index <= end)]

    def load_state(self) -> SoilState | None:
        return self.state


class Recorder:
    """``Outputs`` that keeps everything and hands it back as frames."""

    def __init__(self) -> None:
        self.chains: list[tuple[dt.datetime, ChainResult]] = []
        self.rows: list[StepResult] = []
        self.plans: list[Plan] = []
        self.saved: list[SoilState] = []

    def chain(self, now: dt.datetime, result: ChainResult) -> None:
        self.chains.append((now, result))

    def steps(self, results: Sequence[StepResult]) -> None:
        self.rows.extend(results)

    def plan(self, plan: Plan) -> None:
        self.plans.append(plan)

    def save_state(self, state: SoilState) -> None:
        self.saved.append(state)

    def to_frames(self) -> dict[str, pd.DataFrame]:
        """``diagnostics`` and ``tension`` indexed by state time, plus the
        concatenated ``shading`` and ``evapotranspiration`` chain outputs."""
        index = pd.DatetimeIndex([s.state.at for s in self.rows])
        frames = {
            "diagnostics": pd.DataFrame([dict(s.diagnostics) for s in self.rows], index=index),
            "tension": pd.DataFrame([dict(s.probe_tension) for s in self.rows], index=index),
        }
        if self.chains:
            frames["shading"] = pd.concat([c.shading for _, c in self.chains])
            frames["evapotranspiration"] = pd.concat([c.evapotranspiration for _, c in self.chains])
        return frames
