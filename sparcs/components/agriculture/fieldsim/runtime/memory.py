# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.fieldsim.runtime.memory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In-memory implementations of the two ports. Shipped adapters, not test
doubles: ``ScenarioRunner`` runs on them, and so do tests, notebooks and
tuning campaigns.
"""

from __future__ import annotations

import datetime as dt
from typing import Mapping, Optional

import pandas as pd

from ..core.state import ChainResult, Plan, SoilState, StepResult
from .ports import InputKey


class FrameInputs:
    """``Inputs`` over one DataFrame per key. Frames are indexed by
    timestamp; ``read`` slices ``[start, end)``."""

    def __init__(self, frames: Mapping[InputKey, pd.DataFrame], state: Optional[SoilState] = None) -> None:
        self.frames = dict(frames)
        self.state = state

    def read(self, key: InputKey, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
        frame = self.frames.get(key)
        if frame is None or frame.empty:
            return pd.DataFrame()
        return frame.loc[(frame.index >= start) & (frame.index < end)]

    def load_state(self) -> SoilState | None:
        return self.state


class Recorder:
    """``Outputs`` that keeps everything and hands it back as frames."""

    def __init__(self) -> None:
        self.chains: list[tuple[dt.datetime, ChainResult]] = []
        self.steps: list[StepResult] = []
        self.plans: list[Plan] = []
        self.saved: list[SoilState] = []

    def chain(self, now: dt.datetime, result: ChainResult) -> None:
        self.chains.append((now, result))

    def step(self, result: StepResult) -> None:
        self.steps.append(result)

    def plan(self, plan: Plan) -> None:
        self.plans.append(plan)

    def save_state(self, state: SoilState) -> None:
        self.saved.append(state)

    def to_frames(self) -> dict[str, pd.DataFrame]:
        """``diagnostics`` and ``tension`` indexed by state time, plus the
        concatenated ``shading`` and ``evapotranspiration`` chain outputs."""
        index = pd.DatetimeIndex([s.state.at for s in self.steps])
        frames = {
            "diagnostics": pd.DataFrame([dict(s.diagnostics) for s in self.steps], index=index),
            "tension": pd.DataFrame([dict(s.probe_tension) for s in self.steps], index=index),
        }
        if self.chains:
            frames["shading"] = pd.concat([c.shading for _, c in self.chains])
            frames["evapotranspiration"] = pd.concat([c.evapotranspiration for _, c in self.chains])
        return frames
