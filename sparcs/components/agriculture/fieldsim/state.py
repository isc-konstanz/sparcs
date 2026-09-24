# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.fieldsim.state
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Value objects that cross the seams: ``SoilState`` in and out of the engine,
``Forcing`` from the weather chain into the engine, ``StepResult`` out of one
advance, ``Plan`` out of the planner. All immutable, all FiPy-free, all
picklable.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SoilState:
    """Everything needed to resume the PDE: relative saturation per cell, the
    surface pond, and the instant the state is valid for."""

    se: np.ndarray
    ponding_m: float
    at: dt.datetime

    def to_blob(self) -> bytes:
        """Serialize for the ``simulation_state`` channel. Wire format is
        the one ``SoilPDECore.save_state_blob`` writes today, unchanged."""
        raise NotImplementedError

    @classmethod
    def from_blob(cls, blob: bytes) -> SoilState:
        raise NotImplementedError


@dataclass(frozen=True)
class Forcing:
    """Surface forcing for one step, per mesh segment name.

    Rates in m/s. ``irrigation`` and ``et`` are keyed by segment; rain is
    uniform. One ``Forcing`` per weather row; the chain produces the list.
    """

    start: dt.datetime
    dt_s: float
    rain_m_s: float
    irrigation_m_s: Mapping[str, float] = field(default_factory=dict)
    et_m_s: Mapping[str, float] = field(default_factory=dict)

    @property
    def end(self) -> dt.datetime:
        return self.start + dt.timedelta(seconds=self.dt_s)


@dataclass(frozen=True)
class StepResult:
    """Outcome of ``SoilEngine.advance`` for one ``Forcing``."""

    state: SoilState
    diagnostics: Mapping[str, float]  # water_total, delta, walk_substeps, skipped_s, ...
    cancelled: bool = False


@dataclass(frozen=True)
class Plan:
    """Outcome of ``IrrigationPlanner.plan``: the chosen candidate plus the
    frames the forecast tables are written from."""

    chosen: Any
    trajectories: Mapping[Any, pd.DataFrame]
    header: pd.DataFrame
    detail: pd.DataFrame
    irrigation: pd.DataFrame
    image: pd.DataFrame | None = None
