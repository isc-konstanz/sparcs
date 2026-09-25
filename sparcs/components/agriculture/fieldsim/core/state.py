# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.fieldsim.core.state
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Value objects that cross the seams: ``SoilState`` in and out of the engine,
``Forcing`` and ``ChainResult`` out of the weather chain, ``StepResult`` out of
one advance, ``Plan`` out of the planner, ``Snapshot`` as the read model the
``Simulation`` keeps for dash and tests. All immutable, all FiPy-free, all
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
class ChainResult:
    """Outputs of one ``WeatherChain.forcing_series`` call, published once
    per weather chunk: shading factors and open-sky irradiance per row, the
    ET intermediates and per-segment ET per row, and the shading progress
    image when one was due."""

    shading: pd.DataFrame
    evapotranspiration: pd.DataFrame
    image: bytes | None = None


@dataclass(frozen=True)
class StepResult:
    """Outcome of ``SoilEngine.advance`` for one ``Forcing``, completed by
    ``Simulation.run`` with the assimilated state and the probe tensions."""

    state: SoilState
    diagnostics: Mapping[str, float]  # water_total, delta, walk_substeps, skipped_s, ...
    probe_tension: Mapping[str, float] = field(default_factory=dict)  # signed negative hPa
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


@dataclass(frozen=True)
class Snapshot:
    """What the ``Simulation`` knows right now. Dash and tests read this;
    nothing reads back through the output channels."""

    state: SoilState | None
    last_chain: ChainResult | None = None
    last_step: StepResult | None = None
    last_plan: Plan | None = None

    @property
    def frontier(self) -> dt.datetime | None:
        return self.state.at if self.state is not None else None
