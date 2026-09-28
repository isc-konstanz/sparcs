# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.fieldsim.core.state
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Immutable, FiPy-free, picklable value objects that cross the core seams.
"""

from __future__ import annotations

import datetime as dt
import io
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import pandas as pd


def encode_state_blob(se: np.ndarray, se_old: np.ndarray, surface_h: Mapping[str, float]) -> bytes:
    """The ``simulation_state`` npz wire format: ``rel_sat``, ``rel_sat_old`` and the pond
    depths as ``surface_names``/``surface_h``, all pickle-free."""
    buf = io.BytesIO()
    names = np.array(list(surface_h.keys()), dtype=np.str_)
    values = np.array([surface_h[k] for k in names], dtype=float)
    np.savez(
        buf,
        rel_sat=np.asarray(se).copy(),
        rel_sat_old=np.asarray(se_old).copy(),
        surface_names=names,
        surface_h=values,
    )
    return buf.getvalue()


def decode_state_blob(blob: bytes) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Inverse of ``encode_state_blob``; a legacy blob carrying only ``rel_sat`` decodes with
    ``rel_sat_old = rel_sat`` and no ponds. Raises ``ValueError`` for a blob that needs pickle."""
    arrays = np.load(io.BytesIO(blob), allow_pickle=False)
    se = np.asarray(arrays["rel_sat"], dtype=float)
    se_old = np.asarray(arrays["rel_sat_old"], dtype=float) if "rel_sat_old" in arrays.files else se.copy()
    surface_h: dict[str, float] = {}
    if "surface_names" in arrays.files and "surface_h" in arrays.files:
        surface_h = {str(n): float(v) for n, v in zip(arrays["surface_names"], arrays["surface_h"])}
    return se, se_old, surface_h


@dataclass(frozen=True)
class SoilState:
    """Everything needed to resume the PDE exactly, plus the instant it is valid for."""

    se: np.ndarray
    se_old: np.ndarray
    surface_h: Mapping[str, float]  # pond depth [m] per open-sky segment + "WateringTopSegment"
    at: dt.datetime

    @property
    def ponding_m(self) -> float:
        return float(sum(self.surface_h.values())) if self.surface_h else 0.0

    def to_blob(self) -> bytes:
        return encode_state_blob(self.se, self.se_old, self.surface_h)

    @classmethod
    def from_blob(cls, blob: bytes, at: dt.datetime) -> SoilState:
        se, se_old, surface_h = decode_state_blob(blob)
        return cls(se=se, se_old=se_old, surface_h=surface_h, at=at)


@dataclass(frozen=True)
class Forcing:
    """Surface forcing held constant over the window ``(at - dt_s, at]``."""

    at: dt.datetime
    dt_s: float
    rain_flux: float = 0.0  # kg/(m^2 s)
    flow_m3s: float = 0.0  # m^3/s per metre of row
    seg_evap: Mapping[str, float] = field(default_factory=dict)  # kg/(m^2 s) per top segment
    seg_transp: Mapping[str, float] = field(default_factory=dict)  # kg/(m^2 s) per top segment

    @property
    def start(self) -> dt.datetime:
        return self.at - dt.timedelta(seconds=self.dt_s)

    @property
    def end(self) -> dt.datetime:
        return self.at


@dataclass(frozen=True)
class ChainResult:
    """Outputs of one ``WeatherChain.forcing_series`` call."""

    shading: pd.DataFrame
    evapotranspiration: pd.DataFrame
    image: bytes | None = None


@dataclass(frozen=True)
class StepResult:
    """Outcome of one advance, completed with the assimilated state and probe tensions."""

    state: SoilState
    diagnostics: Mapping[str, float]
    probe_tension: Mapping[str, float] = field(default_factory=dict)  # signed negative hPa
    cancelled: bool = False


@dataclass(frozen=True)
class Plan:
    """Outcome of ``IrrigationPlanner.plan``: the chosen candidate and the forecast frames."""

    chosen: Any
    trajectories: Mapping[Any, pd.DataFrame]
    header: pd.DataFrame
    detail: pd.DataFrame
    irrigation: pd.DataFrame
    image: pd.DataFrame | None = None


@dataclass(frozen=True)
class Snapshot:
    """What the ``Simulation`` knows right now; the read model for dash and tests."""

    state: SoilState | None
    last_chain: ChainResult | None = None
    last_step: StepResult | None = None
    last_plan: Plan | None = None

    @property
    def frontier(self) -> dt.datetime | None:
        return self.state.at if self.state is not None else None
