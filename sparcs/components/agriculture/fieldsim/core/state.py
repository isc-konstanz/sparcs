# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.fieldsim.core.state
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Value objects that cross the seams: ``SoilState`` in and out of the engine,
``Forcing`` and ``ChainResult`` out of the weather chain, ``StepResult`` out of
one advance, ``Plan`` out of the planner, ``Snapshot`` as the read model the
``Simulation`` keeps for dash and tests. All immutable, all FiPy-free, all
picklable.

Units follow the live PDE core (``simulation._soil.FluxRates``): surface
fluxes in kg/(m^2 s), drip flow in m^3/s per metre of row, pond depths in m.
"""

from __future__ import annotations

import datetime as dt
import io
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SoilState:
    """Everything needed to resume the PDE exactly: the relative-saturation
    field, its previous-step copy, the pond depth per surface segment, and
    the instant the state is valid for. Mirrors what
    ``SoilPDECore.save_state_blob`` persists."""

    se: np.ndarray  # rel_sat.value
    se_old: np.ndarray  # rel_sat._old.value
    surface_h: Mapping[str, float]  # pond depth [m] per open-sky segment + "WateringTopSegment"
    at: dt.datetime

    @property
    def ponding_m(self) -> float:
        return float(sum(self.surface_h.values())) if self.surface_h else 0.0

    def to_blob(self) -> bytes:
        """The ``simulation_state`` wire format: npz with ``rel_sat``,
        ``rel_sat_old``, ``surface_names`` (fixed-width unicode, no pickle)
        and ``surface_h``. Byte-compatible with ``SoilPDECore.save_state_blob``."""
        buf = io.BytesIO()
        names = np.array(list(self.surface_h.keys()), dtype=np.str_)
        values = np.array([self.surface_h[k] for k in names], dtype=float)
        np.savez(
            buf,
            rel_sat=np.asarray(self.se).copy(),
            rel_sat_old=np.asarray(self.se_old).copy(),
            surface_names=names,
            surface_h=values,
        )
        return buf.getvalue()

    @classmethod
    def from_blob(cls, blob: bytes, at: dt.datetime) -> SoilState:
        """Inverse of ``to_blob``; tolerates legacy blobs that carry only
        ``rel_sat`` (then ``se_old = se`` and no ponds), as the live core does."""
        arrays = np.load(io.BytesIO(blob), allow_pickle=True)
        se = np.asarray(arrays["rel_sat"], dtype=float)
        se_old = np.asarray(arrays["rel_sat_old"], dtype=float) if "rel_sat_old" in arrays.files else se.copy()
        surface_h: dict[str, float] = {}
        if "surface_names" in arrays.files and "surface_h" in arrays.files:
            surface_h = {str(n): float(v) for n, v in zip(arrays["surface_names"], arrays["surface_h"])}
        return cls(se=se, se_old=se_old, surface_h=surface_h, at=at)


@dataclass(frozen=True)
class Forcing:
    """Surface forcing held constant over one window, in the PDE core's
    units. One ``Forcing`` per weather row; the chain produces the list.
    Maps one-to-one onto ``simulation._soil.FluxRates``."""

    start: dt.datetime
    dt_s: float
    rain_flux: float = 0.0  # kg/(m^2 s), uniform over the open-sky top
    flow_m3s: float = 0.0  # m^3/s per metre of row, into the drip strip
    seg_evap: Mapping[str, float] = field(default_factory=dict)  # kg/(m^2 s) per top segment
    seg_transp: Mapping[str, float] = field(default_factory=dict)  # kg/(m^2 s) per top segment

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
    ``Simulation.run`` with the assimilated state and the probe tensions.

    ``diagnostics`` carries the mass-balance terms in kg/(m^2 h)
    (``top_out, transpiration, top_in, bottom_out, runoff, demand_unmet,
    balance_residual``) plus ``water_total``, ``surface_water``,
    ``delta_storage``, ``skipped_s``, ``retries`` and ``walk_ok``."""

    state: SoilState
    diagnostics: Mapping[str, float]
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
