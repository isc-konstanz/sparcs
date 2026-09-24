# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.fieldsim.core.shading
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ground shading under the PV rows, as a pure model: geometry in, per-segment
shade factors out. This is the pvfactors part of today's ``ground_shading.py``
(the numpy-2 compat patch, ``_PVSetup``, the ground report and combine
helpers, the horizontal / trackable / as-is setup builders) with every
``self.data`` call and the progress figure removed. Publishing goes through
``FieldIO.publish_chain``; rendering lives in ``plots``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import pandas as pd

MODE_FREE_FIELD = "free_field"
MODE_FIXED = "fixed"
MODE_TRACKED = "tracked"


@dataclass(frozen=True)
class TrackerConfig:
    axis_azimuth: float
    max_angle: float
    backtrack: bool = True


@dataclass(frozen=True)
class ShadingConfig:
    """``[ground_shading]`` block plus the PV geometry it is evaluated against.

    ``segment_ranges`` maps soil-mesh top-segment names to x-ranges in
    pvfactors coordinates; None means no mesh is wired and factors are
    reported per bay only.
    """

    mode: str = MODE_FREE_FIELD
    albedo: float = 0.2
    surface_azimuth: float = 180.0
    surface_tilt: float = 0.0
    mirrored: bool = False
    bay_width: float = 3.5
    tracker: Optional[TrackerConfig] = None
    pv_rows: Sequence[Mapping[str, float]] = ()  # row geometry from the PV system
    segment_ranges: Optional[Mapping[str, tuple[float, float]]] = None

    @classmethod
    def from_mapping(cls, block: Mapping[str, Any], *, pv_geometry: Any, bay_width: float) -> ShadingConfig:
        """Today ``GroundShading._configure_geometry`` + ``_resolve_segment_ranges``."""
        raise NotImplementedError

    def __post_init__(self) -> None:
        if self.mode not in (MODE_FREE_FIELD, MODE_FIXED, MODE_TRACKED):
            raise ValueError(f"unknown shading mode {self.mode!r}")
        if not 0.0 <= self.albedo <= 1.0:
            raise ValueError("albedo must be within [0, 1]")


class ShadingModel:
    """Per-row pvfactors evaluation, aggregated to soil segments."""

    def __init__(self, config: ShadingConfig) -> None:
        self.config = config
        self._setups: list[Any] = []  # one _PVSetup per PV row arrangement

    def evaluate(self, weather: pd.DataFrame) -> pd.DataFrame:
        """Return one row per weather row with a ``<segment>`` column per
        soil top segment (shade factor in [0, 1]) plus ``open_sky_ghi``.

        Free-field mode returns all ones without touching pvfactors
        (today ``GroundShading.evaluate`` + ``_evaluate_free_field``).
        """
        raise NotImplementedError

    def pv_rows_at(self, ts: pd.Timestamp) -> list[tuple]:
        """PV-row geometry for the plot at ``ts``; night frames reuse the
        last sun-up geometry (today ``_synthesize_pv_rows``)."""
        raise NotImplementedError

    def _build_setups(self) -> None:
        """Today ``_build_horizontal_setups`` / ``_build_trackable_setups`` /
        ``_build_as_is_setups``, chosen by ``config.mode``."""
        raise NotImplementedError

    def _aggregate_per_segment(self, ground: list[tuple]) -> dict[str, float]:
        """Today ``_aggregate_per_segment`` over ``_qinc_in_range``."""
        raise NotImplementedError
