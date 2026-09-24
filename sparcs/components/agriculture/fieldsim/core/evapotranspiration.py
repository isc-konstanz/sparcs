# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.fieldsim.core.evapotranspiration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Penman-Monteith evapotranspiration per soil segment, as a pure model. This is
today's ``evapotranspiration.py`` without the Component base and the channel
registration; the per-term helpers keep their names so the move is a cut.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd


@dataclass(frozen=True)
class SegmentProperties:
    """Per-segment vegetation + radiation state for one ET evaluation.

    ``shade_factor`` in [0, 1] scales bulk GHI to local incoming shortwave.
    ``face_length`` [m] weights bulk means across segments.
    """

    name: str
    lai: float
    plant_height: float = 0.1
    ndvi: float = 0.25
    roughness: float = 0.002
    shade_factor: float = 1.0
    face_length: float = 0.0
    is_canopy: bool = False


class ETModel:
    BEER_K: float = 0.6
    TEMP_GROUND_LIFT: float = 0.012

    REQUIRED_WEATHER_COLUMNS: tuple[str, ...] = (
        "ghi",
        "temp_air",
        "relative_humidity",
        "wind_speed",
        "pressure",
    )

    def evaluate(self, weather: pd.DataFrame, segments: Sequence[SegmentProperties]) -> pd.DataFrame:
        """Return one row per weather row with the bulk intermediates
        (``sat_vapor_pressure``, ``net_irradiance``, ``aerodynamic_resistance``,
        ``radiation_term``, ``aerodynamic_term``, ...) and one
        ``et_<segment>`` column in kg/(m^2 h) per segment.
        """
        raise NotImplementedError

    @staticmethod
    def _sat_vapor_pressure(temp_air: pd.Series) -> pd.Series:
        raise NotImplementedError

    @staticmethod
    def _net_irradiance(weather: pd.DataFrame, segment: SegmentProperties) -> pd.Series:
        raise NotImplementedError

    @staticmethod
    def _aerodynamic_resistance(wind_speed: pd.Series, segment: SegmentProperties) -> pd.Series:
        raise NotImplementedError

    @staticmethod
    def _resistance_surface(segment: SegmentProperties) -> float:
        raise NotImplementedError
