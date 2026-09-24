# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.fieldsim.chain
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Weather frame in, forcing series out. Wraps the ground-shading and
evapotranspiration evaluators as plain functions; neither is a lories
Component any more, and neither publishes. What today's
``FieldSimulation._run_chain`` + ``_build_segments`` + ``_irrigation_flow_lpm``
compute, minus the reads and the channel writes.
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from .config import FieldConfig
from .state import Forcing


class WeatherChain:
    def __init__(self, config: FieldConfig) -> None:
        self.config = config

    def forcing_series(self, weather: pd.DataFrame, irrigation_lpm: pd.Series) -> Sequence[Forcing]:
        """One ``Forcing`` per weather row.

        1. ``_prepare_weather``: rename, fill vegetation defaults.
        2. ``shading(weather)``: per-segment shading factors (pvfactors).
        3. ``segments(weather, shading)``: ``SegmentProperties`` per row.
        4. ``evapotranspiration(weather, segments)``: ET per segment.
        5. Fold rain, irrigation flow (l/min over the drip line length) and ET
           into ``Forcing`` values in m/s.
        """
        raise NotImplementedError

    def shading(self, weather: pd.DataFrame) -> pd.DataFrame:
        """Ground-shading evaluator (today ``GroundShading.evaluate`` without publish)."""
        raise NotImplementedError

    def evapotranspiration(self, weather: pd.DataFrame, segments: Sequence[object]) -> pd.DataFrame:
        """ET evaluator (today ``Evapotranspiration.evaluate`` without publish)."""
        raise NotImplementedError
