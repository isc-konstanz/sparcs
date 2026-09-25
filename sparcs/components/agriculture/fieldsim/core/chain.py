# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.fieldsim.core.chain
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Weather frame in, forcing series plus chain outputs out. Orchestration only:
prepare weather, shading, segments, evapotranspiration, fold into ``Forcing``.
The models live in ``shading`` and ``evapotranspiration``; nothing here
publishes. What today's ``FieldSimulation._run_chain`` + ``_build_segments`` +
``_irrigation_flow_lpm`` compute, minus the reads and the channel writes.
"""

from __future__ import annotations

from typing import Optional, Sequence

import pandas as pd

from .config import FieldSetup
from .evapotranspiration import ETModel, SegmentProperties
from .plots import PlotConfig, render_due, render_shading_png
from .shading import ShadingModel
from .state import ChainResult, Forcing


class WeatherChain:
    def __init__(
        self,
        setup: FieldSetup,
        shading: ShadingModel,
        et: ETModel,
        plots: Optional[PlotConfig] = None,
    ) -> None:
        self.setup = setup
        self.shading = shading
        self.et = et
        self.plots = plots
        self._last_plot: Optional[pd.Timestamp] = None

    def forcing_series(self, weather: pd.DataFrame, irrigation_lpm: pd.Series) -> tuple[Sequence[Forcing], ChainResult]:  # noqa: E501
        weather = self._prepare_weather(weather)
        shading = self.shading.evaluate(weather)
        segments = self._segments(weather, shading)
        et = self.et.evaluate(weather, segments)
        forcing = self._forcings(weather, shading, et, irrigation_lpm)
        image = None
        ts = weather.index[-1]
        if self.plots is not None and render_due(self._last_plot, ts, self.plots):
            factors = {s.name: float(shading[s.name].iloc[-1]) for s in segments}
            image = render_shading_png(ts, self.shading.pv_rows_at(ts), factors, envelope=None)
            self._last_plot = ts
        return forcing, ChainResult(shading=shading, evapotranspiration=et, image=image)

    def _prepare_weather(self, weather: pd.DataFrame) -> pd.DataFrame:
        """Rename connector columns, fill vegetation defaults, validate the
        required columns (today ``_prepare_weather`` + ``_weather_frame_valid``)."""
        raise NotImplementedError

    def _segments(self, weather: pd.DataFrame, shading: pd.DataFrame) -> Sequence[SegmentProperties]:
        """One ``SegmentProperties`` per soil top segment from ``setup.field``
        (lai_type, plant_height, bare_*) and the shading factors
        (today ``_populate_vegetation`` + ``_build_segments``)."""
        raise NotImplementedError

    def _forcings(
        self,
        weather: pd.DataFrame,
        shading: pd.DataFrame,
        et: pd.DataFrame,
        irrigation_lpm: pd.Series,
    ) -> Sequence[Forcing]:
        """Rain from ``weather``, irrigation l/min over the drip line length
        into m/s per drip segment, ET per segment into m/s; one ``Forcing``
        per row (today ``SoilSimulation._compute_flux_rates`` + ``flow_m3s_per_m``)."""
        raise NotImplementedError
