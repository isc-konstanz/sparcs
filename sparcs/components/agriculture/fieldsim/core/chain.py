# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.fieldsim.core.chain
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Weather frame in, forcing series plus chain outputs out. Orchestration only:
prepare weather, shading, segments, evapotranspiration, fold into ``Forcing``.
The models live in ``shading`` and ``evapotranspiration``; nothing here
publishes. What today's ``FieldSimulation._run_chain`` + ``_build_segments`` +
``SoilSimulation._compute_flux_rates`` compute, minus the reads and the
channel writes.
"""

from __future__ import annotations

import logging
from typing import Mapping, Optional, Sequence

import pandas as pd
from lories.components.weather import Weather
from sparcs.components.weather import validate_meteo_inputs

from .config import FieldSetup
from .evapotranspiration import ETModel, SegmentProperties
from .plots import PlotConfig, render_due
from .shading import ShadingModel
from .state import ChainResult, Forcing

logger = logging.getLogger(__name__)

# Monthly LAI lookup tables keyed by ``field.lai_type`` (today's ``_LAI_BY_TYPE``).
_LAI_BY_TYPE: dict[str, list[float]] = {
    "fao": [3.0] * 12,
    "grass": [0.2, 0.2, 0.2, 0.3, 0.6, 0.8, 0.9, 1.2, 1.4, 1.2, 0.8, 0.6],
    "apple": [0.2, 0.4, 1.2, 2.5, 3.0, 3.2, 3.0, 2.8, 2.0, 1.0, 0.5, 0.2],
}

# Weather keys filled with a default when the feed doesn't supply them (today's ``_WEATHER_DEFAULTS``).
_WEATHER_DEFAULTS: dict[str, float] = {
    Weather.CLEAR_SKY_INDEX: 0.5,
    Weather.HUMIDITY_REL: 60.0,  # %
}

_CANOPY_SEGMENT_NAMES = ("PlantTopLeftSegment", "PlantTopRightSegment")

# Vegetation column names populated onto the weather frame; match today's
# ``FieldSimulation`` Constant keys (LAI/ROUGHNESS/PLANT_HEIGHT/NDVI).
_LAI = "lai"
_ROUGHNESS = "roughness"
_PLANT_HEIGHT = "plant_height"
_NDVI = "ndvi"


def design_flow_lpm(nozzle_count: int, nozzle_flow_lph: float) -> float:
    """Whole-field design flow [l/min] from the drip layout (today's ``simulation._soil.design_flow_lpm``)."""
    return nozzle_count * nozzle_flow_lph / 60.0


def flow_m3s_per_m(flow_lpm: float, total_drip_line_length_m: float) -> float:
    """Whole-field flow [l/min] normalized to m^3/s per out-of-plane metre of
    row (today's ``simulation._soil.flow_m3s_per_m``)."""
    return flow_lpm / (60_000.0 * total_drip_line_length_m)


def segment_flux_dicts(
    seg_et: Mapping[str, pd.DataFrame],
    ts: pd.Timestamp,
) -> tuple[dict[str, float], dict[str, float]]:
    """Per-segment ET flux dicts at ``ts``: negative ET is clipped to zero and
    zero-flux segments are skipped (today's ``simulation._soil.segment_flux_dicts``)."""
    seg_evap: dict[str, float] = {}
    seg_transp: dict[str, float] = {}
    for name, frame in seg_et.items():
        if ts not in frame.index:
            continue
        evap = max(0.0, float(frame.loc[ts, "evap"]))
        transp = max(0.0, float(frame.loc[ts, "transp"]))
        if evap > 0.0:
            seg_evap[name] = evap
        if transp > 0.0:
            seg_transp[name] = transp
    return seg_evap, seg_transp


def rain_flux(weather: pd.DataFrame, ts: pd.Timestamp, elapsed_s: float) -> float:
    """Rain flux density [kg/(m^2*s)] for the interval ending at ``ts``:
    ``precip_mm / elapsed_s`` (today's ``simulation._soil.rain_flux``); reads
    ``lories.components.weather.Weather.PRECIPITATION``."""
    col = Weather.PRECIPITATION
    if elapsed_s <= 0 or col not in weather.columns or ts not in weather.index:
        return 0.0
    precip = weather.loc[ts, col]
    if pd.isna(precip) or precip <= 0:
        return 0.0
    return float(precip) / elapsed_s  # mm/s == kg/(m^2*s)


class WeatherChain:
    def __init__(
        self,
        setup: FieldSetup,
        shading: ShadingModel,
        et: ETModel,
        plots: Optional[PlotConfig] = None,
        *,
        top_segment_names: Sequence[str] = (),
        segment_face_length: Optional[Mapping[str, float]] = None,
    ) -> None:
        self.setup = setup
        self.shading = shading
        self.et = et
        self.plots = plots
        self._top_segment_names = tuple(top_segment_names)
        self._segment_face_length = dict(segment_face_length or {})
        self._last_plot: Optional[pd.Timestamp] = None
        self._vegetation_placeholder_warned = False
        self._weather_default_warned: set[str] = set()

    def forcing_series(self, weather: pd.DataFrame, irrigation_lpm: pd.Series) -> tuple[Sequence[Forcing], ChainResult]:  # noqa: E501
        df = self._prepare_weather(weather)
        shading = self.shading.evaluate(df)
        segments = self._segments(df, shading)
        bulk, seg_et = self.et.evaluate(df, segments)
        forcing = self._forcings(df, shading, seg_et, irrigation_lpm)

        if self.plots is not None and not df.empty:
            ts = df.index[-1]
            if render_due(self._last_plot, ts, self.plots):
                # Rendering itself (render_shading_png) is a later unit's
                # job; this only keeps the due-cadence bookkeeping current.
                self._last_plot = ts

        return forcing, ChainResult(shading=shading, evapotranspiration=bulk, image=None)

    def horizon_inputs(self, forecast: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
        """What the planner rolls over: the prepared forecast frame (the live
        rollout reads its index and precipitation column) and the per-segment
        ET frames. Same chain as ``forcing_series``, minus the forcings."""
        df = self._prepare_weather(forecast)
        shading = self.shading.evaluate(df)
        segments = self._segments(df, shading)
        _, seg_et = self.et.evaluate(df, segments)
        return df, dict(seg_et)

    def _prepare_weather(self, weather: pd.DataFrame) -> pd.DataFrame:
        """Derive solar-position/irradiance columns when a ``Location`` is
        known, then fill the placeholder vegetation state (today's
        ``_prepare_weather`` + ``_populate_vegetation``). Required-column
        validation is ``ETModel.evaluate``'s job, not this seam's."""
        if self.setup.location is not None:
            weather = validate_meteo_inputs(weather, self.setup.location)
        return self._populate_vegetation(weather)

    def _populate_vegetation(self, weather: pd.DataFrame) -> pd.DataFrame:
        """Today's ``_populate_vegetation`` (``publish=False`` path): fill the
        placeholder canopy state and the optional weather defaults."""
        field = self.setup.field
        df = weather.copy()
        if not self._vegetation_placeholder_warned:
            logger.warning(
                "WeatherChain: using placeholder vegetation state (LAI from monthly table '%s', "
                "ROUGHNESS=%s, PLANT_HEIGHT=%s, NDVI=%s); no Crop subcomponent or field-level "
                "sensors are publishing these channels yet.",
                field.lai_type,
                field.roughness,
                field.plant_height,
                field.ndvi,
            )
            self._vegetation_placeholder_warned = True
        df[_LAI] = pd.array(_LAI_BY_TYPE[field.lai_type])[df.index.month - 1].astype(float)
        df[_ROUGHNESS] = field.roughness
        df[_PLANT_HEIGHT] = field.plant_height
        df[_NDVI] = field.ndvi

        for key, default in _WEATHER_DEFAULTS.items():
            if key not in df.columns or df[key].isna().all():
                if key not in self._weather_default_warned:
                    logger.warning("WeatherChain: weather feed does not supply '%s'; defaulting to %s.", key, default)
                    self._weather_default_warned.add(key)
                df[key] = default
            elif df[key].isna().any():
                df[key] = df[key].fillna(default)
        return df

    def _segments(self, weather: pd.DataFrame, shading: pd.DataFrame) -> Sequence[SegmentProperties]:
        """One ``SegmentProperties`` per soil top segment from ``setup.field``
        (lai_type, plant_height, bare_*) and the shading factors
        (today ``_populate_vegetation`` + ``_build_segments``)."""
        field = self.setup.field
        canopy_lai = float(weather[_LAI].iloc[-1])
        canopy_plant_height = float(weather[_PLANT_HEIGHT].iloc[-1])
        canopy_ndvi = float(weather[_NDVI].iloc[-1])
        canopy_roughness = float(weather[_ROUGHNESS].iloc[-1])

        if not self._top_segment_names:
            return [
                SegmentProperties(
                    name="_bulk",
                    lai=canopy_lai,
                    plant_height=canopy_plant_height,
                    ndvi=canopy_ndvi,
                    roughness=canopy_roughness,
                    shade_factor=1.0,
                    face_length=1.0,
                    is_canopy=True,
                )
            ]

        seg_props: list[SegmentProperties] = []
        for name in self._top_segment_names:
            is_canopy = name in _CANOPY_SEGMENT_NAMES
            shade_factor = float(shading[name].iloc[-1]) if name in shading.columns else 1.0
            seg_props.append(
                SegmentProperties(
                    name=name,
                    lai=canopy_lai if is_canopy else field.bare_lai,
                    plant_height=canopy_plant_height if is_canopy else field.bare_plant_height,
                    ndvi=canopy_ndvi if is_canopy else field.bare_ndvi,
                    roughness=canopy_roughness if is_canopy else field.bare_roughness,
                    shade_factor=shade_factor,
                    face_length=float(self._segment_face_length.get(name, 0.0)),
                    is_canopy=is_canopy,
                )
            )
        return seg_props

    def _forcings(
        self,
        weather: pd.DataFrame,
        shading: pd.DataFrame,
        seg_et: Mapping[str, pd.DataFrame],
        irrigation_lpm: pd.Series,
    ) -> Sequence[Forcing]:
        """Rain from ``weather``, irrigation l/min over the drip line length
        into m^3/s per drip segment, ET per segment; one ``Forcing`` per row
        (today ``SoilSimulation._compute_flux_rates`` + ``flow_m3s_per_m``).
        ``shading`` isn't needed for the flux math -- the per-segment shade
        factor is already folded into ``seg_et`` by ``ETModel.evaluate`` --
        but stays positional so it matches ``forcing_series``'s call shape."""
        index = weather.index
        irrigation = irrigation_lpm.reindex(index, method="ffill").fillna(0.0)
        total_drip_line_length_m = self.setup.soil.total_drip_line_length_m

        forcings: list[Forcing] = []
        for i, ts in enumerate(index):
            if i + 1 < len(index):
                dt_s = (index[i + 1] - ts).total_seconds()
            elif i > 0:
                dt_s = (index[i] - index[i - 1]).total_seconds()
            else:
                dt_s = 3600.0
            seg_evap, seg_transp = segment_flux_dicts(seg_et, ts)
            forcings.append(
                Forcing(
                    start=ts.to_pydatetime(),
                    dt_s=dt_s,
                    rain_flux=rain_flux(weather, ts, dt_s),
                    flow_m3s=flow_m3s_per_m(float(irrigation.loc[ts]), total_drip_line_length_m),
                    seg_evap=seg_evap,
                    seg_transp=seg_transp,
                )
            )
        return forcings
