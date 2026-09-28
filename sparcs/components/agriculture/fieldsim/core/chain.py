# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.fieldsim.core.chain
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Weather frame in, forcing series plus chain outputs out. Orchestration only.
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Mapping, Optional, Sequence

import pandas as pd
from lories.components.weather import Weather
from sparcs.components.weather import validate_meteo_inputs

from .config import FieldSetup
from .evapotranspiration import ETModel, SegmentProperties
from .pde import flow_m3s_per_m, rain_flux, segment_flux_dicts
from .shading import ShadingModel
from .state import ChainResult, Forcing

logger = logging.getLogger(__name__)

STRIP_FLUX_WARN_MM_H = 500.0

_LAI_BY_TYPE: dict[str, list[float]] = {
    "fao": [3.0] * 12,
    "grass": [0.2, 0.2, 0.2, 0.3, 0.6, 0.8, 0.9, 1.2, 1.4, 1.2, 0.8, 0.6],
    "apple": [0.2, 0.4, 1.2, 2.5, 3.0, 3.2, 3.0, 2.8, 2.0, 1.0, 0.5, 0.2],
}

# Filled with a default when the weather feed does not supply them.
_WEATHER_DEFAULTS: dict[str, float] = {
    Weather.CLEAR_SKY_INDEX: 0.5,
    Weather.HUMIDITY_REL: 60.0,  # %
}

_CANOPY_SEGMENT_NAMES = ("PlantTopLeftSegment", "PlantTopRightSegment")

_LAI = "lai"
_ROUGHNESS = "roughness"
_PLANT_HEIGHT = "plant_height"
_NDVI = "ndvi"


class WeatherChain:
    def __init__(
        self,
        setup: FieldSetup,
        shading: ShadingModel,
        et: ETModel,
        *,
        top_segment_names: Sequence[str] = (),
        segment_face_length: Optional[Mapping[str, float]] = None,
    ) -> None:
        self.setup = setup
        self.shading = shading
        self.et = et
        self._top_segment_names = tuple(top_segment_names)
        self._segment_face_length = dict(segment_face_length or {})
        self._envelope = shading.envelope(setup.soil.mesh.height)
        self._strip_flux_warned = False
        self._vegetation_placeholder_warned = False
        self._weather_default_warned: set[str] = set()

    def forcing_series(
        self,
        weather: pd.DataFrame,
        irrigation_lpm: pd.Series,
        *,
        frontier: Optional[dt.datetime] = None,
        first_dt_s: float = 0.0,
    ) -> tuple[Sequence[Forcing], ChainResult]:
        """Chain outputs for every weather row, plus one ``Forcing`` per row after ``frontier``.

        Without a frontier the first row is the cold-start anchor and spans ``first_dt_s``.
        """
        df = self._prepare_weather(weather)
        shading = self.shading.evaluate(df)
        segments = self._segments(df, shading)
        bulk, seg_et = self.et.evaluate(df, segments)
        forcing = self._forcings(df, shading, seg_et, irrigation_lpm, frontier=frontier, first_dt_s=first_dt_s)
        ground, pv_rows, sun_state = self.shading.render_inputs_at(df.index[-1])
        result = ChainResult(
            shading=shading,
            evapotranspiration=bulk,
            ground=ground,
            pv_rows=pv_rows,
            sun_state=sun_state,
            envelope=self._envelope,
        )
        return forcing, result

    def horizon_inputs(self, forecast: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
        """The prepared forecast frame and the per-segment ET frames the planner rolls
        over; the shading model keeps the live tick's frame inputs."""
        df = self._prepare_weather(forecast)
        shading = self.shading.evaluate(df, remember=False)
        segments = self._segments(df, shading)
        _, seg_et = self.et.evaluate(df, segments)
        return df, dict(seg_et)

    def _prepare_weather(self, weather: pd.DataFrame) -> pd.DataFrame:
        """Derive solar-position/irradiance columns when a ``Location`` is known, then vegetation."""
        if self.setup.location is not None:
            weather = validate_meteo_inputs(weather, self.setup.location)
        return self._populate_vegetation(weather)

    def _populate_vegetation(self, weather: pd.DataFrame) -> pd.DataFrame:
        """Fill the placeholder canopy state and the optional weather defaults."""
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
        """One ``SegmentProperties`` per soil top segment from the field config and shading."""
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

    def _warn_absurd_strip_flux(self, flow_m3s: float) -> None:
        """Warn once when the strip flux is far beyond drip rates (a unit or length error)."""
        watering_width = self.setup.soil.mesh.watering_width
        if self._strip_flux_warned or flow_m3s <= 0.0 or watering_width <= 0.0:
            return
        strip_mm_h = flow_m3s / watering_width * 3.6e6
        if strip_mm_h > STRIP_FLUX_WARN_MM_H:
            self._strip_flux_warned = True
            logger.warning(
                "irrigation strip flux %.0f mm/h exceeds %.0f mm/h. Check the irrigation_flow values "
                "(must be true l/min, whole-field total) and total_drip_line_length_m "
                "(%.1f m; must be n_rows * row_length).",
                strip_mm_h,
                STRIP_FLUX_WARN_MM_H,
                self.setup.soil.total_drip_line_length_m,
            )

    def _forcings(
        self,
        weather: pd.DataFrame,
        shading: pd.DataFrame,
        seg_et: Mapping[str, pd.DataFrame],
        irrigation_lpm: pd.Series,
        *,
        frontier: Optional[dt.datetime],
        first_dt_s: float = 0.0,
    ) -> Sequence[Forcing]:
        """One ``Forcing`` per row after ``frontier``, each window reaching back to the previous row."""
        index = weather.index
        irrigation = irrigation_lpm.reindex(index, method="ffill").fillna(0.0)
        total_drip_line_length_m = self.setup.soil.total_drip_line_length_m
        cutoff = pd.Timestamp(frontier) if frontier is not None else None

        forcings: list[Forcing] = []
        previous = cutoff
        for ts in index:
            if cutoff is not None and ts <= cutoff:
                continue
            dt_s = first_dt_s if previous is None else (ts - previous).total_seconds()
            previous = ts
            seg_evap, seg_transp = segment_flux_dicts(seg_et, ts)
            flow_m3s = flow_m3s_per_m(float(irrigation.loc[ts]), total_drip_line_length_m)
            self._warn_absurd_strip_flux(flow_m3s)
            forcings.append(
                Forcing(
                    at=ts.to_pydatetime(),
                    dt_s=dt_s,
                    rain_flux=rain_flux(weather, ts, dt_s),
                    flow_m3s=flow_m3s,
                    seg_evap=seg_evap,
                    seg_transp=seg_transp,
                )
            )
        return forcings
