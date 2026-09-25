# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.fieldsim.core.shading
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ground shading under the PV rows, as a pure model: geometry in, per-segment
shade factors out. Free-field mode (no PV array) is a direct port of today's
``GroundShading._evaluate_free_field``, needing nothing beyond the weather
GHI column. The pvfactors modes (as_is / horizontal / trackable) reuse the
live ``ground_shading.py``'s geometry primitives directly -- ``_PVSetup`` and
the pure ground/report helpers -- rather than re-deriving them here: that
module pulls in ``lories.Component``/``Constant`` at import time (for its own
channel registration) but never FiPy, so importing it is safe for this
FiPy-free chain; the remaining glue (matplotlib progress plots, channel
writes) stays behind ``publish=True`` in that module and is never touched
here. A later unit may inline the geometry builders and drop this import.
"""

from __future__ import annotations

import logging
from typing import Mapping, Optional, Sequence

from pvlib.tracking import singleaxis

import numpy as np
import pandas as pd
from lories.components.weather import Weather
from lories.core.configs.parameters import Parameter, SelectParameter

from ...simulation.ground_shading import (
    _combine_grounds,
    _open_sky_ghi,
    _patch_pvfactors_numpy2_compat,
    _pvfactors_is_pointing_right,
    _PVSetup,
    _qinc_in_range,
    _TrackerConfig,
)
from .config import Config

logger = logging.getLogger(__name__)

_patch_pvfactors_numpy2_compat()

MODE_AS_IS = "as_is"  # fixed-tilt rows as configured; supports mirrored
MODE_HORIZONTAL = "horizontal"  # row geometry forced flat (surface_tilt = 0)
MODE_TRACKABLE = "trackable"  # single-axis tracker via pvlib.tracking.singleaxis
MODE_FREE_FIELD = "free_field"  # no PV array; open-sky reference baseline
MODES = (MODE_AS_IS, MODE_HORIZONTAL, MODE_TRACKABLE, MODE_FREE_FIELD)

# 7 rows (3 on each side of the centre) gives the middle row representative inter-row shading.
_N_ROWS = 7

# Solar zenith [deg] above which pvfactors is unstable; skip and treat as open sky.
_ZENITH_DAYTIME_LIMIT = 89.0


class ShadingConfig(Config):
    """``[ground_shading]`` plus the PV geometry it is evaluated against.

    Keys mirror today's ``GroundShading._configure_geometry`` /
    ``_build_horizontal_setups`` / ``_build_trackable_setups`` /
    ``_build_as_is_setups`` exactly: ``height``/``width``/``distance``/
    ``axis_azimuth`` are common to every non-free-field mode; ``axis_tilt``/
    ``max_angle``/``backtrack`` are read only in trackable mode and
    ``surface_tilt``/``mirrored`` only in as_is mode -- but all live flat at
    the top of ``[ground_shading]``, never under a ``[tracker]`` table (that
    grouping in an earlier draft of this class was a guess, not what the live
    parser reads).

    ``bay_width``, ``pv_rows`` and ``segment_ranges`` are derived by the
    adapter from the field config, the PV system and the soil mesh; they are
    attributes, not config keys.
    """

    _CONFIGS_ALLOWED_KEYS = Config._CONFIGS_ALLOWED_KEYS | {"plot"}

    mode = SelectParameter(choices=list(MODES), default=MODE_AS_IS, desc="as_is, horizontal, trackable or free_field")
    albedo = Parameter(type=float, default=0.2, min=0.0, max=1.0, desc="Ground albedo")
    height = Parameter(type=float, default=3.770, min=0.0, desc="PV row mounting height (m)")
    width = Parameter(type=float, default=1.134, min=0.0, desc="PV row/module width (m)")
    distance = Parameter(
        type=float, default=None, required=False, desc="Inter-row spacing (m); defaults to the field bay_width"
    )
    axis_azimuth = Parameter(type=float, default=100.0, min=0.0, max=360.0, desc="Row-axis bearing (deg)")
    surface_tilt = Parameter(type=float, default=10.0, min=0.0, max=90.0, desc="PV surface tilt (deg); as_is mode")
    surface_azimuth = Parameter(type=float, default=180.0, min=0.0, max=360.0, desc="PV surface azimuth (deg)")
    mirrored = Parameter(type=bool, default=False, desc="Mirror the row arrangement about the bay centre (as_is)")
    axis_tilt = Parameter(type=float, default=0.0, min=-90.0, max=90.0, desc="Tracker axis tilt from horizontal (deg)")
    max_angle = Parameter(type=float, default=60.0, min=0.0, max=90.0, desc="Tracker maximum rotation (deg)")
    backtrack = Parameter(type=bool, default=True, desc="Tracker backtracking to avoid row-to-row shading")

    # derived, set by the adapter
    bay_width: float = 3.5
    pv_rows: Sequence[Mapping[str, float]] = ()
    segment_ranges: Optional[Mapping[str, tuple[float, float]]] = None

    def derive(
        self, *, bay_width: float, pv_rows: Sequence[Mapping[str, float]] = (), segment_ranges=None
    ) -> ShadingConfig:
        """Today ``GroundShading._configure_geometry``'s ``distance`` default
        plus ``_resolve_segment_ranges``."""
        self.bay_width = bay_width
        self.pv_rows = tuple(pv_rows)
        self.segment_ranges = segment_ranges
        if self.distance is None:
            self.distance = float(bay_width)
        return self


class ShadingModel:
    """Per-row pvfactors evaluation, aggregated to soil segments."""

    def __init__(self, config: ShadingConfig) -> None:
        self.config = config
        self._setups: list[_PVSetup] = []
        self._tracker: Optional[_TrackerConfig] = None
        self._mirrored: bool = False
        self._surface_tilt: float = config.surface_tilt
        self._surface_azimuth: float = config.surface_azimuth
        self._last_pv_rows: list[tuple] = []
        self._pvfactors_failure_warned = False
        if config.mode != MODE_FREE_FIELD:
            self._build_setups()

    def evaluate(self, weather: pd.DataFrame) -> pd.DataFrame:
        """Return one row per weather row with a ``<segment>`` column per
        soil top segment (shade factor in [0, 1]), ``open_sky_ghi`` and a
        ``ghi_<segment>`` column per segment.

        Free-field mode returns all ones with ``open_sky_ghi`` taken straight
        from the weather GHI column, without touching pvfactors (today
        ``GroundShading.evaluate`` + ``_evaluate_free_field``). Any pvfactors
        exception falls back to the same open-sky result, warning once.
        """
        if weather.empty:
            return pd.DataFrame(index=weather.index)
        if self.config.mode == MODE_FREE_FIELD or not self._setups:
            return self._open_sky_result(weather)

        pv_df = self._build_pvfactors_input(weather)
        if pv_df.empty:
            return self._open_sky_result(weather)

        try:
            ghi_open = _open_sky_ghi(pv_df)
            per_setup_ground: list[list[list[tuple]]] = []
            for setup in self._setups:
                if self.config.mode == MODE_TRACKABLE:
                    setup_df = pv_df
                else:
                    setup_df = pv_df.copy()
                    setup_df["surface_tilt"] = setup.surface_tilt
                    setup_df["surface_azimuth"] = setup.surface_azimuth
                report = setup.run(setup_df)
                per_setup_ground.append(list(report["ground"].values))
                pv_rows = list(report["pv_rows"].values)
                if pv_rows and pv_rows[-1]:
                    self._last_pv_rows = pv_rows[-1]
        except Exception:  # noqa: BLE001
            if not self._pvfactors_failure_warned:
                logger.warning("ShadingModel: pvfactors raised; falling back to open-sky.", exc_info=True)
                self._pvfactors_failure_warned = True
            return self._open_sky_result(weather)

        n_t = len(pv_df.index)
        combined_per_t = [
            _combine_grounds([per_setup_ground[s][t] for s in range(len(per_setup_ground))]) for t in range(n_t)
        ]
        seg_factors, seg_ghi = self._aggregate_per_segment(combined_per_t, ghi_open)

        out = pd.DataFrame(index=weather.index, dtype=float)
        out["open_sky_ghi"] = float(np.mean(ghi_open)) if ghi_open.size else 0.0
        for name, factor in seg_factors.items():
            out[name] = factor
            out[f"ghi_{name}"] = seg_ghi.get(name, 0.0)
        return out

    def pv_rows_at(self, ts: pd.Timestamp) -> list[tuple]:
        """PV-row geometry for the plot at ``ts``; night frames reuse the
        last sun-up geometry (today ``_synthesize_pv_rows``)."""
        return self._last_pv_rows if self._last_pv_rows else self._synthesize_pv_rows()

    def _build_setups(self) -> None:
        """Today ``_build_horizontal_setups`` / ``_build_trackable_setups`` /
        ``_build_as_is_setups``, chosen by ``config.mode``."""
        cfg = self.config
        distance = cfg.distance if cfg.distance is not None else cfg.bay_width
        common = dict(
            n_rows=_N_ROWS,
            height=cfg.height,
            width=cfg.width,
            distance=distance,
            axis_azimuth=cfg.axis_azimuth,
        )
        if cfg.mode == MODE_HORIZONTAL:
            self._mirrored = False
            self._surface_tilt = 0.0
            self._surface_azimuth = cfg.surface_azimuth
            self._setups = [_PVSetup(surface_tilt=0.0, surface_azimuth=self._surface_azimuth, offset_x=0.0, **common)]
        elif cfg.mode == MODE_TRACKABLE:
            self._mirrored = False
            self._surface_tilt = 0.0
            self._surface_azimuth = cfg.surface_azimuth
            self._tracker = _TrackerConfig(
                axis_tilt=cfg.axis_tilt,
                axis_azimuth=common["axis_azimuth"],
                max_angle=cfg.max_angle,
                backtrack=cfg.backtrack,
                gcr=common["width"] / common["distance"],
            )
            self._setups = [_PVSetup(surface_tilt=0.0, surface_azimuth=self._surface_azimuth, offset_x=0.0, **common)]
        else:  # MODE_AS_IS
            surface_tilt = cfg.surface_tilt
            surface_azimuth = cfg.surface_azimuth
            mirrored = cfg.mirrored
            self._surface_tilt = surface_tilt
            self._surface_azimuth = surface_azimuth
            self._mirrored = mirrored
            common_with_az = {**common, "surface_azimuth": surface_azimuth}
            if not mirrored:
                self._setups = [_PVSetup(surface_tilt=surface_tilt, offset_x=0.0, **common_with_az)]
            else:
                tilt = abs(surface_tilt)
                half = common["width"] * np.cos(np.radians(tilt)) / 2.0
                left_sign = 1.0 if _pvfactors_is_pointing_right(surface_azimuth, common["axis_azimuth"]) else -1.0
                self._setups = [
                    _PVSetup(surface_tilt=left_sign * tilt, offset_x=-half, **common_with_az),
                    _PVSetup(surface_tilt=-left_sign * tilt, offset_x=+half, **common_with_az),
                ]

    def _aggregate_per_segment(
        self,
        combined_per_t: list[list[tuple]],
        ghi_open: np.ndarray,
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Today ``_aggregate_per_segment`` over ``_qinc_in_range``: time-mean
        shade factor (sun-up rows only) and GHI [W/m^2] per segment."""
        seg_factors: dict[str, float] = {}
        seg_ghi: dict[str, float] = {}
        for name, (x0, x1) in (self.config.segment_ranges or {}).items():
            factor_vals: list[float] = []
            ghi_vals: list[float] = []
            for t, ground in enumerate(combined_per_t):
                qinc = _qinc_in_range(ground, x0, x1)
                if not np.isfinite(qinc):
                    continue
                ghi_vals.append(qinc)
                ref = ghi_open[t]
                if ref <= 0:
                    continue
                factor_vals.append(min(1.0, qinc / ref))
            seg_factors[name] = float(np.mean(factor_vals)) if factor_vals else 1.0
            seg_ghi[name] = float(np.mean(ghi_vals)) if ghi_vals else 0.0
        return seg_factors, seg_ghi

    def _open_sky_result(self, weather: pd.DataFrame) -> pd.DataFrame:
        """Factor 1.0 everywhere, per-segment GHI = open-sky GHI (today
        ``_evaluate_free_field`` / ``_publish_open_sky``)."""
        ghi = weather[Weather.GHI]
        out = pd.DataFrame(index=weather.index)
        out["open_sky_ghi"] = ghi
        for name in self.config.segment_ranges or {}:
            out[name] = 1.0
            out[f"ghi_{name}"] = ghi
        return out

    def _build_pvfactors_input(self, weather: pd.DataFrame) -> pd.DataFrame:
        """Select/derive the columns pvfactors needs; drop night rows (today
        ``_build_pvfactors_input``)."""
        df = pd.DataFrame(index=weather.index)
        df["solar_zenith"] = weather.get("solar_zenith")
        df["solar_azimuth"] = weather.get("solar_azimuth")
        df["dni"] = weather.get(Weather.DNI)
        df["dhi"] = weather.get(Weather.DHI)
        df["albedo"] = self.config.albedo

        df = df.dropna()
        if df.empty:
            return df
        df = df[df["solar_zenith"] < _ZENITH_DAYTIME_LIMIT]
        df = df[(df["dni"] > 0) | (df["dhi"] > 0)]
        if df.empty:
            return df

        if self.config.mode == MODE_TRACKABLE:
            tracking = singleaxis(
                apparent_zenith=df["solar_zenith"],
                apparent_azimuth=df["solar_azimuth"],
                axis_tilt=self._tracker.axis_tilt,
                axis_azimuth=self._tracker.axis_azimuth,
                max_angle=self._tracker.max_angle,
                backtrack=self._tracker.backtrack,
                gcr=self._tracker.gcr,
            )
            df["surface_tilt"] = tracking["surface_tilt"]
            df["surface_azimuth"] = tracking["surface_azimuth"]
            df = df.dropna(subset=["surface_tilt", "surface_azimuth"])
        else:
            df["surface_tilt"] = self._surface_tilt
            df["surface_azimuth"] = self._surface_azimuth
        return df

    def _synthesize_pv_rows(self) -> list[tuple]:
        """Compute PV-row endpoints analytically from setup config, no
        pvfactors (today ``_synthesize_pv_rows``)."""
        if not self._setups:
            return []
        rows: list[tuple] = []
        for setup in self._setups:
            lean = (
                -setup.surface_tilt
                if _pvfactors_is_pointing_right(setup.surface_azimuth, setup.axis_azimuth)
                else setup.surface_tilt
            )
            tilt_rad = np.radians(lean)
            half_x = setup.width / 2.0 * np.cos(tilt_rad)
            half_y = setup.width / 2.0 * np.sin(tilt_rad)
            for i in range(setup.n_rows):
                cx = i * setup.distance + setup.offset_x
                rows.append(
                    (
                        (cx - half_x, setup.height + half_y),
                        (cx + half_x, setup.height - half_y),
                        {"qinc_front": 0.0, "qinc_back": 0.0},
                    )
                )
        return rows
