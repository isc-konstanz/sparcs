# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.simulation.ground_shading
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Per-segment ground shading from a PV array using ``solarfactors``.
Publishes a shade factor in ``[0, 1]`` (1 = open sky) and time-mean
local irradiance (W/m²) for each soil-mesh top segment.
"""

from __future__ import annotations

import io
import logging
from typing import Any, Optional

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pvlib.tracking import singleaxis

import numpy as np
import pandas as pd
from lories import Component
from lories.components.weather import Weather
from lories.typing import Configurations

from ..fieldsim.components import GroundShading as _FsGroundShading
from ..fieldsim.core.pv import (  # noqa: F401
    _DEFAULT_PLOT_INTERVAL,
    _GROUND_SHADING_N_ROWS,
    _VALID_MODES,
    _ZENITH_DAYTIME_LIMIT,
    MODE_AS_IS,
    MODE_FREE_FIELD,
    MODE_HORIZONTAL,
    MODE_TRACKABLE,
    _combine_grounds,
    _open_sky_ghi,
    _pvfactors_is_pointing_right,
    _PVSetup,
    _qinc_in_range,
    _TrackerConfig,
)
from . import plot_style

logger = logging.getLogger(__name__)
# 4. GroundShading component


class GroundShading(Component):
    TYPE: str = "ground_shading"

    # Mean shade factor [-]: 0 = full PV shade, 1 = open sky.
    SHADING_FACTOR = _FsGroundShading.SHADING_FACTOR

    # PNG bytes of the most recent shading-pattern plot.
    SHADING_PROGRESS_IMAGE = _FsGroundShading.SHADING_PROGRESS_IMAGE

    CHANNELS = [SHADING_FACTOR]

    # --- Geometry / mode state ------------------------------------------------
    _mode: str
    _pv_setups: list[_PVSetup]
    _albedo: float
    _surface_azimuth: float
    _surface_tilt: float
    _mirrored: bool
    _tracker: Optional[_TrackerConfig] = None

    # --- Mesh ↔ PV-coordinate state ------------------------------------------
    # Soil mesh top-segment x-ranges in pvfactors coords; None if no mesh is wired.
    _segment_ranges: Optional[dict[str, tuple[float, float]]] = None

    # --- Plot state (_plot_config is None when plotting is disabled) ----------
    _plot_config: Optional[plot_style.PlotConfig] = None
    _plot_fig: Any = None
    _plot_axes: Any = None
    _last_plot_ts: Optional[pd.Timestamp] = None
    # Consecutive render failures toward plot_style.PLOT_DISABLE_AFTER (W2.9).
    _plot_strikes: int = 0
    # Last sun-up PV-row geometry; reused for night structure-only frames.
    _last_pv_rows: list

    # Static plot envelope computed once at activate so PNG size stays stable.
    _plot_x_half: float = 0.0
    _plot_y_min: float = -1.0
    _plot_y_max: float = 1.0

    # 4a. Channel registration

    def configure(self, configs: Configurations) -> None:
        super().configure(configs)
        self._last_pv_rows = []
        self._register_channels()
        self._configure_geometry(configs)
        self._configure_plot(configs)

    def activate(self) -> None:
        super().activate()
        self._segment_ranges = self._resolve_segment_ranges()
        self._compute_plot_envelope()

    def _compute_plot_envelope(self) -> None:
        """Set x/y plot limits: 3 bays around the middle row, worst-case panel height above, soil bottom below."""
        if self._pv_setups:
            distance = self._pv_setups[0].distance
            self._plot_x_half = distance * 1.5
            if self._mode == MODE_TRACKABLE and self._tracker is not None:
                tilt_max = abs(self._tracker.max_angle)
            else:
                tilt_max = abs(self._surface_tilt)
            tilt_rad = np.radians(tilt_max)
            y_panel = max(setup.height + (setup.width / 2.0) * np.sin(tilt_rad) for setup in self._pv_setups)
            self._plot_y_max = y_panel + 1.0
        else:
            bay_width = float(getattr(self.context, "bay_width", 3.5))
            self._plot_x_half = bay_width * 1.5
            self._plot_y_max = 1.0

        mesh = getattr(self.context, "mesh_config", None)
        self._plot_y_min = (-mesh.height - 0.5) if mesh is not None else -1.0

    def _register_channels(self) -> None:
        """Register the bulk SHADING_FACTOR channel plus the in-memory
        plot_strikes strike counter."""
        for c in self.CHANNELS:
            self.data.add(c, aggregate="mean", logger={"enabled": False})
        # In-memory strike counter (W2.9): registered or it surfaces nowhere.
        self.data.add("plot_strikes", type=float, name="Plot Strikes", aggregate="last", logger={"enabled": False})

    def _configure_plot(self, configs: Configurations) -> None:
        """Read the ``[plot]`` block and register SHADING_PROGRESS_IMAGE when enabled."""
        self._plot_config = plot_style.load_plot_config(configs, default_interval=_DEFAULT_PLOT_INTERVAL)
        if self._plot_config is None:
            return
        self.data.add(
            GroundShading.SHADING_PROGRESS_IMAGE,
            aggregate="last",
            logger={"enabled": True},
        )

    # 4b. Geometry configuration

    def _configure_geometry(self, configs: Configurations) -> None:
        """Parse the [ground_shading] block and build the PV setups."""
        mode = str(configs.get("mode", default=MODE_AS_IS)).lower()
        if mode not in _VALID_MODES:
            raise ValueError(f"Unsupported ground_shading mode '{mode}'. Must be one of: {sorted(_VALID_MODES)}")
        self._mode = mode
        self._albedo = configs.get_float("albedo", default=0.2)

        if mode == MODE_FREE_FIELD:
            self._configure_free_field()
            return

        # ``distance`` defaults to the parent FieldSimulation's ``bay_width``.
        default_distance = float(getattr(self.context, "bay_width", 3.5))
        common = dict(
            n_rows=_GROUND_SHADING_N_ROWS,
            height=configs.get_float("height", default=3.770),
            width=configs.get_float("width", default=1.134),
            distance=configs.get_float("distance", default=default_distance),
            axis_azimuth=configs.get_float("axis_azimuth", default=100.0),
        )

        if mode == MODE_HORIZONTAL:
            self._pv_setups = self._build_horizontal_setups(configs, common)
        elif mode == MODE_TRACKABLE:
            self._pv_setups = self._build_trackable_setups(configs, common)
        else:  # MODE_AS_IS
            self._pv_setups = self._build_as_is_setups(configs, common)

    def _configure_free_field(self) -> None:
        """Open-sky baseline: no array, every segment sees full irradiance."""
        self._pv_setups = []
        self._mirrored = False
        self._surface_tilt = 0.0
        self._surface_azimuth = 180.0

    def _build_horizontal_setups(
        self,
        configs: Configurations,
        common: dict[str, Any],
    ) -> list[_PVSetup]:
        """Row geometry with surface_tilt forced to 0 (flat)."""
        self._mirrored = False
        self._surface_tilt = 0.0
        self._surface_azimuth = configs.get_float("surface_azimuth", default=180.0)
        return [
            _PVSetup(
                surface_tilt=self._surface_tilt,
                surface_azimuth=self._surface_azimuth,
                offset_x=0.0,
                **common,
            )
        ]

    def _build_trackable_setups(
        self,
        configs: Configurations,
        common: dict[str, Any],
    ) -> list[_PVSetup]:
        """Single-axis tracker; actual rotation computed per-timestep via pvlib.singleaxis."""
        self._mirrored = False
        # Placeholder tilt; overridden per-timestep in ``_build_pvfactors_input``.
        self._surface_tilt = 0.0
        self._surface_azimuth = configs.get_float("surface_azimuth", default=common["axis_azimuth"])
        self._tracker = _TrackerConfig(
            axis_tilt=configs.get_float("axis_tilt", default=0.0),
            axis_azimuth=common["axis_azimuth"],
            max_angle=configs.get_float("max_angle", default=60.0),
            backtrack=configs.get_bool("backtrack", default=True),
            gcr=common["width"] / common["distance"],
        )
        return [
            _PVSetup(
                surface_tilt=0.0,
                surface_azimuth=self._surface_azimuth,
                offset_x=0.0,
                **common,
            )
        ]

    def _build_as_is_setups(
        self,
        configs: Configurations,
        common: dict[str, Any],
    ) -> list[_PVSetup]:
        """Fixed-tilt rows. ``mirrored=True`` builds an A-frame pair tilted in opposite directions."""
        surface_tilt = configs.get_float("surface_tilt", default=10.0)
        surface_azimuth = configs.get_float("surface_azimuth", default=180.0)
        mirrored = configs.get_bool("mirrored", default=False)
        self._surface_tilt = surface_tilt
        self._surface_azimuth = surface_azimuth
        self._mirrored = mirrored

        common_with_az = {**common, "surface_azimuth": surface_azimuth}
        if not mirrored:
            return [
                _PVSetup(
                    surface_tilt=surface_tilt,
                    offset_x=0.0,
                    **common_with_az,
                )
            ]

        # A-frame: both panels' high edges lean toward x=0 (a peak). Which tilt
        # sign leans a row right vs. left depends on pvfactors' azimuth
        # convention, so the pairing must flip with ``is_pointing_right`` —
        # hard-coding -left/+right inverts the roof for some axis_azimuth
        # (e.g. 180). We want the left panel's high edge on its right ("/") and
        # the right panel's high edge on its left ("\"): rotation>0 → "/",
        # rotation<0 → "\", with rotation = tilt if pointing_right else -tilt.
        tilt = abs(surface_tilt)
        half = common["width"] * np.cos(np.radians(tilt)) / 2.0
        left_sign = 1.0 if _pvfactors_is_pointing_right(surface_azimuth, common["axis_azimuth"]) else -1.0
        return [
            _PVSetup(surface_tilt=left_sign * tilt, offset_x=-half, **common_with_az),
            _PVSetup(surface_tilt=-left_sign * tilt, offset_x=+half, **common_with_az),
        ]

    # 4c. Segment-range resolution

    def _resolve_segment_ranges(self) -> Optional[dict[str, tuple[float, float]]]:
        """Compute soil-mesh top-segment x-ranges in pvfactors coordinates.

        Aligns the PV array centre over the plant centre of the soil mesh
        (shift = plant_center - pv_center), then maps each mesh segment.
        """
        mesh = self.context.mesh_config
        if mesh is None:
            return None

        dx = mesh.dx
        plant_width = mesh.plant_width
        watering_width = mesh.watering_width
        n_pv_segments = int((mesh.width - plant_width) / (2 * dx))

        plant_left = n_pv_segments * dx
        plant_right = plant_left + plant_width
        watering_left = plant_left + (plant_width - watering_width) / 2
        watering_right = watering_left + watering_width
        plant_center = (plant_left + plant_right) / 2.0

        if self._pv_setups:
            pv = self._pv_setups[0]
            pv_center = (pv.n_rows - 1) * pv.distance / 2.0
            shift = plant_center - pv_center
        else:
            shift = plant_center

        ranges: dict[str, tuple[float, float]] = {}
        for i in range(n_pv_segments):
            ranges[f"LeftTopSegment_{i}"] = (i * dx - shift, (i + 1) * dx - shift)
        ranges["PlantTopLeftSegment"] = (plant_left - shift, watering_left - shift)
        ranges["PlantTopRightSegment"] = (watering_right - shift, plant_right - shift)
        for i in range(n_pv_segments):
            x0 = plant_right + i * dx
            ranges[f"RightTopSegment_{i}"] = (x0 - shift, x0 + dx - shift)
        return ranges

    # 4d. Evaluation pipeline

    def evaluate(
        self,
        data: Optional[pd.DataFrame] = None,
        *,
        publish: bool = True,
    ) -> Optional[dict[str, float]]:
        """Compute per-segment shade factors for ``data``.

        Returns per-segment factors when a mesh is wired, else None.
        ``publish=False`` suppresses channel writes and live-plot capture.
        In MODE_FREE_FIELD, pvfactors is skipped; all factors are 1.0.
        """
        if data is None or data.empty:
            return None

        pv_df = self._build_pvfactors_input(data)
        if pv_df.empty:
            return self._publish_open_sky(data, publish=publish)

        ts = data.index[-1]
        ghi_open = _open_sky_ghi(pv_df)

        if self._mode == MODE_FREE_FIELD:
            return self._evaluate_free_field(ts, ghi_open, publish=publish)

        # Run each setup with its own surface_tilt/azimuth; mirrored setups use per-setup overrides.
        per_setup_ground: list[list[list[tuple]]] = []
        per_setup_pv_rows: list[list[list[tuple]]] = []
        try:
            for setup in self._pv_setups:
                if self._mode == MODE_TRACKABLE:
                    setup_df = pv_df
                else:
                    setup_df = pv_df.copy()
                    setup_df["surface_tilt"] = setup.surface_tilt
                    setup_df["surface_azimuth"] = setup.surface_azimuth
                report = setup.run(setup_df)
                per_setup_ground.append(list(report["ground"].values))
                per_setup_pv_rows.append(list(report["pv_rows"].values))
        except Exception:  # noqa: BLE001
            # pvfactors/numpy>=2 compat: collapsed zero-area surfaces can raise
            # inhomogeneous-shape ValueError even with the import-time patch
            # above. Fall back to open-sky; retry next tick.
            logger.warning(
                "%s: pvfactors raised; falling back to open-sky for this tick.",
                self.name,
                exc_info=True,
            )
            return self._publish_open_sky(data, publish=publish)
        n_t = len(pv_df.index)
        combined_per_t = [
            _combine_grounds([per_setup_ground[s][t] for s in range(len(per_setup_ground))]) for t in range(n_t)
        ]
        pv_rows_per_t = [[row for s in per_setup_pv_rows for row in s[t]] for t in range(n_t)]

        if self._segment_ranges:
            seg_factors, seg_ghi = self._aggregate_per_segment(combined_per_t, ghi_open)
            if publish:
                self._publish_per_segment_ghi(ts, seg_ghi)
        else:
            seg_factors = None
            seg_ghi = {}

        # Bulk SHADING_FACTOR: length-weighted mean over one bay centred on the middle row.
        mean_factor = self._bay_mean_factor(combined_per_t, ghi_open)
        if publish:
            self.data[GroundShading.SHADING_FACTOR].set(ts, mean_factor)
            last_idx = pv_df.index[-1]
            sun_state = (
                float(pv_df.at[last_idx, "solar_zenith"]),
                float(pv_df.at[last_idx, "solar_azimuth"]),
                self._pv_setups[0].axis_azimuth if self._pv_setups else None,
            )
            last_pv_rows = pv_rows_per_t[-1] if pv_rows_per_t else []
            if last_pv_rows:
                self._last_pv_rows = last_pv_rows
            self._capture_progress(
                ts=last_idx,
                ground=combined_per_t[-1] if combined_per_t else [],
                pv_rows=last_pv_rows,
                sun_state=sun_state,
            )
        return seg_factors

    def _aggregate_per_segment(
        self,
        combined_per_t: list[list[tuple]],
        ghi_open: np.ndarray,
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Time-mean shade factor (sun-up rows only) and GHI [W/m²] per segment.
        Returns ``(seg_factors, seg_ghi)``.
        """
        seg_factors: dict[str, float] = {}
        seg_ghi: dict[str, float] = {}
        for name, (x0, x1) in self._segment_ranges.items():
            factor_vals: list[float] = []
            ghi_vals: list[float] = []
            for t, ground in enumerate(combined_per_t):
                qinc = _qinc_in_range(ground, x0, x1)
                if not np.isfinite(qinc):
                    # A stray non-finite qinc (pvfactors edge case) must not
                    # poison the time means for the whole segment.
                    continue
                ghi_vals.append(qinc)
                ref = ghi_open[t]
                if ref <= 0:
                    continue
                factor_vals.append(min(1.0, qinc / ref))
            seg_factors[name] = float(np.mean(factor_vals)) if factor_vals else 1.0
            seg_ghi[name] = float(np.mean(ghi_vals)) if ghi_vals else 0.0
        return seg_factors, seg_ghi

    def _evaluate_free_field(
        self,
        ts: pd.Timestamp,
        ghi_open: np.ndarray,
        *,
        publish: bool = True,
    ) -> Optional[dict[str, float]]:
        """Open-sky shortcut: factor 1.0 everywhere, per-segment GHI = open-sky."""
        ghi_mean = float(np.mean(ghi_open)) if ghi_open.size else 0.0
        if publish:
            self.data[GroundShading.SHADING_FACTOR].set(ts, 1.0)
        if not self._segment_ranges:
            return None
        seg_ghi = {name: ghi_mean for name in self._segment_ranges}
        if publish:
            self._publish_per_segment_ghi(ts, seg_ghi)
        return {name: 1.0 for name in self._segment_ranges}

    def _build_pvfactors_input(self, data: pd.DataFrame) -> pd.DataFrame:
        """Select/derive the columns pvfactors needs; drop night rows (zenith >= limit)."""
        df = pd.DataFrame(index=data.index)
        df["solar_zenith"] = data.get("solar_zenith")
        df["solar_azimuth"] = data.get("solar_azimuth")
        df["dni"] = data.get(Weather.DNI)
        df["dhi"] = data.get(Weather.DHI)
        df["albedo"] = self._albedo

        df = df.dropna()
        if df.empty:
            return df
        df = df[df["solar_zenith"] < _ZENITH_DAYTIME_LIMIT]
        # Lightless rows carry no shading information, and the Perez
        # transposition is undefined at DHI=0 (sky-clearness epsilon is 0/0:
        # pvlib returns NaN and solarfactors' ``poa_sky_diffuse == 0``
        # luminance guard misses NaN) -- one such twilight row poisons the
        # whole chunk's per-segment GHI mean with NaN.
        df = df[(df["dni"] > 0) | (df["dhi"] > 0)]
        if df.empty:
            return df

        if self._mode == MODE_FREE_FIELD:
            return df

        if self._mode == MODE_TRACKABLE:
            tracking = singleaxis(
                apparent_zenith=df["solar_zenith"],
                apparent_azimuth=df["solar_azimuth"],
                axis_tilt=self._tracker.axis_tilt,
                axis_azimuth=self._tracker.axis_azimuth,
                max_angle=self._tracker.max_angle,
                backtrack=self._tracker.backtrack,
                gcr=self._tracker.gcr,
            )
            # Drop rows where pvlib returns NaN (sun outside tracker envelope).
            df["surface_tilt"] = tracking["surface_tilt"]
            df["surface_azimuth"] = tracking["surface_azimuth"]
            df = df.dropna(subset=["surface_tilt", "surface_azimuth"])
        else:
            df["surface_tilt"] = self._surface_tilt
            df["surface_azimuth"] = self._surface_azimuth
        return df

    def _publish_open_sky(
        self,
        data: pd.DataFrame,
        *,
        publish: bool = True,
    ) -> Optional[dict[str, float]]:
        """No usable rows (all night): publish factor 1 and a structure-only progress frame."""
        ts = data.index[-1]
        if publish:
            self.data[GroundShading.SHADING_FACTOR].set(ts, 1.0)

        if publish:
            pv_rows = self._last_pv_rows or self._synthesize_pv_rows()
            if pv_rows:

                def _last(col: str, default: float) -> float:
                    series = data.get(col)
                    if series is None or not pd.notna(series.iloc[-1]):
                        return default
                    return float(series.iloc[-1])

                axis_az = self._pv_setups[0].axis_azimuth if self._pv_setups else None
                self._capture_progress(
                    ts=ts,
                    ground=[],
                    pv_rows=pv_rows,
                    sun_state=(_last("solar_zenith", 90.0), _last("solar_azimuth", 0.0), axis_az),
                )

        if not self._segment_ranges:
            return None
        if publish:
            self._publish_per_segment_ghi(ts, {name: 0.0 for name in self._segment_ranges})
        return {name: 1.0 for name in self._segment_ranges}

    def _synthesize_pv_rows(self) -> list[tuple]:
        """Compute PV-row endpoints analytically from setup config (no pvfactors).
        Used for cold-start night renders; trackers drawn at rest position (tilt=0).
        """
        if not self._pv_setups:
            return []
        rows: list[tuple] = []
        for setup in self._pv_setups:
            # Match pvfactors' drawn geometry: rotation = tilt when "pointing
            # right" draws "/" (high edge right), while a positive lean in the
            # endpoint math below draws "\" (high edge left) -- so the sign
            # flips when pointing right. Without this the night render disagrees
            # with the daytime pvfactors render for some axis_azimuth (e.g. 180).
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

    def _publish_per_segment_ghi(self, ts: pd.Timestamp, seg_ghi: dict[str, float]) -> None:
        """Publish per-segment GHI [W/m²] to SEG_GHI; non-finite values become 0.0 with a warning.

        The placeholder must be 0.0, not NaN: a NaN in a VALID-state channel
        write raises ResourceError in lories (Channel._is_empty), which kills
        the tick and stalls the frontier on the same chunk forever.
        """
        cleaned = {}
        missing = []
        for name, v in seg_ghi.items():
            if v is not None and np.isfinite(v):
                cleaned[name] = float(v)
            else:
                cleaned[name] = 0.0
                missing.append(name)
        if missing:
            logger.warning(
                "%s: SEG_GHI has no value for segment(s) %s at %s; writing 0.0 placeholder.",
                self.name,
                sorted(missing),
                ts,
            )
        self.context.set_segment_values(self.context.SEG_GHI, ts, cleaned)

    def _bay_mean_factor(
        self,
        combined_per_t: list[list[tuple]],
        ghi_open: np.ndarray,
    ) -> float:
        """Length-weighted mean shade factor over one inter-row bay centred on the middle row.
        Averaged over sun-up timesteps.
        """
        if not self._pv_setups:
            return 1.0
        middles = [(setup.n_rows - 1) / 2.0 * setup.distance + setup.offset_x for setup in self._pv_setups]
        center_x = float(np.mean(middles))
        distance = self._pv_setups[0].distance
        bay_lo = center_x - distance / 2.0
        bay_hi = center_x + distance / 2.0

        vals: list[float] = []
        for t, ground in enumerate(combined_per_t):
            ref = ghi_open[t]
            if ref <= 0 or not ground:
                continue
            qinc_avg = _qinc_in_range(ground, bay_lo, bay_hi)
            vals.append(min(1.0, qinc_avg / ref))
        return float(np.mean(vals)) if vals else 1.0

    # 4e. Progress plotting

    # Colormap ceiling [W/m²]: midday clear-sky GHI on a horizontal surface.
    _PLOT_QINC_MAX: float = 1000.0

    def _capture_progress(
        self,
        ts: pd.Timestamp,
        ground: list[tuple],
        pv_rows: list[tuple],
        sun_state: tuple[float, float, Optional[float]],
    ) -> None:
        """Throttle renders by PlotConfig.interval and forward to _render_progress.
        ``sun_state`` is ``(solar_zenith, solar_azimuth, axis_azimuth)`` for shadow projection.
        """
        if self._plot_config is None:
            return
        if not plot_style.render_due(self._last_plot_ts, ts, self._plot_config.interval):
            return
        self._last_plot_ts = ts
        try:
            self._render_progress(ts, ground, pv_rows, sun_state)
        except Exception:  # noqa: BLE001
            self._plot_strikes, disable = plot_style.count_render_failure(logger, self.name, self._plot_strikes)
            plot_style.set_strike_channel(self, "plot_strikes", self._plot_strikes)
            if disable:
                self._plot_config = None
            return
        if self._plot_strikes:
            self._plot_strikes = 0
            plot_style.set_strike_channel(self, "plot_strikes", 0)

    def _init_progress_figure(self) -> None:
        """Create the matplotlib figure once; reuse across renders.
        The evaluate() call runs on the field tick's worker thread, so render headless.
        """
        if matplotlib.get_backend().lower() not in ("agg", "module://matplotlib_inline.backend_inline"):
            matplotlib.use("Agg", force=True)
        x_extent = 2.0 * self._plot_x_half
        y_extent = self._plot_y_max - self._plot_y_min
        fig, ax = plt.subplots(
            figsize=plot_style.compute_fig_size(x_extent, y_extent),
            dpi=plot_style.DPI,
        )
        cmap = plt.get_cmap(plot_style.COLORMAP)
        norm = mcolors.PowerNorm(gamma=0.5, vmin=0.0, vmax=self._PLOT_QINC_MAX)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, shrink=plot_style.CBAR_SHRINK, label="incident irradiance [W/m²]")
        plot_style.apply_subplots_adjust(fig)
        self._plot_fig = fig
        self._plot_axes = (ax, cmap, norm)

    def _render_progress(
        self,
        ts: pd.Timestamp,
        ground: list[tuple],
        pv_rows: list[tuple],
        sun_state: tuple[float, float, Optional[float]],
    ) -> None:
        """Draw the 2-D scene: ground coloured by qinc, PV rows in black, shadow projection lines.
        Persists the PNG to the SHADING_PROGRESS_IMAGE DB blob channel.
        """
        if self._plot_fig is None:
            self._init_progress_figure()
        ax, cmap, norm = self._plot_axes
        ax.clear()

        # Re-centre x on the middle row so x=0 is the middle row in the plot.
        if self._pv_setups:
            middles = [(setup.n_rows - 1) / 2.0 * setup.distance + setup.offset_x for setup in self._pv_setups]
            center_x = float(np.mean(middles))
        else:
            center_x = 0.0

        def rx(x: float) -> float:
            return x - center_x

        ax.axhline(
            y=0.0,
            color="black",
            linewidth=0.8,
            zorder=0.5,
        )

        for seg in ground:
            qinc = max(0.0, seg[2]["qinc"])
            ax.plot(
                [rx(seg[0][0]), rx(seg[1][0])],
                [0.0, 0.0],
                color=cmap(norm(qinc)),
                linewidth=6,
                solid_capstyle="butt",
                zorder=2,
            )

        # Shadow projection: shadow_x = px - py·tan(zenith)·sin(sun_az - axis_az)
        sun_zen, sun_az, axis_az = sun_state
        if axis_az is not None and sun_zen < _ZENITH_DAYTIME_LIMIT and pv_rows:
            sun_x_per_y = float(np.tan(np.radians(sun_zen)) * np.sin(np.radians(sun_az - axis_az)))
            seen: set[tuple[float, float]] = set()
            for seg in pv_rows:
                for endpoint in (seg[0], seg[1]):
                    px, py = endpoint
                    if py <= 0:
                        continue
                    key = (round(px, 4), round(py, 4))
                    if key in seen:
                        continue
                    seen.add(key)
                    shadow_x = px - py * sun_x_per_y
                    ax.plot(
                        [rx(px), rx(shadow_x)],
                        [py, 0.0],
                        color="gray",
                        linewidth=0.6,
                        linestyle="--",
                        alpha=0.45,
                        zorder=1.5,
                    )

        for seg in pv_rows:
            ax.plot(
                [rx(seg[0][0]), rx(seg[1][0])],
                [seg[0][1], seg[1][1]],
                color="black",
                linewidth=2,
            )

        # Soil cross-section: brown rectangle for full soil depth, green for plant block.
        mesh = getattr(self.context, "mesh_config", None)
        if mesh is not None and self._segment_ranges:
            seg_xs = [x for pair in self._segment_ranges.values() for x in pair]
            ground_left = min(seg_xs)
            ground_right = max(seg_xs)
            plant_left = self._segment_ranges["PlantTopLeftSegment"][0]
            plant_right = self._segment_ranges["PlantTopRightSegment"][1]

            soil_left_plot = rx(ground_left)
            soil_width_plot = rx(ground_right) - soil_left_plot
            ax.add_patch(
                Rectangle(
                    (soil_left_plot, -mesh.height),
                    soil_width_plot,
                    mesh.height,
                    facecolor="saddlebrown",
                    edgecolor="saddlebrown",
                    alpha=0.18,
                    linewidth=1.0,
                    zorder=0,
                )
            )

            plant_left_plot = rx(plant_left)
            plant_width_plot = rx(plant_right) - plant_left_plot
            ax.add_patch(
                Rectangle(
                    (plant_left_plot, -mesh.plant_height),
                    plant_width_plot,
                    mesh.plant_height,
                    facecolor="forestgreen",
                    edgecolor="darkgreen",
                    alpha=0.35,
                    linewidth=1.2,
                    zorder=1,
                )
            )

        ax.set_xlim(-self._plot_x_half, +self._plot_x_half)
        ax.set_ylim(self._plot_y_min, self._plot_y_max)

        plot_style.apply_axes_style(ax)
        timezone = getattr(getattr(self.context, "location", None), "timezone", None)
        ax.set_title(plot_style.format_progress_title("Ground shading", ts, tz=timezone))

        buf = io.BytesIO()
        self._plot_fig.savefig(buf, dpi=plot_style.DPI, format="png")
        png_bytes = buf.getvalue()

        self.data[GroundShading.SHADING_PROGRESS_IMAGE].set(ts, png_bytes)
