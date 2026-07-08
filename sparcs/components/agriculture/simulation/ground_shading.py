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
import os
import threading
from dataclasses import dataclass
from typing import Any, Optional

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pvfactors.engine import PVEngine
from pvfactors.geometry import OrderedPVArray
from pvlib.tracking import singleaxis

import numpy as np
import pandas as pd


# pvfactors builds rho_mat from a mix of scalar and array reflectivities;
# numpy>=2 rejects the inhomogeneous list. Broadcast scalars to n_states
# so the radiosity matrix stays rectangular. Patched once at import time.
def _patch_pvfactors_numpy2_compat() -> None:
    from pvfactors.irradiance.models import SKY_REFLECTIVITY_DUMMY, HybridPerezOrdered

    if getattr(HybridPerezOrdered, "_lories_numpy2_patched", False):
        return

    def get_full_ts_modeling_vectors(self, pvarray):
        irradiance_mat, rho_mat, inv_rho_mat, total_perez_mat = self.get_ts_modeling_vectors(pvarray)
        irradiance_mat.append(self.isotropic_luminance)
        total_perez_mat.append(self.isotropic_luminance)
        rho_mat.append(SKY_REFLECTIVITY_DUMMY * np.ones(pvarray.n_states))
        inv_rho_mat.append(SKY_REFLECTIVITY_DUMMY * np.ones(pvarray.n_states))

        n = pvarray.n_states

        def _normalize(lst):
            normalized = []
            for x in lst:
                arr = np.asarray(x, dtype=float)
                if arr.ndim == 0 or arr.size == 1:
                    arr = np.full(n, arr.item(), dtype=float)
                elif arr.shape[0] != n:
                    # pad or truncate to keep the matrix rectangular
                    if arr.shape[0] < n:
                        pad = np.zeros(n - arr.shape[0], dtype=float)
                        arr = np.concatenate([arr, pad])
                    else:
                        arr = arr[:n]
                normalized.append(arr)
            return np.array(normalized)

        return (
            _normalize(irradiance_mat),
            _normalize(rho_mat),
            _normalize(inv_rho_mat),
            _normalize(total_perez_mat),
        )

    HybridPerezOrdered.get_full_ts_modeling_vectors = get_full_ts_modeling_vectors
    HybridPerezOrdered._lories_numpy2_patched = True


_patch_pvfactors_numpy2_compat()

from lories import Component, Constant
from lories.components.weather import Weather
from lories.typing import Configurations

from . import plot_style

logger = logging.getLogger(__name__)

# 1. Constants

# 7 rows (3 on each side of the centre) gives the middle row representative inter-row shading.
_GROUND_SHADING_N_ROWS = 7

# Solar zenith [deg] above which pvfactors is unstable; skip and use factor 1.
_ZENITH_DAYTIME_LIMIT = 89.0

# Outer-edge clamp for ground segment x-coordinates [m].
_GROUND_X_CLAMP = 100.0

# Geometry modes selected via ``mode = ...`` in the [ground_shading] block.
MODE_AS_IS = "as_is"  # fixed-tilt rows; supports `mirrored`
MODE_HORIZONTAL = "horizontal"  # row geometry forced flat (surface_tilt = 0)
MODE_TRACKABLE = "trackable"  # single-axis tracker via pvlib.tracking.singleaxis
MODE_FREE_FIELD = "free_field"  # no PV array; open-sky reference baseline
_VALID_MODES = (MODE_AS_IS, MODE_HORIZONTAL, MODE_TRACKABLE, MODE_FREE_FIELD)


# 2. Free helpers


def _qinc_in_range(ground: list[tuple], x_start: float, x_end: float) -> float:
    """Length-weighted mean ``qinc`` over ``[x_start, x_end]`` in ``ground``."""
    if not ground or x_end <= x_start:
        return 0.0
    total_len = 0.0
    weighted = 0.0
    for seg in ground:
        a = seg[0][0]
        b = seg[1][0]
        lo = max(a, x_start)
        hi = min(b, x_end)
        if hi <= lo:
            continue
        length = hi - lo
        weighted += seg[2]["qinc"] * length
        total_len += length
    if total_len <= 0:
        return 0.0
    return weighted / total_len


def _combine_grounds(grounds_per_setup: list[list[tuple]]) -> list[tuple]:
    """Merge per-setup ground segments at one timestep into a unified ground.

    Direct components multiply across setups (independent shading);
    isotropic and reflection components average.
    Combined qinc = direct_frac·direct_max + ⟨reflection⟩ + ⟨isotropic⟩.
    """
    if not grounds_per_setup:
        return []

    def edges_of(ground: list[tuple]) -> np.ndarray:
        if not ground:
            return np.array([-_GROUND_X_CLAMP, _GROUND_X_CLAMP])
        edge = [seg[0][0] for seg in ground] + [seg[1][0] for seg in ground]
        edge[0] = -_GROUND_X_CLAMP
        edge[-1] = _GROUND_X_CLAMP
        return np.unique(edge)

    edges = np.unique(np.concatenate([edges_of(g) for g in grounds_per_setup]))
    # Maximum direct component across all setups; open-sky segments set the ceiling.
    direct_max = max(
        (max(0.0, seg[2]["qinc"] - seg[2]["reflection"] - seg[2]["isotropic"]) for g in grounds_per_setup for seg in g),
        default=0.0,
    )
    n = len(grounds_per_setup)

    combined: list[tuple] = []
    for x_start, x_end in zip(edges[:-1], edges[1:]):
        direct_frac = 1.0
        reflection = 0.0
        isotropic = 0.0
        for ground in grounds_per_setup:
            seg_qinc = 0.0
            seg_refl = 0.0
            seg_iso = 0.0
            for seg in ground:
                if seg[0][0] <= x_start and seg[1][0] >= x_end:
                    seg_qinc = seg[2]["qinc"]
                    seg_refl = seg[2]["reflection"]
                    seg_iso = seg[2]["isotropic"]
                    break
            seg_direct = max(0.0, seg_qinc - seg_refl - seg_iso)
            direct_frac *= (seg_direct / direct_max) if direct_max > 0 else 0.0
            reflection += seg_refl
            isotropic += seg_iso

        params = {
            "qinc": direct_frac * direct_max + reflection / n + isotropic / n,
            "reflection": reflection / n,
            "isotropic": isotropic / n,
        }
        combined.append(((x_start, 0.0), (x_end, 0.0), params))
    return combined


def _open_sky_ghi(pv_df: pd.DataFrame) -> np.ndarray:
    """Open-sky GHI [W/m²] per row of ``pv_df``: ``dni·cos(zenith) + dhi``."""
    cosz = np.cos(np.radians(pv_df["solar_zenith"].to_numpy()))
    return pv_df["dni"].to_numpy() * cosz + pv_df["dhi"].to_numpy()


# 3. Internal types


@dataclass
class PlotConfig:
    """Plot output settings.

    ``interval``: minimum time between renders.
    ``live``: overwrite a single ``ground_shading.png``.
    ``save``: archive one timestamped PNG per render.
    ``show``: pop a matplotlib window (main thread only).
    """

    def __init__(self, configs: Configurations, default_dir: str):
        interval = configs.get("interval", default="1h")
        if isinstance(interval, (int, float)):
            self.interval: pd.Timedelta = pd.Timedelta(seconds=float(interval))
        else:
            self.interval: pd.Timedelta = pd.Timedelta(interval)
        self.live: bool = configs.get_bool("live", default=True)
        self.save: bool = configs.get_bool("save", default=False)
        self.show: bool = configs.get_bool("show", default=False)
        self.dir: str = configs.get("dir", default=default_dir)


@dataclass
class _TrackerConfig:
    axis_tilt: float  # rotation-axis tilt from horizontal [deg]
    axis_azimuth: float  # rotation-axis bearing [deg], 180 = N–S axis
    max_angle: float  # tracker rotation limit [deg]
    backtrack: bool
    gcr: float  # ground coverage ratio used for backtracking


def _pvfactors_is_pointing_right(surface_azimuth: float, axis_azimuth: float) -> bool:
    """pvfactors' tilt-sign convention, mirroring
    ``pvfactors.geometry.base._get_rotation_from_tilt_azimuth``: it derives a
    signed ``rotation = tilt if is_pointing_right else -tilt`` and the row
    geometry follows ``rotation``'s sign. So the *same* signed surface_tilt
    leans the row opposite ways depending on this flag — which is why the
    A-frame sign pairing must key off it instead of being hard-coded.
    """
    return (surface_azimuth - axis_azimuth) % 360.0 > 180.0


class _PVSetup:
    """One PV-array geometry fed to a single solarfactors engine."""

    def __init__(
        self,
        n_rows: int,
        height: float,
        width: float,
        distance: float,
        axis_azimuth: float,
        surface_tilt: float,
        surface_azimuth: float,
        offset_x: float,
    ):
        self.n_rows = n_rows
        self.height = height
        self.width = width
        self.distance = distance
        self.axis_azimuth = axis_azimuth
        self.surface_tilt = surface_tilt
        self.surface_azimuth = surface_azimuth
        self.offset_x = offset_x

        self._pv_array = OrderedPVArray.init_from_dict(
            {
                "n_pvrows": n_rows,
                "pvrow_height": height,
                "pvrow_width": width,
                "axis_azimuth": axis_azimuth,
                "gcr": width / distance,
            }
        )
        self._engine = PVEngine(self._pv_array)

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the engine on ``df``; return a DataFrame with columns
        ``ground`` (segment tuples with qinc/reflection/isotropic) and
        ``pv_rows`` (row endpoint tuples with qinc_front/qinc_back).
        """
        self._engine.fit(
            df.index,
            df["dni"],
            df["dhi"],
            df["solar_zenith"],
            df["solar_azimuth"],
            df["surface_tilt"],
            df["surface_azimuth"],
            df["albedo"],
        )
        # Suppress 0/0 from zero-length collapsed surfaces; _report_ground drops them.
        with np.errstate(invalid="ignore", divide="ignore"):
            return self._engine.run_full_mode(fn_build_report=self._build_report)

    def _build_report(self, pvarray: Any) -> pd.DataFrame:
        return pd.concat(
            [self._report_ground(pvarray), self._report_pv_rows(pvarray)],
            axis=1,
        )

    def _report_ground(self, pvarray: Any) -> pd.DataFrame:
        ground = pvarray.ts_ground
        all_elements = ground.illum_elements + ground.shadow_elements

        per_surface_rows: list[list[tuple]] = []
        for sfc in all_elements:
            qinc = sfc.get_param_weighted("qinc").tolist()
            refl = sfc.get_param_weighted("reflection").tolist()
            iso = sfc.get_param_weighted("isotropic").tolist()
            xs = sfc.b1.x
            ys = sfc.b1.y
            xe = sfc.b2.x
            ye = sfc.b2.y
            per_surface_rows.append(
                [
                    (
                        (float(a) + self.offset_x, float(b)),
                        (float(c) + self.offset_x, float(d)),
                        {"qinc": float(q), "reflection": float(r), "isotropic": float(i)},
                    )
                    for a, b, c, d, q, r, i in zip(xs, ys, xe, ye, qinc, refl, iso)
                ]
            )

        # Transpose to (timestep, surface), drop zero-length, sort by x, clamp outer edges.
        per_timestep = list(map(list, zip(*per_surface_rows)))
        cleaned: list[list[tuple]] = []
        for grounds in per_timestep:
            grounds = [g for g in grounds if g[0][0] != g[1][0]]
            grounds.sort(key=lambda g: g[0][0])
            if grounds:
                _, y0 = grounds[0][0]
                grounds[0] = ((-_GROUND_X_CLAMP, y0), grounds[0][1], grounds[0][2])
                _, y1 = grounds[-1][1]
                grounds[-1] = (grounds[-1][0], (_GROUND_X_CLAMP, y1), grounds[-1][2])
            cleaned.append(grounds)

        return pd.DataFrame({"ground": cleaned})

    def _report_pv_rows(self, pvarray: Any) -> pd.DataFrame:
        """Per-timestep PV row segments with physical x/y coordinates and qinc."""
        rows_per_pvrow: list[list[tuple]] = []
        for pvrow in pvarray.ts_pvrows:
            b1, b2 = pvrow.full_pvrow_coords.b1, pvrow.full_pvrow_coords.b2
            qinc_front = pvrow.front.get_param_weighted("qinc").tolist()
            qinc_back = pvrow.back.get_param_weighted("qinc").tolist()
            rows_per_pvrow.append(
                [
                    (
                        (float(xs) + self.offset_x, float(ys)),
                        (float(xe) + self.offset_x, float(ye)),
                        {"qinc_front": float(qf), "qinc_back": float(qb)},
                    )
                    for xs, ys, xe, ye, qf, qb in zip(
                        b1.x,
                        b1.y,
                        b2.x,
                        b2.y,
                        qinc_front,
                        qinc_back,
                    )
                ]
            )

        per_timestep = list(map(list, zip(*rows_per_pvrow)))
        return pd.DataFrame({"pv_rows": per_timestep})


# 4. GroundShading component


class GroundShading(Component):
    TYPE: str = "ground_shading"

    # Mean shade factor [-]: 0 = full PV shade, 1 = open sky.
    SHADING_FACTOR = Constant(float, "shading_factor", "Mean Ground Shading Factor", "-")

    # PNG bytes of the most recent shading-pattern plot.
    SHADING_PROGRESS_IMAGE = Constant(bytes, "shading_progress_image", "Ground Shading Progress Image", "png")

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

    # --- Plot state ----------------------------------------------------------
    _plot_progress: bool = False
    _plot_config: Optional[PlotConfig] = None
    _plot_fig: Any = None
    _plot_axes: Any = None
    _last_plot_ts: Optional[pd.Timestamp] = None
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
        """Register the bulk SHADING_FACTOR channel."""
        for c in self.CHANNELS:
            self.data.add(c, aggregate="mean", logger={"enabled": False})

    def _configure_plot(self, configs: Configurations) -> None:
        """Read the ``[plot]`` block and register SHADING_PROGRESS_IMAGE when enabled."""
        self._plot_progress = configs.get_bool("plot_progress", default=True)
        if not self._plot_progress:
            return

        default_dir = str(configs.dirs.data.joinpath("ground_shading"))
        self._plot_config = PlotConfig(
            configs.get_member("plot", defaults={}, ensure_exists=True),
            default_dir=default_dir,
        )
        if self._plot_config.save or self._plot_config.live:
            os.makedirs(self._plot_config.dir, exist_ok=True)
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
        """Publish per-segment GHI [W/m²] to SEG_GHI; non-finite values become NaN with a warning."""
        cleaned = {}
        missing = []
        for name, v in seg_ghi.items():
            if v is not None and np.isfinite(v):
                cleaned[name] = float(v)
            else:
                cleaned[name] = float("nan")
                missing.append(name)
        if missing:
            logger.warning(
                "%s: SEG_GHI has no value for segment(s) %s at %s; writing NaN placeholder.",
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
        if not self._plot_progress or self._plot_config is None:
            return
        if self._last_plot_ts is not None and (ts - self._last_plot_ts) < self._plot_config.interval:
            return
        self._last_plot_ts = ts
        try:
            self._render_progress(ts, ground, pv_rows, sun_state)
        except Exception:  # noqa: BLE001
            logger.exception("%s: progress-plot render failed; disabling.", self.name)
            self._plot_progress = False

    def _init_progress_figure(self) -> None:
        """Create the matplotlib figure once; reuse across renders.
        Forces Agg backend on worker threads (interactive backends require the main thread).
        """
        on_main_thread = threading.current_thread() is threading.main_thread()
        if not on_main_thread:
            if self._plot_config.show:
                logger.warning(
                    "%s: progress plot 'show' disabled; runs on a worker "
                    "thread (matplotlib GUI requires the main thread). Use "
                    "'live = true' to view ground_shading.png in a browser.",
                    self.name,
                )
                self._plot_config.show = False
            if matplotlib.get_backend().lower() not in ("agg", "module://matplotlib_inline.backend_inline"):
                matplotlib.use("Agg", force=True)

        if self._plot_config.show:
            plt.ion()
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
        Persists PNG to SHADING_PROGRESS_IMAGE and optionally to disk per PlotConfig.
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
        ax.set_title(plot_style.format_progress_title("Ground shading", ts, suffix=f"mode: {self._mode}"))

        if self._plot_config.show:
            try:
                self._plot_fig.canvas.draw_idle()
                plt.pause(0.001)
            except Exception:  # noqa: BLE001
                pass

        buf = io.BytesIO()
        self._plot_fig.savefig(buf, dpi=plot_style.DPI, format="png")
        png_bytes = buf.getvalue()

        self.data[GroundShading.SHADING_PROGRESS_IMAGE].set(ts, png_bytes)

        if self._plot_config.live:
            target = os.path.join(self._plot_config.dir, "ground_shading.png")
            tmp = target + ".tmp"
            with open(tmp, "wb") as f:
                f.write(png_bytes)
            os.replace(tmp, target)

        if self._plot_config.save:
            fname = ts.strftime("%Y%m%dT%H%M%S") + ".png"
            with open(os.path.join(self._plot_config.dir, fname), "wb") as f:
                f.write(png_bytes)
