# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.ground_shading
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Per-segment ground shading from a PV array using ``solarfactors`` (a
maintained fork of ``pvfactors``). Couples to ``SoilSimulation`` by
publishing two things for each soil-mesh top segment:

* a shade factor in ``[0, 1]``  — 1 = open sky, 0 = full PV shade.
  Returned from :meth:`GroundShading.evaluate` and applied to local
  irradiance by ``Evapotranspiration``.
* time-mean local irradiance (``ghi_<segment>``, W/m²) — published as
  one channel per top segment so dashboards / loggers / downstream
  callbacks can see the absolute incoming shortwave each segment
  receives, not just the relative shade factor.

File layout:

    1. Module-level constants & geometry-mode names
    2. Free helpers (segment naming, ground-segment math)
    3. Internal classes (``_TrackerConfig``, ``_PVSetup``, ``PlotConfig``)
    4. ``GroundShading`` component
        4a. Channel registration
        4b. Geometry configuration (one builder per mode)
        4c. Segment-range resolution (mesh ↔ pvfactors coordinates)
        4d. Evaluation pipeline
        4e. Progress plotting (SHADING_PROGRESS_IMAGE channel + optional file output)

Ported from ``ground_shading.ipynb``; the notebook keeps the
unconstrained exploratory version.
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
import numpy as np
import pandas as pd
from pvfactors.engine import PVEngine
from pvfactors.geometry import OrderedPVArray
from pvlib.tracking import singleaxis

# pvfactors 1.6.1 (and the ``solarfactors`` fork) was written against
# numpy < 2. ``HybridPerezOrdered.get_full_ts_modeling_vectors`` builds
# ``rho_mat`` / ``inv_rho_mat`` by appending each timeseries surface's
# ``get_param('rho')``, which is a scalar for surfaces with a constant
# reflectivity and an array of length ``n_states`` for the rest. numpy<2
# silently boxed the mixed list as ``dtype=object``; numpy>=2 raises
# ``ValueError: setting an array element with a sequence ... inhomogeneous
# shape``. Broadcast each scalar back to length ``n_states`` so the list
# is homogeneous and the downstream radiosity math (which expects a 2-D
# numeric matrix) keeps working. Patch in-place once at import time.
def _patch_pvfactors_numpy2_compat() -> None:
    from pvfactors.irradiance.models import HybridPerezOrdered, SKY_REFLECTIVITY_DUMMY

    if getattr(HybridPerezOrdered, "_lories_numpy2_patched", False):
        return

    def get_full_ts_modeling_vectors(self, pvarray):
        irradiance_mat, rho_mat, inv_rho_mat, total_perez_mat = (
            self.get_ts_modeling_vectors(pvarray)
        )
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
                    # ``.item()`` handles both 0-d and 1-d size-1 arrays;
                    # ``float(arr)`` raises on the latter under numpy >= 2.
                    arr = np.full(n, arr.item(), dtype=float)
                elif arr.shape[0] != n:
                    # Surface produced a different-length array than n_states;
                    # pad or truncate to keep the matrix rectangular. This
                    # branch is defensive — the scalar case is what we have
                    # seen in the wild.
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


# ---------------------------------------------------------------------------
# 1. Constants
# ---------------------------------------------------------------------------

# We always model 7 rows (3 on each side of the middle row) so the middle
# row sees representative inter-row shading instead of edge-of-array
# behavior. The soil mesh maps onto the ground under the middle row via
# :meth:`GroundShading._resolve_segment_ranges` (which centers the array
# over the plant block). Any user-supplied ``n_rows`` is ignored.
_GROUND_SHADING_N_ROWS = 7

# Solar zenith above which pvfactors is unstable (sun at/below horizon).
# Skip those rows and report a shading factor of 1 for them — at night the
# factor does not change the soil energy balance anyway.
_ZENITH_DAYTIME_LIMIT = 89.0

# Outer-edge clamp for the per-timestep ground segments (m). Keeps every
# downstream x-lookup inside a covering segment without needing bounds
# checks at the call site.
_GROUND_X_CLAMP = 100.0

# Geometry modes selected via ``mode = ...`` in the [ground_shading] block.
MODE_AS_IS = "as_is"            # fixed-tilt rows; supports `mirrored`
MODE_HORIZONTAL = "horizontal"  # row geometry forced flat (surface_tilt = 0)
MODE_TRACKABLE = "trackable"    # single-axis tracker via pvlib.tracking.singleaxis
MODE_FREE_FIELD = "free_field"  # no PV array — open-sky reference baseline
_VALID_MODES = (MODE_AS_IS, MODE_HORIZONTAL, MODE_TRACKABLE, MODE_FREE_FIELD)


# ---------------------------------------------------------------------------
# 2. Free helpers
# ---------------------------------------------------------------------------

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

    pvfactors's ``qinc`` is the total (direct + isotropic + reflection)
    incident on the patch. Combine across setups (e.g. mirrored A-frame):

    - **Direct** multiplies — each setup independently shades the same
      patch: ``combined_direct_frac = ∏ (direct_s / direct_max)``.
    - **Isotropic / reflection** average — same sky and neighbour geometry,
      not independent shading events.

    Combined qinc is then ``combined_direct + ⟨isotropic⟩ + ⟨reflection⟩``.
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
    # Max **direct** component (qinc minus its diffuse parts) — this is
    # what direct fractions normalise against. Open-sky segments where the
    # panels cast no shadow set the ceiling.
    direct_max = max(
        (
            max(0.0, seg[2]["qinc"] - seg[2]["reflection"] - seg[2]["isotropic"])
            for g in grounds_per_setup for seg in g
        ),
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
    """Open-sky GHI on a horizontal surface (W/m²) per row of ``pv_df``.

    ``dni · cos(zenith) + dhi`` — the GHI a horizontal patch would see
    without the array. Used as the ratio reference when converting
    per-segment ``qinc`` to a shade factor; also serves as the per-segment
    irradiance in :data:`MODE_FREE_FIELD`.
    """
    cosz = np.cos(np.radians(pv_df["solar_zenith"].to_numpy()))
    return pv_df["dni"].to_numpy() * cosz + pv_df["dhi"].to_numpy()


# ---------------------------------------------------------------------------
# 3. Internal types
# ---------------------------------------------------------------------------

@dataclass
class PlotConfig:
    """Plot output settings, mirroring SoilSimulation's PlotConfig.

    ``interval`` throttles re-rendering: skip evaluations whose ts is closer
    than ``interval`` to the last rendered one.
    ``live`` overwrites a single ``ground_shading.png`` (paired with an
    auto-generated HTML viewer).
    ``save`` archives one timestamped PNG per render.
    ``show`` pops a matplotlib window — only works on the main thread; auto
    -disabled otherwise.
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
    axis_tilt: float       # rotation-axis tilt from horizontal [deg]
    axis_azimuth: float    # rotation-axis bearing [deg], 180 = N–S axis
    max_angle: float       # tracker rotation limit [deg]
    backtrack: bool
    gcr: float             # ground coverage ratio used for backtracking


class _PVSetup:
    """One PV-array geometry fed to a single solarfactors engine.

    Multiple ``_PVSetup`` instances are combined when the user models a
    structure with two adjacent rows tilted in opposite directions
    (vertical-bifacial / A-frame agrivoltaics).
    """

    def __init__(
        self,
        n_rows: int,
        height: float,
        width: float,
        distance: float,
        axis_azimuth: float,
        surface_tilt: float,
        surface_azimuth: float,
        albedo: float,
        offset_x: float,
    ):
        self.n_rows = n_rows
        self.height = height
        self.width = width
        self.distance = distance
        self.axis_azimuth = axis_azimuth
        self.surface_tilt = surface_tilt
        self.surface_azimuth = surface_azimuth
        self.albedo = albedo
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
        """Run the engine on ``df`` and return per-timestep geometry.

        The returned frame has two columns:

        - ``ground`` — list of ``((x_start, 0), (x_end, 0), {"qinc",
          "reflection", "isotropic"})`` tuples sorted by ``x_start``,
          x-coordinates already shifted by ``self.offset_x``. The first
          segment's ``x_start`` and the last segment's ``x_end`` are
          clamped to ``±_GROUND_X_CLAMP`` so downstream lookups always
          find a covering segment.
        - ``pv_rows`` — list of ``((x_start, y_start), (x_end, y_end),
          {"qinc_front", "qinc_back"})`` tuples, one per PV row, with
          their actual elevated coordinates. Used by the progress plot
          to draw the array geometry on top of the ground.
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
        # pvfactors keeps a fixed surface-per-zone layout per timestep; zones
        # that collapse give zero-length surfaces and trip 0/0 inside
        # TsGroundElement.get_param_weighted. _report_ground drops those, so
        # the divide is noise — suppress locally.
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

        # Each surface yields a length-T array per parameter (T = #timesteps).
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

        # Transpose to (timestep, surface) layout, drop zero-length surfaces,
        # sort by x, and clamp the outermost edges so any downstream x lookup
        # always finds a covering segment.
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
        """Per-timestep PV row segments with their physical x/y coordinates.

        Mirrors ``PVSimulation._build_report_pvrows`` from the notebook so
        the progress plot can draw black row segments on the same scene as
        the ground irradiance.
        """
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
                        b1.x, b1.y, b2.x, b2.y, qinc_front, qinc_back,
                    )
                ]
            )

        # Transpose to (timestep, pvrow) layout — one PV-row segment list
        # per timestep.
        per_timestep = list(map(list, zip(*rows_per_pvrow)))
        return pd.DataFrame({"pv_rows": per_timestep})


# ---------------------------------------------------------------------------
# 4. GroundShading component
# ---------------------------------------------------------------------------

class GroundShading(Component):
    TYPE: str = "ground_shading"

    # Mean exposure of the ground to direct sky: 0 = full shade under PV,
    # 1 = open sky. The per-segment vector is returned from `evaluate`
    # and picked up by `FieldSimulation._segment_shade`; the mean is
    # published here for downstream visibility (logging, dashboards).
    SHADING_FACTOR = Constant(float, "shading_factor", "Mean Ground Shading Factor", "-")

    # PNG bytes of the most recent shading-pattern plot. Logged with
    # ``aggregate="last"`` so a database / UI consumer always sees the
    # newest snapshot. Disabled when ``plot_progress = false``.
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
    # Soil mesh segment x-ranges, resolved at activate from the parent
    # FieldSimulation's MeshConfig. ``None`` means "no soil simulation
    # wired" — we still publish the mean shading factor.
    _segment_ranges: Optional[dict[str, tuple[float, float]]] = None

    # --- Plot state ----------------------------------------------------------
    _plot_progress: bool = False
    _plot_config: Optional[PlotConfig] = None
    _plot_fig: Any = None
    _plot_axes: Any = None
    _last_plot_ts: Optional[pd.Timestamp] = None
    # Last sun-up snapshot of per-timestep PV-row geometry. Reused at
    # night so ``_publish_open_sky`` renders a structure-only frame.
    _last_pv_rows: list

    # Static plot envelope (x: 3 bays around the middle row; y: worst-case
    # panel reach above to soil bottom below). Computed once at activate so
    # the PNG dimensions don't wobble between renders.
    _plot_x_half: float = 0.0
    _plot_y_min: float = -1.0
    _plot_y_max: float = 1.0

    # =========================================================================
    # 4a. Channel registration
    # =========================================================================

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
        """Static plot extent: 3 bays around the middle row (x), worst-case
        panel reach above and soil bottom below (y).

        Computed once so the rendered PNG keeps the same dimensions across
        every tick (the dashboard ``<img>`` would otherwise reflow on each
        update). Worst-case y for trackable mode uses ``max_angle``; for
        fixed-tilt modes the configured ``surface_tilt``; for free_field
        there are no panels at all.
        """
        if self._pv_setups:
            distance = self._pv_setups[0].distance
            self._plot_x_half = distance * 1.5
            if self._mode == MODE_TRACKABLE and self._tracker is not None:
                tilt_max = abs(self._tracker.max_angle)
            else:
                tilt_max = abs(self._surface_tilt)
            tilt_rad = np.radians(tilt_max)
            y_panel = max(
                setup.height + (setup.width / 2.0) * np.sin(tilt_rad)
                for setup in self._pv_setups
            )
            self._plot_y_max = y_panel + 1.0
        else:
            # Free-field: no panels. Pick a reasonable open-sky envelope
            # that matches the bay scale of the soil mesh.
            bay_width = float(getattr(self.context, "bay_width", 3.5))
            self._plot_x_half = bay_width * 1.5
            self._plot_y_max = 1.0

        mesh = getattr(self.context, "mesh_config", None)
        self._plot_y_min = (-mesh.height - 0.5) if mesh is not None else -1.0

    def _register_channels(self) -> None:
        """Register only the bulk SHADING_FACTOR. Per-segment GHI channels
        live on the parent FieldSimulation (single home for segment-scoped
        data — see ``FieldSimulation._register_segment_channels``)."""
        for c in self.CHANNELS:
            self.data.add(c, aggregate="mean", logger={"enabled": False})

    def _configure_plot(self, configs: Configurations) -> None:
        """Read the ``[plot]`` block + ``plot_progress`` flag and register the
        ``SHADING_PROGRESS_IMAGE`` channel when enabled. The output directory
        defaults to ``<data_dir>/ground_shading``."""
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

    # =========================================================================
    # 4b. Geometry configuration
    # =========================================================================

    def _configure_geometry(self, configs: Configurations) -> None:
        """Parse the [ground_shading] block and build the PV setups."""
        mode = str(configs.get("mode", default=MODE_AS_IS)).lower()
        if mode not in _VALID_MODES:
            raise ValueError(
                f"Unsupported ground_shading mode '{mode}'. "
                f"Must be one of: {sorted(_VALID_MODES)}"
            )
        self._mode = mode
        self._albedo = configs.get_float("albedo", default=0.2)

        # Free-field has no array — short-circuit before reading row geometry.
        if mode == MODE_FREE_FIELD:
            self._configure_free_field()
            return

        # Common row geometry; n_rows is always 7 — see ``_GROUND_SHADING_N_ROWS``.
        # ``distance`` (inter-row spacing) defaults to the parent
        # FieldSimulation's ``bay_width`` so the PV bay and the soil mesh
        # always describe the same physical strip. Setting ``distance``
        # explicitly in the [ground_shading] block still wins.
        default_distance = float(getattr(self.context, "bay_width", 3.5))
        common = dict(
            n_rows=_GROUND_SHADING_N_ROWS,
            height=configs.get_float("height", default=3.770),
            width=configs.get_float("width", default=1.134),
            distance=configs.get_float("distance", default=default_distance),
            axis_azimuth=configs.get_float("axis_azimuth", default=100.0),
            albedo=self._albedo,
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
        self, configs: Configurations, common: dict[str, Any],
    ) -> list[_PVSetup]:
        """Same row geometry as `as_is` but tilt forced flat. ``mirrored`` is
        meaningless at zero tilt and is dropped."""
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
        self, configs: Configurations, common: dict[str, Any],
    ) -> list[_PVSetup]:
        """Single-axis tracker — pvfactors accepts per-timestep
        surface_tilt / surface_azimuth, so a single setup is enough; the
        actual rotation is computed in ``_build_pvfactors_input`` via
        :func:`pvlib.tracking.singleaxis`. The pvfactors ``axis_azimuth``
        here is the rotation-axis bearing and matches
        ``tracker.axis_azimuth``.
        """
        self._mirrored = False
        # Placeholder; overridden per-timestep in ``_build_pvfactors_input``.
        self._surface_tilt = 0.0
        self._surface_azimuth = configs.get_float(
            "surface_azimuth", default=common["axis_azimuth"]
        )
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
        self, configs: Configurations, common: dict[str, Any],
    ) -> list[_PVSetup]:
        """Fixed-tilt rows. When ``mirrored = true`` the array is modeled as
        two sub-arrays tilted in opposite directions and offset by half a
        module width along the row-perpendicular axis (vertical-bifacial /
        A-frame agrivoltaics)."""
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

        # A-frame "roof" geometry — both panels lean toward the centerline
        # so their high edges meet at x=0:
        #   left  (offset_x = -half): -tilt → high edge on its right (toward 0)
        #   right (offset_x = +half): +tilt → high edge on its left  (toward 0)
        # In pvfactors's tilt convention with ``surface_azimuth=180`` and
        # ``axis_azimuth ≈ 100``, +tilt produces a ``\`` panel (high edge
        # on the left in 2-D cross-section) and -tilt produces ``/``.
        # Matches the notebook (ground_shading.ipynb cell 9).
        half = common["width"] * np.cos(np.radians(abs(surface_tilt))) / 2.0
        return [
            _PVSetup(surface_tilt=-abs(surface_tilt), offset_x=-half, **common_with_az),
            _PVSetup(surface_tilt=+abs(surface_tilt), offset_x=+half, **common_with_az),
        ]

    # =========================================================================
    # 4c. Segment-range resolution
    # =========================================================================

    def _resolve_segment_ranges(self) -> Optional[dict[str, tuple[float, float]]]:
        """Compute soil-mesh top-segment x-ranges in pvfactors coordinates.

        Pvfactors places PV row 0 at x=0 with rows spaced by ``distance``;
        the array center sits at ``(n_rows - 1) * distance / 2``. We align
        that center over the plant center of the soil mesh, so a segment
        whose mesh x-range is ``[a, b]`` (mesh origin at the field's left
        edge) maps to pvfactors x-range ``[a - shift, b - shift]`` where
        ``shift = plant_center - pv_center``.

        In :data:`MODE_FREE_FIELD` there are no PV rows; the segment
        ranges are still computed from the mesh (free-field reuses the
        mesh layout — only the irradiance source differs) using the
        plant center as the origin.
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
            # Free-field has no array — anchor segment x-coords to the plant
            # center so downstream code can treat ranges uniformly.
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

    # =========================================================================
    # 4d. Evaluation pipeline
    # =========================================================================

    def evaluate(
        self,
        data: Optional[pd.DataFrame] = None,
        *,
        publish: bool = True,
    ) -> Optional[dict[str, float]]:
        """Compute per-segment shading factor for the rows in ``data``.

        Returns the per-segment factors when a soil mesh is wired (so the
        caller can apply them), or ``None`` when there is no mesh — in
        that case only the spatial-mean SHADING_FACTOR is published.

        ``publish=False`` skips every channel write and the live-plot
        capture so callers (e.g. ``SoilPredictor``) can run the chain on
        forecast data without polluting the live dashboards.

        In :data:`MODE_FREE_FIELD` pvfactors is skipped entirely; every
        segment sees factor 1.0 and per-segment GHI = open-sky GHI.
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

        # Run each setup, then collapse the per-setup results into one
        # ground per timestep. Each setup gets its own ``surface_tilt`` /
        # ``surface_azimuth`` written into the frame just before the call —
        # the shared ``pv_df`` carries the component-level scalars (or, in
        # MODE_TRACKABLE, the per-timestep tracker output we don't want to
        # clobber). Without this override the mirrored A-frame collapses
        # to two identical-tilt panels.
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
            # pvfactors 1.6.1 vs numpy ≥ 2 mismatch: surfaces can collapse to
            # zero-area for some sun geometries, producing inhomogeneous-shape
            # arrays that numpy 2's strict ``np.array`` rejects (raises
            # ``ValueError: setting an array element with a sequence``).
            # Length-mismatch / IndexError variants of the same root cause
            # also surface on multi-row inputs. Fall back to open-sky for
            # this evaluation rather than break the chain — the next tick
            # gets a fresh attempt.
            logging.warning(
                "%s: pvfactors raised — falling back to open-sky for this "
                "tick. Underlying cause is usually a pvfactors/numpy>=2 "
                "compat bug; pin numpy<2 to restore real shading.",
                self.name,
                exc_info=True,
            )
            return self._publish_open_sky(data, publish=publish)
        n_t = len(pv_df.index)
        combined_per_t = [
            _combine_grounds([per_setup_ground[s][t] for s in range(len(per_setup_ground))])
            for t in range(n_t)
        ]
        # Concatenate all setups' PV rows per timestep so mirrored arrays
        # show both row sets.
        pv_rows_per_t = [
            [row for s in per_setup_pv_rows for row in s[t]]
            for t in range(n_t)
        ]

        if self._segment_ranges:
            seg_factors, seg_ghi = self._aggregate_per_segment(combined_per_t, ghi_open)
            if publish:
                self._publish_per_segment_ghi(ts, seg_ghi)
        else:
            seg_factors = None
            seg_ghi = {}

        # Bulk SHADING_FACTOR: spatial mean over one bay around the middle
        # row, then time-mean over sun-up rows. Decoupled from
        # ``seg_factors`` (which excludes the watering strip directly under
        # the row — the most-shaded patch) and from the full ground domain
        # (which is clamped to ±_GROUND_X_CLAMP and would otherwise be
        # dominated by far-field open sky).
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
        """Time-mean shade factor + time-mean local GHI for each segment.

        Shade factor uses sun-up rows only (``ghi_open > 0``); GHI averages
        every row so nighttime zeros are reflected. Returns
        ``(seg_factors, seg_ghi)``.
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
        """Project ``data`` onto the columns pvfactors expects.

        ``solar_zenith`` / ``solar_azimuth`` are produced upstream by
        ``validate_meteo_inputs``; ``dni`` / ``dhi`` likewise. Surface
        tilt and azimuth come from the configured PV geometry. Rows with
        the sun at or below the horizon are dropped (pvfactors is
        unstable there). In :data:`MODE_FREE_FIELD` no surface fields
        are populated — the frame is only used to read solar position
        and irradiance.
        """
        df = pd.DataFrame(index=data.index)
        df["solar_zenith"] = data.get("solar_zenith")
        df["solar_azimuth"] = data.get("solar_azimuth")
        df["dni"] = data.get(Weather.DNI)
        df["dhi"] = data.get(Weather.DHI)
        df["albedo"] = self._albedo

        # Drop sun-position / radiation NaNs first so the tracker call sees a
        # clean frame; then derive surface_tilt / surface_azimuth.
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
            # pvlib returns NaN tilt/azimuth when the sun is below the
            # tracker's working envelope. Drop those rows — pvfactors needs
            # a numeric tilt for every timestep it integrates.
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
        """No usable rows (e.g. all night): publish a factor of 1, render a
        structure-only progress frame so the live plot keeps refreshing,
        and return the open-sky per-segment factors so the caller can
        reset the soil sibling to bulk evaporation."""
        ts = data.index[-1]
        if publish:
            self.data[GroundShading.SHADING_FACTOR].set(ts, 1.0)

        # Structure-only frame: panels in their last sun-up orientation,
        # no ground qinc. The shadow-line block in ``_render_progress``
        # self-skips at sun_zen >= _ZENITH_DAYTIME_LIMIT, so 90° is an
        # inert sentinel. On cold-start nights with no cached panels the
        # geometry is synthesized analytically from the setup config so
        # the dashboard still shows the structure.
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
        # Sun below horizon → no incoming irradiance per segment; the
        # shade factor is a ratio against zero so it stays at the
        # neutral 1.0 sentinel.
        if publish:
            self._publish_per_segment_ghi(ts, {name: 0.0 for name in self._segment_ranges})
        return {name: 1.0 for name in self._segment_ranges}

    def _synthesize_pv_rows(self) -> list[tuple]:
        """Analytical PV-row endpoints from the setup config — no pvfactors run.

        Used to render a structure-only frame at night when the cache of
        the last daytime geometry is empty (e.g. cold start after dark).
        For ``MODE_TRACKABLE`` the actual tilt is sun-dependent; we draw
        the rest position (tilt = 0). The returned tuples mirror the
        shape of ``_PVSetup._report_pv_rows`` so ``_render_progress``
        can consume them transparently.
        """
        if not self._pv_setups:
            return []
        rows: list[tuple] = []
        for setup in self._pv_setups:
            tilt_rad = np.radians(setup.surface_tilt)
            half_x = setup.width / 2.0 * np.cos(tilt_rad)
            half_y = setup.width / 2.0 * np.sin(tilt_rad)
            for i in range(setup.n_rows):
                cx = i * setup.distance + setup.offset_x
                # +tilt → high edge on the left (matches the pvfactors
                # convention used in `_build_as_is_setups`).
                rows.append(
                    (
                        (cx - half_x, setup.height + half_y),
                        (cx + half_x, setup.height - half_y),
                        {"qinc_front": 0.0, "qinc_back": 0.0},
                    )
                )
        return rows

    def _publish_per_segment_ghi(self, ts: pd.Timestamp, seg_ghi: dict[str, float]) -> None:
        """Publish per-segment time-mean GHI [W/m²] to the parent
        FieldSimulation's bundled ``SEG_GHI`` channel. Non-finite values
        become NaN within the list — and emit a warning so missing entries
        are visible during debugging instead of silently propagating."""
        cleaned = {}
        missing = []
        for name, v in seg_ghi.items():
            if v is not None and np.isfinite(v):
                cleaned[name] = float(v)
            else:
                cleaned[name] = float("nan")
                missing.append(name)
        if missing:
            logging.warning(
                "%s: SEG_GHI has no value for segment(s) %s at %s — writing NaN placeholder.",
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
        """Mean shade factor over one bay around the middle (4th) row.

        Per timestep, computes the length-weighted mean of ``qinc`` over
        ``[bay_x_min, bay_x_max]`` (= the inter-row distance centered on
        the middle row), divides by the open-sky reference, and clips to
        ``[0, 1]``. Then averages over sun-up rows.

        Restricting to the bay avoids two earlier biases: (a) per-segment
        mean missing the watering strip directly under the row, and (b)
        full-ground mean dominated by the ``±_GROUND_X_CLAMP`` open-sky
        tails outside the array.
        """
        if not self._pv_setups:
            return 1.0
        middles = [
            (setup.n_rows - 1) / 2.0 * setup.distance + setup.offset_x
            for setup in self._pv_setups
        ]
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

    # =========================================================================
    # 4e. Progress plotting
    # =========================================================================

    # Colormap range for the ground-irradiance scene plot. 0..1000 W/m² is
    # the standard "midday clear sky" envelope on a horizontal surface and
    # matches the notebook (cell 12 in ground_shading.ipynb).
    _PLOT_QINC_MAX: float = 1000.0

    def _capture_progress(
        self,
        ts: pd.Timestamp,
        ground: list[tuple],
        pv_rows: list[tuple],
        sun_state: tuple[float, float, Optional[float]],
    ) -> None:
        """Throttle by ``PlotConfig.interval`` and forward to the renderer.

        Wraps the render so a matplotlib failure (missing backend, file
        permission, …) cannot break the rest of ``evaluate`` — the bulk
        SHADING_FACTOR and per-segment publishes already ran before this.
        ``sun_state`` is ``(solar_zenith, solar_azimuth, axis_azimuth)``
        for the last timestep, used to project shadow lines."""
        if not self._plot_progress or self._plot_config is None:
            return
        if (
            self._last_plot_ts is not None
            and (ts - self._last_plot_ts) < self._plot_config.interval
        ):
            return
        self._last_plot_ts = ts
        try:
            self._render_progress(ts, ground, pv_rows, sun_state)
        except Exception:  # noqa: BLE001
            logging.exception("%s: progress-plot render failed; disabling.", self.name)
            self._plot_progress = False

    def _init_progress_figure(self) -> None:
        """Create the matplotlib figure once; reuse the canvas across renders.

        On worker threads we force the ``Agg`` backend regardless of the
        ``show`` setting — interactive backends require the main thread and
        would otherwise hang or raise inside ``plt.subplots``.
        Layout matches the notebook ``plot_pv``: a single wide axis with
        equal aspect so the PV / ground geometry is physically faithful.
        """
        on_main_thread = threading.current_thread() is threading.main_thread()
        if not on_main_thread:
            if self._plot_config.show:
                logging.warning(
                    "%s: progress plot 'show' disabled — runs on a worker "
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
        # ``plasma`` is perceptually uniform and high-contrast across the
        # full range, so partial-shade qinc values around 200–600 W/m²
        # (typical mid-day under-canopy) read clearly instead of fading
        # into a dark band the way the old blue→yellow ramp did.
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
        """Draw the 2-D scene (ground colored by qinc + black PV-row segments)
        following the notebook ``plot_pv``. Persist the PNG bytes to the
        ``SHADING_PROGRESS_IMAGE`` channel and mirror to ``ground_shading.png``
        / archived frames per :class:`PlotConfig`.

        Also overlays shadow projection lines from each PV row endpoint to
        its ground footprint along the sun direction so the geometric
        relationship between rows and shaded patches is visible.
        """
        if self._plot_fig is None:
            self._init_progress_figure()
        ax, cmap, norm = self._plot_axes
        ax.clear()

        # --- Rebase x-axis on the middle (4th) row --------------------------
        # All geometry comes from pvfactors in absolute coordinates (row 0 at
        # x=0, row 3 at x = 3·distance). Subtract that offset so the plot's
        # x=0 sits under the middle row, with the symmetry axes at the
        # left/right edges of the view. For mirrored arrays the per-setup
        # middles average to the same point.
        if self._pv_setups:
            middles = [
                (setup.n_rows - 1) / 2.0 * setup.distance + setup.offset_x
                for setup in self._pv_setups
            ]
            center_x = float(np.mean(middles))
        else:
            center_x = 0.0

        def rx(x: float) -> float:
            """Re-center x on the middle row."""
            return x - center_x

        # --- Ground reference line at y=0 ----------------------------------
        # Thin gray baseline so the structure-only night frames still have a
        # readable ground reference. Drawn behind the shadow projection lines
        # and the qinc-colored segments via zorder.
        ax.axhline(
            y=0.0, color="black", linewidth=0.8, zorder=0.5,
        )

        # --- Ground at y=0, colored by qinc ---------------------------------
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

        # --- Shadow projection lines from PV endpoints toward the ground ----
        # Project each PV row endpoint (px, py) to where the line from the
        # sun through that point meets y=0. In the 2-D cross-section
        # perpendicular to the row axis the ground-x of the shadow is:
        #   shadow_x = px - py · tan(zenith) · sin(sun_az − axis_az)
        # Drawn as thin dashed gray lines so the qinc/PV layers stay legible.
        sun_zen, sun_az, axis_az = sun_state
        if axis_az is not None and sun_zen < _ZENITH_DAYTIME_LIMIT and pv_rows:
            sun_x_per_y = float(
                np.tan(np.radians(sun_zen))
                * np.sin(np.radians(sun_az - axis_az))
            )
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
                        [rx(px), rx(shadow_x)], [py, 0.0],
                        color="gray", linewidth=0.6, linestyle="--", alpha=0.45,
                        zorder=1.5,
                    )

        # --- PV rows in their actual coordinates (black) --------------------
        for seg in pv_rows:
            ax.plot(
                [rx(seg[0][0]), rx(seg[1][0])],
                [seg[0][1], seg[1][1]],
                color="black", linewidth=2,
            )

        # --- Soil mesh cross-section: full ground rectangle + plant block ---
        # Mesh extent comes from _segment_ranges (already in pvfactors x-coords,
        # so rx() handles the re-centering). Soil is a single brown rectangle
        # spanning all top segments down to y=-height; the plant block sits on
        # top of it as a green rectangle [-plant_width/2, +plant_width/2] ×
        # [-plant_height, 0]. The qinc-coloured strip at y=0 already drawn above
        # stays visible because both rectangles use alpha < 1.
        mesh = getattr(self.context, "mesh_config", None)
        if mesh is not None and self._segment_ranges:
            seg_xs = [x for pair in self._segment_ranges.values() for x in pair]
            ground_left = min(seg_xs)
            ground_right = max(seg_xs)
            plant_left = self._segment_ranges["PlantTopLeftSegment"][0]
            plant_right = self._segment_ranges["PlantTopRightSegment"][1]

            soil_left_plot = rx(ground_left)
            soil_width_plot = rx(ground_right) - soil_left_plot
            ax.add_patch(Rectangle(
                (soil_left_plot, -mesh.height),
                soil_width_plot, mesh.height,
                facecolor="saddlebrown", edgecolor="saddlebrown",
                alpha=0.18, linewidth=1.0, zorder=0,
            ))

            plant_left_plot = rx(plant_left)
            plant_width_plot = rx(plant_right) - plant_left_plot
            ax.add_patch(Rectangle(
                (plant_left_plot, -mesh.plant_height),
                plant_width_plot, mesh.plant_height,
                facecolor="forestgreen", edgecolor="darkgreen",
                alpha=0.35, linewidth=1.2, zorder=1,
            ))

        # --- Static window: locked at activate so PNG dimensions stay stable
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

        # Render the figure to PNG once, then fan out to all sinks (channel,
        # live overwrite, archived per-frame file).
        buf = io.BytesIO()
        self._plot_fig.savefig(buf, dpi=120, format="png")
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
