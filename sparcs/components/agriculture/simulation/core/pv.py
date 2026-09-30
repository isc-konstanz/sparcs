# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.simulation.core.pv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pvfactors ground-irradiance helpers shared by the shading models.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from pvfactors.engine import PVEngine
from pvfactors.geometry import OrderedPVArray

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


logger = logging.getLogger(__name__)
# 1. Constants

# 7 rows (3 on each side of the centre) gives the middle row representative inter-row shading.
_GROUND_SHADING_N_ROWS = 7

# Solar zenith [deg] above which pvfactors is unstable; skip and use factor 1.
_ZENITH_DAYTIME_LIMIT = 89.0

# Outer-edge clamp for ground segment x-coordinates [m].
_GROUND_X_CLAMP = 100.0

# Default cadence for [plot] progress-image snapshots, absent a [plot] interval override.
_DEFAULT_PLOT_INTERVAL: str = "1h"

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
