# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.fieldsim.core.plots
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Progress images as pure functions from data to PNG bytes: create the
figure, draw, save to bytes, close it -- no channel, component, or
figure-reuse state. Ported from ``simulation.plot_render`` +
``simulation.plot_style`` (relative-saturation cross-section) and
``simulation.ground_shading`` (shading pattern frame); the strike counter
and the ``plot_strikes``/image channel writes stay in the IO layer.
``PlotConfig`` itself is declared in ``config`` with the other sections.
"""

from __future__ import annotations

import io
import logging
import os
import sys
import threading
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, PowerNorm
from scipy.interpolate import griddata

import numpy as np
import pandas as pd

from .config import PlotConfig

PlotConfig = PlotConfig  # declared in config.py; re-exported for the chain

# --------------------------------------------------------------------------- backend

_NON_GUI_BACKENDS = ("agg", "module://matplotlib_inline.backend_inline")


def _ensure_safe_backend() -> None:
    """Force ``Agg`` when the caller can't drive a GUI (port of
    ``plot_render._ensure_safe_backend``)."""
    backend = matplotlib.get_backend().lower()
    if backend in _NON_GUI_BACKENDS:
        return
    on_main_thread = threading.current_thread() is threading.main_thread()
    has_display = sys.platform != "linux" or bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    if on_main_thread and has_display:
        return
    logging.getLogger(__name__).debug(
        "Switching matplotlib backend %s -> Agg for headless / off-thread render.",
        backend,
    )
    matplotlib.use("Agg", force=True)


# --------------------------------------------------------------------------- shared style (port of plot_style.py)

_FIG_WIDTH_IN: float = 8.0
_DPI: int = 120
_MARGIN = {"left": 0.9, "right": 1.2, "bottom": 0.55, "top": 0.45}
_AXES_WIDTH_IN: float = _FIG_WIDTH_IN - _MARGIN["left"] - _MARGIN["right"]
_VERTICAL_CHROME_IN: float = _MARGIN["top"] + _MARGIN["bottom"]

_GRID_ALPHA: float = 0.3
_COLORMAP: str = "plasma"
_CBAR_SHRINK: float = 0.8

_AXIS_LABEL_X: str = "x [m]"
_AXIS_LABEL_Y: str = "y [m]"

_TIMESTAMP_FORMAT: str = "%Y-%m-%d %H:%M"

_MODE_TRACKABLE: str = "trackable"
_ZENITH_DAYTIME_LIMIT: float = 89.0
_PLOT_QINC_MAX: float = 1000.0


def _compute_fig_size(x_extent: float, y_extent: float) -> tuple[float, float]:
    """Figure ``(width_in, height_in)`` preserving the data aspect ratio with fixed width."""
    if x_extent <= 0:
        x_extent = 1.0
    if y_extent <= 0:
        y_extent = 1.0
    axes_h_in = _AXES_WIDTH_IN * y_extent / x_extent
    return _FIG_WIDTH_IN, axes_h_in + _VERTICAL_CHROME_IN


def _apply_subplots_adjust(fig: Any) -> None:
    """Convert inch margins to figure-relative fractions and call ``subplots_adjust``."""
    w, h = fig.get_size_inches()
    fig.subplots_adjust(
        left=_MARGIN["left"] / w,
        right=1.0 - _MARGIN["right"] / w,
        bottom=_MARGIN["bottom"] / h,
        top=1.0 - _MARGIN["top"] / h,
    )


def _apply_axes_style(ax: Any) -> None:
    """Bracketed-unit labels, light grid, equal aspect (shared on both plots)."""
    ax.set_xlabel(_AXIS_LABEL_X)
    ax.set_ylabel(_AXIS_LABEL_Y)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=_GRID_ALPHA)


def _localize_timestamp(ts: pd.Timestamp, tz: Any) -> pd.Timestamp:
    """Return ``ts`` in the site timezone ``tz``. A naive ``ts`` is assumed UTC;
    ``tz=None`` returns ``ts`` unchanged (naive stays naive -> no offset shown)."""
    if tz is None:
        return ts
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert(tz)


def _offset_suffix(ts: pd.Timestamp) -> str:
    """`` +HH:MM`` for a tz-aware ``ts`` (colon-form offset), or ``""`` when naive."""
    raw = ts.strftime("%z")
    return f" {raw[:3]}:{raw[3:]}" if raw else ""


def _format_progress_title(label: str, ts: pd.Timestamp, *, tz: Any = None) -> str:
    """``"<label> -- YYYY-MM-DD HH:MM[ +HH:MM]"`` -- the shared title format."""
    ts = _localize_timestamp(ts, tz)
    return f"{label} — {ts.strftime(_TIMESTAMP_FORMAT)}{_offset_suffix(ts)}"


class _SmoothstepNorm(Normalize):
    """Smoothstep colormap norm ``f(x) = 3x^2 - 2x^3`` (port of ``plot_style.SmoothstepNorm``)."""

    def __call__(self, value, clip=None):
        v_min = float(self.vmin)
        v_max = float(self.vmax)
        denom = max(v_max - v_min, 1e-12)
        v = (np.asarray(value, dtype=float) - v_min) / denom
        v = np.clip(v, 0.0, 1.0)
        return 3.0 * v * v - 2.0 * v * v * v

    def inverse(self, value):
        v_min = float(self.vmin)
        v_max = float(self.vmax)
        y = np.clip(np.asarray(value, dtype=float), 0.0, 1.0)
        x = 0.5 - np.sin(np.arcsin(1.0 - 2.0 * y) / 3.0)
        return x * (v_max - v_min) + v_min


def _saturation_norm(vmin: float = 0.0, vmax: float = 1.0) -> Normalize:
    return _SmoothstepNorm(vmin=vmin, vmax=vmax)


# --------------------------------------------------------------------------- render_due


def render_due(last: Optional[pd.Timestamp], now: pd.Timestamp, config: Optional[PlotConfig]) -> bool:
    return config is not None and (last is None or now - last >= config.interval)


# --------------------------------------------------------------------------- shading envelope


@dataclass(frozen=True)
class ShadingEnvelope:
    """Static plot extent, computed once so PNG size stays stable (port of
    ``GroundShading._compute_plot_envelope``)."""

    x_half: float
    y_min: float
    y_max: float


def shading_envelope(
    *,
    mode: str,
    pv_setups: Sequence[Any],
    tracker: Any,
    surface_tilt: float,
    bay_width: float,
    mesh_height: float,
) -> ShadingEnvelope:
    """3 bays around the middle row, worst-case panel height above, soil
    bottom below (exact port of ``GroundShading._compute_plot_envelope``
    over the same inputs it reads from ``self``). ``pv_setups`` items need
    only ``.distance``, ``.height``, ``.width``; ``tracker`` needs only
    ``.max_angle``."""
    if pv_setups:
        distance = pv_setups[0].distance
        x_half = distance * 1.5
        if mode == _MODE_TRACKABLE and tracker is not None:
            tilt_max = abs(tracker.max_angle)
        else:
            tilt_max = abs(surface_tilt)
        tilt_rad = np.radians(tilt_max)
        y_panel = max(setup.height + (setup.width / 2.0) * np.sin(tilt_rad) for setup in pv_setups)
        y_max = y_panel + 1.0
    else:
        x_half = bay_width * 1.5
        y_max = 1.0

    y_min = -mesh_height - 0.5
    return ShadingEnvelope(x_half=x_half, y_min=y_min, y_max=y_max)


# --------------------------------------------------------------------------- renderers


def render_shading_png(
    ts: pd.Timestamp,
    ground: Sequence[tuple],
    pv_rows: Sequence[tuple],
    sun_state: tuple[float, float, Optional[float]],
    envelope: ShadingEnvelope,
    *,
    title: str = "Ground shading",
    tz: Any = None,
) -> bytes:
    """Shading pattern frame: ground coloured by qinc, PV rows in black,
    shadow projection lines (port of ``GroundShading._render_progress``,
    minus the middle-row re-centring and soil cross-section rectangles,
    which need geometry this pure function isn't given). Figure is created
    and closed per call."""
    _ensure_safe_backend()
    x_extent = 2.0 * envelope.x_half
    y_extent = envelope.y_max - envelope.y_min
    fig, ax = plt.subplots(figsize=_compute_fig_size(x_extent, y_extent), dpi=_DPI)
    try:
        cmap = plt.get_cmap(_COLORMAP)
        norm = PowerNorm(gamma=0.5, vmin=0.0, vmax=_PLOT_QINC_MAX)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, shrink=_CBAR_SHRINK, label="incident irradiance [W/m²]")
        _apply_subplots_adjust(fig)

        ax.axhline(y=0.0, color="black", linewidth=0.8, zorder=0.5)

        for seg in ground:
            qinc = max(0.0, seg[2]["qinc"])
            ax.plot(
                [seg[0][0], seg[1][0]],
                [0.0, 0.0],
                color=cmap(norm(qinc)),
                linewidth=6,
                solid_capstyle="butt",
                zorder=2,
            )

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
                        [px, shadow_x],
                        [py, 0.0],
                        color="gray",
                        linewidth=0.6,
                        linestyle="--",
                        alpha=0.45,
                        zorder=1.5,
                    )

        for seg in pv_rows:
            ax.plot(
                [seg[0][0], seg[1][0]],
                [seg[0][1], seg[1][1]],
                color="black",
                linewidth=2,
            )

        ax.set_xlim(-envelope.x_half, envelope.x_half)
        ax.set_ylim(envelope.y_min, envelope.y_max)
        _apply_axes_style(ax)
        ax.set_title(_format_progress_title(title, ts, tz=tz))

        buf = io.BytesIO()
        fig.savefig(buf, dpi=_DPI, format="png")
        return buf.getvalue()
    finally:
        plt.close(fig)


def render_rel_sat_png(
    mesh: Any,
    rel_sat_values: np.ndarray,
    sim_t: pd.Timestamp,
    *,
    width_m: float,
    height_m: float,
    title: str = "Relative saturation",
    tz: Any = None,
) -> bytes:
    """Relative-saturation cross-section (port of
    ``plot_render.init_rel_sat_figure`` + ``render_rel_sat_png``, folded
    into one call). ``mesh`` only needs ``.cellCenters`` (array shape
    ``(2, N)``). Figure is created and closed per call."""
    _ensure_safe_backend()
    fig, ax = plt.subplots(figsize=_compute_fig_size(width_m, height_m), dpi=_DPI)
    try:
        norm = _saturation_norm(vmin=0.0, vmax=1.0)
        sm = plt.cm.ScalarMappable(cmap=_COLORMAP, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, shrink=_CBAR_SHRINK, label="relative saturation [-]")
        _apply_subplots_adjust(fig)

        x, y = mesh.cellCenters
        xi = np.linspace(np.min(x), np.max(x), 100)
        yi = np.linspace(np.min(y), np.max(y), 100)
        zi = griddata(
            (np.asarray(x), np.asarray(y)),
            rel_sat_values,
            (xi[None, :], yi[:, None]),
            method="cubic",
        )

        ax.contourf(xi, yi, zi, levels=15, cmap=_COLORMAP, norm=norm)
        ax.contour(xi, yi, zi, levels=15, linewidths=0.5, colors="k")
        _apply_axes_style(ax)
        ax.set_title(_format_progress_title(title, sim_t, tz=tz))

        buf = io.BytesIO()
        fig.savefig(buf, dpi=_DPI, format="png")
        return buf.getvalue()
    finally:
        plt.close(fig)
