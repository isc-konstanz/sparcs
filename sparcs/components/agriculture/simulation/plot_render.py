# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.simulation.plot_render
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Matplotlib helpers for rendering soil saturation fields as PNG.
Shared by SoilSimulation and SoilPredictor. Switches to Agg automatically
when running off the main thread or on a headless host.
"""

from __future__ import annotations

import io
import logging
import os
import sys
import threading
from typing import Any, Tuple

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.interpolate import griddata

import numpy as np
import pandas as pd

from . import plot_style

_NON_GUI_BACKENDS = ("agg", "module://matplotlib_inline.backend_inline")


def _ensure_safe_backend() -> None:
    """Force ``Agg`` when the caller can't drive a GUI.

    No-op when the current backend is already non-interactive, or when
    the caller is on the main thread with a display attached (the only
    case where an interactive window is actually possible).
    """
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


def init_rel_sat_figure(
    width_m: float,
    height_m: float,
) -> Tuple[Any, Any, Normalize]:
    """Build a fig/ax/norm triple for a ``width_m × height_m`` soil cross-section with saturation colorbar."""
    _ensure_safe_backend()
    fig, ax = plt.subplots(
        figsize=plot_style.compute_fig_size(width_m, height_m),
        dpi=plot_style.DPI,
    )
    norm = plot_style.saturation_norm(vmin=0.0, vmax=1.0)
    sm = plt.cm.ScalarMappable(cmap=plot_style.COLORMAP, norm=norm)
    sm.set_array([])
    fig.colorbar(
        sm,
        ax=ax,
        shrink=plot_style.CBAR_SHRINK,
        label="relative saturation [-]",
    )
    plot_style.apply_subplots_adjust(fig)
    return fig, ax, norm


def render_rel_sat_png(
    fig: Any,
    ax: Any,
    norm: Normalize,
    mesh: Any,
    rel_sat_values: np.ndarray,
    sim_t: pd.Timestamp,
    *,
    title: str = "Relative saturation",
    tz=None,
) -> bytes:
    """Render ``rel_sat_values`` (one value per FiPy cell) onto ``fig``/``ax`` and return PNG bytes."""
    x, y = mesh.cellCenters
    xi = np.linspace(np.min(x), np.max(x), 100)
    yi = np.linspace(np.min(y), np.max(y), 100)
    zi = griddata(
        (np.asarray(x), np.asarray(y)),
        rel_sat_values,
        (xi[None, :], yi[:, None]),
        method="cubic",
    )

    ax.clear()
    ax.contourf(xi, yi, zi, levels=15, cmap=plot_style.COLORMAP, norm=norm)
    ax.contour(xi, yi, zi, levels=15, linewidths=0.5, colors="k")
    plot_style.apply_axes_style(ax)
    ax.set_title(plot_style.format_progress_title(title, sim_t, tz=tz))

    buf = io.BytesIO()
    fig.savefig(buf, dpi=plot_style.DPI, format="png")
    return buf.getvalue()
