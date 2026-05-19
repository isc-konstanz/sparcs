# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.simulation.plot_render
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Reusable matplotlib helpers for rendering a saturation field on the soil
PDE cross-section as a PNG. Shared by :class:`SoilSimulation` (live
progress plot, fans the PNG to file sinks + a logger channel) and
:class:`SoilPredictor` (per-snapshot prediction plots emitted on the
``predict_plot`` bytes channel).

The module also owns a small "safe backend" negotiation: matplotlib's
default backend on macOS is ``macosx`` which can only build figures
from the main thread, so the moment a listener / callback worker tries
to render — which is exactly what both callers do — ``plt.subplots``
raises ``RuntimeError: Cannot create a GUI FigureManager outside the
main thread``. ``init_rel_sat_figure`` switches to ``Agg`` on first use
when the caller isn't on the main thread (or the host is headless).
Callers do not need to repeat the dance.
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
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from scipy.interpolate import griddata

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
    has_display = sys.platform != "linux" or bool(
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    )
    if on_main_thread and has_display:
        return
    logging.getLogger(__name__).debug(
        "Switching matplotlib backend %s -> Agg for headless / off-thread render.",
        backend,
    )
    matplotlib.use("Agg", force=True)


def init_rel_sat_figure(
    width_m: float, height_m: float,
) -> Tuple[Any, Any, Normalize]:
    """Build a fig/ax/norm triple sized for a ``width_m × height_m`` soil
    cross-section, with the shared smoothstep saturation colorbar attached.

    Forces Agg on first call if running off the main thread or on a
    headless host, so pyplot doesn't raise from a listener callback.
    """
    _ensure_safe_backend()
    fig, ax = plt.subplots(
        figsize=plot_style.compute_fig_size(width_m, height_m),
        dpi=plot_style.DPI,
    )
    # Smoothstep norm stretches the mid-saturation range visually so
    # typical operating values (Se ≈ 0.3–0.7) get most of the colorbar's
    # contrast, while near-dry / near-saturated bands are compressed.
    # Colorbar tick labels stay in physical Se units because
    # ``SmoothstepNorm.inverse`` is the analytic inverse.
    norm = plot_style.saturation_norm(vmin=0.0, vmax=1.0)
    sm = plt.cm.ScalarMappable(cmap=plot_style.COLORMAP, norm=norm)
    sm.set_array([])
    fig.colorbar(
        sm, ax=ax, shrink=plot_style.CBAR_SHRINK,
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
) -> bytes:
    """Render ``rel_sat_values`` (one value per FiPy cell) on ``mesh`` into
    ``fig``/``ax`` and return PNG bytes.

    Mutates the passed-in axes (clear + redraw) so the same fig/ax can
    be reused across many calls without leaking artists.
    """
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
    # Contour levels stay linear in Se (the iso-saturation lines are
    # physically meaningful at evenly spaced Se values); the colour
    # mapping is reshaped by ``norm`` so the same 15 bands give finer
    # mid-range gradient and coarser endpoint gradient.
    ax.contourf(xi, yi, zi, levels=15, cmap=plot_style.COLORMAP, norm=norm)
    ax.contour(xi, yi, zi, levels=15, linewidths=0.5, colors="k")
    plot_style.apply_axes_style(ax)
    # Mesh y is positive depth; flip so the surface (y=0) sits at the top
    # of the figure and depth grows downward.
    ax.invert_yaxis()
    ax.set_title(plot_style.format_progress_title(title, sim_t))

    buf = io.BytesIO()
    fig.savefig(buf, dpi=120, format="png")
    return buf.getvalue()
