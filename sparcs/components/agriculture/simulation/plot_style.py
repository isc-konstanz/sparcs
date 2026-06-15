# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.simulation.plot_style
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Shared dashboard-progress-plot style for the field-simulation chain.

The two PNG-emitting components in this package — ``SoilSimulation`` and
``GroundShading`` — render different scenes (a contour cross-section vs.
a row-and-shadow geometry view) but live next to each other in the
sparcs Dash UI. This module pins down the look-and-feel both share so
they read as siblings in the dashboard:

* fixed PNG width with ``aspect="equal"`` so 1 m on x = 1 m on y inside
  each plot, with the figure height derived from the y-data extent.
* identical inch margins via :data:`MARGIN`, locked with
  ``fig.subplots_adjust(...)`` so the axes box sits in the same place
  across both PNGs.
* the same font/grid/colorbar treatment (``plasma`` ramp, 0.3-alpha
  grid, bracketed unit labels, ``%Y-%m-%d %H:%M`` titles).

Each renderer still owns its own data → axes mapping; this module is
the visual contract, not a renderer.
"""

from __future__ import annotations

from typing import Optional

from matplotlib.colors import Normalize

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

# 8 inches × 120 DPI = 960 px wide. Dashboard cards use ``maxWidth: 100%``,
# so this is an upper bound — the browser scales down as needed and the
# PNG aspect (locked by `aspect="equal"`) is what stays stable.
FIG_WIDTH_IN: float = 8.0
DPI: int = 120

# Inch margins around the axes. Fixed values (rather than ``tight_layout``)
# so the axes box sits at the same figure-relative position in every
# render — both renderers' PNGs end up framed identically.
# ``right`` reserves room for the colorbar.
MARGIN = {
    "left": 0.9,
    "right": 1.2,
    "bottom": 0.55,
    "top": 0.45,
}

# Inner usable axes width in inches, after subtracting left/right margins.
AXES_WIDTH_IN: float = FIG_WIDTH_IN - MARGIN["left"] - MARGIN["right"]

# Vertical inches consumed by title + xlabel (title at top, x-tick labels
# and xlabel at bottom). Used when deriving figure height from data.
VERTICAL_CHROME_IN: float = MARGIN["top"] + MARGIN["bottom"]

GRID_ALPHA: float = 0.3
COLORMAP: str = "plasma"
CBAR_SHRINK: float = 0.8

AXIS_LABEL_X: str = "x [m]"
AXIS_LABEL_Y: str = "y [m]"

# Human-friendly timestamp shown in the plot title. Drops ISO seconds and
# timezone — at the dashboard refresh cadence (1 h default) these are noise.
TIMESTAMP_FORMAT: str = "%Y-%m-%d %H:%M"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def compute_fig_size(x_extent: float, y_extent: float) -> tuple[float, float]:
    """Figure ``(width_in, height_in)`` for the shared layout.

    Width is fixed at :data:`FIG_WIDTH_IN`. Height is derived from the
    data's ``y_extent / x_extent`` aspect ratio so that, with
    ``aspect="equal"`` and the shared :data:`MARGIN`, the axes box
    exactly fits the data without distortion.
    """
    if x_extent <= 0:
        x_extent = 1.0
    if y_extent <= 0:
        y_extent = 1.0
    axes_h_in = AXES_WIDTH_IN * y_extent / x_extent
    return FIG_WIDTH_IN, axes_h_in + VERTICAL_CHROME_IN


def apply_subplots_adjust(fig) -> None:
    """Lock the axes box at the shared inch margins.

    Translates the inch-margin dict into the figure-relative
    ``left/right/bottom/top`` ``subplots_adjust`` accepts.
    """
    w, h = fig.get_size_inches()
    fig.subplots_adjust(
        left=MARGIN["left"] / w,
        right=1.0 - MARGIN["right"] / w,
        bottom=MARGIN["bottom"] / h,
        top=1.0 - MARGIN["top"] / h,
    )


def apply_axes_style(ax) -> None:
    """Bracketed-unit labels, light grid, equal aspect — shared on both plots."""
    ax.set_xlabel(AXIS_LABEL_X)
    ax.set_ylabel(AXIS_LABEL_Y)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=GRID_ALPHA)


def format_progress_title(label: str, ts: pd.Timestamp, suffix: Optional[str] = None) -> str:
    """``"<label> — YYYY-MM-DD HH:MM[ (<suffix>)]"`` — the shared title format."""
    base = f"{label} — {ts.strftime(TIMESTAMP_FORMAT)}"
    if suffix:
        return f"{base} ({suffix})"
    return base


class SmoothstepNorm(Normalize):
    """Stretch the middle of ``[vmin, vmax]``, compress the extremes.

    Applies the smoothstep transform ``f(x) = 3x² - 2x³`` to values
    normalised into ``[0, 1]`` before they reach the colormap. The
    forward derivative is zero at the endpoints and peaks at 1.5 in
    the middle — visually, small differences around Se ≈ 0.5 produce
    bigger colour swings than the same differences at Se ≈ 0 or
    Se ≈ 1. Useful for soil-moisture plots where the operating
    regime sits in the middle band and saturation extremes are rare
    but still worth a sentinel colour.

    The inverse uses the closed-form smoothstep inverse
    ``g(y) = 0.5 - sin(arcsin(1 - 2y) / 3)`` so the colorbar tick
    labels stay in physical Se units.
    """

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


def saturation_norm(vmin: float = 0.0, vmax: float = 1.0) -> Normalize:
    """The shared :class:`SmoothstepNorm` for soil-saturation plots.

    Wrapped as a function so callers (currently
    ``SoilSimulation._render_progress``) can pin the range explicitly
    without touching the class API. Returning a fresh instance per
    call keeps matplotlib's internal ``Normalize`` state isolated
    between figures.
    """
    return SmoothstepNorm(vmin=vmin, vmax=vmax)
