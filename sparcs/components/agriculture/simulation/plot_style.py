# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.simulation.plot_style
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Shared visual contract for progress-plot PNGs (SoilSimulation, GroundShading):
fixed width, equal-aspect axes, shared margins, plasma colormap, and timestamp titles.
"""

from __future__ import annotations

from matplotlib.colors import Normalize

import numpy as np
import pandas as pd

# Style constants

FIG_WIDTH_IN: float = 8.0  # inches; 960 px at DPI=120
DPI: int = 120

# Fixed inch margins so axes sit at the same position across all renders;
# ``right`` reserves space for the colorbar.
MARGIN = {
    "left": 0.9,
    "right": 1.2,
    "bottom": 0.55,
    "top": 0.45,
}

AXES_WIDTH_IN: float = FIG_WIDTH_IN - MARGIN["left"] - MARGIN["right"]
VERTICAL_CHROME_IN: float = MARGIN["top"] + MARGIN["bottom"]  # title + xlabel inches

GRID_ALPHA: float = 0.3
COLORMAP: str = "plasma"
CBAR_SHRINK: float = 0.8

AXIS_LABEL_X: str = "x [m]"
AXIS_LABEL_Y: str = "y [m]"

TIMESTAMP_FORMAT: str = "%Y-%m-%d %H:%M"


# Helpers


def compute_fig_size(x_extent: float, y_extent: float) -> tuple[float, float]:
    """Figure ``(width_in, height_in)`` preserving the data aspect ratio with fixed width."""
    if x_extent <= 0:
        x_extent = 1.0
    if y_extent <= 0:
        y_extent = 1.0
    axes_h_in = AXES_WIDTH_IN * y_extent / x_extent
    return FIG_WIDTH_IN, axes_h_in + VERTICAL_CHROME_IN


def apply_subplots_adjust(fig) -> None:
    """Convert inch margins to figure-relative fractions and call ``subplots_adjust``."""
    w, h = fig.get_size_inches()
    fig.subplots_adjust(
        left=MARGIN["left"] / w,
        right=1.0 - MARGIN["right"] / w,
        bottom=MARGIN["bottom"] / h,
        top=1.0 - MARGIN["top"] / h,
    )


def apply_axes_style(ax) -> None:
    """Bracketed-unit labels, light grid, equal aspect (shared on both plots)."""
    ax.set_xlabel(AXIS_LABEL_X)
    ax.set_ylabel(AXIS_LABEL_Y)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=GRID_ALPHA)


def _localize_timestamp(ts: pd.Timestamp, tz) -> pd.Timestamp:
    """Return ``ts`` in the site timezone ``tz``. A naive ``ts`` is assumed UTC;
    ``tz=None`` returns ``ts`` unchanged (naive stays naive → no offset shown)."""
    if tz is None:
        return ts
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert(tz)


def _offset_suffix(ts: pd.Timestamp) -> str:
    """`` +HH:MM`` for a tz-aware ``ts`` (colon-form offset), or ``""`` when naive."""
    raw = ts.strftime("%z")  # "+0200" / "-0500", or "" when naive
    return f" {raw[:3]}:{raw[3:]}" if raw else ""


def format_progress_title(label: str, ts: pd.Timestamp, *, tz=None) -> str:
    """``"<label> — YYYY-MM-DD HH:MM[ +HH:MM]"`` — the shared title format.

    ``tz`` (an IANA name or ``tzinfo``, from ``location.timezone``) renders the
    timestamp in site-local time with a colon-form offset; a naive ``ts`` is
    assumed UTC. ``tz=None`` keeps ``ts`` as-is with no offset."""
    ts = _localize_timestamp(ts, tz)
    return f"{label} — {ts.strftime(TIMESTAMP_FORMAT)}{_offset_suffix(ts)}"


class SmoothstepNorm(Normalize):
    """Smoothstep colormap norm ``f(x) = 3x² - 2x³``: stretches mid-range, compresses extremes.

    Inverse ``g(y) = 0.5 - sin(arcsin(1 - 2y) / 3)`` keeps colorbar ticks in physical Se units.
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
    """Return a fresh :class:`SmoothstepNorm` for soil-saturation plots."""
    return SmoothstepNorm(vmin=vmin, vmax=vmax)
