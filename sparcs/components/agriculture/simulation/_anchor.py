# -*- coding: utf-8 -*-
"""Pure constant-gain analysis update for anchoring the soil saturation field.

The simulation runs Richards forward from weather and irrigation alone, so it
drifts from the real bay over a season. This nudges the simulated
effective-saturation (Se) field toward tensiometer readings near each sensor: a
constant-gain (alpha) analysis step -- a simplified Kalman update with a fixed
steady-state gain -- localized to an anisotropic ellipse around each sensor and
combined across sensors by a precision-weighted mean.

This module is deliberately free of FiPy and the retention model: it operates on
plain numpy arrays in Se space. The pF->Se measurement conversion and the
per-sensor variance live in the caller (the observation adapter), so this core is
unit-testable on a synthetic cell grid with no mesh and no soil model. See the
``[anchor]`` design in ``.scratch/soil-sensor-anchoring/PRD.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

# Sharpness of the Gaussian localization taper, in units of normalised distance
# (d = 1 at the ellipse edge). The taper is a Gaussian of this std, shifted to
# vanish continuously at d = 1 so the next solve is not handed a saturation step.
# Gaspari-Cohn is the noted upgrade if strict compact support is later wanted.
_TAPER_STD = 0.5


@dataclass(frozen=True)
class AnchorObservation:
    """One tensiometer reading mapped into Se space at a mesh location.

    ``x_m``, ``y_m``: sensor location in mesh coordinates (metres; ``y`` negative
    downward), matching ``mesh.cellCenters``. ``se_meas``: measured effective
    saturation (from tension via the retention model's ``se_from_psi``).
    ``variance``: measurement variance R in Se^2 units (must be > 0), formed by
    the caller from the per-sensor pF std.
    """

    x_m: float
    y_m: float
    se_meas: float
    variance: float


def _localization_weights(cell_centers: np.ndarray, x_m: float, y_m: float, r_h: float, r_v: float) -> np.ndarray:
    """Anisotropic Gaussian ellipse taper: 1 at the sensor, smoothly 0 at d >= 1.

    Normalised squared distance ``d^2 = (dx / r_h)^2 + (dy / r_v)^2``; the
    Gaussian is shifted and rescaled so the weight is exactly 1 at the sensor and
    falls continuously to 0 at the ellipse edge, then held at 0 beyond it.
    """
    dx = cell_centers[0] - x_m
    dy = cell_centers[1] - y_m
    d2 = (dx / r_h) ** 2 + (dy / r_v) ** 2
    g = np.exp(-d2 / (2.0 * _TAPER_STD**2))
    g_edge = np.exp(-1.0 / (2.0 * _TAPER_STD**2))
    w = (g - g_edge) / (1.0 - g_edge)
    return np.where(d2 < 1.0, w, 0.0)


def anchor_field(
    se: np.ndarray,
    cell_centers: np.ndarray,
    observations: Iterable[AnchorObservation],
    sigma_sys: float,
    r_h: float,
    r_v: float,
    se_min: float,
    se_max: float,
) -> np.ndarray:
    """Pull the Se field toward the observations and return the corrected field.

    For each cell touched by one or more fresh sensors, blend the model state and
    the sensor-implied state by a precision-weighted mean::

        Se_new(c) = ( Se(c)/sigma_sys^2 + sum_s w_s(c) Se_meas,s / R_s )
                    / ( 1/sigma_sys^2     + sum_s w_s(c) / R_s )

    where ``w_s(c)`` is the localization weight. Order-independent, cannot
    overshoot either reading, and reduces to ``Se + k (Se_meas - Se)`` with
    ``k = sigma_sys^2 / (R + sigma_sys^2)`` for a single sensor at its own cell.
    Cells out of every sensor's reach are returned unchanged. The result is
    clipped to ``[se_min, se_max]``.

    ``se``: (N,) effective-saturation field. ``cell_centers``: (2, N) mesh cell
    centres in metres (row 0 x, row 1 y, y < 0 down). ``sigma_sys``: model std in
    Se units. ``r_h``, ``r_v``: localization radii in metres. ``se_min``,
    ``se_max``: physical clip bounds (live callers pass ``_soil.SE_MIN`` /
    ``SE_MAX``). A non-positive ``sigma_sys`` or radius makes the update a no-op.
    """
    se = np.asarray(se, dtype=float)
    cell_centers = np.asarray(cell_centers, dtype=float)
    observations = list(observations)

    if sigma_sys <= 0.0 or r_h <= 0.0 or r_v <= 0.0 or not observations:
        return se.copy()

    inv_sys = 1.0 / sigma_sys**2
    numerator = se * inv_sys
    denominator = np.full_like(se, inv_sys)

    for obs in observations:
        precision = _localization_weights(cell_centers, obs.x_m, obs.y_m, r_h, r_v) / obs.variance
        numerator = numerator + precision * obs.se_meas
        denominator = denominator + precision

    return np.clip(numerator / denominator, se_min, se_max)
