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

import math
from dataclasses import dataclass, replace
from typing import Any, Iterable

import numpy as np

# Sharpness of the Gaussian localization taper, in units of normalised distance
# (d = 1 at the ellipse edge). The taper is a Gaussian of this std, shifted to
# vanish continuously at d = 1 so the next solve is not handed a saturation step.
# Gaspari-Cohn is the noted upgrade if strict compact support is later wanted.
_TAPER_STD = 0.5

# 1 hPa of pressure equals this many metres of water column (4 degC).
_HPA_TO_M_WATER = 0.0101972

_LN10 = math.log(10.0)

# Floor on the measurement variance in Se^2 units. Near saturation the pF->Se
# Jacobian collapses -- the flat retention curve maps a fixed pF error to a
# vanishing Se error -- which would hand a near-saturated sensor near-infinite
# trust and let it overwrite the field. The floor keeps the gain finite. This is
# the PRD's recorded watch-point (.scratch/soil-sensor-anchoring/PRD.md).
_MIN_VARIANCE = 1.0e-6

# dh_dse is singular at the saturation bounds (Se = 0 or 1); evaluate the slope a
# hair inside them. At the bounds head_m is extreme or zero anyway, so the floor
# governs the variance regardless of the exact clipped slope.
_SE_EPS = 1.0e-9


@dataclass(frozen=True)
class AnchorObservation:
    """One tensiometer reading mapped into Se space at a mesh location.

    ``x_m``, ``y_m``: sensor location in mesh coordinates (metres; ``y`` negative
    downward), matching ``mesh.cellCenters``. ``se_meas``: measured effective
    saturation (from tension via the retention model's ``se_from_psi``).
    ``variance``: measurement variance R in Se^2 units (must be > 0), formed by
    the caller from the per-sensor pF std. ``r_h``, ``r_v``: this sensor's
    localization radii in metres (default 0 = no reach), attached by
    :func:`anchor_update` from the per-sensor ``[anchor]`` settings so each
    tensiometer carries its own reach into the shared :func:`anchor_field`.
    """

    x_m: float
    y_m: float
    se_meas: float
    variance: float
    r_h: float = 0.0
    r_v: float = 0.0


def sensor_xy_m(x_offset_cm: float, depth_cm: float, width_m: float) -> tuple[float, float]:
    """Sensor mesh position in metres, mirroring the probe resolver's convention.

    Bay-centered ``x_offset`` (cm, left negative) maps to absolute mesh x
    ``x_offset * 0.01 + width / 2``; positive-downward ``depth`` (cm) maps to mesh
    ``y = -depth * 0.01`` (the mesh uses negative y for depth). Same mapping as
    ``_soil._nearest_cell_m``, but kept continuous so the localization ellipse
    centres on the sensor rather than its snapped cell.
    """
    return x_offset_cm * 0.01 + width_m / 2.0, -(depth_cm * 0.01)


def observation_from_tension(
    tension_hpa: float,
    x_offset_cm: float,
    depth_cm: float,
    width_m: float,
    model: Any,
    sigma_meas_pf: float,
) -> AnchorObservation:
    """Map a tensiometer reading (hPa) at a sensor into an ``AnchorObservation``.

    The shared observation adapter both backends feed: the live channel value in
    ``advance()`` and the loaded-history lookup in the soil_tuning worker. The
    measured tension converts to effective saturation through the retention model
    (``se_from_psi``); the measurement std, specified in pF, converts to an Se
    variance at the measured state via the curve slope::

        sigma_se = sigma_meas_pf * |dSe/dpF| = sigma_meas_pf * head_m * ln10 / dh_dse

    where ``head_m`` is the measured head in metres of water and ``dh_dse`` is the
    model's |d head / dSe| in metres. The variance is floored (``_MIN_VARIANCE``)
    so a near-saturated sensor cannot acquire near-infinite trust. ``model`` is
    duck-typed (any retention model exposing ``se_from_psi`` and ``dh_dse``), so
    this stays free of any FiPy import.
    """
    se_meas = float(model.se_from_psi(tension_hpa))
    head_m = abs(tension_hpa) * _HPA_TO_M_WATER
    se_eval = min(max(se_meas, _SE_EPS), 1.0 - _SE_EPS)
    slope = float(model.dh_dse(se_eval))
    sigma_se = sigma_meas_pf * head_m * _LN10 / slope
    variance = max(sigma_se * sigma_se, _MIN_VARIANCE)
    x_m, y_m = sensor_xy_m(x_offset_cm, depth_cm, width_m)
    return AnchorObservation(x_m=x_m, y_m=y_m, se_meas=se_meas, variance=variance)


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
    se_min: float,
    se_max: float,
) -> np.ndarray:
    """Pull the Se field toward the observations and return the corrected field.

    For each cell touched by one or more fresh sensors, blend the model state and
    the sensor-implied state by a precision-weighted mean::

        Se_new(c) = ( Se(c)/sigma_sys^2 + sum_s w_s(c) Se_meas,s / R_s )
                    / ( 1/sigma_sys^2     + sum_s w_s(c) / R_s )

    where ``w_s(c)`` is the localization weight over sensor ``s``'s own reach
    ``(obs.r_h, obs.r_v)``. Order-independent, cannot overshoot either reading,
    and reduces to ``Se + k (Se_meas - Se)`` with
    ``k = sigma_sys^2 / (R + sigma_sys^2)`` for a single sensor at its own cell.
    Cells out of every sensor's reach are returned unchanged. The result is
    clipped to ``[se_min, se_max]``.

    ``se``: (N,) effective-saturation field. ``cell_centers``: (2, N) mesh cell
    centres in metres (row 0 x, row 1 y, y < 0 down). ``sigma_sys``: model std in
    Se units; each observation carries its own localization radii. ``se_min``,
    ``se_max``: physical clip bounds (live callers pass ``_soil.SE_MIN`` /
    ``SE_MAX``). A non-positive ``sigma_sys`` makes the update a no-op, as does a
    non-positive radius on an individual observation (that sensor is skipped).
    """
    se = np.asarray(se, dtype=float)
    cell_centers = np.asarray(cell_centers, dtype=float)
    observations = list(observations)

    if sigma_sys <= 0.0 or not observations:
        return se.copy()

    inv_sys = 1.0 / sigma_sys**2
    numerator = se * inv_sys
    denominator = np.full_like(se, inv_sys)

    for obs in observations:
        if obs.r_h <= 0.0 or obs.r_v <= 0.0:
            continue
        precision = _localization_weights(cell_centers, obs.x_m, obs.y_m, obs.r_h, obs.r_v) / obs.variance
        numerator = numerator + precision * obs.se_meas
        denominator = denominator + precision

    return np.clip(numerator / denominator, se_min, se_max)


# --- Orchestration layer: the freshness gate + shared update entry point. -------
# Both backends (live advance() and the soil_tuning replay) call anchor_update;
# they differ only in the read_tension closure they pass. Timestamps/durations are
# duck-typed (Any) so this module pulls in no pandas/lories/FiPy.


@dataclass(frozen=True)
class SensorOverrides:
    """Per-sensor ``[anchor.sensors.<key>]`` overrides; each field ``None`` inherits.

    A sensor may carry its own trust (``sigma_meas_pf``), freshness tolerance
    (``staleness``), and reach (``r_horizontal``/``r_vertical``); any field left
    ``None`` falls back to the corresponding ``[anchor]`` global. ``sigma_sys``
    stays global -- it is the model/cell prior precision shared by every sensor
    reaching a cell, and per-sensor pull strength is already fully expressed
    through ``sigma_meas_pf`` (a larger measurement std pulls that cell less).
    """

    sigma_meas_pf: float | None = None
    staleness: Any = None
    r_horizontal: float | None = None
    r_vertical: float | None = None


@dataclass(frozen=True)
class AnchorConfig:
    """Resolved ``[anchor]`` settings (parsing lives at the call site in soil.py).

    ``sensors`` is the allowlist: sensor key -> its :class:`SensorOverrides` (or
    ``None`` for an all-inherit entry). ``sigma_sys`` is the model std in Se units,
    ``sigma_meas_pf``/``r_horizontal``/``r_vertical``/``staleness`` are the global
    defaults each sensor inherits unless it overrides them.

    ``min_tension_hpa`` is the dead-sensor floor: a reading whose magnitude is below
    it (a disconnected tensiometer reading ~0 hPa = saturation) is rejected rather
    than trusted (``0.0`` = disabled).
    """

    enabled: bool
    sigma_sys: float
    sigma_meas_pf: float
    r_horizontal: float
    r_vertical: float
    staleness: Any
    sensors: dict[str, SensorOverrides | None]
    min_tension_hpa: float = 0.0

    def sensor_sigma(self, key: str) -> float:
        """The pF measurement std for ``key``: its per-sensor override or the global."""
        override = self.sensors.get(key)
        if override is None or override.sigma_meas_pf is None:
            return self.sigma_meas_pf
        return override.sigma_meas_pf

    def sensor_staleness(self, key: str) -> Any:
        """The freshness tolerance for ``key``: its per-sensor override or the global."""
        override = self.sensors.get(key)
        if override is None or override.staleness is None:
            return self.staleness
        return override.staleness

    def sensor_radii(self, key: str) -> tuple[float, float]:
        """The localization radii ``(r_h, r_v)`` for ``key``: overrides or globals."""
        override = self.sensors.get(key)
        if override is None:
            return self.r_horizontal, self.r_vertical
        r_h = self.r_horizontal if override.r_horizontal is None else override.r_horizontal
        r_v = self.r_vertical if override.r_vertical is None else override.r_vertical
        return r_h, r_v


@dataclass(frozen=True)
class AnchorSensor:
    """A discovered tension sensor's static anchoring geometry (cm, bay-centered)."""

    key: str
    x_offset_cm: float
    depth_cm: float


@dataclass(frozen=True)
class AnchorResult:
    """Outcome of one anchor update.

    ``se_new`` is the corrected field; ``anchored_at`` maps each sensor that
    contributed to the reading timestamp it consumed (merge into the caller's
    persistent ``last_anchored`` only after the field is committed); ``innovations``
    maps each to ``se_meas - se_model`` at its nearest cell, for diagnostics.
    """

    se_new: np.ndarray
    anchored_at: dict[str, Any]
    innovations: dict[str, float]


def latest_reading_at(series: Any, now: Any) -> tuple[Any, float]:
    """The tension reading contemporaneous with ``now``: the latest ``(timestamp,
    value)`` in ``series`` at or before ``now``, or ``(None, nan)`` when the series
    is empty, has nothing at/before ``now``, or that value is non-finite.

    The single lookup both anchor backends share -- the live tick and the offline
    soil_tuning worker each range-read a per-sensor tension ``series`` and call this
    per step, so a reading is assimilated at its own time, not wherever the frontier
    happens to be. ``series`` is a pandas Series indexed by timestamp; kept
    duck-typed so this core stays free of a pandas import.
    """
    if series is None or len(series) == 0:
        return None, float("nan")
    prior = series.loc[:now]
    if len(prior) == 0:
        return None, float("nan")
    value = float(prior.iloc[-1])
    if not np.isfinite(value):
        return None, float("nan")
    return prior.index[-1], value


def _nearest_cell(cell_centers: np.ndarray, x_m: float, y_m: float) -> int:
    dx = cell_centers[0] - x_m
    dy = cell_centers[1] - y_m
    return int(np.argmin(dx * dx + dy * dy))


def anchor_update(
    se: np.ndarray,
    cell_centers: np.ndarray,
    sensors: Iterable[AnchorSensor],
    read_tension: Any,
    now: Any,
    cfg: AnchorConfig,
    model: Any,
    width_m: float,
    last_anchored: dict[str, Any],
    se_min: float,
    se_max: float,
) -> AnchorResult | None:
    """Gather fresh sensor readings and return the corrected field, or ``None``.

    For each sensor, ``read_tension(sensor)`` yields ``(timestamp, tension_hpa)`` --
    the reading contemporaneous with ``now``, i.e. the latest at or before it, from
    both backends' per-tick ranged read. A reading anchors only if it is present and
    finite, above the dead-sensor floor (``cfg.min_tension_hpa``), strictly newer than
    ``last_anchored[key]``, and within that sensor's staleness tolerance of ``now`` --
    the event-driven cadence that stops a stale, frozen, or dead-at-zero reading from
    dragging the field. Because the reading is looked up at/before ``now`` it is
    assimilated at its own timestamp, never back-dated onto a different step. Each
    qualifying
    reading carries its own localization radii into the blend. If no sensor
    qualifies the step is a no-op (``None``); otherwise the readings are blended in one
    precision-weighted :func:`anchor_field` call. ``sensors`` is the set to use,
    already filtered to the run's allowlist (the live path passes ``cfg.sensors``;
    soil_tuning passes its per-run set so a sensor can be held out).
    """
    se = np.asarray(se, dtype=float)
    fresh: list[AnchorObservation] = []
    anchored_at: dict[str, Any] = {}
    innovations: dict[str, float] = {}

    for sensor in sensors:
        ts, tension = read_tension(sensor)
        if ts is None or tension is None or not np.isfinite(tension):
            continue
        # Dead-sensor floor: a disconnected tensiometer reads ~0 hPa, which maps to
        # saturation and would drag the probe there. Reject it rather than trust it.
        if cfg.min_tension_hpa and abs(tension) < cfg.min_tension_hpa:
            continue
        previous = last_anchored.get(sensor.key)
        if previous is not None and ts <= previous:
            continue
        if now - ts > cfg.sensor_staleness(sensor.key):
            continue
        obs = observation_from_tension(
            tension, sensor.x_offset_cm, sensor.depth_cm, width_m, model, cfg.sensor_sigma(sensor.key)
        )
        r_h, r_v = cfg.sensor_radii(sensor.key)
        obs = replace(obs, r_h=r_h, r_v=r_v)
        fresh.append(obs)
        anchored_at[sensor.key] = ts
        innovations[sensor.key] = obs.se_meas - float(se[_nearest_cell(cell_centers, obs.x_m, obs.y_m)])

    if not fresh:
        return None

    se_new = anchor_field(se, cell_centers, fresh, cfg.sigma_sys, se_min, se_max)
    return AnchorResult(se_new=se_new, anchored_at=anchored_at, innovations=innovations)
