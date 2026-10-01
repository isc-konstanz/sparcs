# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.simulation.core.anchor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Constant-gain (alpha) analysis update that nudges the simulated Se field toward tensiometer readings.
Each sensor reaches an anisotropic ellipse and sensors combine by a precision-weighted mean; no FiPy here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Iterable

import numpy as np

# Std of the Gaussian localization taper in normalised distance (d = 1 at the ellipse edge);
# the taper is shifted to vanish continuously at d = 1 so the next solve gets no saturation step.
_TAPER_STD = 0.5

# 1 hPa of pressure equals this many metres of water column (4 degC).
_HPA_TO_M_WATER = 0.0101972

_LN10 = math.log(10.0)

# Floor on the measurement variance in Se^2 units: near saturation the pF->Se Jacobian collapses
# and would give a near-saturated sensor near-infinite trust.
_MIN_VARIANCE = 1.0e-6

# dh_dse is singular at Se = 0 or 1, so the slope is evaluated a hair inside them.
_SE_EPS = 1.0e-9


@dataclass(frozen=True)
class AnchorObservation:
    """One tensiometer reading in Se space at a mesh location (metres, ``y`` negative downward).
    ``variance`` is R in Se^2 units (> 0); ``r_h``/``r_v`` are localization radii in metres, 0 = no reach."""

    x_m: float
    y_m: float
    se_meas: float
    variance: float
    r_h: float = 0.0
    r_v: float = 0.0


def sensor_xy_m(x_offset_cm: float, depth_cm: float, width_m: float) -> tuple[float, float]:
    """Sensor mesh position in metres from bay-centred ``x_offset`` (cm, left negative) and ``depth`` (cm, down).
    Kept continuous so the localization ellipse centres on the sensor rather than its snapped cell."""
    return x_offset_cm * 0.01 + width_m / 2.0, -(depth_cm * 0.01)


def observation_from_tension(
    tension_hpa: float,
    x_offset_cm: float,
    depth_cm: float,
    width_m: float,
    model: Any,
    sigma_meas_pf: float,
) -> AnchorObservation:
    """Map a tension reading (hPa) to an ``AnchorObservation``, its pF std to an Se variance via the curve slope.
    The variance is floored at ``_MIN_VARIANCE``; ``model`` is any retention model with ``se_from_psi``/``dh_dse``."""
    se_meas = float(model.se_from_psi(tension_hpa))
    head_m = abs(tension_hpa) * _HPA_TO_M_WATER
    se_eval = min(max(se_meas, _SE_EPS), 1.0 - _SE_EPS)
    slope = float(model.dh_dse(se_eval))
    sigma_se = sigma_meas_pf * head_m * _LN10 / slope
    variance = max(sigma_se * sigma_se, _MIN_VARIANCE)
    x_m, y_m = sensor_xy_m(x_offset_cm, depth_cm, width_m)
    return AnchorObservation(x_m=x_m, y_m=y_m, se_meas=se_meas, variance=variance)


def _localization_weights(cell_centers: np.ndarray, x_m: float, y_m: float, r_h: float, r_v: float) -> np.ndarray:
    """Anisotropic Gaussian ellipse taper: 1 at the sensor, falling continuously to 0 at normalised distance d >= 1."""
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
    """Pull the Se field toward the observations by a precision-weighted mean of model and sensor Se.
    Out-of-reach cells stay unchanged, the result is clipped to [se_min, se_max]; ``sigma_sys <= 0`` is a no-op."""
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


# Freshness gate and the update entry point. Timestamps and durations are duck-typed (Any)
# so this module pulls in no pandas, lories or FiPy.


@dataclass(frozen=True)
class SensorOverrides:
    """Per-sensor ``[anchor.sensors.<key>]`` overrides; a ``None`` field inherits the ``[anchor]`` global.
    ``sigma_sys`` stays global: it is the model prior precision shared by every sensor reaching a cell."""

    sigma_meas_pf: float | None = None
    staleness: Any = None
    r_horizontal: float | None = None
    r_vertical: float | None = None


@dataclass(frozen=True)
class AnchorConfig:
    """Resolved ``[anchor]`` settings; ``sensors`` is the allowlist, key to overrides (``None`` inherits all).
    ``sigma_sys`` is the model std in Se units; the other scalars are the globals a sensor inherits."""

    enabled: bool
    sigma_sys: float
    sigma_meas_pf: float
    r_horizontal: float
    r_vertical: float
    staleness: Any
    sensors: dict[str, SensorOverrides | None]

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
    """Outcome of one anchor update; ``innovations`` is ``se_meas - se_model`` at each sensor's nearest cell.
    Merge ``anchored_at`` into the caller's ``last_anchored`` only after the field is committed."""

    se_new: np.ndarray
    anchored_at: dict[str, Any]
    innovations: dict[str, float]


def latest_reading_at(series: Any, now: Any) -> tuple[Any, float]:
    """Latest ``(timestamp, value)`` in ``series`` at or before ``now``; ``(None, nan)`` if none or non-finite.
    ``series`` is a pandas Series indexed by timestamp, duck-typed to keep pandas out of this module."""
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
    """Blend fresh readings into the Se field in one :func:`anchor_field` call; ``None`` when none qualify.
    A reading qualifies if finite, newer than ``last_anchored[key]`` and within its sensor's staleness of ``now``."""
    se = np.asarray(se, dtype=float)
    fresh: list[AnchorObservation] = []
    anchored_at: dict[str, Any] = {}
    innovations: dict[str, float] = {}

    for sensor in sensors:
        ts, tension = read_tension(sensor)
        if ts is None or tension is None or not np.isfinite(tension):
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
