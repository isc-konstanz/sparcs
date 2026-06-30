# -*- coding: utf-8 -*-
"""Unit tests for the anchor observation adapter (tension hPa -> Se observation).

Exercises ``observation_from_tension`` and ``sensor_xy_m`` against a real
retention model, with no FiPy mesh -- the shared conversion both the live and the
soil_tuning backends feed (``.scratch/soil-sensor-anchoring/PRD.md`` step 1).
"""

import numpy as np
from sparcs.components.agriculture.simulation._anchor import (
    _MIN_VARIANCE,
    observation_from_tension,
    sensor_xy_m,
)
from sparcs.components.agriculture.soil.models import Genuchten

# Uni Hohenheim ground-truth curve carried by copperhead's field_2 config.
MODEL = Genuchten(theta_r=0.0, theta_s=0.43, alpha=0.02, n=1.14, k_s=1e-5)


def test_sensor_xy_m_matches_resolver_convention():
    # width 3 m -> bay center at mesh x 1.5; depth 30 cm -> mesh y -0.3.
    assert sensor_xy_m(x_offset_cm=0.0, depth_cm=30.0, width_m=3.0) == (1.5, -0.3)
    # x_offset -100 cm = -1.0 m -> absolute mesh x 0.5 (left of center).
    assert sensor_xy_m(x_offset_cm=-100.0, depth_cm=30.0, width_m=3.0) == (0.5, -0.3)
    # +100 cm -> absolute mesh x 2.5 (right of center).
    assert sensor_xy_m(x_offset_cm=100.0, depth_cm=60.0, width_m=3.0)[0] == 2.5


def test_se_meas_uses_retention_model():
    obs = observation_from_tension(300.0, 0.0, 30.0, 3.0, MODEL, sigma_meas_pf=0.1)
    assert obs.se_meas == MODEL.se_from_psi(300.0)
    assert 0.0 < obs.se_meas < 1.0
    assert np.isfinite(obs.variance) and obs.variance > 0.0
    # geometry threaded through unchanged.
    assert (obs.x_m, obs.y_m) == (1.5, -0.3)


def test_variance_scales_with_pf_std_squared():
    """R is a variance: doubling the pF std quadruples it (away from the floor)."""
    lo = observation_from_tension(300.0, 0.0, 30.0, 3.0, MODEL, sigma_meas_pf=0.1)
    hi = observation_from_tension(300.0, 0.0, 30.0, 3.0, MODEL, sigma_meas_pf=0.2)
    assert hi.variance > _MIN_VARIANCE  # genuinely above the floor, so the ratio is real
    assert np.isclose(hi.variance / lo.variance, 4.0)


def test_near_saturation_variance_hits_the_floor():
    """A saturated reading (tension -> 0) would give zero Se variance; floored."""
    obs = observation_from_tension(0.0, 0.0, 30.0, 3.0, MODEL, sigma_meas_pf=0.1)
    assert obs.variance == _MIN_VARIANCE
