# -*- coding: utf-8 -*-
"""Unit tests for the pure constant-gain anchor analysis update.

Exercises ``anchor_field`` and its localization taper on a synthetic cell grid,
in Se space, with no FiPy mesh and no retention model -- the seam the PRD pins as
the primary one (``.scratch/soil-sensor-anchoring/PRD.md`` Testing Decisions).
Asserts external behaviour (the corrected field), not the internal call sequence.
"""

import numpy as np
from sparcs.components.agriculture.simulation._anchor import (
    AnchorObservation,
    _localization_weights,
    anchor_field,
)

# A two-cell grid: a cell at the sensor (mesh x 0.0, 0.3 m deep) and a far cell
# (mesh x 1.0) well outside a 0.5 m horizontal reach.
NEAR_FAR = np.array([[0.0, 1.0], [-0.3, -0.3]])
SE_MIN, SE_MAX = 1e-6, 0.999


def _obs(se_meas=0.8, variance=0.04, x_m=0.0, y_m=-0.3, r_h=0.5, r_v=0.5):
    return AnchorObservation(x_m=x_m, y_m=y_m, se_meas=se_meas, variance=variance, r_h=r_h, r_v=r_v)


def test_single_sensor_reduces_to_constant_gain_blend():
    """At its own cell a lone sensor gives Se + k (Se_meas - Se), k = s2/(R+s2)."""
    se = np.array([0.5, 0.5])
    sigma_sys = 0.1  # sigma^2 = 0.01
    out = anchor_field(se, NEAR_FAR, [_obs(variance=0.04)], sigma_sys, SE_MIN, SE_MAX)

    k = 0.01 / (0.04 + 0.01)  # 0.2
    assert np.isclose(out[0], 0.5 + k * (0.8 - 0.5))  # 0.56
    assert np.isclose(out[0], 0.56)
    # The far cell is out of reach and untouched.
    assert out[1] == 0.5


def test_zero_radius_is_a_noop():
    """A non-positive reach on the observation skips that sensor entirely."""
    se = np.array([0.5, 0.5])
    out_h = anchor_field(se, NEAR_FAR, [_obs(r_h=0.0)], 0.1, SE_MIN, SE_MAX)
    out_v = anchor_field(se, NEAR_FAR, [_obs(r_v=0.0)], 0.1, SE_MIN, SE_MAX)
    assert np.array_equal(out_h, se)
    assert np.array_equal(out_v, se)


def test_zero_gain_is_a_noop():
    """sigma_sys = 0 means infinite trust in the model -> no correction."""
    se = np.array([0.5, 0.5])
    out = anchor_field(se, NEAR_FAR, [_obs()], 0.0, SE_MIN, SE_MAX)
    assert np.array_equal(out, se)


def test_output_stays_within_physical_bounds():
    se = np.array([0.5, 0.5])
    high = anchor_field(se, NEAR_FAR, [_obs(se_meas=5.0)], 10.0, SE_MIN, SE_MAX)
    low = anchor_field(se, NEAR_FAR, [_obs(se_meas=-5.0)], 10.0, SE_MIN, SE_MAX)
    assert np.all(high <= SE_MAX) and np.all(high >= SE_MIN)
    assert np.all(low <= SE_MAX) and np.all(low >= SE_MIN)


def test_two_agreeing_sensors_pull_harder_without_overshoot():
    """Coincident agreeing sensors match the precision-weighted mean, no overshoot."""
    se = np.array([0.5, 0.5])
    one = anchor_field(se, NEAR_FAR, [_obs()], 0.1, SE_MIN, SE_MAX)
    two = anchor_field(se, NEAR_FAR, [_obs(), _obs()], 0.1, SE_MIN, SE_MAX)

    # (0.5/0.01 + 2*0.8/0.04) / (1/0.01 + 2/0.04) = 90/150 = 0.6
    assert np.isclose(two[0], 0.6)
    assert one[0] < two[0] < 0.8  # pulls harder than one, never past the reading


def test_per_observation_radii_set_each_sensor_reach():
    """Two sensors, different vertical reach: only the wider one touches a deep cell."""
    # Sensor at mesh (0, 0); a cell 0.4 m below it.
    cells = np.array([[0.0], [-0.4]])
    se = np.array([0.5])
    near = anchor_field(se, cells, [_obs(x_m=0.0, y_m=0.0, r_h=0.5, r_v=0.2)], 0.1, SE_MIN, SE_MAX)
    far = anchor_field(se, cells, [_obs(x_m=0.0, y_m=0.0, r_h=0.5, r_v=0.6)], 0.1, SE_MIN, SE_MAX)
    assert np.array_equal(near, se)  # 0.4 m is outside r_v = 0.2 -> untouched
    assert not np.isclose(far[0], 0.5)  # 0.4 m is inside r_v = 0.6 -> pulled


def test_order_independent():
    se = np.array([0.5, 0.5])
    a = anchor_field(se, NEAR_FAR, [_obs(se_meas=0.8), _obs(se_meas=0.2)], 0.1, SE_MIN, SE_MAX)
    b = anchor_field(se, NEAR_FAR, [_obs(se_meas=0.2), _obs(se_meas=0.8)], 0.1, SE_MIN, SE_MAX)
    assert np.allclose(a, b)


def test_taper_is_one_at_sensor_continuous_and_zero_at_edge():
    # Cells marching outward in x from the sensor; r_h = r_v = 1.0 puts the edge
    # at mesh x = 1.0 (d = 1). The last cell sits just past the edge.
    xs = np.array([0.0, 0.25, 0.5, 0.9, 0.999, 1.0, 1.5])
    cells = np.vstack([xs, np.zeros_like(xs)])
    w = _localization_weights(cells, x_m=0.0, y_m=0.0, r_h=1.0, r_v=1.0)

    assert np.isclose(w[0], 1.0)  # full weight at the sensor
    assert np.all(np.diff(w[:5]) < 0)  # monotonic decrease toward the edge
    assert w[4] < 0.02  # continuous: weight just inside the edge is ~0
    assert w[5] == 0.0 and w[6] == 0.0  # zero at and beyond the ellipse edge
