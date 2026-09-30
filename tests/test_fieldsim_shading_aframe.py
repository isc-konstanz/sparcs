# -*- coding: utf-8 -*-
"""
tests.test_fieldsim_shading_aframe
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The mirrored ``as_is`` roof must be a peak ("/\\"), not a valley ("\\/"), for
any ``axis_azimuth``: pvfactors derives a signed rotation from
``(surface_azimuth - axis_azimuth)`` and the row geometry follows that sign.
Ground truth is pvfactors itself, so the pins are not a tautology against the
builder's own sign formula.
"""

import pytest

import numpy as np

_pvgeom = pytest.importorskip("pvfactors.geometry")

from sparcs.components.agriculture.simulation.core.pv import _pvfactors_is_pointing_right  # noqa: E402
from sparcs.components.agriculture.simulation.core.shading import ShadingConfig, ShadingModel  # noqa: E402

_WIDTH = 1.134
_DISTANCE = 3.4
_HEIGHT = 3.77
_TILT = 10.0

# (label, surface_azimuth, axis_azimuth). copperhead is the one that regressed.
CONFIGS = [
    ("test_agri_sim", 90.0, 0.0),
    ("copperhead", 90.0, 180.0),
    ("soil_shading_both_sided", 180.0, 100.0),
    ("stress_axis_270", 90.0, 270.0),
]


def _high_edge_side(surface_tilt, surface_azimuth, axis_azimuth):
    """Independently ask pvfactors which end of the row is higher: "/" (high
    edge on the right) or "\\" (high edge on the left)."""
    arr = _pvgeom.OrderedPVArray.fit_from_dict_of_scalars(
        {
            "n_pvrows": 1,
            "pvrow_height": 2.0,
            "pvrow_width": _WIDTH,
            "axis_azimuth": axis_azimuth,
            "gcr": _WIDTH / _DISTANCE,
            "surface_tilt": surface_tilt,
            "surface_azimuth": surface_azimuth,
            "solar_zenith": 30.0,
            "solar_azimuth": 180.0,
            "rho_ground": 0.2,
        }
    )
    coords = arr.ts_pvrows[0].full_pvrow_coords
    (lx, ly), (rx, ry) = sorted(
        [
            (float(np.ravel(coords.b1.x)[0]), float(np.ravel(coords.b1.y)[0])),
            (float(np.ravel(coords.b2.x)[0]), float(np.ravel(coords.b2.y)[0])),
        ]
    )
    assert abs(ly - ry) > 1e-9, "row is flat; tilt sign is unobservable"
    return "\\" if ly > ry else "/"


def _mirrored_model(surface_azimuth, axis_azimuth) -> ShadingModel:
    config = ShadingConfig.from_dict(
        {
            "mode": "as_is",
            "mirrored": True,
            "surface_tilt": _TILT,
            "surface_azimuth": surface_azimuth,
            "axis_azimuth": axis_azimuth,
            "height": _HEIGHT,
            "width": _WIDTH,
            "distance": _DISTANCE,
        }
    )
    return ShadingModel(config)


@pytest.mark.parametrize("label,surface_azimuth,axis_azimuth", CONFIGS)
def test_mirrored_aframe_is_a_peak_not_a_valley(label, surface_azimuth, axis_azimuth):
    left, right = _mirrored_model(surface_azimuth, axis_azimuth)._setups

    assert left.offset_x < 0 < right.offset_x

    left_side = _high_edge_side(left.surface_tilt, surface_azimuth, axis_azimuth)
    right_side = _high_edge_side(right.surface_tilt, surface_azimuth, axis_azimuth)

    assert (left_side, right_side) == (
        "/",
        "\\",
    ), f"{label}: roof rendered {left_side + right_side}, expected /\\ (peak)"


@pytest.mark.parametrize("label,surface_azimuth,axis_azimuth", CONFIGS)
def test_synthesized_night_rows_match_pvfactors_orientation(label, surface_azimuth, axis_azimuth):
    model = _mirrored_model(surface_azimuth, axis_azimuth)
    setups = model._setups
    rows = model._synthesize_pv_rows()
    assert len(rows) == sum(s.n_rows for s in setups)

    per_setup = [rows[i * setups[0].n_rows : (i + 1) * setups[0].n_rows] for i in range(len(setups))]
    for setup, setup_rows in zip(setups, per_setup):
        pvf_side = _high_edge_side(setup.surface_tilt, setup.surface_azimuth, setup.axis_azimuth)
        for start, end, _params in setup_rows:
            (lx, ly), (rx, ry) = sorted([start, end])
            synth_side = "\\" if ly > ry else "/"
            assert synth_side == pvf_side, f"{label}: night render {synth_side} disagrees with pvfactors {pvf_side}"


def test_pointing_right_matches_pvfactors_rule():
    # is_pointing_right = (surface_azimuth - axis_azimuth) % 360 > 180
    assert _pvfactors_is_pointing_right(90.0, 180.0) is True  # 270 > 180
    assert _pvfactors_is_pointing_right(90.0, 0.0) is False  # 90
    assert _pvfactors_is_pointing_right(180.0, 100.0) is False  # 80
    assert _pvfactors_is_pointing_right(90.0, 270.0) is False  # 180, not > 180
