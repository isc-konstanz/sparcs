# -*- coding: utf-8 -*-
"""Regression tests for the mirrored A-frame roof orientation.

The mirrored ``as_is`` builder must place the two sub-arrays so their high
edges meet in a peak ("/\\"), not a valley ("\\/"), regardless of
``axis_azimuth``. pvfactors derives a signed rotation from
``(surface_azimuth - axis_azimuth)`` and the row geometry follows that sign,
so a hard-coded tilt-sign pairing inverts the roof for some azimuths (the
copperhead config uses ``axis_azimuth = 180`` and used to render inverted).

Ground truth here is pvfactors itself (an independent geometry build), so the
test is not a tautology against the builder's own sign formula. pvfactors is
imported lazily and the module skips if it is unavailable.
"""

import importlib
from types import SimpleNamespace

import pytest

import numpy as np

gs = pytest.importorskip("sparcs.components.agriculture.simulation.ground_shading")
_pvgeom = importlib.import_module("pvfactors.geometry")


# (label, surface_azimuth, axis_azimuth). copperhead is the one that regressed.
CONFIGS = [
    ("test_agri_sim", 90.0, 0.0),
    ("copperhead", 90.0, 180.0),
    ("soil_shading_both_sided", 180.0, 100.0),
    ("stress_axis_270", 90.0, 270.0),
]


def _high_edge_side(surface_tilt, surface_azimuth, axis_azimuth):
    """Independently ask pvfactors which end of the row is higher.

    Returns "/" (high edge on the right) or "\\" (high edge on the left).
    """
    arr = _pvgeom.OrderedPVArray.fit_from_dict_of_scalars(
        {
            "n_pvrows": 1,
            "pvrow_height": 2.0,
            "pvrow_width": 1.134,
            "axis_azimuth": axis_azimuth,
            "gcr": 1.134 / 3.4,
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


def _build_mirrored_setups(surface_azimuth, axis_azimuth, surface_tilt=10.0):
    """Drive the real builder with a stub self + minimal config object."""
    common = dict(
        n_rows=3,
        height=3.77,
        width=1.134,
        distance=3.4,
        axis_azimuth=axis_azimuth,
    )
    values = {"surface_tilt": surface_tilt, "surface_azimuth": surface_azimuth}
    configs = SimpleNamespace(
        get_float=lambda key, default=None: values.get(key, default),
        get_bool=lambda key, default=None: True if key == "mirrored" else default,
    )
    stub = SimpleNamespace()
    return gs.GroundShading._build_as_is_setups(stub, configs, common)


@pytest.mark.parametrize("label,surface_azimuth,axis_azimuth", CONFIGS)
def test_mirrored_aframe_is_a_peak_not_a_valley(label, surface_azimuth, axis_azimuth):
    left, right = _build_mirrored_setups(surface_azimuth, axis_azimuth)

    # The left sub-array sits at x < 0, the right at x > 0.
    assert left.offset_x < 0 < right.offset_x

    left_side = _high_edge_side(left.surface_tilt, surface_azimuth, axis_azimuth)
    right_side = _high_edge_side(right.surface_tilt, surface_azimuth, axis_azimuth)

    # Peak: left panel's high edge on its right, right panel's high edge on its
    # left -> "/" then "\" -> "/\\". A valley ("\\/") is the inverted-roof bug.
    assert (left_side, right_side) == (
        "/",
        "\\",
    ), f"{label}: roof rendered {left_side + right_side}, expected /\\ (peak)"


@pytest.mark.parametrize("label,surface_azimuth,axis_azimuth", CONFIGS)
def test_synthesized_night_rows_match_pvfactors_orientation(label, surface_azimuth, axis_azimuth):
    setups = _build_mirrored_setups(surface_azimuth, axis_azimuth)
    stub = SimpleNamespace(_pv_setups=setups)
    rows = gs.GroundShading._synthesize_pv_rows(stub)
    assert len(rows) == sum(s.n_rows for s in setups)

    # Group synthesized rows by setup (rows are emitted setup-by-setup).
    per_setup = [rows[i * setups[0].n_rows : (i + 1) * setups[0].n_rows] for i in range(len(setups))]
    for setup, setup_rows in zip(setups, per_setup):
        pvf_side = _high_edge_side(setup.surface_tilt, setup.surface_azimuth, setup.axis_azimuth)
        for start, end, _params in setup_rows:
            (lx, ly), (rx, ry) = sorted([start, end])
            synth_side = "\\" if ly > ry else "/"
            assert synth_side == pvf_side, f"{label}: night render {synth_side} disagrees with pvfactors {pvf_side}"


def test_pointing_right_matches_pvfactors_rule():
    # is_pointing_right = (surface_azimuth - axis_azimuth) % 360 > 180
    assert gs._pvfactors_is_pointing_right(90.0, 180.0) is True  # 270 > 180
    assert gs._pvfactors_is_pointing_right(90.0, 0.0) is False  # 90
    assert gs._pvfactors_is_pointing_right(180.0, 100.0) is False  # 80
    assert gs._pvfactors_is_pointing_right(90.0, 270.0) is False  # 180, not > 180
