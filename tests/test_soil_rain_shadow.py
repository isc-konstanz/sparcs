# -*- coding: utf-8 -*-
"""Unit tests for the rain-shadow passthrough fraction (imperfect PV cover).

Exercises ``SoilPDECore._compute_rain_open_fractions`` on a synthetic MeshConfig
with no FiPy mesh: passthrough 0 fully blocks the shaded plant segments (the
existing behaviour), passthrough f admits fraction f of the rain there, and
open-sky segments outside the shadow are untouched. The bench sweeps this
parameter to let rain reach the bay-center probe column without the unphysical
k_s the fully-blocked shadow otherwise forces.
"""

import types

import pytest

from lories import Configurations
from sparcs.components.agriculture.simulation._soil import (
    MeshConfig,
    PDEConfig,
    SoilPDECore,
    top_segment_names_from_mesh,
)


def _configs(tmp_dir: str, **values) -> Configurations:
    return Configurations.load("test.conf", conf_dir=tmp_dir, require=False, **values)


def _fractions(tmp_dir: str, passthrough: float) -> dict:
    # width 3.0 / plant 1.0 / watering 0.5 / dx 0.5 -> plant top segments span
    # (1.0, 1.25) and (1.75, 2.0); a shadow of width 1.0 centered at 1.5 -> [1.0, 2.0]
    # covers both plant segments fully. No mesh is built.
    mc = MeshConfig(_configs(tmp_dir, width=3.0, plant_width=1.0, watering_width=0.5, d_x=0.5, dl=0.2))
    oc = PDEConfig(_configs(tmp_dir, rain_shadow_width=1.0, rain_shadow_passthrough=passthrough))
    stub = types.SimpleNamespace(mesh_config=mc, ode_config=oc)
    return SoilPDECore._compute_rain_open_fractions(stub, top_segment_names_from_mesh(mc))


def test_passthrough_zero_blocks_shaded_segments(tmp_path):
    """Default passthrough 0 fully blocks the shaded plant segments (unchanged)."""
    f = _fractions(str(tmp_path), 0.0)
    assert f["PlantTopLeftSegment"] == pytest.approx(0.0)
    assert f["PlantTopRightSegment"] == pytest.approx(0.0)


def test_passthrough_admits_fraction_to_shaded_segments(tmp_path):
    """passthrough f lets fraction f of the rain reach the shaded plant column."""
    f = _fractions(str(tmp_path), 0.25)
    assert f["PlantTopLeftSegment"] == pytest.approx(0.25)
    assert f["PlantTopRightSegment"] == pytest.approx(0.25)


def test_open_sky_segments_unaffected_by_passthrough(tmp_path):
    """Segments outside the shadow stay fully open regardless of passthrough."""
    assert _fractions(str(tmp_path), 0.0)["LeftTopSegment_0"] == pytest.approx(1.0)
    assert _fractions(str(tmp_path), 0.5)["LeftTopSegment_0"] == pytest.approx(1.0)


def test_passthrough_clamped_to_unit_interval(tmp_path):
    """Out-of-range passthrough clamps to [0, 1] (>1 => shaded segment fully open)."""
    f = _fractions(str(tmp_path), 5.0)
    assert f["PlantTopLeftSegment"] == pytest.approx(1.0)
