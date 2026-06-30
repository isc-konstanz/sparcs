# -*- coding: utf-8 -*-
"""Unit tests for the sensor/probe coordinate resolver.

Pins the cm->m convention and the bay-center / depth-sign mapping that the
``A Sensor is-a Probe`` vocabulary relies on (see context/sparcs.md and
docs/adr/0001-soil-coordinate-units-cm.md). These exercise the pure mapping
helpers with a synthetic cell grid, so no Gmsh mesh or FiPy solve is built.

The SoilMoisture-side attributes (`x_offset` default, `has_measured_tension`)
are covered by the box integration suite, which can construct a full
component tree.
"""

from types import SimpleNamespace

import numpy as np
from sparcs.components.agriculture.simulation._soil import (
    _coords_to_cell,
    _nearest_cell_m,
    resolve_probe_from_sensor,
)


def _grid(xs, ys):
    """A FiPy-like stand-in exposing ``cellCenters`` as a (2, N) array."""
    cell_x = np.array(xs, dtype=float)
    cell_y = np.array(ys, dtype=float)
    return SimpleNamespace(cellCenters=np.vstack([cell_x, cell_y]))


def test_nearest_cell_m_applies_bay_center_shift_and_depth_sign():
    # cells laid out across mesh x in [0, 3], at mesh y = -0.5 (0.5 m deep).
    cell_x = np.array([0.0, 1.5, 3.0])
    cell_y = np.array([-0.5, -0.5, -0.5])
    # bay-centered x = 0 with x_offset (width/2) = 1.5 -> absolute mesh x 1.5,
    # depth 0.5 m -> mesh y -0.5: the middle cell.
    idx = _nearest_cell_m(cell_x, cell_y, x_m=0.0, depth_m=0.5, x_offset=1.5)
    assert idx == 1
    # bay-centered x = -1.5 -> absolute 0.0: the left cell.
    assert _nearest_cell_m(cell_x, cell_y, x_m=-1.5, depth_m=0.5, x_offset=1.5) == 0
    # bay-centered x = +1.5 -> absolute 3.0: the right cell.
    assert _nearest_cell_m(cell_x, cell_y, x_m=1.5, depth_m=0.5, x_offset=1.5) == 2


def test_coords_to_cell_converts_cm_to_m():
    # Two candidate cells at 30 cm and 60 cm depth, on the bay-center axis.
    mesh = _grid([1.5, 1.5], [-0.3, -0.6])
    mesh_config = SimpleNamespace(width=3.0)
    # depth 30 cm, x_offset 0 cm -> 0.3 m deep at center -> first cell.
    assert _coords_to_cell(mesh, mesh_config, x_offset_cm=0.0, depth_cm=30.0) == 0
    # depth 60 cm -> 0.6 m deep -> second cell.
    assert _coords_to_cell(mesh, mesh_config, x_offset_cm=0.0, depth_cm=60.0) == 1


def test_coords_to_cell_signed_x_offset():
    # Cells left and right of the bay center (width 3 -> center at mesh x 1.5).
    mesh = _grid([0.5, 2.5], [-0.3, -0.3])
    mesh_config = SimpleNamespace(width=3.0)
    # x_offset -100 cm = -1.0 m -> absolute mesh x 0.5 -> left cell.
    assert _coords_to_cell(mesh, mesh_config, x_offset_cm=-100.0, depth_cm=30.0) == 0
    # x_offset +100 cm = +1.0 m -> absolute mesh x 2.5 -> right cell.
    assert _coords_to_cell(mesh, mesh_config, x_offset_cm=100.0, depth_cm=30.0) == 1


def test_resolve_probe_from_sensor_builds_point_spec():
    mesh = _grid([1.5, 1.5], [-0.3, -0.6])
    mesh_config = SimpleNamespace(width=3.0)
    sensor = SimpleNamespace(key="bay1_30cm", x_offset=0.0, depth=30.0)

    spec = resolve_probe_from_sensor(sensor, mesh, mesh_config)

    assert spec.channel_id == "bay1_30cm"
    assert spec.cell_indices.tolist() == [0]
    assert spec.weights.tolist() == [1.0]
    # name carries the cm coordinates for log readability.
    assert "30.0cm" in spec.name
    assert "bay1_30cm" in spec.name
