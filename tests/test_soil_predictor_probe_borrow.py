# -*- coding: utf-8 -*-
"""sparcs.tests.test_soil_predictor_probe_borrow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Unit tests for ``SoilPredictor._borrow_probes`` -- the shared-probes borrow
seam (issue 1.4, soil-refactor Wave 1). Adopting the sim's already-resolved
``ProbeSpec``s instead of re-resolving ``[soil_simulation.probes]`` against a
second mesh instance is only safe when the two meshes are the SAME
configuration. ``_borrow_probes`` is a pure ``@staticmethod`` (mirrors
``_resolve_drip_layout``'s idiom, see test_soil_predictor_flow_schedule.py)
so the adopt-vs-fallback decision is unit-testable without running
``configure()`` -- no test here calls ``SoilPredictor.configure()``.

Importing ``soil_predictor``/``_soil`` pulls the full lories + soil
(FiPy/Gmsh) stack; ``importorskip`` keeps this out of environments that lack
it (the full check runs on the box).
"""

from types import SimpleNamespace

import pytest

import numpy as np

soil_predictor = pytest.importorskip("sparcs.components.agriculture.simulation.soil_predictor")
_soil = pytest.importorskip("sparcs.components.agriculture.simulation._soil")
SoilPredictor = soil_predictor.SoilPredictor
ProbeSpec = _soil.ProbeSpec


def _mesh(**overrides) -> SimpleNamespace:
    """A MeshConfig-shaped stand-in exposing only the eight attributes
    ``_mesh_configs_equivalent`` compares."""
    attrs = dict(
        filename="soil.msh",
        dl=0.1,
        width=3.5,
        height=5.0,
        plant_width=2.0,
        plant_height=2.0,
        watering_width=1.0,
        dx=0.5,
    )
    attrs.update(overrides)
    return SimpleNamespace(**attrs)


def _probe(channel_id: str) -> ProbeSpec:
    return ProbeSpec(
        name=f"Probe {channel_id}",
        channel_id=channel_id,
        cell_indices=np.array([0], dtype=int),
        weights=np.array([1.0]),
    )


def test_adopts_sim_probes_on_mesh_identity():
    """Non-empty get_probes() + context.mesh_config IS the predictor's own
    mesh_config (identity, the common in-context case) -> the SAME
    ProbeSpec objects, no re-resolution."""
    mesh = _mesh()
    probes = [_probe("p1"), _probe("p2")]
    context = SimpleNamespace(get_probes=lambda: list(probes), mesh_config=mesh)

    borrowed = SoilPredictor._borrow_probes(context, mesh)

    assert borrowed is not None
    assert len(borrowed) == len(probes)
    assert all(b is p for b, p in zip(borrowed, probes))


def test_adopts_sim_probes_on_mesh_attribute_equality():
    """Different MeshConfig instances but identical __init__-set attribute
    values (no shared instance, e.g. standalone construction) -> still
    adopts -- the non-identity half of the mesh guard."""
    probes = [_probe("p1")]
    context_mesh = _mesh()
    predictor_mesh = _mesh()  # separate object, same attribute values
    context = SimpleNamespace(get_probes=lambda: list(probes), mesh_config=context_mesh)

    borrowed = SoilPredictor._borrow_probes(context, predictor_mesh)

    assert borrowed is not None
    assert borrowed[0] is probes[0]


def test_falls_back_when_mesh_attributes_differ():
    """Attribute-differing MeshConfig (different width, distinct objects) ->
    None; caller falls back to its own resolve_probes -- never adopts
    against a mismatched mesh."""
    probes = [_probe("p1")]
    context_mesh = _mesh(width=7.0)
    predictor_mesh = _mesh(width=3.5)
    context = SimpleNamespace(get_probes=lambda: list(probes), mesh_config=context_mesh)

    assert SoilPredictor._borrow_probes(context, predictor_mesh) is None


def test_falls_back_when_sim_probes_empty():
    """Mesh matches (identity) but the sim has resolved no probes (yet) ->
    None -- the known first-configure ordering case (predictor configures
    before the sim)."""
    mesh = _mesh()
    context = SimpleNamespace(get_probes=lambda: [], mesh_config=mesh)

    assert SoilPredictor._borrow_probes(context, mesh) is None


def test_falls_back_when_context_has_no_get_probes():
    """Bare context stub with no get_probes at all -> None, never raises
    (the callable-guard)."""
    mesh = _mesh()
    context = SimpleNamespace(mesh_config=mesh)

    assert SoilPredictor._borrow_probes(context, mesh) is None
