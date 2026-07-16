# -*- coding: utf-8 -*-
"""sparcs.tests.test_field_simulation_get_probes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Unit tests for ``FieldSimulation.get_probes()`` -- the shared-probes borrow
seam (issue 1.4, soil-refactor Wave 1). Method, not property: ``ProbeSpec``
carries numpy arrays and would trip lories' ``get_members`` reflection (see
.scratch/lories-frictions/issues/01-reflection-truth-test-trap.md and the
method's own docstring). getattr-guards like ``mesh_config``/
``soil_pde_config``: ``[]`` when ``soil_simulation`` is absent or not yet
configured (``SoilSimulation._probes`` is a bare class annotation until
``_configure_probes`` runs). Bare ``object.__new__`` instances exercise this
without the Component/PDE stack, matching the sibling
``test_field_simulation_*`` tests (e.g. test_field_simulation_soil_pde_config.py).
"""

import types

import pytest

import numpy as np

base = pytest.importorskip("sparcs.components.agriculture.simulation.base")
_soil = pytest.importorskip("sparcs.components.agriculture.simulation._soil")
FieldSimulation = base.FieldSimulation
ProbeSpec = _soil.ProbeSpec


def _sim() -> FieldSimulation:
    sim = object.__new__(FieldSimulation)
    sim.soil_simulation = None
    return sim


def _probe(channel_id: str) -> ProbeSpec:
    return ProbeSpec(
        name=f"Probe {channel_id}",
        channel_id=channel_id,
        cell_indices=np.array([0], dtype=int),
        weights=np.array([1.0]),
    )


def test_returns_empty_list_when_soil_simulation_absent():
    sim = _sim()
    assert sim.get_probes() == []


def test_returns_empty_list_when_soil_simulation_unconfigured():
    """Sibling exists but hasn't run _configure_probes yet (bare class
    annotation, no _probes attribute set at all) -- must never raise."""
    sim = _sim()
    sim.soil_simulation = types.SimpleNamespace()  # no _probes attribute

    assert sim.get_probes() == []


def test_delegates_to_configured_sim_same_objects_fresh_container():
    """Configured sim -- delegate returns the SAME ProbeSpec objects
    (identity; ProbeSpec is eq=False) inside a FRESH list container (per
    SoilSimulation.get_probes()'s own list(self._probes) contract)."""
    sim = _sim()
    probes = [_probe("p1"), _probe("p2")]

    class _Sim:
        _probes = probes

        def get_probes(self):
            return list(self._probes)

    sim.soil_simulation = _Sim()

    result = sim.get_probes()

    assert result is not probes
    assert len(result) == len(probes)
    assert all(r is p for r, p in zip(result, probes))
