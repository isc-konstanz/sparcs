# -*- coding: utf-8 -*-
"""sparcs.tests.test_field_simulation_soil_pde_config
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Unit tests for ``FieldSimulation.soil_pde_config`` -- the resolved-config
handoff seam mirroring ``mesh_config``: prefers the live ``SoilSimulation``'s
``_ode_config`` once configured, falls back to the eagerly-parsed copy built
at the field's own configure time (children configure in alphanumeric id
order, so ``soil_predictor`` configures BEFORE ``soil_simulation`` and never
sees the live object). Bare ``object.__new__`` instances exercise the
property without the Component/PDE stack, matching the sibling
``test_field_simulation_*`` tests.
"""

import types

import pytest

base = pytest.importorskip("sparcs.components.agriculture.simulation.base")
FieldSimulation = base.FieldSimulation


def _sim() -> FieldSimulation:
    sim = object.__new__(FieldSimulation)
    sim.soil_simulation = None
    sim._soil_pde_config = None
    return sim


def test_prefers_live_sim_ode_config():
    """Sim configured and carrying its own ``_ode_config`` -> that object wins,
    even when an eager copy is also present."""
    sim = _sim()
    live_ode = object()
    sim.soil_simulation = types.SimpleNamespace(_ode_config=live_ode)
    sim._soil_pde_config = object()  # must be ignored while the live object exists

    assert sim.soil_pde_config is live_ode


def test_falls_back_to_eager_copy_when_sim_absent():
    """No SoilSimulation sibling at all (predictor configures first) -> the
    eagerly-parsed copy from FieldSimulation.configure() is served."""
    sim = _sim()
    eager = object()
    sim._soil_pde_config = eager

    assert sim.soil_pde_config is eager


def test_falls_back_to_eager_copy_when_sim_unconfigured():
    """Sim sibling exists but hasn't configured yet (no ``_ode_config`` attribute
    set) -> falls back to the eager copy, same as the absent-sim case."""
    sim = _sim()
    sim.soil_simulation = types.SimpleNamespace()  # no _ode_config attribute
    eager = object()
    sim._soil_pde_config = eager

    assert sim.soil_pde_config is eager


def test_returns_none_when_neither_is_available():
    """No sim sibling and no eager parse ran (e.g. field has no
    [soil_simulation] block at all) -> None, matching mesh_config's contract."""
    sim = _sim()

    assert sim.soil_pde_config is None
