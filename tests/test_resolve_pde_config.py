# -*- coding: utf-8 -*-
"""sparcs.tests.test_resolve_pde_config
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Unit tests for ``resolve_pde_config`` -- the helper that collapses the
construct-``PDEConfig``-then-``apply_surface_forcing`` sequence shared by
``SoilSimulation.configure``, ``FieldSimulation.configure``'s eager
``_soil_pde_config`` parse, and ``SoilPredictor._resolve_ode_config``'s
own-``[pde]`` branch. Exercised directly -- pure config parsing, no mesh.
"""

import pytest

from lories import Configurations

_soil = pytest.importorskip("sparcs.components.agriculture.simulation._soil")
resolve_pde_config = _soil.resolve_pde_config
apply_surface_forcing = _soil.apply_surface_forcing
PDEConfig = _soil.PDEConfig


def _configs(tmp_path, name="t.conf", **values) -> Configurations:
    return Configurations.load(name, conf_dir=str(tmp_path), require=False, **values)


def test_fresh_parse_no_inherit_uses_own_forcing_blocks(tmp_path):
    """No ``inherit_forcing_from`` -> forcing comes from the component's own
    sibling ``[ponding]``/``[feddes]`` blocks; an absent block leaves the
    hardcoded ``PDEConfig`` defaults untouched."""
    model_block = _configs(tmp_path, name="model.conf")
    component_block = _configs(
        tmp_path,
        name="component.conf",
        pde={"dt": "5min"},
        ponding={"enabled": True, "h_max_mm": 8.0},
        # no [feddes] block
    )

    cfg = resolve_pde_config(component_block, model_block)

    assert cfg.dt == 300.0
    assert cfg.ponding.enabled is True
    assert cfg.ponding.h_max_mm == 8.0
    assert cfg.feddes.enabled is False  # PDEConfig/FeddesConfig hardcoded default


def test_inherit_branch_no_local_forcing_is_identity(tmp_path):
    """``inherit_forcing_from`` given, component states neither ``[ponding]``
    nor ``[feddes]`` -> ``cfg.ponding``/``.feddes`` ARE the inherited objects
    (``is`` identity) -- the seed-before-apply ordering this helper exists for."""
    model_block = _configs(tmp_path, name="model.conf")
    sim_block = _configs(
        tmp_path,
        name="sim.conf",
        pde={},
        ponding={"h_max_mm": 8.0, "watering_h_max_mm": 50.0},
        feddes={"enabled": True, "p2_pf": 2.5},
    )
    inherit_from = PDEConfig(sim_block.get_member("pde"), model_configs=model_block)
    apply_surface_forcing(inherit_from, sim_block)

    predictor_block = _configs(tmp_path, name="predictor.conf", pde={"dt": "5min"})

    cfg = resolve_pde_config(predictor_block, model_block, inherit_forcing_from=inherit_from)

    assert cfg is not inherit_from
    assert cfg.ponding is inherit_from.ponding
    assert cfg.feddes is inherit_from.feddes
    assert cfg.dt == 300.0  # solver keys still come from the component's own [pde]


def test_inherit_branch_local_partial_ponding_merges_against_inherited_base(tmp_path):
    """``inherit_forcing_from`` given, component states a PARTIAL ``[ponding]``
    -> the explicit key wins, unset keys follow ``inherit_forcing_from``'s
    resolved ponding (the ``ponding_base=`` key-level merge), and the
    inherited object itself is left untouched (no in-place mutation)."""
    model_block = _configs(tmp_path, name="model.conf")
    sim_block = _configs(
        tmp_path,
        name="sim.conf",
        pde={},
        ponding={"enabled": True, "h_max_mm": 8.0, "watering_h_max_mm": 50.0},
    )
    inherit_from = PDEConfig(sim_block.get_member("pde"), model_configs=model_block)
    apply_surface_forcing(inherit_from, sim_block)
    orig_ponding = inherit_from.ponding
    assert orig_ponding.watering_h_max_mm == 50.0  # guard: fixture is set up right

    predictor_block = _configs(
        tmp_path,
        name="predictor.conf",
        pde={"dt": "5min"},
        ponding={"watering_h_max_mm": 17.0},
    )

    cfg = resolve_pde_config(predictor_block, model_block, inherit_forcing_from=inherit_from)

    assert cfg.ponding is not orig_ponding
    assert cfg.ponding.watering_h_max_mm == 17.0  # explicit key wins
    assert cfg.ponding.enabled is True  # unset key follows inherit_from's base
    assert cfg.ponding.h_max_mm == 8.0  # unset key follows inherit_from's base
    # the inherited object itself is untouched
    assert inherit_from.ponding is orig_ponding
    assert inherit_from.ponding.watering_h_max_mm == 50.0
