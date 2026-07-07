# -*- coding: utf-8 -*-
"""sparcs.tests.test_soil_predictor_ponding_inheritance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Unit tests for ``SoilPredictor._resolve_ode_config`` -- the guard that keeps the
predictor's watering-pond cap consistent with the live ``soil_simulation``.

The predictor builds its PDE from its OWN ``[pde]`` block, so any key it does not
restate silently takes the ``PDEConfig`` default. For ponding that default is
5 mm ``watering_h_max_mm`` while the live sim ponds to 50 mm, so a candidate
watering roll overflows ~10x sooner and the horizon reads too dry. The guard
inherits the sim's ``PondingConfig`` when (and only when) the predictor omits its
own ``[pde.ponding]``. These tests pin the three-branch contract without building
a mesh or running FiPy -- ``PDEConfig`` construction is pure config parsing.
"""

import pytest

from lories import Configurations

soil_predictor = pytest.importorskip("sparcs.components.agriculture.simulation.soil_predictor")
SoilPredictor = soil_predictor.SoilPredictor
PDEConfig = soil_predictor.PDEConfig


def _configs(tmp_path, name="t.conf", **values) -> Configurations:
    return Configurations.load(name, conf_dir=str(tmp_path), require=False, **values)


def _live_soil_pde(tmp_path) -> PDEConfig:
    """A parsed sim PDEConfig with a distinctive, non-default ponding (50 mm) and
    a non-default rain shadow, standing in for the live ``soil_simulation``."""
    model_block = _configs(tmp_path, name="model.conf")
    soil_pde_cfg = _configs(
        tmp_path,
        name="soil.conf",
        rain_shadow_width=1.5,
        ponding={"h_max_mm": 8.0, "watering_h_max_mm": 50.0},
    )
    return PDEConfig(soil_pde_cfg, model_configs=model_block)


def test_own_pde_without_ponding_inherits_sim_ponding(tmp_path):
    """Own ``[pde]`` but no ``[pde.ponding]`` -> the sim's PondingConfig is
    inherited verbatim (same object), so the 50 mm cap survives; every other key
    still comes from the predictor's own block (ponding is the ONLY key inherited)."""
    soil_pde = _live_soil_pde(tmp_path)
    predictor_cfg = _configs(tmp_path, pde={"dt": "40s", "rain_shadow_width": 0.7})
    model_block = _configs(tmp_path, name="model.conf")

    ode = SoilPredictor._resolve_ode_config(predictor_cfg, soil_pde, model_block)

    assert ode is not soil_pde  # own [pde] -> a distinct PDEConfig
    assert ode.ponding is soil_pde.ponding  # ...but ponding is the sim's object
    assert ode.ponding.watering_h_max_mm == 50.0
    # Non-ponding keys stay predictor-local -- rain shadow is NOT inherited.
    assert ode.rain_shadow_width == 0.7
    assert ode.dt == 40.0


def test_own_ponding_block_wins(tmp_path):
    """An explicit ``[pde.ponding]`` on the predictor overrides the sim's cap; the
    predictor's own PondingConfig is used and no inheritance occurs."""
    soil_pde = _live_soil_pde(tmp_path)
    predictor_cfg = _configs(tmp_path, pde={"dt": "40s", "ponding": {"watering_h_max_mm": 17.0}})
    model_block = _configs(tmp_path, name="model.conf")

    ode = SoilPredictor._resolve_ode_config(predictor_cfg, soil_pde, model_block)

    assert ode.ponding is not soil_pde.ponding
    assert ode.ponding.watering_h_max_mm == 17.0


def test_no_pde_block_inherits_soil_pde_wholesale(tmp_path):
    """No ``[pde]`` block at all -> the predictor inherits the sim's PDEConfig
    object outright (unchanged from before the guard), ponding included."""
    soil_pde = _live_soil_pde(tmp_path)
    predictor_cfg = _configs(tmp_path)  # no pde member
    model_block = _configs(tmp_path, name="model.conf")

    ode = SoilPredictor._resolve_ode_config(predictor_cfg, soil_pde, model_block)

    assert ode is soil_pde
    assert ode.ponding is soil_pde.ponding
    assert ode.ponding.watering_h_max_mm == 50.0
