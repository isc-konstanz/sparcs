# -*- coding: utf-8 -*-
"""sparcs.tests.test_fieldsim_planner_model_cascade
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The retention parameters the planner rolls on are the live sim's: both read one
``SoilEngine``, built from ``[soil_simulation]``'s model block after the
field-level ``[model]`` cascade (``Component._build_defaults(includes=["model"])``,
the merge ``FieldSimulation.configure`` performs). A key-level
``[soil_simulation.model]`` override therefore wins while unstated keys still
fall back to the field block -- a tuned ``k_s`` can never diverge between the
forecast and the sim. Pure config parsing; no mesh is built.
"""

from lories import Configurations
from lories.components import Component
from sparcs.components.agriculture.fieldsim.core.pde import PDEConfig


def _configs(tmp_path, name="t.conf", **values) -> Configurations:
    return Configurations.load(name, conf_dir=str(tmp_path), require=False, **values)


def _model_block(configs: Configurations) -> Configurations:
    """The model block ``SoilEngine.build`` reads off the configured soil section."""
    defaults = Component._build_defaults(configs, includes=["model", "plot"], strict=True)
    soil_block = configs.get_member("soil_simulation", defaults=defaults)
    return soil_block.get_member("model", defaults={}, ensure_exists=True)


def test_soil_simulation_model_override_reaches_the_engine(tmp_path):
    """A ``[soil_simulation.model]`` override wins for the key it restates; a key
    it leaves unset still falls back to the field-level ``[model]`` block."""
    configs = _configs(tmp_path, model={"k_s": 1.0e-4, "alpha": 0.08}, soil_simulation={"model": {"k_s": 5.0e-5}})

    model_block = _model_block(configs)

    assert model_block.get("k_s") == 5.0e-5  # soil-level override wins
    assert model_block.get("alpha") == 0.08  # field default still applies


def test_field_model_only_still_honored(tmp_path):
    """No ``[soil_simulation.model]`` at all -> the resolved model block is
    exactly the field-level ``[model]`` values."""
    configs = _configs(tmp_path, model={"k_s": 3.0e-4, "alpha": 0.09}, soil_simulation={})

    model_block = _model_block(configs)

    assert model_block.get("k_s") == 3.0e-4
    assert model_block.get("alpha") == 0.09


def test_soil_model_override_without_field_model(tmp_path):
    """A ``[soil_simulation.model]`` override with NO field-level ``[model]``
    resolves to exactly the override's values (no default injection)."""
    configs = _configs(tmp_path, soil_simulation={"model": {"k_s": 5.0e-5}})

    model_block = _model_block(configs)

    assert model_block.get("k_s") == 5.0e-5
    assert "alpha" not in model_block


def test_no_model_blocks_falls_back_to_pde_config_defaults(tmp_path):
    """Neither field ``[model]`` nor ``[soil_simulation.model]`` -> ``PDEConfig``
    built-in defaults apply."""
    configs = _configs(tmp_path, soil_simulation={})

    pde_config = PDEConfig(_configs(tmp_path, name="pde.conf"), model_configs=_model_block(configs))

    assert pde_config.k_s == 1.0e-4  # PDEConfig's built-in default


def test_soil_member_refetch_is_the_same_stored_object(tmp_path):
    """The cascade rests on a lories guarantee: ``get_member(defaults=)`` mutates
    the stored member in place and returns the SAME object, so an earlier
    defaults-merging fetch stays visible to every later one. A lories-side change
    must fail loudly here instead of silently splitting the sim's and planner's
    view of ``[model]``."""
    configs = _configs(tmp_path, model={"k_s": 1.0e-4, "alpha": 0.08}, soil_simulation={"model": {"k_s": 5.0e-5}})
    stored = configs.get_member("soil_simulation", defaults={"cascade_marker": 1})

    defaults = Component._build_defaults(configs, includes=["model", "plot"], strict=True)
    soil_block = configs.get_member("soil_simulation", defaults=defaults)

    assert soil_block is stored
    assert "cascade_marker" in soil_block
    assert soil_block.get_member("model", defaults={}, ensure_exists=True).get("k_s") == 5.0e-5
