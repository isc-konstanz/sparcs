# -*- coding: utf-8 -*-
"""sparcs.tests.test_soil_predictor_model_block
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Unit tests for ``SoilPredictor._resolve_model_block`` -- the guard that keeps the
predictor's resolved ``[model]`` block consistent with the live ``soil_simulation``.

The sim reads ``[model]`` through the ``[soil_simulation]`` cascade
(``Component._build_defaults(includes=["model", "plot"])``, base.py:134,160-162,202),
so a ``[soil_simulation.model]`` key-level override (e.g. a tuned ``k_s``) wins over
the field-level ``[model]`` block. Before this, the predictor read
``self.context.configs.get_member("model")`` -- the raw field block only -- so a
tuned retention param set under ``[soil_simulation.model]`` silently diverged
between the live sim and the predictor's forecast (same bug class as the
5mm/50mm ponding trap fixed for surface forcing).
"""

import pytest

from lories import Configurations

soil_predictor = pytest.importorskip("sparcs.components.agriculture.simulation.soil_predictor")
SoilPredictor = soil_predictor.SoilPredictor
PDEConfig = soil_predictor.PDEConfig


def _configs(tmp_path, name="t.conf", **values) -> Configurations:
    return Configurations.load(name, conf_dir=str(tmp_path), require=False, **values)


def test_soil_simulation_model_override_reaches_predictor(tmp_path):
    """A ``[soil_simulation.model]`` override wins for the key it restates; a key
    it leaves unset still falls back to the field-level ``[model]`` block."""
    context_configs = _configs(
        tmp_path,
        model={"k_s": 1.0e-4, "alpha": 0.08},
        soil_simulation={"model": {"k_s": 5.0e-5}},
    )

    _, model_block = SoilPredictor._resolve_model_block(context_configs)

    assert model_block.get("k_s") == 5.0e-5  # soil-level override wins
    assert model_block.get("alpha") == 0.08  # field default still applies


def test_field_model_only_still_honored(tmp_path):
    """No ``[soil_simulation.model]`` at all -> the resolved model block is
    exactly the field-level ``[model]`` values, unchanged from today's behavior."""
    context_configs = _configs(tmp_path, model={"k_s": 3.0e-4, "alpha": 0.09})

    _, model_block = SoilPredictor._resolve_model_block(context_configs)

    assert model_block.get("k_s") == 3.0e-4
    assert model_block.get("alpha") == 0.09


def test_soil_model_override_without_field_model(tmp_path):
    """A ``[soil_simulation.model]`` override with NO field-level ``[model]`` at
    all resolves to exactly the override's values (no default injection)."""
    context_configs = _configs(tmp_path, soil_simulation={"model": {"k_s": 5.0e-5}})

    _, model_block = SoilPredictor._resolve_model_block(context_configs)

    assert model_block.get("k_s") == 5.0e-5
    assert "alpha" not in model_block


def test_no_model_blocks_falls_back_to_pde_config_defaults(tmp_path):
    """Neither field ``[model]`` nor ``[soil_simulation.model]`` -> ``PDEConfig``
    built-in defaults apply, identical to today."""
    context_configs = _configs(tmp_path)

    _, model_block = SoilPredictor._resolve_model_block(context_configs)
    pde_config = PDEConfig(_configs(tmp_path, name="pde.conf"), model_configs=model_block)

    assert pde_config.k_s == 1.0e-4  # PDEConfig's built-in default
