# -*- coding: utf-8 -*-
"""
tests.test_fieldsim_planner_model_cascade
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The retention parameters the planner rolls on are the live sim's: both read one
``SoilEngine``, built from ``[soil_simulation]``'s model block after the
field-level ``[model]`` cascade (``Component._build_defaults(includes=["model"])``,
the merge ``FieldSimulation.configure`` performs). A key-level
``[soil_simulation.model]`` override therefore wins while unstated keys still
fall back to the field block -- a tuned ``k_s`` can never diverge between the
forecast and the sim. Pure config parsing; the mesh and the FiPy core are stubbed.
"""

from types import SimpleNamespace

from conftest import MESH_KW, load_configs

from lories import Configurations
from lories.components import Component
from sparcs.components.agriculture.simulation.core import engine as engine_module
from sparcs.components.agriculture.simulation.core.config import SoilConfig
from sparcs.components.agriculture.simulation.core.engine import SoilEngine
from sparcs.components.agriculture.simulation.core.pde import PDEConfig


def _configs(tmp_path, soil_simulation: dict, **field) -> Configurations:
    """Field-level ``**field`` plus a ``[soil_simulation]`` carrying the standard test mesh."""
    mesh = {**MESH_KW, "filename": str(tmp_path / "soil.msh")}
    return load_configs(tmp_path, soil_simulation={**soil_simulation, "mesh": mesh}, **field)


def _soil_section(configs: Configurations) -> SoilConfig:
    """``[soil_simulation]`` as ``FieldSimulation.configure`` builds it: the field-level
    ``[model]`` cascades in as a default before the section is configured."""
    defaults = Component._build_defaults(configs, includes=["model", "plot"], strict=True)
    soil = SoilConfig()
    soil.configure(configs.get_member("soil_simulation", defaults=defaults).copy())
    soil.mesh.derive(bay_width=3.0)
    return soil


def _model_block(monkeypatch, configs: Configurations) -> Configurations:
    """The model block ``SoilEngine.build`` resolves for the section and hands to
    ``resolve_pde_config``."""
    seen = {}
    resolve_pde_config = engine_module.resolve_pde_config

    def spy(component_block, model_block, inherit_forcing_from=None):
        seen["model"] = model_block
        return resolve_pde_config(component_block, model_block, inherit_forcing_from)

    monkeypatch.setattr(engine_module, "ensure_mesh", lambda mesh_config: None)
    monkeypatch.setattr(engine_module, "SoilPDECore", lambda *args, **kwargs: SimpleNamespace(soil_model=None))
    monkeypatch.setattr(engine_module, "resolve_pde_config", spy)
    SoilEngine.build(_soil_section(configs))
    return seen["model"]


def test_soil_simulation_model_override_reaches_the_engine(tmp_path, monkeypatch):
    """A ``[soil_simulation.model]`` override wins for the key it restates; a key
    it leaves unset still falls back to the field-level ``[model]`` block."""
    configs = _configs(tmp_path, {"model": {"k_s": 5.0e-5}}, model={"k_s": 1.0e-4, "alpha": 0.08})

    model_block = _model_block(monkeypatch, configs)

    assert model_block.get("k_s") == 5.0e-5  # soil-level override wins
    assert model_block.get("alpha") == 0.08  # field default still applies


def test_field_model_only_still_honored(tmp_path, monkeypatch):
    """No ``[soil_simulation.model]`` at all -> the resolved model block is
    exactly the field-level ``[model]`` values."""
    configs = _configs(tmp_path, {}, model={"k_s": 3.0e-4, "alpha": 0.09})

    model_block = _model_block(monkeypatch, configs)

    assert model_block.get("k_s") == 3.0e-4
    assert model_block.get("alpha") == 0.09


def test_soil_model_override_without_field_model(tmp_path, monkeypatch):
    """A ``[soil_simulation.model]`` override with NO field-level ``[model]``
    resolves to exactly the override's values (no default injection)."""
    configs = _configs(tmp_path, {"model": {"k_s": 5.0e-5}})

    model_block = _model_block(monkeypatch, configs)

    assert model_block.get("k_s") == 5.0e-5
    assert "alpha" not in model_block


def test_no_model_blocks_falls_back_to_pde_config_defaults(tmp_path, monkeypatch):
    """Neither field ``[model]`` nor ``[soil_simulation.model]`` -> ``PDEConfig``
    built-in defaults apply."""
    configs = _configs(tmp_path, {})

    pde_config = PDEConfig(load_configs(tmp_path, name="pde.conf"), model_configs=_model_block(monkeypatch, configs))

    assert pde_config.k_s == 1.0e-4  # PDEConfig's built-in default


def test_soil_member_refetch_is_the_same_stored_object(tmp_path):
    """The cascade rests on a lories guarantee: ``get_member(defaults=)`` mutates
    the stored member in place and returns the SAME object, so an earlier
    defaults-merging fetch stays visible to every later one. A lories-side change
    must fail loudly here instead of silently splitting the sim's and planner's
    view of ``[model]``."""
    configs = load_configs(tmp_path, model={"k_s": 1.0e-4, "alpha": 0.08}, soil_simulation={"model": {"k_s": 5.0e-5}})
    stored = configs.get_member("soil_simulation", defaults={"cascade_marker": 1})

    defaults = Component._build_defaults(configs, includes=["model", "plot"], strict=True)
    soil_block = configs.get_member("soil_simulation", defaults=defaults)

    assert soil_block is stored
    assert "cascade_marker" in soil_block
    assert soil_block.get_member("model", defaults={}, ensure_exists=True).get("k_s") == 5.0e-5
