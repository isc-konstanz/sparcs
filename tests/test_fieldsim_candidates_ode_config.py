# -*- coding: utf-8 -*-
"""
tests.test_fieldsim_candidates_ode_config
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``resolve_ode_config`` -- the guard that keeps a predictor-local PDE config's
surface forcing (ponding + feddes) consistent with the live sim's.

``[ponding]`` and ``[feddes]`` are sibling blocks of ``[pde]`` (not nested under
it), so a whole-block ``[pde]`` override cannot silently drop them: the sim's
forcing is inherited unless an own sibling block is supplied, which then wins.
Pure config parsing -- no mesh is built.
"""

from conftest import load_configs

from sparcs.components.agriculture.fieldsim.core.candidates import resolve_ode_config
from sparcs.components.agriculture.fieldsim.core.pde import PDEConfig, apply_surface_forcing


def _live_soil_pde(tmp_path) -> PDEConfig:
    """A parsed sim PDEConfig standing in for the live ``soil_simulation``: its
    ``[pde]`` plus the sibling ``[ponding]`` (50 mm) and ``[feddes]`` (enabled),
    attached exactly as ``SoilSimulation.configure`` does."""
    model_block = load_configs(tmp_path, name="model.conf")
    soil_block = load_configs(
        tmp_path,
        name="soil.conf",
        pde={"rain_shadow_width": 1.5},
        ponding={"h_max_mm": 8.0, "watering_h_max_mm": 50.0},
        feddes={"enabled": True, "p2_pf": 2.5},
    )
    soil_pde = PDEConfig(soil_block.get_member("pde"), model_configs=model_block)
    apply_surface_forcing(soil_pde, soil_block)
    assert soil_pde.ponding.watering_h_max_mm == 50.0  # guard: fixture is set up right
    assert soil_pde.feddes.enabled is True
    return soil_pde


def test_own_pde_without_forcing_inherits_sim_forcing(tmp_path):
    """Own ``[pde]`` but no ``[ponding]`` / ``[feddes]`` -> both are inherited from
    the sim verbatim (same objects); the solver keys still come from the
    predictor's own ``[pde]``, so the two concerns are decoupled."""
    soil_pde = _live_soil_pde(tmp_path)
    predictor_cfg = load_configs(tmp_path, pde={"dt": "5min", "rain_shadow_width": 0.7})
    model_block = load_configs(tmp_path, name="model.conf")

    ode = resolve_ode_config(predictor_cfg, soil_pde, model_block)

    assert ode is not soil_pde  # own [pde] -> a distinct PDEConfig
    assert ode.ponding is soil_pde.ponding  # ...but forcing is inherited
    assert ode.ponding.watering_h_max_mm == 50.0
    assert ode.feddes is soil_pde.feddes
    assert ode.feddes.enabled is True
    # Solver keys stay predictor-local -- a [pde] override no longer touches forcing.
    assert ode.rain_shadow_width == 0.7
    assert ode.dt == 300.0


def test_own_ponding_block_wins_and_does_not_touch_feddes(tmp_path):
    """An explicit ``[ponding]`` on the predictor overrides the sim's cap; feddes,
    left unspecified, is still inherited (the blocks are independent)."""
    soil_pde = _live_soil_pde(tmp_path)
    predictor_cfg = load_configs(tmp_path, pde={"dt": "5min"}, ponding={"watering_h_max_mm": 17.0})
    model_block = load_configs(tmp_path, name="model.conf")

    ode = resolve_ode_config(predictor_cfg, soil_pde, model_block)

    assert ode.ponding is not soil_pde.ponding
    assert ode.ponding.watering_h_max_mm == 17.0
    assert ode.feddes is soil_pde.feddes  # feddes untouched -> inherited
    assert ode.feddes.enabled is True


def test_own_feddes_block_wins_and_does_not_touch_ponding(tmp_path):
    """The mirror case: an explicit ``[feddes]`` overrides while ponding, left
    unspecified, is inherited -- confirming ponding and feddes are decoupled."""
    soil_pde = _live_soil_pde(tmp_path)
    predictor_cfg = load_configs(tmp_path, pde={"dt": "5min"}, feddes={"enabled": False})
    model_block = load_configs(tmp_path, name="model.conf")

    ode = resolve_ode_config(predictor_cfg, soil_pde, model_block)

    assert ode.feddes is not soil_pde.feddes
    assert ode.feddes.enabled is False
    assert ode.ponding is soil_pde.ponding  # ponding untouched -> inherited
    assert ode.ponding.watering_h_max_mm == 50.0


def test_no_pde_block_inherits_soil_pde_wholesale(tmp_path):
    """No ``[pde]`` block at all -> the predictor inherits the sim's PDEConfig
    object outright, forcing included."""
    soil_pde = _live_soil_pde(tmp_path)
    predictor_cfg = load_configs(tmp_path)  # no pde / ponding / feddes members
    model_block = load_configs(tmp_path, name="model.conf")

    ode = resolve_ode_config(predictor_cfg, soil_pde, model_block)

    assert ode is soil_pde
    assert ode.ponding is soil_pde.ponding
    assert ode.ponding.watering_h_max_mm == 50.0
    assert ode.feddes is soil_pde.feddes


def test_predictor_partial_ponding_override_merges_with_sim(tmp_path):
    """Predictor sets ONLY watering_h_max_mm; enabled and h_max_mm must inherit
    from the sim's resolved ponding, not reset to PondingConfig's hardcoded
    defaults (enabled=False, h_max_mm=5.0) as a base=None re-parse would."""
    model_block = load_configs(tmp_path, name="model.conf")
    soil_block = load_configs(
        tmp_path,
        name="soil.conf",
        pde={},
        ponding={"enabled": True, "h_max_mm": 8.0},
    )
    soil_pde = PDEConfig(soil_block.get_member("pde"), model_configs=model_block)
    apply_surface_forcing(soil_pde, soil_block)
    assert soil_pde.ponding.enabled is True  # guard: fixture is set up right
    assert soil_pde.ponding.h_max_mm == 8.0

    predictor_cfg = load_configs(tmp_path, pde={"dt": "5min"}, ponding={"watering_h_max_mm": 50.0})
    ode = resolve_ode_config(predictor_cfg, soil_pde, model_block)

    assert ode.ponding.enabled is True
    assert ode.ponding.h_max_mm == 8.0
    assert ode.ponding.watering_h_max_mm == 50.0


def test_no_pde_ponding_override_does_not_rewrite_sim_forcing(tmp_path):
    """HAZARD (B4 review): no [pde] but WITH a [ponding] override -- before the
    fix, ode_config WAS soil_pde in the no-[pde] branch, so apply_surface_forcing's
    wholesale .ponding replacement silently rewrote the SIM's own resolved
    forcing. The returned ode must now be a distinct object, and soil_pde's
    ponding/feddes must stay untouched -- same identity AND same values."""
    soil_pde = _live_soil_pde(tmp_path)
    orig_ponding = soil_pde.ponding
    orig_feddes = soil_pde.feddes
    predictor_cfg = load_configs(tmp_path, ponding={"watering_h_max_mm": 17.0})
    model_block = load_configs(tmp_path, name="model.conf")

    ode = resolve_ode_config(predictor_cfg, soil_pde, model_block)

    assert ode is not soil_pde
    assert ode.ponding.watering_h_max_mm == 17.0
    assert soil_pde.ponding is orig_ponding
    assert soil_pde.ponding.watering_h_max_mm == 50.0
    assert soil_pde.feddes is orig_feddes
    assert soil_pde.feddes.enabled is True


def test_no_pde_feddes_override_does_not_rewrite_sim_forcing(tmp_path):
    """Mirror of the ponding HAZARD test: a [feddes]-only override (still no
    [pde]) must also isolate via copy -- soil_pde.feddes AND soil_pde.ponding
    stay untouched."""
    soil_pde = _live_soil_pde(tmp_path)
    orig_ponding = soil_pde.ponding
    orig_feddes = soil_pde.feddes
    predictor_cfg = load_configs(tmp_path, feddes={"enabled": False})
    model_block = load_configs(tmp_path, name="model.conf")

    ode = resolve_ode_config(predictor_cfg, soil_pde, model_block)

    assert ode is not soil_pde
    assert ode.feddes.enabled is False
    assert soil_pde.feddes is orig_feddes
    assert soil_pde.feddes.enabled is True
    assert soil_pde.ponding is orig_ponding
    assert soil_pde.ponding.watering_h_max_mm == 50.0
