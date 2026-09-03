# -*- coding: utf-8 -*-
"""sparcs.tests.test_predictor_rollout_aliases
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Module-identity pins for the ``_predictor_rollout`` extraction:
``SoilPredictor`` keeps ``_segment_flux_dicts`` / ``_rain_flux`` as
delegating ``staticmethod`` aliases onto the module-level functions in
``_predictor_rollout`` (attribute access on the class unwraps
``staticmethod`` to the underlying function, so ``is`` catches
copy-instead-of-move drift), and ``_worker_init`` stashes a
``RolloutEngine`` -- not a ``SoilPredictor`` -- carrying exactly the six
loose rollout fields the spawn worker's independent roll reads.

``soil_predictor`` pulls the full lories + soil (FiPy/Gmsh) stack via
``_soil.py``; ``importorskip`` keeps this file out of environments that
lack it. ``_predictor_rollout`` is imported PLAINLY on purpose: its
absence must be a hard failure (the extraction landing is exactly what
this file pins), never a silent skip.
"""

import pytest

soil_predictor = pytest.importorskip("sparcs.components.agriculture.simulation.soil_predictor")

from sparcs.components.agriculture.simulation import _predictor_rollout  # noqa: E402

SoilPredictor = soil_predictor.SoilPredictor


@pytest.mark.parametrize(
    ("pinned_name", "module_name"),
    [
        ("_segment_flux_dicts", "_segment_flux_dicts"),
        ("_rain_flux", "_rain_flux"),
    ],
)
def test_pinned_flux_static_is_module_function(pinned_name, module_name):
    assert getattr(SoilPredictor, pinned_name) is getattr(_predictor_rollout, module_name)


def test_worker_init_stashes_rollout_engine_with_six_fields(monkeypatch):
    class _FakePDE:
        def __init__(self, mesh_config, ode_config, *, rel_sat_name):
            self.rel_sat_name = rel_sat_name

    monkeypatch.setattr(_predictor_rollout, "SoilPDECore", _FakePDE)
    monkeypatch.setattr(_predictor_rollout, "ensure_mesh", lambda mesh_config: None)

    probes = [object()]
    windows = [object()]
    _predictor_rollout._worker_init(
        mesh_config=object(),
        ode_config=object(),
        rel_sat_name="predictor relative saturation",
        name="worker",
        probes=probes,
        windows=windows,
        flow_m3s=1.0e-5,
        grid_mode="fill_order",
        ic_rel_sat=None,
        et_data=None,
        seg_et={},
        horizon_start=None,
        horizon_end=None,
    )

    engine = _predictor_rollout._WORKER["engine"]
    assert isinstance(engine, _predictor_rollout.RolloutEngine)
    assert engine.name == "worker"
    assert isinstance(engine.pde, _FakePDE)
    assert engine.probes is probes
    assert engine.windows is windows
    assert engine.flow_m3s == 1.0e-5
    assert engine.grid_mode == "fill_order"


def test_module_flux_functions_are_soil_functions():
    """W4.2: the keyed flux helpers live in ``_soil`` (shared by the sim's
    ``_compute_flux_rates`` delegation); ``_predictor_rollout`` keeps the
    pinned underscore names as import-aliases of the same function objects."""
    from sparcs.components.agriculture.simulation import _soil

    assert _predictor_rollout._segment_flux_dicts is _soil.segment_flux_dicts
    assert _predictor_rollout._rain_flux is _soil.rain_flux
