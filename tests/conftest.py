# -*- coding: utf-8 -*-
"""sparcs.tests.conftest
~~~~~~~~~~~~~~~~~~~~~~~~

Shared fixtures for the solver-backed predictor tests -- the standard
small-mesh ``SoilPDECore`` recipe and the bare-PDE ``SoilPredictor``
factory that ``test_soil_predictor_ladder_rollout.py`` and
``test_soil_predictor_zero_window.py`` previously copy-pasted -- and the
electrical-component ``configure()`` driver shared by
``test_electrical_components.py`` and ``test_sunspec_components.py``,
which must agree on the harness to be comparable.

Deliberately minimal: the other ``object.__new__`` helpers --
predict_wiring's stub-collaborator factory, the gate/bare/equivalence
predictors -- are each file's pin mechanism and stay local to their files.

Heavy imports (FiPy via ``_soil``) happen inside the fixtures, not at
module level, so collecting or running the fast tests stays light.
"""

import pytest


@pytest.fixture(scope="module")
def pde_core_factory(tmp_path_factory):
    """Factory for the standard small-mesh test core.

    Each call builds a FRESH ``SoilPDECore`` in its own tmp dir (callers
    such as the caterpillar-vs-independent parity tests need two isolated
    cores rolled from the same IC), running ``ensure_mesh`` per call.
    ``dt`` stays parameterizable -- ``test_soil_core_integration.py`` keeps
    its own dt='50s' fixture.
    """
    from lories import Configurations
    from sparcs.components.agriculture.simulation._soil import (
        MeshConfig,
        PDEConfig,
        SoilPDECore,
        ensure_mesh,
    )

    def make_core(subdir: str, dt: str = "30s") -> SoilPDECore:
        tmp_path = tmp_path_factory.mktemp(subdir)

        def _configs(**values) -> Configurations:
            return Configurations.load(
                "test.conf",
                conf_dir=str(tmp_path),
                require=False,
                **values,
            )

        mesh_config = MeshConfig(
            _configs(
                filename=str(tmp_path / "soil_test.msh"),
                dl=0.2,
                width=3.0,
                height=1.5,
                plant_width=1.0,
                plant_height=0.5,
                watering_width=0.5,
                d_x=0.5,
            )
        )
        ode_config = PDEConfig(
            _configs(
                dt=dt,
                dt_min="1s",
            )
        )
        ensure_mesh(mesh_config)
        return SoilPDECore(mesh_config, ode_config, rel_sat_name="Se_test")

    return make_core


@pytest.fixture
def strip_probe_factory():
    """Point probe under the watering strip (bay-center, just below the
    surface), where irrigation ponding directly affects the sampled Se."""
    import numpy as np
    from sparcs.components.agriculture.simulation._soil import (
        ProbeSpec,
        SoilPDECore,
        _coords_to_cell,
    )

    def make_probe(core: SoilPDECore) -> ProbeSpec:
        idx = _coords_to_cell(core.mesh, core.mesh_config, x_offset_cm=0.0, depth_cm=5.0)
        return ProbeSpec(
            name="watering strip probe",
            channel_id="strip",
            cell_indices=np.array([idx], dtype=int),
            weights=np.array([1.0]),
        )

    return make_probe


@pytest.fixture
def bare_pde_predictor():
    """Bare SoilPredictor exposing only what the roll-out mechanics touch --
    same ``object.__new__`` pattern as
    ``test_soil_predictor_scheduling_gate.py``'s ``_make_gate_only_predictor``.
    Callers assign ``_windows``/``_window_durations`` after construction."""
    from sparcs.components.agriculture.simulation.soil_predictor import SoilPredictor

    def make_predictor(
        core,
        probes,
        flow_m3s: float,
        grid_mode: str = "fill_order",
        name: str = "bare_pde_predictor",
    ) -> SoilPredictor:
        predictor = object.__new__(SoilPredictor)
        predictor._name = name
        predictor._pde = core
        predictor._probes = probes
        predictor._flow_m3s = flow_m3s
        predictor._grid_mode = grid_mode
        return predictor

    return make_predictor


@pytest.fixture
def configure_component(monkeypatch, tmp_path):
    """Drive a component's ``configure()`` and return the channels it declared.

    Full component construction needs an application context, so this runs
    ``configure()`` on a ``__new__``-bypassed instance with the lories base
    ``configure()`` no-op'ed and name/key/data patched. The recorded
    ``data.add`` calls are the assertion surface: the whole inheritance
    chain still runs, only lories' own ``Component.configure`` is stubbed.
    """
    from lories.components import Component
    from lories.core.configs.configurations import Configurations

    def _configure(component_class, toml_text: str) -> dict:
        (tmp_path / "test.conf").write_text(toml_text)
        configs = Configurations.load("test.conf", data_dir=str(tmp_path), flat=True)

        calls = []

        class _Data:
            def add(self, **channel):
                calls.append(channel)

        monkeypatch.setattr(Component, "configure", lambda self, c: None)
        monkeypatch.setattr(component_class, "name", property(lambda self: "Device 1"), raising=False)
        monkeypatch.setattr(component_class, "key", property(lambda self: "dev1"), raising=False)
        monkeypatch.setattr(component_class, "data", property(lambda self: _Data()), raising=False)

        component = component_class.__new__(component_class)
        component.configure(configs)
        return {channel["key"]: channel for channel in calls}

    return _configure
