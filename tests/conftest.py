# -*- coding: utf-8 -*-
"""
tests.conftest
~~~~~~~~~~~~~~

Shared helpers and fixtures for the fieldsim tests: the standard test mesh
kwargs, a minute ``Timedelta``, a ``Configurations`` loader, the standard
small-mesh ``SoilPDECore`` recipe, a watering-strip probe, and the bare
``RolloutEngine`` factory the ladder/zero-window roll-out pins share.

Heavy imports (FiPy via ``core.pde``) happen inside the fixtures, not at
module level, so collecting or running the fast tests stays light.
"""

import pytest

import pandas as pd
from lories import Configurations

MESH_KW = {
    "dl": 0.2,
    "width": 3.0,
    "height": 1.5,
    "plant_width": 1.0,
    "plant_height": 0.5,
    "watering_width": 0.5,
    "d_x": 0.5,
}


def td(minutes: int) -> pd.Timedelta:
    return pd.Timedelta(minutes=minutes)


def load_configs(tmp_path, name="t.conf", **values) -> Configurations:
    return Configurations.load(name, conf_dir=str(tmp_path), require=False, **values)


@pytest.fixture(scope="module")
def pde_core_factory(tmp_path_factory):
    """Factory for the standard small-mesh test core.

    Each call builds a FRESH ``SoilPDECore`` in its own tmp dir (callers
    such as the caterpillar-vs-independent parity tests need two isolated
    cores rolled from the same IC), running ``ensure_mesh`` per call.
    ``dt`` stays parameterizable -- ``test_soil_core_integration.py`` keeps
    its own dt='50s' fixture.
    """
    from sparcs.components.agriculture.simulation.core.pde import (
        MeshConfig,
        PDEConfig,
        SoilPDECore,
        ensure_mesh,
    )

    def make_core(subdir: str, dt: str = "30s", **ode_values) -> SoilPDECore:
        tmp_path = tmp_path_factory.mktemp(subdir)
        mesh_config = MeshConfig(
            load_configs(tmp_path, "test.conf", filename=str(tmp_path / "soil_test.msh"), **MESH_KW)
        )
        ode_config = PDEConfig(load_configs(tmp_path, "test.conf", dt=dt, dt_min="1s", **ode_values))
        ensure_mesh(mesh_config)
        return SoilPDECore(mesh_config, ode_config, rel_sat_name="Se_test")

    return make_core


@pytest.fixture
def strip_probe_factory():
    """Point probe under the watering strip (bay-center, just below the
    surface), where irrigation ponding directly affects the sampled Se."""
    import numpy as np
    from sparcs.components.agriculture.simulation.core.pde import (
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
def rollout_engine_factory():
    """Bare ``RolloutEngine`` over a core/probe pair, carrying only the loose
    fields the roll-out methods read."""
    from sparcs.components.agriculture.simulation.core.rollout import RolloutEngine

    def make_engine(
        core,
        probes,
        flow_m3s: float,
        grid_mode: str = "fill_order",
        name: str = "rollout_engine",
        **fields,
    ) -> RolloutEngine:
        return RolloutEngine(pde=core, probes=probes, flow_m3s=flow_m3s, grid_mode=grid_mode, name=name, **fields)

    return make_engine
