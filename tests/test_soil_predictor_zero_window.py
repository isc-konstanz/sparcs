# -*- coding: utf-8 -*-
"""Regression: a middle watering window whose durations are only ``0min``
contributes no ladder rungs, so the caterpillar's max-branch save never runs
for it. Pre-fix, the shared prefix then stalled at that window's segment start
and every later window's roll silently skipped the weather in between --
diverging from the independent reference roll and dropping the skipped
segment's timestamps from the trajectory.

Heavy (builds a real Gmsh mesh and runs FiPy): marked slow.
"""

import datetime

import pytest

import numpy as np
import pandas as pd

pytestmark = pytest.mark.slow

from lories import Configurations  # noqa: E402
from sparcs.components.agriculture.simulation._soil import (  # noqa: E402
    MeshConfig,
    PDEConfig,
    ProbeSpec,
    SoilPDECore,
    _coords_to_cell,
    ensure_mesh,
)
from sparcs.components.agriculture.simulation.soil_predictor import (  # noqa: E402
    SoilPredictor,
    WateringWindow,
)


def _configs(tmp_dir: str, **values) -> Configurations:
    return Configurations.load(
        "test.conf",
        conf_dir=tmp_dir,
        require=False,
        **values,
    )


def _build_core(tmp_path) -> SoilPDECore:
    """Same small-mesh recipe as ``test_soil_predictor_ladder_rollout.py``."""
    mesh_config = MeshConfig(
        _configs(
            str(tmp_path),
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
            str(tmp_path),
            dt="30s",
            dt_min="1s",
        )
    )
    ensure_mesh(mesh_config)
    return SoilPDECore(mesh_config, ode_config, rel_sat_name="Se_test")


def _strip_probe(core: SoilPDECore) -> ProbeSpec:
    idx = _coords_to_cell(core.mesh, core.mesh_config, x_offset_cm=0.0, depth_cm=5.0)
    return ProbeSpec(
        name="watering strip probe",
        channel_id="strip",
        cell_indices=np.array([idx], dtype=int),
        weights=np.array([1.0]),
    )


def _make_predictor(core: SoilPDECore, flow_m3s: float) -> SoilPredictor:
    predictor = object.__new__(SoilPredictor)
    predictor._name = "test_soil_predictor_zero_window"
    predictor._pde = core
    predictor._probes = [_strip_probe(core)]
    predictor._flow_m3s = flow_m3s
    predictor._grid_mode = "fill_order"
    return predictor


# Same extreme-flow scale as the ladder-rollout test: guarantees visible effect.
_EXTREME_FLOW = 2000.0e-3 / 3600.0 * 0.5


def test_zero_only_middle_window_still_advances_the_prefix(tmp_path):
    horizon_start = pd.Timestamp("2026-07-03 08:00", tz="Europe/Berlin")
    idx = pd.DatetimeIndex(
        [horizon_start + pd.Timedelta(minutes=m) for m in (0, 10, 20, 30, 35)],
        name="timestamp",
    )
    horizon_end = idx[-1]
    et_data = pd.DataFrame(index=idx)
    seg_et: dict[str, pd.DataFrame] = {}

    windows = [
        WateringWindow(start=datetime.time(8, 10)),
        WateringWindow(start=datetime.time(8, 20)),  # degenerate: 0min only
        WateringWindow(start=datetime.time(8, 30)),
    ]
    window_durations = [
        [pd.Timedelta(0), pd.Timedelta(minutes=5)],
        [pd.Timedelta(0)],
        [pd.Timedelta(0), pd.Timedelta(minutes=5)],
    ]
    ladder = SoilPredictor._build_ladder(window_durations, grid_mode="fill_order")
    candidate = (pd.Timedelta(minutes=5), pd.Timedelta(0), pd.Timedelta(minutes=5))
    assert candidate in ladder

    ladder_dir = tmp_path / "ladder"
    ladder_dir.mkdir()
    core_ladder = _build_core(ladder_dir)
    ic_rel_sat = core_ladder.snapshot()
    predictor_ladder = _make_predictor(core_ladder, _EXTREME_FLOW)
    predictor_ladder._windows = windows
    predictor_ladder._window_durations = window_durations

    ladder_results = predictor_ladder._rollout_ladder(
        ic_rel_sat, ladder, et_data, seg_et, _EXTREME_FLOW, horizon_start, horizon_end
    )
    ladder_timestamps, ladder_traj = ladder_results[candidate]

    # Pre-fix failure mode 1: the degenerate window's segment timestamps vanish
    # from every later candidate's trajectory.
    assert list(ladder_timestamps) == list(idx)

    independent_dir = tmp_path / "independent"
    independent_dir.mkdir()
    core_independent = _build_core(independent_dir)
    predictor_independent = _make_predictor(core_independent, _EXTREME_FLOW)
    predictor_independent._windows = windows
    predictor_independent._window_durations = window_durations

    ref_timestamps, ref_traj = predictor_independent._rollout_independent(
        ic_rel_sat, candidate, et_data, seg_et, _EXTREME_FLOW, horizon_start, horizon_end
    )

    # Pre-fix failure mode 2: the skipped segment's drainage never happens, so
    # the shared-prefix roll diverges from the independent reference.
    assert list(ref_timestamps) == list(idx)
    np.testing.assert_allclose(ladder_traj["strip"], ref_traj["strip"], rtol=1e-6, atol=1e-9)
