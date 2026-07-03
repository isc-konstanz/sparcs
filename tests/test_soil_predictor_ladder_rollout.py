# -*- coding: utf-8 -*-
"""sparcs.tests.test_soil_predictor_ladder_rollout
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PDE-backed guard for issue 04's prefix-shared caterpillar roll-out
(``SoilPredictor._rollout_ladder``): a watering candidate (active emitters, so
``surface_h`` ponds form) rolled through the caterpillar's shared-prefix path must
match an independent, no-sharing roll of the same candidate from the same IC
(``SoilPredictor._rollout_independent``), within solver tolerance.

This is the exact test that would have caught a ``snapshot()``/``set_state()``
branch-save bug: those round-trip only ``rel_sat`` and silently drop the
``surface_h`` ponds a watering branch accumulates, so a later branch would
inherit ponds from the wrong earlier duration. ``_rollout_ladder`` instead uses
``save_state_blob()``/``load_state_blob()``, which round-trip ``surface_h`` too.

Heavy (builds a real Gmsh mesh and runs FiPy): marked slow.
"""

import datetime

import pytest

import numpy as np
import pandas as pd

pytestmark = pytest.mark.slow

from lories import Configurations  # noqa: E402
from sparcs.components.agriculture.simulation._soil import (  # noqa: E402
    FluxRates,
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

WATERING = "WateringTopSegment"


def _configs(tmp_dir: str, **values) -> Configurations:
    return Configurations.load(
        "test.conf",
        conf_dir=tmp_dir,
        require=False,
        **values,
    )


def _build_core(tmp_path, dt: str = "30s") -> SoilPDECore:
    """Same small-mesh recipe as ``test_soil_strip_ponding.py``'s ``_build_core``."""
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
            dt=dt,
            dt_min="1s",
        )
    )
    ensure_mesh(mesh_config)
    return SoilPDECore(mesh_config, ode_config, rel_sat_name="Se_test")


def _watering_strip_probe(core: SoilPDECore) -> ProbeSpec:
    """A point probe under the watering strip (bay-center, just below the surface),
    where irrigation ponding directly affects the sampled Se."""
    idx = _coords_to_cell(core.mesh, core.mesh_config, x_offset_cm=0.0, depth_cm=5.0)
    return ProbeSpec(
        name="watering strip probe",
        channel_id="strip",
        cell_indices=np.array([idx], dtype=int),
        weights=np.array([1.0]),
    )


def _make_predictor(
    core: SoilPDECore,
    probes: list[ProbeSpec],
    flow_m3s: float,
    grid_mode: str = "fill_order",
) -> SoilPredictor:
    """Bare SoilPredictor instance exposing only what _rollout_ladder /
    _rollout_independent touch -- same object.__new__ pattern as
    test_soil_predictor_scheduling_gate.py's _make_gate_only_predictor."""
    predictor = object.__new__(SoilPredictor)
    predictor._name = "test_soil_predictor_ladder_rollout"
    predictor._pde = core
    predictor._probes = probes
    predictor._flow_m3s = flow_m3s
    predictor._grid_mode = grid_mode
    return predictor


# ~2000 mm/h over the 0.5 m strip -- far beyond intake; must pond, matching
# test_soil_strip_ponding.py's EXTREME_FLOW scale.
_EXTREME_FLOW = 2000.0e-3 / 3600.0 * 0.5


def test_prefix_shared_rollout_matches_independent_rollout_with_ponding(tmp_path):
    horizon_start = pd.Timestamp("2026-07-03 08:00", tz="Europe/Berlin")
    # horizon_end lands exactly at window 2's off-edge (8:20 + 5min), so the roll
    # ends right after the second pulse -- before the pond has time to drain --
    # making the ponding assertion below robust rather than a timing coincidence.
    idx = pd.DatetimeIndex(
        [horizon_start + pd.Timedelta(minutes=m) for m in (0, 10, 20, 25)],
        name="timestamp",
    )
    horizon_end = idx[-1]
    et_data = pd.DataFrame(index=idx)
    seg_et: dict[str, pd.DataFrame] = {}

    windows = [
        WateringWindow(start=datetime.time(8, 10)),
        WateringWindow(start=datetime.time(8, 20)),
    ]
    window_durations = [
        [pd.Timedelta(0), pd.Timedelta(minutes=5)],
        [pd.Timedelta(0), pd.Timedelta(minutes=5)],
    ]
    ladder = SoilPredictor._build_ladder(window_durations, grid_mode="fill_order")
    # Both windows active -- NOT the all-zero candidate.
    candidate = (pd.Timedelta(minutes=5), pd.Timedelta(minutes=5))
    assert candidate in ladder

    ladder_dir = tmp_path / "ladder"
    ladder_dir.mkdir()
    independent_dir = tmp_path / "independent"
    independent_dir.mkdir()

    core_ladder = _build_core(ladder_dir)
    ic_rel_sat = core_ladder.snapshot()
    probes_ladder = [_watering_strip_probe(core_ladder)]
    predictor_ladder = _make_predictor(core_ladder, probes_ladder, _EXTREME_FLOW)
    predictor_ladder._windows = windows
    predictor_ladder._window_durations = window_durations

    ladder_results = predictor_ladder._rollout_ladder(
        ic_rel_sat, ladder, et_data, seg_et, _EXTREME_FLOW, horizon_start, horizon_end
    )
    # Every ladder candidate must come back under its own, correctly-labelled key
    # (catches a candidate mislabelled with the wrong window's max duration).
    assert set(ladder_results.keys()) == set(ladder)
    ladder_timestamps, ladder_traj = ladder_results[candidate]

    # Reference roll from the same IC, no prefix sharing; the pond assertion below
    # confirms the scenario actually ponds (the case this guard needs to exercise).
    core_independent = _build_core(independent_dir)
    probes_independent = [_watering_strip_probe(core_independent)]
    predictor_independent = _make_predictor(core_independent, probes_independent, _EXTREME_FLOW)
    predictor_independent._windows = windows
    predictor_independent._window_durations = window_durations

    independent_timestamps, independent_traj = predictor_independent._rollout_independent(
        ic_rel_sat, candidate, et_data, seg_et, _EXTREME_FLOW, horizon_start, horizon_end
    )

    assert core_independent.surface_h[WATERING] > 0.0, (
        "test scenario must actually pond -- otherwise it cannot catch the " "snapshot()/set_state() pond-loss bug"
    )

    assert ladder_timestamps == independent_timestamps
    np.testing.assert_allclose(
        ladder_traj["strip"],
        independent_traj["strip"],
        atol=1e-6,
        err_msg="prefix-shared caterpillar roll must match an independent roll "
        "for the same watering candidate (ponding preserved via "
        "save_state_blob/load_state_blob)",
    )


def test_all_zero_candidate_reproduces_zero_irrigation_rollout(tmp_path):
    """The all-0min rung must equal a no-window zero-flow roll -- no regression
    against the pre-ladder behaviour (User Story 11)."""
    horizon_start = pd.Timestamp("2026-07-03 08:00", tz="Europe/Berlin")
    idx = pd.DatetimeIndex(
        [horizon_start + pd.Timedelta(minutes=m) for m in (0, 10, 20)],
        name="timestamp",
    )
    horizon_end = idx[-1]
    et_data = pd.DataFrame(index=idx)
    seg_et: dict[str, pd.DataFrame] = {}

    windows = [WateringWindow(start=datetime.time(8, 10))]
    window_durations = [[pd.Timedelta(0), pd.Timedelta(minutes=5)]]
    ladder = SoilPredictor._build_ladder(window_durations, grid_mode="fill_order")
    zero_candidate = (pd.Timedelta(0),)

    core = _build_core(tmp_path)
    ic_rel_sat = core.snapshot()
    probes = [_watering_strip_probe(core)]
    predictor = _make_predictor(core, probes, _EXTREME_FLOW)
    predictor._windows = windows
    predictor._window_durations = window_durations

    ladder_results = predictor._rollout_ladder(
        ic_rel_sat, ladder, et_data, seg_et, _EXTREME_FLOW, horizon_start, horizon_end
    )
    _, zero_traj = ladder_results[zero_candidate]

    control_dir = tmp_path / "control"
    control_dir.mkdir()
    core_control = _build_core(control_dir)
    core_control.set_state(ic_rel_sat)
    probe_control = _watering_strip_probe(core_control)
    control_traj = [core_control.sample(probe_control)]
    for ts_prev, ts_next in zip(idx[:-1], idx[1:]):
        core_control.walk_window(
            rates=FluxRates(seg_evap={}, seg_transp={}, flow_m3s=0.0, rain_flux=0.0),
            window_s=(ts_next - ts_prev).total_seconds(),
        )
        control_traj.append(core_control.sample(probe_control))

    np.testing.assert_allclose(zero_traj["strip"], control_traj, atol=1e-9)


def test_fill_order_candidate_keys_carry_each_window_own_max(tmp_path):
    """Windows with DIFFERENT max durations: window 0's max (10min) must label the
    key at position 0 for every window-1 rung, not window 1's own max (5min) --
    the exact mislabelling defect a same-max scenario cannot expose."""
    horizon_start = pd.Timestamp("2026-07-03 08:00", tz="Europe/Berlin")
    # window 1 (8:30) starts well after window 0's 10min pulse (8:10-8:20) ends,
    # so the two pulses never overlap; horizon_end lands at window 1's off-edge.
    idx = pd.DatetimeIndex(
        [horizon_start + pd.Timedelta(minutes=m) for m in (0, 10, 30, 35)],
        name="timestamp",
    )
    horizon_end = idx[-1]
    et_data = pd.DataFrame(index=idx)
    seg_et: dict[str, pd.DataFrame] = {}

    windows = [
        WateringWindow(start=datetime.time(8, 10)),
        WateringWindow(start=datetime.time(8, 30)),
    ]
    window_durations = [
        [pd.Timedelta(0), pd.Timedelta(minutes=10)],
        [pd.Timedelta(0), pd.Timedelta(minutes=5)],
    ]
    ladder = SoilPredictor._build_ladder(window_durations, grid_mode="fill_order")
    # (max0=10min, d1=5min) -- DEFECT 1 would key this as (5min, 5min) instead.
    candidate = (pd.Timedelta(minutes=10), pd.Timedelta(minutes=5))
    assert candidate in ladder

    ladder_dir = tmp_path / "ladder"
    ladder_dir.mkdir()
    independent_dir = tmp_path / "independent"
    independent_dir.mkdir()

    core_ladder = _build_core(ladder_dir)
    ic_rel_sat = core_ladder.snapshot()
    predictor_ladder = _make_predictor(core_ladder, [_watering_strip_probe(core_ladder)], _EXTREME_FLOW)
    predictor_ladder._windows = windows
    predictor_ladder._window_durations = window_durations

    ladder_results = predictor_ladder._rollout_ladder(
        ic_rel_sat, ladder, et_data, seg_et, _EXTREME_FLOW, horizon_start, horizon_end
    )
    assert set(ladder_results.keys()) == set(ladder)
    ladder_timestamps, ladder_traj = ladder_results[candidate]

    core_independent = _build_core(independent_dir)
    predictor_independent = _make_predictor(core_independent, [_watering_strip_probe(core_independent)], _EXTREME_FLOW)
    predictor_independent._windows = windows
    predictor_independent._window_durations = window_durations

    independent_timestamps, independent_traj = predictor_independent._rollout_independent(
        ic_rel_sat, candidate, et_data, seg_et, _EXTREME_FLOW, horizon_start, horizon_end
    )

    assert ladder_timestamps == independent_timestamps
    np.testing.assert_allclose(
        ladder_traj["strip"],
        independent_traj["strip"],
        atol=1e-6,
        err_msg="the (max0, d1) candidate must match an independent roll of the "
        "same (window0=max, window1=d1) schedule",
    )


def test_full_grid_mode_rolls_every_candidate_independently(tmp_path):
    """grid_mode='full' must produce the full Cartesian product (not the
    fill_order subset the caterpillar chain would silently substitute)."""
    horizon_start = pd.Timestamp("2026-07-03 08:00", tz="Europe/Berlin")
    idx = pd.DatetimeIndex(
        [horizon_start + pd.Timedelta(minutes=m) for m in (0, 10, 30, 35)],
        name="timestamp",
    )
    horizon_end = idx[-1]
    et_data = pd.DataFrame(index=idx)
    seg_et: dict[str, pd.DataFrame] = {}

    windows = [
        WateringWindow(start=datetime.time(8, 10)),
        WateringWindow(start=datetime.time(8, 30)),
    ]
    window_durations = [
        [pd.Timedelta(0), pd.Timedelta(minutes=10)],
        [pd.Timedelta(0), pd.Timedelta(minutes=5)],
    ]
    ladder = SoilPredictor._build_ladder(window_durations, grid_mode="full")
    assert len(ladder) == 4  # full 2x2 product, including the back-loaded combo
    # Back-loaded combo that fill_order would have dropped -- proves "full" is
    # really the product, not silently the fill_order subset.
    back_loaded = (pd.Timedelta(0), pd.Timedelta(minutes=5))
    assert back_loaded in ladder

    ladder_dir = tmp_path / "ladder"
    ladder_dir.mkdir()
    independent_dir = tmp_path / "independent"
    independent_dir.mkdir()

    core_ladder = _build_core(ladder_dir)
    ic_rel_sat = core_ladder.snapshot()
    predictor_ladder = _make_predictor(
        core_ladder, [_watering_strip_probe(core_ladder)], _EXTREME_FLOW, grid_mode="full"
    )
    predictor_ladder._windows = windows
    predictor_ladder._window_durations = window_durations

    ladder_results = predictor_ladder._rollout_ladder(
        ic_rel_sat, ladder, et_data, seg_et, _EXTREME_FLOW, horizon_start, horizon_end
    )
    assert set(ladder_results.keys()) == set(ladder)

    core_independent = _build_core(independent_dir)
    predictor_independent = _make_predictor(
        core_independent, [_watering_strip_probe(core_independent)], _EXTREME_FLOW, grid_mode="full"
    )
    predictor_independent._windows = windows
    predictor_independent._window_durations = window_durations

    independent_timestamps, independent_traj = predictor_independent._rollout_independent(
        ic_rel_sat, back_loaded, et_data, seg_et, _EXTREME_FLOW, horizon_start, horizon_end
    )
    ladder_timestamps, ladder_traj = ladder_results[back_loaded]

    assert ladder_timestamps == independent_timestamps
    np.testing.assert_allclose(
        ladder_traj["strip"],
        independent_traj["strip"],
        atol=1e-6,
        err_msg="grid_mode='full' must roll each candidate independently, "
        "matching _rollout_independent for the same candidate",
    )
