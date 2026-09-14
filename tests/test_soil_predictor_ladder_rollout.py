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

from sparcs.components.agriculture.simulation._soil import FluxRates  # noqa: E402
from sparcs.components.agriculture.simulation.soil_predictor import (  # noqa: E402
    SoilPredictor,
    WateringWindow,
)

WATERING = "WateringTopSegment"

# ~2000 mm/h over the 0.5 m strip -- far beyond intake; must pond, matching
# test_soil_strip_ponding.py's EXTREME_FLOW scale.
_EXTREME_FLOW = 2000.0e-3 / 3600.0 * 0.5


def test_prefix_shared_rollout_matches_independent_rollout_with_ponding(
    pde_core_factory, strip_probe_factory, bare_pde_predictor
):
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

    core_ladder = pde_core_factory("ladder")
    ic_rel_sat = core_ladder.snapshot()
    probes_ladder = [strip_probe_factory(core_ladder)]
    predictor_ladder = bare_pde_predictor(core_ladder, probes_ladder, _EXTREME_FLOW)
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
    core_independent = pde_core_factory("independent")
    probes_independent = [strip_probe_factory(core_independent)]
    predictor_independent = bare_pde_predictor(core_independent, probes_independent, _EXTREME_FLOW)
    predictor_independent._windows = windows
    predictor_independent._window_durations = window_durations

    independent_timestamps, independent_traj = predictor_independent._rollout_independent(
        ic_rel_sat, candidate, et_data, seg_et, _EXTREME_FLOW, horizon_start, horizon_end
    )

    assert (
        core_independent.surface_h[WATERING] > 0.0
    ), "test scenario must actually pond -- otherwise it cannot catch the snapshot()/set_state() pond-loss bug"

    assert ladder_timestamps == independent_timestamps
    np.testing.assert_allclose(
        ladder_traj["strip"],
        independent_traj["strip"],
        atol=1e-6,
        err_msg="prefix-shared caterpillar roll must match an independent roll "
        "for the same watering candidate (ponding preserved via "
        "save_state_blob/load_state_blob)",
    )


def test_all_zero_candidate_reproduces_zero_irrigation_rollout(
    pde_core_factory, strip_probe_factory, bare_pde_predictor
):
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

    core = pde_core_factory("zero")
    ic_rel_sat = core.snapshot()
    probes = [strip_probe_factory(core)]
    predictor = bare_pde_predictor(core, probes, _EXTREME_FLOW)
    predictor._windows = windows
    predictor._window_durations = window_durations

    ladder_results = predictor._rollout_ladder(
        ic_rel_sat, ladder, et_data, seg_et, _EXTREME_FLOW, horizon_start, horizon_end
    )
    _, zero_traj = ladder_results[zero_candidate]

    core_control = pde_core_factory("control")
    core_control.set_state(ic_rel_sat)
    probe_control = strip_probe_factory(core_control)
    control_traj = [core_control.sample(probe_control)]
    for ts_prev, ts_next in zip(idx[:-1], idx[1:]):
        core_control.walk_window(
            rates=FluxRates(seg_evap={}, seg_transp={}, flow_m3s=0.0, rain_flux=0.0),
            window_s=(ts_next - ts_prev).total_seconds(),
        )
        control_traj.append(core_control.sample(probe_control))

    np.testing.assert_allclose(zero_traj["strip"], control_traj, atol=1e-9)


def test_fill_order_candidate_keys_carry_each_window_own_max(pde_core_factory, strip_probe_factory, bare_pde_predictor):
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

    core_ladder = pde_core_factory("ladder")
    ic_rel_sat = core_ladder.snapshot()
    predictor_ladder = bare_pde_predictor(core_ladder, [strip_probe_factory(core_ladder)], _EXTREME_FLOW)
    predictor_ladder._windows = windows
    predictor_ladder._window_durations = window_durations

    ladder_results = predictor_ladder._rollout_ladder(
        ic_rel_sat, ladder, et_data, seg_et, _EXTREME_FLOW, horizon_start, horizon_end
    )
    assert set(ladder_results.keys()) == set(ladder)
    ladder_timestamps, ladder_traj = ladder_results[candidate]

    core_independent = pde_core_factory("independent")
    predictor_independent = bare_pde_predictor(core_independent, [strip_probe_factory(core_independent)], _EXTREME_FLOW)
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


def test_full_grid_mode_rolls_every_candidate_independently(pde_core_factory, strip_probe_factory, bare_pde_predictor):
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

    core_ladder = pde_core_factory("ladder")
    ic_rel_sat = core_ladder.snapshot()
    predictor_ladder = bare_pde_predictor(
        core_ladder, [strip_probe_factory(core_ladder)], _EXTREME_FLOW, grid_mode="full"
    )
    predictor_ladder._windows = windows
    predictor_ladder._window_durations = window_durations

    ladder_results = predictor_ladder._rollout_ladder(
        ic_rel_sat, ladder, et_data, seg_et, _EXTREME_FLOW, horizon_start, horizon_end
    )
    assert set(ladder_results.keys()) == set(ladder)

    core_independent = pde_core_factory("independent")
    predictor_independent = bare_pde_predictor(
        core_independent, [strip_probe_factory(core_independent)], _EXTREME_FLOW, grid_mode="full"
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


def test_collapsed_segment_bounds_fall_back_to_independent_rolls(
    pde_core_factory, strip_probe_factory, bare_pde_predictor
):
    """When two window starts floor to the SAME forecast timestamp (a window pair
    inside one forecast interval), the caterpillar's segment save/restore would
    silently drop the earlier window's water. The strictly-increasing-bounds guard
    must instead fall back to independent per-candidate rolls, so a two-window
    watering candidate still matches its independent roll (and actually ponds)."""
    horizon_start = pd.Timestamp("2026-07-03 08:00", tz="Europe/Berlin")
    # Grid points at 0/10/20/23 min. Windows at 8:12 and 8:18 BOTH floor to 8:10 ->
    # collapsed segment bounds -> fallback. Non-overlapping pulses (8:12-8:17, 8:18-8:23);
    # the horizon ends with the last pulse so the strip pond has not infiltrated yet.
    idx = pd.DatetimeIndex(
        [horizon_start + pd.Timedelta(minutes=m) for m in (0, 10, 20, 23)],
        name="timestamp",
    )
    horizon_end = idx[-1]
    et_data = pd.DataFrame(index=idx)
    seg_et: dict[str, pd.DataFrame] = {}

    windows = [
        WateringWindow(start=datetime.time(8, 12)),
        WateringWindow(start=datetime.time(8, 18)),
    ]
    window_durations = [
        [pd.Timedelta(0), pd.Timedelta(minutes=5)],
        [pd.Timedelta(0), pd.Timedelta(minutes=5)],
    ]
    ladder = SoilPredictor._build_ladder(window_durations, grid_mode="fill_order")
    candidate = (pd.Timedelta(minutes=5), pd.Timedelta(minutes=5))  # both windows active
    assert candidate in ladder

    core_ladder = pde_core_factory("ladder")
    ic_rel_sat = core_ladder.snapshot()
    predictor_ladder = bare_pde_predictor(core_ladder, [strip_probe_factory(core_ladder)], _EXTREME_FLOW)
    predictor_ladder._windows = windows
    predictor_ladder._window_durations = window_durations

    ladder_results = predictor_ladder._rollout_ladder(
        ic_rel_sat, ladder, et_data, seg_et, _EXTREME_FLOW, horizon_start, horizon_end
    )
    assert set(ladder_results.keys()) == set(ladder)
    ladder_timestamps, ladder_traj = ladder_results[candidate]

    core_independent = pde_core_factory("independent")
    predictor_independent = bare_pde_predictor(core_independent, [strip_probe_factory(core_independent)], _EXTREME_FLOW)
    predictor_independent._windows = windows
    predictor_independent._window_durations = window_durations

    independent_timestamps, independent_traj = predictor_independent._rollout_independent(
        ic_rel_sat, candidate, et_data, seg_et, _EXTREME_FLOW, horizon_start, horizon_end
    )

    assert core_independent.surface_h[WATERING] > 0.0, "scenario must pond to exercise the dropped-water bug"
    assert ladder_timestamps == independent_timestamps
    np.testing.assert_allclose(
        ladder_traj["strip"],
        independent_traj["strip"],
        atol=1e-6,
        err_msg="collapsed segment bounds must fall back to an independent roll that "
        "still applies BOTH windows' water (pre-fix the caterpillar dropped the earlier window)",
    )
