# -*- coding: utf-8 -*-
"""
tests.test_fieldsim_rollout_ladder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PDE-backed guard for the prefix-shared caterpillar (``rollout_ladder``): every
candidate rolled through the shared prefix must match an independent roll of
the same candidate from the same IC (``rollout_independent``), within solver
tolerance -- including when a watering branch ponds, when the windows carry
different maxima, when a middle window is zero-only, and when two window starts
collapse onto one forecast timestamp.

Heavy (builds a real Gmsh mesh and runs FiPy): marked slow.
"""

import datetime

import pytest

import numpy as np
import pandas as pd

pytestmark = pytest.mark.slow

from sparcs.components.agriculture.simulation.core.candidates import (  # noqa: E402
    WateringWindow,
    build_candidate_grid,
)
from sparcs.components.agriculture.simulation.core.pde import FluxRates  # noqa: E402

WATERING = "WateringTopSegment"

# ~2000 mm/h over the 0.5 m strip -- far beyond intake; must pond, matching
# test_soil_strip_ponding.py's EXTREME_FLOW scale.
_EXTREME_FLOW = 2000.0e-3 / 3600.0 * 0.5

_TZ = "Europe/Berlin"


def _index(horizon_start, minutes) -> pd.DatetimeIndex:
    return pd.DatetimeIndex([horizon_start + pd.Timedelta(minutes=m) for m in minutes], name="timestamp")


def _windows(*starts) -> list:
    return [WateringWindow(start=datetime.time(*start)) for start in starts]


def _engine(factory, core_factory, probe_factory, tag, windows, window_durations, grid_mode="fill_order"):
    core = core_factory(tag)
    engine = factory(
        core,
        [probe_factory(core)],
        _EXTREME_FLOW,
        grid_mode=grid_mode,
        windows=windows,
        window_durations=window_durations,
    )
    return core, engine


def test_prefix_shared_rollout_matches_independent_rollout_with_ponding(
    pde_core_factory, strip_probe_factory, rollout_engine_factory
):
    horizon_start = pd.Timestamp("2026-07-03 08:00", tz=_TZ)
    # horizon_end lands exactly at window 2's off-edge (8:20 + 5min), so the roll
    # ends right after the second pulse -- before the pond has time to drain.
    idx = _index(horizon_start, (0, 10, 20, 25))
    horizon_end = idx[-1]
    et_data = pd.DataFrame(index=idx)
    seg_et: dict[str, pd.DataFrame] = {}

    windows = _windows((8, 10), (8, 20))
    window_durations = [
        [pd.Timedelta(0), pd.Timedelta(minutes=5)],
        [pd.Timedelta(0), pd.Timedelta(minutes=5)],
    ]
    ladder = build_candidate_grid(window_durations, grid_mode="fill_order")
    candidate = (pd.Timedelta(minutes=5), pd.Timedelta(minutes=5))  # both windows active
    assert candidate in ladder

    core_ladder, engine_ladder = _engine(
        rollout_engine_factory, pde_core_factory, strip_probe_factory, "ladder", windows, window_durations
    )
    ic_rel_sat = core_ladder.snapshot()

    ladder_results = engine_ladder.rollout_ladder(
        ic_rel_sat, ladder, et_data, seg_et, _EXTREME_FLOW, horizon_start, horizon_end
    )
    # Every ladder candidate must come back under its own, correctly-labelled key.
    assert set(ladder_results.keys()) == set(ladder)
    ladder_timestamps, ladder_traj = ladder_results[candidate]

    core_independent, engine_independent = _engine(
        rollout_engine_factory, pde_core_factory, strip_probe_factory, "independent", windows, window_durations
    )
    independent_timestamps, independent_traj = engine_independent.rollout_independent(
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
        "for the same watering candidate (ponding preserved via save/load_state_blob)",
    )


def test_all_zero_candidate_reproduces_zero_irrigation_rollout(
    pde_core_factory, strip_probe_factory, rollout_engine_factory
):
    """The all-0min rung must equal a no-window zero-flow roll."""
    horizon_start = pd.Timestamp("2026-07-03 08:00", tz=_TZ)
    idx = _index(horizon_start, (0, 10, 20))
    horizon_end = idx[-1]
    et_data = pd.DataFrame(index=idx)
    seg_et: dict[str, pd.DataFrame] = {}

    windows = _windows((8, 10))
    window_durations = [[pd.Timedelta(0), pd.Timedelta(minutes=5)]]
    ladder = build_candidate_grid(window_durations, grid_mode="fill_order")
    zero_candidate = (pd.Timedelta(0),)

    core, engine = _engine(
        rollout_engine_factory, pde_core_factory, strip_probe_factory, "zero", windows, window_durations
    )
    ic_rel_sat = core.snapshot()

    ladder_results = engine.rollout_ladder(
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


def test_fill_order_candidate_keys_carry_each_window_own_max(
    pde_core_factory, strip_probe_factory, rollout_engine_factory
):
    """Windows with DIFFERENT max durations: window 0's max (10min) must label the
    key at position 0 for every window-1 rung, not window 1's own max (5min)."""
    horizon_start = pd.Timestamp("2026-07-03 08:00", tz=_TZ)
    # window 1 (8:30) starts well after window 0's 10min pulse (8:10-8:20) ends,
    # so the two pulses never overlap; horizon_end lands at window 1's off-edge.
    idx = _index(horizon_start, (0, 10, 30, 35))
    horizon_end = idx[-1]
    et_data = pd.DataFrame(index=idx)
    seg_et: dict[str, pd.DataFrame] = {}

    windows = _windows((8, 10), (8, 30))
    window_durations = [
        [pd.Timedelta(0), pd.Timedelta(minutes=10)],
        [pd.Timedelta(0), pd.Timedelta(minutes=5)],
    ]
    ladder = build_candidate_grid(window_durations, grid_mode="fill_order")
    candidate = (pd.Timedelta(minutes=10), pd.Timedelta(minutes=5))  # (max0, d1)
    assert candidate in ladder

    core_ladder, engine_ladder = _engine(
        rollout_engine_factory, pde_core_factory, strip_probe_factory, "ladder", windows, window_durations
    )
    ic_rel_sat = core_ladder.snapshot()

    ladder_results = engine_ladder.rollout_ladder(
        ic_rel_sat, ladder, et_data, seg_et, _EXTREME_FLOW, horizon_start, horizon_end
    )
    assert set(ladder_results.keys()) == set(ladder)
    ladder_timestamps, ladder_traj = ladder_results[candidate]

    _core_independent, engine_independent = _engine(
        rollout_engine_factory, pde_core_factory, strip_probe_factory, "independent", windows, window_durations
    )
    independent_timestamps, independent_traj = engine_independent.rollout_independent(
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


def test_full_grid_mode_rolls_every_candidate_independently(
    pde_core_factory, strip_probe_factory, rollout_engine_factory
):
    """grid_mode='full' must produce the full Cartesian product, not the
    fill_order subset the caterpillar chain would silently substitute."""
    horizon_start = pd.Timestamp("2026-07-03 08:00", tz=_TZ)
    idx = _index(horizon_start, (0, 10, 30, 35))
    horizon_end = idx[-1]
    et_data = pd.DataFrame(index=idx)
    seg_et: dict[str, pd.DataFrame] = {}

    windows = _windows((8, 10), (8, 30))
    window_durations = [
        [pd.Timedelta(0), pd.Timedelta(minutes=10)],
        [pd.Timedelta(0), pd.Timedelta(minutes=5)],
    ]
    ladder = build_candidate_grid(window_durations, grid_mode="full")
    assert len(ladder) == 4  # full 2x2 product, including the back-loaded combo
    back_loaded = (pd.Timedelta(0), pd.Timedelta(minutes=5))
    assert back_loaded in ladder

    core_ladder, engine_ladder = _engine(
        rollout_engine_factory,
        pde_core_factory,
        strip_probe_factory,
        "ladder",
        windows,
        window_durations,
        grid_mode="full",
    )
    ic_rel_sat = core_ladder.snapshot()

    ladder_results = engine_ladder.rollout_ladder(
        ic_rel_sat, ladder, et_data, seg_et, _EXTREME_FLOW, horizon_start, horizon_end
    )
    assert set(ladder_results.keys()) == set(ladder)

    _core_independent, engine_independent = _engine(
        rollout_engine_factory,
        pde_core_factory,
        strip_probe_factory,
        "independent",
        windows,
        window_durations,
        grid_mode="full",
    )
    independent_timestamps, independent_traj = engine_independent.rollout_independent(
        ic_rel_sat, back_loaded, et_data, seg_et, _EXTREME_FLOW, horizon_start, horizon_end
    )
    ladder_timestamps, ladder_traj = ladder_results[back_loaded]

    assert ladder_timestamps == independent_timestamps
    np.testing.assert_allclose(
        ladder_traj["strip"],
        independent_traj["strip"],
        atol=1e-6,
        err_msg="grid_mode='full' must roll each candidate independently, "
        "matching rollout_independent for the same candidate",
    )


def test_collapsed_segment_bounds_fall_back_to_independent_rolls(
    pde_core_factory, strip_probe_factory, rollout_engine_factory
):
    """When two window starts floor to the SAME forecast timestamp, the
    caterpillar's segment save/restore would silently drop the earlier window's
    water. The strictly-increasing-bounds guard must fall back to independent
    per-candidate rolls instead."""
    horizon_start = pd.Timestamp("2026-07-03 08:00", tz=_TZ)
    # Grid points at 0/10/20/23 min. Windows at 8:12 and 8:18 BOTH floor to 8:10.
    idx = _index(horizon_start, (0, 10, 20, 23))
    horizon_end = idx[-1]
    et_data = pd.DataFrame(index=idx)
    seg_et: dict[str, pd.DataFrame] = {}

    windows = _windows((8, 12), (8, 18))
    window_durations = [
        [pd.Timedelta(0), pd.Timedelta(minutes=5)],
        [pd.Timedelta(0), pd.Timedelta(minutes=5)],
    ]
    ladder = build_candidate_grid(window_durations, grid_mode="fill_order")
    candidate = (pd.Timedelta(minutes=5), pd.Timedelta(minutes=5))
    assert candidate in ladder

    core_ladder, engine_ladder = _engine(
        rollout_engine_factory, pde_core_factory, strip_probe_factory, "ladder", windows, window_durations
    )
    ic_rel_sat = core_ladder.snapshot()

    ladder_results = engine_ladder.rollout_ladder(
        ic_rel_sat, ladder, et_data, seg_et, _EXTREME_FLOW, horizon_start, horizon_end
    )
    assert set(ladder_results.keys()) == set(ladder)
    ladder_timestamps, ladder_traj = ladder_results[candidate]

    core_independent, engine_independent = _engine(
        rollout_engine_factory, pde_core_factory, strip_probe_factory, "independent", windows, window_durations
    )
    independent_timestamps, independent_traj = engine_independent.rollout_independent(
        ic_rel_sat, candidate, et_data, seg_et, _EXTREME_FLOW, horizon_start, horizon_end
    )

    assert core_independent.surface_h[WATERING] > 0.0, "scenario must pond to exercise the dropped-water bug"
    assert ladder_timestamps == independent_timestamps
    np.testing.assert_allclose(
        ladder_traj["strip"],
        independent_traj["strip"],
        atol=1e-6,
        err_msg="collapsed segment bounds must fall back to an independent roll that still applies BOTH windows' water",
    )


def test_zero_only_middle_window_still_advances_the_prefix(
    pde_core_factory, strip_probe_factory, rollout_engine_factory
):
    """A middle window whose durations are only ``0min`` contributes no ladder
    rungs, so the caterpillar's max-branch save never runs for it: the shared
    prefix must still advance through that window's segment instead of stalling
    and dropping the skipped segment's weather and timestamps."""
    horizon_start = pd.Timestamp("2026-07-03 08:00", tz=_TZ)
    idx = _index(horizon_start, (0, 10, 20, 30, 35))
    horizon_end = idx[-1]
    et_data = pd.DataFrame(index=idx)
    seg_et: dict[str, pd.DataFrame] = {}

    windows = _windows((8, 10), (8, 20), (8, 30))  # the middle window is 0min-only
    window_durations = [
        [pd.Timedelta(0), pd.Timedelta(minutes=5)],
        [pd.Timedelta(0)],
        [pd.Timedelta(0), pd.Timedelta(minutes=5)],
    ]
    ladder = build_candidate_grid(window_durations, grid_mode="fill_order")
    candidate = (pd.Timedelta(minutes=5), pd.Timedelta(0), pd.Timedelta(minutes=5))
    assert candidate in ladder

    core_ladder, engine_ladder = _engine(
        rollout_engine_factory, pde_core_factory, strip_probe_factory, "ladder", windows, window_durations
    )
    ic_rel_sat = core_ladder.snapshot()

    ladder_results = engine_ladder.rollout_ladder(
        ic_rel_sat, ladder, et_data, seg_et, _EXTREME_FLOW, horizon_start, horizon_end
    )
    ladder_timestamps, ladder_traj = ladder_results[candidate]
    assert list(ladder_timestamps) == list(idx)

    _core_independent, engine_independent = _engine(
        rollout_engine_factory, pde_core_factory, strip_probe_factory, "independent", windows, window_durations
    )
    ref_timestamps, ref_traj = engine_independent.rollout_independent(
        ic_rel_sat, candidate, et_data, seg_et, _EXTREME_FLOW, horizon_start, horizon_end
    )

    assert list(ref_timestamps) == list(idx)
    np.testing.assert_allclose(ladder_traj["strip"], ref_traj["strip"], rtol=1e-6, atol=1e-9)
