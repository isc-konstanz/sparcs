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

from sparcs.components.agriculture.simulation.soil_predictor import (  # noqa: E402
    SoilPredictor,
    WateringWindow,
)

# Same extreme-flow scale as the ladder-rollout test: guarantees visible effect.
_EXTREME_FLOW = 2000.0e-3 / 3600.0 * 0.5


def test_zero_only_middle_window_still_advances_the_prefix(pde_core_factory, strip_probe_factory, bare_pde_predictor):
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

    core_ladder = pde_core_factory("ladder")
    ic_rel_sat = core_ladder.snapshot()
    predictor_ladder = bare_pde_predictor(core_ladder, [strip_probe_factory(core_ladder)], _EXTREME_FLOW)
    predictor_ladder._windows = windows
    predictor_ladder._window_durations = window_durations

    ladder_results = predictor_ladder._rollout_ladder(
        ic_rel_sat, ladder, et_data, seg_et, _EXTREME_FLOW, horizon_start, horizon_end
    )
    ladder_timestamps, ladder_traj = ladder_results[candidate]

    # Pre-fix failure mode 1: the degenerate window's segment timestamps vanish
    # from every later candidate's trajectory.
    assert list(ladder_timestamps) == list(idx)

    core_independent = pde_core_factory("independent")
    predictor_independent = bare_pde_predictor(core_independent, [strip_probe_factory(core_independent)], _EXTREME_FLOW)
    predictor_independent._windows = windows
    predictor_independent._window_durations = window_durations

    ref_timestamps, ref_traj = predictor_independent._rollout_independent(
        ic_rel_sat, candidate, et_data, seg_et, _EXTREME_FLOW, horizon_start, horizon_end
    )

    # Pre-fix failure mode 2: the skipped segment's drainage never happens, so
    # the shared-prefix roll diverges from the independent reference.
    assert list(ref_timestamps) == list(idx)
    np.testing.assert_allclose(ladder_traj["strip"], ref_traj["strip"], rtol=1e-6, atol=1e-9)
