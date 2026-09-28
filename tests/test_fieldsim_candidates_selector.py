# -*- coding: utf-8 -*-
"""
tests.test_fieldsim_candidates_selector
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Candidate scoring and selection: ``score_candidate`` (RMS-to-setpoint),
``select_candidate`` (argmin, least-water tie-break) for both grid modes,
and the planner's Se -> signed water tension conversion.
"""

import types

import pytest
from conftest import td

import numpy as np
import pandas as pd
from sparcs.components.agriculture.fieldsim.core.candidates import score_candidate, select_candidate
from sparcs.components.agriculture.fieldsim.core.planner import IrrigationPlanner
from sparcs.components.agriculture.soil.models import Genuchten

_TZ = "Europe/Berlin"


def _trajectory(probe_values: dict) -> tuple:
    """A minimal ``(timestamps, {probe: [...]})`` trajectory; timestamps are
    irrelevant to the pure scoring math and only match the interface shape."""
    length = len(next(iter(probe_values.values())))
    timestamps = [pd.Timestamp("2026-07-03 00:00", tz=_TZ) + pd.Timedelta(hours=h) for h in range(length)]
    return timestamps, probe_values


# --- psi_from_se sign --------------------------------------------------------


def test_psi_from_se_dry_se_yields_negative_matric_potential():
    """A dry Se (0.2) must yield a NEGATIVE signed matric potential -- psi_from_se
    returns the physical psi (drier -> more negative), not its magnitude."""
    model = Genuchten(theta_r=0.05, theta_s=0.43, alpha=0.08, n=1.6, k_s=1.0e-4)

    psi = model.psi_from_se(0.2)

    assert psi < 0.0
    assert 10.0 < abs(psi) < 100_000.0


# --- score_candidate: RMS distance to the setpoint ---------------------------


def test_score_candidate_is_rms_distance_to_setpoint():
    """The score is the RMS of (tension - threshold) over the horizon; tension
    ABOVE and BELOW the setpoint both add to it (setpoint, not ceiling)."""
    trajectory = _trajectory({"root_20": [100.0, 300.0, 500.0]})  # deviations -200, 0, +200

    score = score_candidate(trajectory, ["root_20"], threshold_hpa=300.0)

    assert score == pytest.approx(float(np.sqrt((200.0**2 + 0.0 + 200.0**2) / 3.0)))


def test_score_candidate_uses_suction_magnitude_of_signed_tension():
    """Production trajectories are signed matric potential (negative hPa); the
    score compares their suction MAGNITUDE to the positive threshold setpoint."""
    trajectory = _trajectory({"root_20": [-100.0, -300.0, -500.0]})  # |.| = 100, 300, 500

    score = score_candidate(trajectory, ["root_20"], threshold_hpa=300.0)

    assert score == pytest.approx(float(np.sqrt((200.0**2 + 0.0 + 200.0**2) / 3.0)))


def test_score_candidate_pools_decision_probes_and_ignores_others():
    """RMS pools every timestep of every decision probe; probes outside
    decision_probes never contribute."""
    trajectory = _trajectory(
        {
            "root_20": [200.0, 400.0],  # deviations -100, +100 vs 300
            "root_40": [300.0, 300.0],  # deviations 0, 0
            "surface": [5000.0, 5000.0],  # ignored (not a decision probe)
        }
    )

    score = score_candidate(trajectory, ["root_20", "root_40"], threshold_hpa=300.0)

    assert score == pytest.approx(float(np.sqrt((100.0**2 + 100.0**2 + 0.0 + 0.0) / 4.0)))


def test_score_candidate_empty_decision_set_is_worst():
    """A decision set matching no present probe scores +inf (fail safe), so it can
    never be the argmin."""
    assert score_candidate(_trajectory({"root_20": [300.0]}), ["not_a_probe"], threshold_hpa=300.0) == float("inf")


def test_score_candidate_rms_distance_over_present_probe():
    """A present decision probe yields the finite RMS-to-setpoint distance; probes
    outside decision_probes are ignored."""
    trajectory = _trajectory({"root_20": [120.0, 480.0], "surface": [900.0, 900.0]})

    assert score_candidate(trajectory, ["root_20"], threshold_hpa=300.0) == pytest.approx(180.0)


# --- select_candidate: argmin of the score, least-water tie-break -------------


def _single_value_trajectories(tensions: dict) -> dict:
    """One synthetic (already-tension) trajectory per rung, a single tension value
    each, so its RMS-to-setpoint score is exactly ``|value - threshold|``."""
    return {candidate: _trajectory({"root_20": [tension]}) for candidate, tension in tensions.items()}


@pytest.mark.parametrize("grid_mode", ["fill_order", "full"])
def test_select_picks_candidate_closest_to_setpoint(grid_mode):
    """Both grid modes reduce to the same rule: the argmin of the RMS-to-setpoint
    score, i.e. the rung whose tension sits closest to threshold_hpa."""
    ladder = [(td(0),), (td(30),), (td(60),)]
    # scores |v - 300| = 200, 20, 200 -> the middle rung is closest.
    trajectories = _single_value_trajectories({ladder[0]: 500.0, ladder[1]: 320.0, ladder[2]: 100.0})

    assert select_candidate(ladder, trajectories, ["root_20"], 300.0, grid_mode=grid_mode) == ladder[1]


def test_select_returns_zero_rung_when_it_tracks_setpoint_best():
    """When doing nothing already sits closest to the setpoint, the all-0min rung
    is chosen -- watering would only overshoot wet."""
    ladder = [(td(0),), (td(30),), (td(60),)]
    trajectories = _single_value_trajectories({ladder[0]: 300.0, ladder[1]: 220.0, ladder[2]: 120.0})

    assert select_candidate(ladder, trajectories, ["root_20"], 300.0, grid_mode="fill_order") == ladder[0]


def test_select_tie_breaks_on_least_total_water():
    """Two rungs equidistant from the setpoint (equal score) -> the one with less
    total watering wins, for a deterministic pick."""
    ladder = [(td(0), td(0)), (td(30), td(0)), (td(30), td(30))]
    # 250 and 350 both score |. - 300| = 50; the all-zero rung is far (score 200).
    trajectories = _single_value_trajectories({ladder[0]: 100.0, ladder[1]: 250.0, ladder[2]: 350.0})

    assert select_candidate(ladder, trajectories, ["root_20"], 300.0, grid_mode="full") == ladder[1]


def test_select_empty_ladder_raises():
    with pytest.raises(ValueError, match="non-empty ladder"):
        select_candidate([], {}, ["root_20"], 300.0, grid_mode="fill_order")


def test_select_unknown_grid_mode_raises():
    ladder = [(td(0),)]
    trajectories = _single_value_trajectories({ladder[0]: 300.0})
    with pytest.raises(ValueError, match="Unknown grid_mode"):
        select_candidate(ladder, trajectories, ["root_20"], 300.0, grid_mode="bogus")


# --- planner Se -> signed water tension --------------------------------------


def test_to_tension_converts_se_to_signed_matric_potential():
    """The roll->publish boundary converter maps per-probe Se trajectories to
    signed matric potential (negative hPa) via the engine model's psi_from_se."""
    planner = object.__new__(IrrigationPlanner)
    model = Genuchten(theta_r=0.05, theta_s=0.43, alpha=0.08, n=1.6, k_s=1.0e-4)
    planner.engine = types.SimpleNamespace(model=model)

    se_traj = [0.8, 0.5, 0.3]  # drying over the horizon
    tension = planner._to_tension({"root_20": se_traj})

    np.testing.assert_allclose(tension["root_20"], model.psi_from_se(np.asarray(se_traj, dtype=float)))
    assert all(v < -1.0 for v in tension["root_20"])  # signed hPa, not Se in [0, 1]
    assert tension["root_20"][-1] < tension["root_20"][0]  # drier -> more negative
