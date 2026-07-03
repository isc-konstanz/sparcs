# -*- coding: utf-8 -*-
"""sparcs.tests.test_soil_predictor_tension_selector
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pure tests for issue 05's tension conversion, feasibility, and ladder selector:
``SoilPredictor._peak_tension``, ``SoilPredictor._feasible``, and
``SoilPredictor._select`` for both ``grid_mode="fill_order"`` and ``"full"``.

Importing ``soil_predictor`` pulls the full lories + soil (FiPy/Gmsh) stack via
``soil.py``; ``importorskip`` keeps this out of environments that lack it (the full
check runs on the box). The methods under test are ``@staticmethod``/``@classmethod``,
called directly off the class with no ``Component``/PDE instantiation needed, and the
selector is exercised over hand-built synthetic trajectories/tensions -- no PDE.
"""

import pytest

import numpy as np
import pandas as pd

soil_predictor = pytest.importorskip("sparcs.components.agriculture.simulation.soil_predictor")
SoilPredictor = soil_predictor.SoilPredictor

soil_models = pytest.importorskip("sparcs.components.agriculture.soil.models")
Genuchten = soil_models.Genuchten


def _td(minutes: int) -> pd.Timedelta:
    return pd.Timedelta(minutes=minutes)


def _trajectory(probe_se: dict) -> tuple:
    """Build a minimal ``(timestamps, {probe: [Se, ...]})`` trajectory; timestamps
    are irrelevant to the pure tension/feasibility math and are only there to match
    the interface shape."""
    length = len(next(iter(probe_se.values())))
    timestamps = [pd.Timestamp("2026-07-03 00:00", tz="Europe/Berlin") + pd.Timedelta(hours=h) for h in range(length)]
    return timestamps, probe_se


class _StubModel:
    """A callable Se -> tension stand-in, for selector tests that must not depend
    on a real Genuchten instance."""

    def __init__(self, tension_by_se: dict):
        self._tension_by_se = tension_by_se

    def psi_from_se(self, se):
        # se may be a scalar or an ndarray; look up each value individually so the
        # stub stays a simple, explicit mapping rather than a fitted curve.
        se_arr = np.atleast_1d(se)
        return np.array([self._tension_by_se[float(v)] for v in se_arr])


# --- psi_from_se sign / feasibility comparison -------------------------------


def test_psi_from_se_dry_se_yields_positive_hpa_magnitude():
    """A dry Se (0.2) must yield a POSITIVE hPa tension of the expected order --
    no sign flip anywhere in the conversion."""
    model = Genuchten(theta_r=0.05, theta_s=0.43, alpha=0.08, n=1.6, k_s=1.0e-4)

    tension = model.psi_from_se(0.2)

    assert tension > 0.0
    # Order-of-magnitude sanity: a dry Se this low on a typical loam curve sits
    # in the hundreds-of-hPa range, not near zero and not negative.
    assert 10.0 < tension < 100_000.0


def test_feasible_compares_directly_no_sign_flip():
    """Feasibility is a direct <= comparison against a positive threshold_hpa;
    a tension above threshold is infeasible, at/below is feasible."""
    model = Genuchten(theta_r=0.05, theta_s=0.43, alpha=0.08, n=1.6, k_s=1.0e-4)
    dry_tension = float(model.psi_from_se(0.2))
    threshold_hpa = dry_tension - 1.0  # threshold just below the dry tension

    assert not SoilPredictor._feasible(dry_tension, threshold_hpa)
    assert SoilPredictor._feasible(dry_tension, dry_tension)  # equality is feasible
    assert SoilPredictor._feasible(dry_tension, dry_tension + 1.0)


def test_peak_tension_is_worst_case_over_probes_and_time():
    """_peak_tension must take the max over BOTH the configured decision probes
    and the whole horizon, ignoring probes outside decision_probes."""
    tension_by_se = {0.9: 50.0, 0.6: 200.0, 0.3: 600.0, 0.1: 1200.0}
    model = _StubModel(tension_by_se)

    trajectory = _trajectory(
        {
            "root_20": [0.9, 0.6, 0.3],  # peak tension 600 at t=2
            "root_40": [0.9, 0.9, 0.9],  # peak tension 50
            "surface": [0.1, 0.1, 0.1],  # peak tension 1200 -- must be ignored (not a decision probe)
        }
    )

    peak = SoilPredictor._peak_tension(trajectory, model, decision_probes=["root_20", "root_40"])

    assert peak == 600.0


# --- selector: fill_order (chain, first-feasible) ----------------------------


def _fill_order_ladder() -> list:
    """A 3-rung fill_order-shaped ladder (single window) with known, monotone
    peak tensions decreasing as water increases."""
    return [(_td(0),), (_td(30),), (_td(60),)]


def _fill_order_trajectories(peak_tensions: dict) -> dict:
    """One synthetic trajectory per rung, engineered so _peak_tension (via the
    stub model identity mapping) returns exactly the given peak tension."""
    trajectories = {}
    for candidate, peak in peak_tensions.items():
        trajectories[candidate] = _trajectory({"root_20": [peak]})
    return trajectories


class _IdentityModel:
    """psi_from_se is the identity -- lets tests author peak tensions directly
    as the "Se" values fed into the trajectory."""

    def psi_from_se(self, se):
        return np.asarray(se, dtype=float)


def test_selector_fill_order_returns_first_feasible_rung():
    ladder = _fill_order_ladder()
    # Monotone decreasing peak tension as water increases: 500 (dry) -> 250 -> 100.
    peak_tensions = {ladder[0]: 500.0, ladder[1]: 250.0, ladder[2]: 100.0}
    trajectories = _fill_order_trajectories(peak_tensions)
    model = _IdentityModel()
    threshold_hpa = 300.0  # rung 0 infeasible (500>300), rung 1 feasible (250<=300)

    chosen, status = SoilPredictor._select(
        ladder, trajectories, model, ["root_20"], threshold_hpa, grid_mode="fill_order"
    )

    assert chosen == ladder[1]
    assert status == "ok"


def test_selector_fill_order_none_needed_when_zero_rung_feasible():
    ladder = _fill_order_ladder()
    peak_tensions = {ladder[0]: 100.0, ladder[1]: 80.0, ladder[2]: 50.0}
    trajectories = _fill_order_trajectories(peak_tensions)
    model = _IdentityModel()
    threshold_hpa = 300.0  # the all-0min rung is already feasible

    chosen, status = SoilPredictor._select(
        ladder, trajectories, model, ["root_20"], threshold_hpa, grid_mode="fill_order"
    )

    assert chosen == ladder[0]
    assert status == "none_needed"


def test_selector_fill_order_infeasible_returns_top_rung():
    ladder = _fill_order_ladder()
    peak_tensions = {ladder[0]: 900.0, ladder[1]: 700.0, ladder[2]: 500.0}
    trajectories = _fill_order_trajectories(peak_tensions)
    model = _IdentityModel()
    threshold_hpa = 300.0  # not even the top (max-water) rung is feasible

    chosen, status = SoilPredictor._select(
        ladder, trajectories, model, ["root_20"], threshold_hpa, grid_mode="fill_order"
    )

    assert chosen == ladder[-1]
    assert status == "infeasible"


# --- selector: full (partial order, least-total-minutes + tie-break) --------


def test_selector_full_picks_least_total_minutes_among_feasible():
    """Two windows; several candidates feasible, the selector must pick the one
    with the smallest total watering minutes, not the first-found or largest."""
    ladder = [
        (_td(0), _td(0)),
        (_td(30), _td(0)),  # 30 min total, feasible
        (_td(0), _td(30)),  # 30 min total, feasible (tie on minutes with the above)
        (_td(60), _td(0)),  # 60 min total, feasible
    ]
    # All non-zero rungs feasible; only the all-zero one is not (needs some water).
    peak_tensions = {
        ladder[0]: 900.0,  # infeasible
        ladder[1]: 200.0,  # feasible, active_count=1, earliest_start=0
        ladder[2]: 200.0,  # feasible, active_count=1, earliest_start=1
        ladder[3]: 100.0,  # feasible, active_count=1, earliest_start=0, more minutes
    }
    trajectories = {c: _trajectory({"root_20": [peak_tensions[c]]}) for c in ladder}
    model = _IdentityModel()
    threshold_hpa = 300.0

    chosen, status = SoilPredictor._select(ladder, trajectories, model, ["root_20"], threshold_hpa, grid_mode="full")

    # Tie between ladder[1] (30 min, earliest_start=0) and ladder[2] (30 min,
    # earliest_start=1): fewer active windows ties (both 1), so tie-break (b)
    # earliest active start picks ladder[1].
    assert chosen == ladder[1]
    assert status == "ok"


def test_selector_full_tie_break_fewer_active_windows():
    """Equal total minutes; the candidate with fewer active (non-zero) windows wins."""
    ladder = [
        (_td(0), _td(0)),
        (_td(30), _td(30)),  # 60 min total, 2 active windows
        (_td(60), _td(0)),  # 60 min total, 1 active window -- must win the tie
    ]
    peak_tensions = {ladder[0]: 900.0, ladder[1]: 150.0, ladder[2]: 150.0}
    trajectories = {c: _trajectory({"root_20": [peak_tensions[c]]}) for c in ladder}
    model = _IdentityModel()
    threshold_hpa = 300.0

    chosen, status = SoilPredictor._select(ladder, trajectories, model, ["root_20"], threshold_hpa, grid_mode="full")

    assert chosen == ladder[2]
    assert status == "ok"


def test_selector_full_tie_break_earliest_active_start():
    """Equal total minutes and equal active-window count; earliest active start wins."""
    ladder = [
        (_td(0), _td(0)),
        (_td(30), _td(0)),  # active window 0 (earlier)
        (_td(0), _td(30)),  # active window 1 (later) -- must lose the tie
    ]
    peak_tensions = {ladder[0]: 900.0, ladder[1]: 150.0, ladder[2]: 150.0}
    trajectories = {c: _trajectory({"root_20": [peak_tensions[c]]}) for c in ladder}
    model = _IdentityModel()
    threshold_hpa = 300.0

    chosen, status = SoilPredictor._select(ladder, trajectories, model, ["root_20"], threshold_hpa, grid_mode="full")

    assert chosen == ladder[1]
    assert status == "ok"


def test_selector_full_tie_break_margin_when_earlier_tiers_tie():
    """Construct two candidates with equal total minutes, equal active-window
    count, AND equal earliest-start index (via two independent single-window
    ladders is impossible in one call, so use a 3-window ladder where both
    contenders' first active window is at the same index) -- the larger tension
    margin must win."""
    ladder = [
        (_td(0), _td(0), _td(0)),
        (_td(30), _td(30), _td(0)),  # active windows {0,1}, earliest=0, margin=300-260=40
        (_td(30), _td(0), _td(30)),  # active windows {0,2}, earliest=0, margin=300-120=180 -- must win
    ]
    peak_tensions = {
        ladder[0]: 900.0,
        ladder[1]: 260.0,
        ladder[2]: 120.0,
    }
    trajectories = {c: _trajectory({"root_20": [peak_tensions[c]]}) for c in ladder}
    model = _IdentityModel()
    threshold_hpa = 300.0

    chosen, status = SoilPredictor._select(ladder, trajectories, model, ["root_20"], threshold_hpa, grid_mode="full")

    assert chosen == ladder[2]
    assert status == "ok"


def test_selector_full_none_needed_when_zero_candidate_chosen():
    ladder = [(_td(0), _td(0)), (_td(30), _td(0))]
    peak_tensions = {ladder[0]: 50.0, ladder[1]: 20.0}  # both feasible; zero is least water
    trajectories = {c: _trajectory({"root_20": [peak_tensions[c]]}) for c in ladder}
    model = _IdentityModel()
    threshold_hpa = 300.0

    chosen, status = SoilPredictor._select(ladder, trajectories, model, ["root_20"], threshold_hpa, grid_mode="full")

    assert chosen == ladder[0]
    assert status == "none_needed"


def test_selector_full_infeasible_returns_largest_total_minutes_candidate():
    ladder = [(_td(0), _td(0)), (_td(30), _td(0)), (_td(60), _td(30))]
    peak_tensions = {ladder[0]: 900.0, ladder[1]: 700.0, ladder[2]: 500.0}  # none feasible
    trajectories = {c: _trajectory({"root_20": [peak_tensions[c]]}) for c in ladder}
    model = _IdentityModel()
    threshold_hpa = 300.0

    chosen, status = SoilPredictor._select(ladder, trajectories, model, ["root_20"], threshold_hpa, grid_mode="full")

    assert chosen == ladder[2]  # 90 min total, the largest
    assert status == "infeasible"
