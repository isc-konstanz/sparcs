# -*- coding: utf-8 -*-
"""sparcs.tests.test_soil_predictor_tension_selector
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pure tests for the tension conversion, candidate scoring, and ladder selector:
``SoilPredictor._score_candidate`` (RMS-to-setpoint) and ``SoilPredictor._select``
(argmin, least-water tie-break) for both ``grid_mode="fill_order"`` and ``"full"``.

Importing ``soil_predictor`` pulls the full lories + soil (FiPy/Gmsh) stack via
``soil.py``; ``importorskip`` keeps this out of environments that lack it (the full
check runs on the box). The methods under test are ``@staticmethod``/``@classmethod``,
called directly off the class with no ``Component``/PDE instantiation needed, and the
selector is exercised over hand-built synthetic trajectories/tensions -- no PDE.
"""

import types

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


# --- psi_from_se sign --------------------------------------------------------


def test_psi_from_se_dry_se_yields_negative_matric_potential():
    """A dry Se (0.2) must yield a NEGATIVE signed matric potential -- psi_from_se
    returns the physical ψ (drier -> more negative), not its magnitude."""
    model = Genuchten(theta_r=0.05, theta_s=0.43, alpha=0.08, n=1.6, k_s=1.0e-4)

    psi = model.psi_from_se(0.2)

    assert psi < 0.0
    # Order-of-magnitude sanity: a dry Se this low on a typical loam curve sits in
    # the hundreds-of-hPa suction range, so |ψ| is well out of the [0, 1] Se range.
    assert 10.0 < abs(psi) < 100_000.0


# --- _score_candidate: RMS distance to the setpoint --------------------------


def test_score_candidate_is_rms_distance_to_setpoint():
    """The score is the RMS of (tension - threshold) over the horizon; tension
    ABOVE and BELOW the setpoint both add to it (setpoint, not ceiling)."""
    trajectory = _trajectory({"root_20": [100.0, 300.0, 500.0]})  # deviations -200, 0, +200

    score = SoilPredictor._score_candidate(trajectory, ["root_20"], threshold_hpa=300.0)

    expected = float(np.sqrt((200.0**2 + 0.0 + 200.0**2) / 3.0))
    assert score == pytest.approx(expected)


def test_score_candidate_uses_suction_magnitude_of_signed_tension():
    """Production trajectories are signed matric potential (negative hPa); the
    score compares their suction MAGNITUDE to the positive threshold setpoint, so
    signed input scores the same as its magnitude."""
    trajectory = _trajectory({"root_20": [-100.0, -300.0, -500.0]})  # |.| = 100, 300, 500

    score = SoilPredictor._score_candidate(trajectory, ["root_20"], threshold_hpa=300.0)

    expected = float(np.sqrt((200.0**2 + 0.0 + 200.0**2) / 3.0))  # deviations -200, 0, +200
    assert score == pytest.approx(expected)


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

    score = SoilPredictor._score_candidate(trajectory, ["root_20", "root_40"], threshold_hpa=300.0)

    expected = float(np.sqrt((100.0**2 + 100.0**2 + 0.0 + 0.0) / 4.0))
    assert score == pytest.approx(expected)


def test_score_candidate_empty_decision_set_is_worst():
    """A decision set matching no present probe scores +inf (fail safe), so it can
    never be the argmin."""
    trajectory = _trajectory({"root_20": [300.0]})

    score = SoilPredictor._score_candidate(trajectory, ["not_a_probe"], threshold_hpa=300.0)

    assert score == float("inf")


# --- selector: argmin of the score, least-water tie-break --------------------


def _single_value_trajectories(tensions: dict) -> dict:
    """One synthetic (already-tension) trajectory per rung, a single tension value
    each, so its RMS-to-setpoint score is exactly ``|value - threshold|``."""
    return {candidate: _trajectory({"root_20": [tension]}) for candidate, tension in tensions.items()}


@pytest.mark.parametrize("grid_mode", ["fill_order", "full"])
def test_select_picks_candidate_closest_to_setpoint(grid_mode):
    """Both grid modes reduce to the same rule: the argmin of the RMS-to-setpoint
    score, i.e. the rung whose tension sits closest to threshold_hpa."""
    ladder = [(_td(0),), (_td(30),), (_td(60),)]
    # scores |v - 300| = 200, 20, 200 -> the middle rung is closest.
    trajectories = _single_value_trajectories({ladder[0]: 500.0, ladder[1]: 320.0, ladder[2]: 100.0})

    chosen = SoilPredictor._select(ladder, trajectories, ["root_20"], 300.0, grid_mode=grid_mode)

    assert chosen == ladder[1]


def test_select_returns_zero_rung_when_it_tracks_setpoint_best():
    """When doing nothing already sits closest to the setpoint, the all-0min rung
    is chosen -- watering would only overshoot wet."""
    ladder = [(_td(0),), (_td(30),), (_td(60),)]
    trajectories = _single_value_trajectories({ladder[0]: 300.0, ladder[1]: 220.0, ladder[2]: 120.0})

    chosen = SoilPredictor._select(ladder, trajectories, ["root_20"], 300.0, grid_mode="fill_order")

    assert chosen == ladder[0]


def test_select_tie_breaks_on_least_total_water():
    """Two rungs equidistant from the setpoint (equal score) -> the one with less
    total watering wins, for a deterministic pick."""
    ladder = [(_td(0), _td(0)), (_td(30), _td(0)), (_td(30), _td(30))]
    # 250 and 350 both score |. - 300| = 50; the all-zero rung is far (score 200).
    trajectories = _single_value_trajectories({ladder[0]: 100.0, ladder[1]: 250.0, ladder[2]: 350.0})

    chosen = SoilPredictor._select(ladder, trajectories, ["root_20"], 300.0, grid_mode="full")

    # ladder[1] (30 min) and ladder[2] (60 min) tie on score; least water wins.
    assert chosen == ladder[1]


def test_select_empty_ladder_raises():
    with pytest.raises(ValueError, match="non-empty ladder"):
        SoilPredictor._select([], {}, ["root_20"], 300.0, grid_mode="fill_order")


def test_select_unknown_grid_mode_raises():
    ladder = [(_td(0),)]
    trajectories = _single_value_trajectories({ladder[0]: 300.0})
    with pytest.raises(ValueError, match="Unknown grid_mode"):
        SoilPredictor._select(ladder, trajectories, ["root_20"], 300.0, grid_mode="bogus")


# --- published predict_<probe> channel is water tension (hPa) ----------------


class _RecordingChannel:
    """Records ``.set(timestamp, value)`` calls; the ``Channel`` surface
    ``_publish_results`` touches."""

    def __init__(self, channel_id: str):
        self.id = channel_id
        self.calls: list[tuple] = []

    def set(self, timestamp, value) -> None:
        self.calls.append((timestamp, value))


class _RecordingData:
    """Minimal ``self.data`` stand-in: auto-creates a recording channel per key."""

    def __init__(self):
        self._channels: dict = {}

    def __getitem__(self, key: str) -> _RecordingChannel:
        return self._channels.setdefault(key, _RecordingChannel(key))


def test_trajectories_to_tension_converts_se_to_signed_matric_potential():
    """The roll->publish boundary converter maps per-probe Se trajectories to
    signed matric potential (negative hPa) via the model's psi_from_se: drier soil
    (lower Se) -> more negative, out of the [0, 1] saturation range."""
    predictor = object.__new__(SoilPredictor)
    model = Genuchten(theta_r=0.05, theta_s=0.43, alpha=0.08, n=1.6, k_s=1.0e-4)
    predictor._pde = types.SimpleNamespace(soil_model=model)

    se_traj = [0.8, 0.5, 0.3]  # drying over the horizon
    tension = predictor._trajectories_to_tension({"root_20": se_traj})

    expected = model.psi_from_se(np.asarray(se_traj, dtype=float))
    np.testing.assert_allclose(tension["root_20"], expected)
    assert all(v < -1.0 for v in tension["root_20"])  # signed hPa, not Se in [0, 1]
    assert tension["root_20"][-1] < tension["root_20"][0]  # drier -> more negative


def test_publish_results_publishes_predict_channel_values_as_is(monkeypatch):
    """_publish_results publishes the predict_<probe> trajectory verbatim -- the
    values are already signed matric potential (negative hPa) from the retention
    model (see _trajectories_to_tension), so this method does not re-convert."""
    predictor = object.__new__(SoilPredictor)
    predictor._name = "test_predict_publish"
    predictor._channel_keys = {"root_20": "predict_root_20"}

    fake = _RecordingData()
    # Component.data is a read-only class property, so patch the CLASS for the
    # bare object.__new__ instance (monkeypatch auto-restores).
    monkeypatch.setattr(SoilPredictor, "data", property(lambda self: fake))

    timestamps = [pd.Timestamp("2026-07-03 00:00", tz="Europe/Berlin") + pd.Timedelta(hours=h) for h in range(3)]
    tension_traj = [-10.0, -40.0, -90.0]  # signed matric potential, more negative as it dries

    predictor._publish_results(
        trajectories={"root_20": tension_traj},
        probes=[types.SimpleNamespace(channel_id="root_20")],
        timestamps=timestamps,
        snapshots={},
        diagnostics={},
        forecast_creation=timestamps[0],
    )

    calls = fake["predict_root_20"].calls
    assert len(calls) == 1
    published = calls[0][1]
    np.testing.assert_allclose(published.to_numpy(), tension_traj)
