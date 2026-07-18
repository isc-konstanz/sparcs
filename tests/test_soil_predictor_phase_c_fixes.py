# -*- coding: utf-8 -*-
"""sparcs.tests.test_soil_predictor_phase_c_fixes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Regression tests for the defects an adversarial review pass surfaced after the
soil-predictor cluster first landed:

- ``_resolve_window_start`` (and therefore ``_build_flow_schedule``) rolled a
  window forward with a fixed ``Timedelta(days=1)``, landing an hour off across a
  DST transition. It now re-resolves the wall-clock on the next calendar day.
- The empty-decision-set fallback returned ``-inf``, making every rung
  vacuously feasible in the feasibility-era selector (the ``_peak_tension``
  helper that carried the original fix is gone). The live scoring path,
  ``_score_candidate``, returns ``+inf`` when the decision-probe subset
  matches no probe in the trajectory (fail-safe worst score).

Each test is red against the pre-fix behavior (a fixed-24h add / a ``-inf``
fallback).
"""

import datetime

import pytest

import pandas as pd

soil_predictor = pytest.importorskip("sparcs.components.agriculture.simulation.soil_predictor")
SoilPredictor = soil_predictor.SoilPredictor
WateringWindow = soil_predictor.WateringWindow

_TZ = "Europe/Berlin"


def test_resolve_window_start_is_dst_correct_on_spring_forward():
    """A window rolled onto the next calendar day must keep its local clock time
    across the spring-forward night, not shift by the fixed 24h that a
    Timedelta(days=1) add would apply."""
    # 2026-03-29 is the German spring-forward day (02:00 -> 03:00, +01:00 -> +02:00).
    horizon_start = pd.Timestamp("2026-03-28 20:00", tz=_TZ)  # evening before, +01:00
    start = datetime.time(18, 0)  # 18:00 has already passed -> rolls to the next day

    on_ts = SoilPredictor._resolve_window_start(start, horizon_start)

    # Correct: 18:00 local on the 29th, at the post-transition +02:00 offset.
    assert on_ts == pd.Timestamp("2026-03-29 18:00", tz=_TZ)
    assert on_ts.hour == 18  # the fixed-24h bug produced 19:00 here
    assert on_ts.utcoffset() == pd.Timedelta(hours=2)


def test_resolve_window_start_is_dst_correct_on_fall_back():
    """Symmetric fall-back case: the fixed-24h add landed an hour early."""
    # 2026-10-25 is the German fall-back day (03:00 -> 02:00, +02:00 -> +01:00).
    horizon_start = pd.Timestamp("2026-10-24 20:00", tz=_TZ)  # evening before, +02:00
    start = datetime.time(18, 0)

    on_ts = SoilPredictor._resolve_window_start(start, horizon_start)

    assert on_ts == pd.Timestamp("2026-10-25 18:00", tz=_TZ)
    assert on_ts.hour == 18
    assert on_ts.utcoffset() == pd.Timedelta(hours=1)


def test_resolve_window_start_same_day_unchanged():
    """The same-day (non-rollover) path is unaffected."""
    horizon_start = pd.Timestamp("2026-07-03 06:00", tz=_TZ)
    on_ts = SoilPredictor._resolve_window_start(datetime.time(8, 0), horizon_start)
    assert on_ts == pd.Timestamp("2026-07-03 08:00", tz=_TZ)


def test_build_flow_schedule_dst_window_integrates_at_correct_local_time():
    """_build_flow_schedule routes through the DST-correct resolver: an evening
    window that rolls across spring-forward starts at 18:00 local, not 19:00."""
    horizon_start = pd.Timestamp("2026-03-28 20:00", tz=_TZ)
    horizon_end = horizon_start + pd.Timedelta("24h")
    window = WateringWindow(start=datetime.time(18, 0))

    intervals = SoilPredictor._build_flow_schedule(
        [window], [pd.Timedelta("30min")], 1.0e-5, horizon_start, horizon_end
    )

    assert len(intervals) == 1
    on_ts, off_ts = intervals[0]
    assert on_ts == pd.Timestamp("2026-03-29 18:00", tz=_TZ)
    assert off_ts == pd.Timestamp("2026-03-29 18:30", tz=_TZ)


def test_score_candidate_empty_decision_set_scores_worst():
    """When decision_probes match no probe in the trajectory, _score_candidate
    returns +inf so the candidate can never be the argmin (fail safe). Trajectory
    values are already water tension (hPa); the score takes no model."""
    trajectory = ([pd.Timestamp("2026-07-03 01:00", tz=_TZ)], {"root_20": [500.0]})

    score = SoilPredictor._score_candidate(trajectory, ["does_not_exist"], threshold_hpa=300.0)

    assert score == float("inf")


def test_score_candidate_rms_distance_over_present_probe():
    """A present decision probe yields the finite RMS-to-setpoint distance over the
    horizon; probes outside decision_probes are ignored."""
    trajectory = (
        [pd.Timestamp("2026-07-03 01:00", tz=_TZ), pd.Timestamp("2026-07-03 02:00", tz=_TZ)],
        {"root_20": [120.0, 480.0], "surface": [900.0, 900.0]},
    )
    score = SoilPredictor._score_candidate(trajectory, ["root_20"], threshold_hpa=300.0)
    # deviations -180, +180 vs 300; surface ignored.
    assert score == pytest.approx(180.0)
