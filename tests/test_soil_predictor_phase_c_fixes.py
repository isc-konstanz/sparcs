# -*- coding: utf-8 -*-
"""sparcs.tests.test_soil_predictor_phase_c_fixes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Regression tests for the defects an adversarial review pass surfaced after the
soil-predictor cluster first landed:

- ``_resolve_window_start`` (and therefore ``_build_flow_schedule``) rolled a
  window forward with a fixed ``Timedelta(days=1)``, landing an hour off across a
  DST transition. It now re-resolves the wall-clock on the next calendar day.
- ``_peak_tension`` returned ``-inf`` when the decision-probe subset matched no
  probe in the trajectory, making every rung vacuously feasible. It now returns
  ``+inf`` (infeasible / fail-safe).

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


class _IdentityModel:
    """psi_from_se(se) = se (monotone, positive) -- enough to exercise feasibility."""

    def psi_from_se(self, se):
        return se


def test_peak_tension_zero_matching_probes_is_infeasible_not_vacuous():
    """When decision_probes match no probe in the trajectory, _peak_tension must
    return +inf so the candidate reads as INFEASIBLE, not vacuously feasible."""
    trajectory = ([pd.Timestamp("2026-07-03 01:00", tz=_TZ)], {"root_20": [500.0]})

    peak = SoilPredictor._peak_tension(trajectory, _IdentityModel(), ["does_not_exist"])

    assert peak == float("inf")
    assert SoilPredictor._feasible(peak, threshold_hpa=300.0) is False  # pre-fix: -inf -> True


def test_peak_tension_matching_probe_still_worst_case():
    """A present decision probe still yields the finite worst-case tension."""
    trajectory = (
        [pd.Timestamp("2026-07-03 01:00", tz=_TZ), pd.Timestamp("2026-07-03 02:00", tz=_TZ)],
        {"root_20": [120.0, 480.0], "surface": [900.0, 900.0]},
    )
    peak = SoilPredictor._peak_tension(trajectory, _IdentityModel(), ["root_20"])
    assert peak == 480.0  # surface ignored (not a decision probe); worst-case over time
