# -*- coding: utf-8 -*-
"""sparcs.tests.test_soil_predictor_flow_schedule
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Unit tests for the three pure building blocks issue 03 adds to ``SoilPredictor``:
drip-flow derivation, the window-schedule builder, and the ``_split_interval``
edge-splitting helper that keeps the all-``0min`` roll behavior-identical.

Importing ``soil_predictor`` pulls the full lories + soil (FiPy/Gmsh) stack via
``soil.py``; ``importorskip`` keeps this out of environments that lack it (the
full check runs on the box). All three targets are ``@staticmethod``s, called
directly off the class with no ``Component``/PDE instantiation needed.
"""

import datetime

import pytest

import pandas as pd

soil_predictor = pytest.importorskip("sparcs.components.agriculture.simulation.soil_predictor")
SoilPredictor = soil_predictor.SoilPredictor
WateringWindow = soil_predictor.WateringWindow


# --- Drip-flow derivation ---------------------------------------------------


def test_drip_flow_derivation_matches_hand_computation_kob_layout():
    """Real kob layout: 31 nozzles x 1.0 L/h over a 12.6 m drip line."""
    flow_lpm = 31 * 1.0 / 60.0
    expected_flow_m3s = flow_lpm / (60_000.0 * 12.6)

    flow_m3s = SoilPredictor._derive_flow_m3s(
        nozzle_count=31,
        nozzle_flow_lph=1.0,
        total_drip_line_length_m=12.6,
    )

    assert flow_lpm == pytest.approx(31 / 60)
    assert flow_m3s == pytest.approx(expected_flow_m3s)
    assert flow_m3s == pytest.approx((31 / 60) / (60_000.0 * 12.6))


def test_drip_flow_derivation_no_live_sim_dependency():
    """Pure function: only takes the three layout numbers, nothing from a live sim."""
    flow_m3s = SoilPredictor._derive_flow_m3s(
        nozzle_count=10,
        nozzle_flow_lph=2.0,
        total_drip_line_length_m=5.0,
    )
    flow_lpm = 10 * 2.0 / 60.0
    assert flow_m3s == pytest.approx(flow_lpm / (60_000.0 * 5.0))


# --- Window schedule builder -------------------------------------------------


def test_schedule_builder_on_off_edges():
    horizon_start = pd.Timestamp("2026-07-03 07:00", tz="Europe/Berlin")
    horizon_end = horizon_start + pd.Timedelta(hours=24)
    windows = [WateringWindow(start=datetime.time(8, 0))]
    durations = [pd.Timedelta(minutes=30)]

    schedule = SoilPredictor._build_flow_schedule(
        windows, durations, flow_m3s=1.0, horizon_start=horizon_start, horizon_end=horizon_end
    )

    assert len(schedule) == 1
    on_ts, off_ts = schedule[0]
    assert on_ts == pd.Timestamp("2026-07-03 08:00", tz="Europe/Berlin")
    assert off_ts == pd.Timestamp("2026-07-03 08:30", tz="Europe/Berlin")


def test_schedule_builder_clamps_to_horizon_end():
    """A window whose start+duration exceeds the horizon clamps to horizon_end."""
    horizon_start = pd.Timestamp("2026-07-03 07:00", tz="Europe/Berlin")
    horizon_end = horizon_start + pd.Timedelta(hours=24)
    windows = [WateringWindow(start=datetime.time(23, 0))]
    durations = [pd.Timedelta(hours=10)]  # would end at 09:00 next day, past horizon_end (07:00)

    schedule = SoilPredictor._build_flow_schedule(
        windows, durations, flow_m3s=1.0, horizon_start=horizon_start, horizon_end=horizon_end
    )

    assert len(schedule) == 1
    on_ts, off_ts = schedule[0]
    assert on_ts == pd.Timestamp("2026-07-03 23:00", tz="Europe/Berlin")
    assert off_ts == horizon_end


def test_schedule_builder_zero_duration_window_contributes_no_interval():
    horizon_start = pd.Timestamp("2026-07-03 07:00", tz="Europe/Berlin")
    horizon_end = horizon_start + pd.Timedelta(hours=24)
    windows = [WateringWindow(start=datetime.time(8, 0)), WateringWindow(start=datetime.time(18, 0))]
    durations = [pd.Timedelta(0), pd.Timedelta(minutes=30)]

    schedule = SoilPredictor._build_flow_schedule(
        windows, durations, flow_m3s=1.0, horizon_start=horizon_start, horizon_end=horizon_end
    )

    assert len(schedule) == 1
    on_ts, off_ts = schedule[0]
    assert on_ts == pd.Timestamp("2026-07-03 18:00", tz="Europe/Berlin")
    assert off_ts == pd.Timestamp("2026-07-03 18:30", tz="Europe/Berlin")


def test_schedule_builder_all_zero_durations_yields_empty_schedule():
    horizon_start = pd.Timestamp("2026-07-03 07:00", tz="Europe/Berlin")
    horizon_end = horizon_start + pd.Timedelta(hours=24)
    windows = [WateringWindow(start=datetime.time(8, 0)), WateringWindow(start=datetime.time(18, 0))]
    durations = [pd.Timedelta(0), pd.Timedelta(0)]

    schedule = SoilPredictor._build_flow_schedule(
        windows, durations, flow_m3s=1.0, horizon_start=horizon_start, horizon_end=horizon_end
    )

    assert schedule == []


# --- _split_interval edge-splitting ------------------------------------------


def test_split_interval_on_edge_matches_forecast_boundary():
    """On-interval [08:00, 08:30] inside an hourly forecast interval [08:00, 09:00]."""
    ts_prev = pd.Timestamp("2026-07-03 08:00", tz="Europe/Berlin")
    ts_next = pd.Timestamp("2026-07-03 09:00", tz="Europe/Berlin")
    on_intervals = [(ts_prev, ts_prev + pd.Timedelta(minutes=30))]
    flow = 2.5e-6

    segments = SoilPredictor._split_interval(on_intervals, ts_prev, ts_next, flow)

    assert segments == [(1800.0, flow), (1800.0, 0.0)]
    total_seconds = sum(w for w, _ in segments)
    assert total_seconds == pytest.approx((ts_next - ts_prev).total_seconds())
    integrated_water = sum(f * w for w, f in segments)
    assert integrated_water == pytest.approx(flow * 1800.0)


def test_split_interval_off_edge_mid_interval():
    """An on-interval that starts before ts_prev and ends mid-interval."""
    ts_prev = pd.Timestamp("2026-07-03 08:00", tz="Europe/Berlin")
    ts_next = pd.Timestamp("2026-07-03 09:00", tz="Europe/Berlin")
    on_intervals = [(ts_prev - pd.Timedelta(minutes=10), ts_prev + pd.Timedelta(minutes=20))]
    flow = 3.0e-6

    segments = SoilPredictor._split_interval(on_intervals, ts_prev, ts_next, flow)

    assert segments == [(1200.0, flow), (2400.0, 0.0)]
    integrated_water = sum(f * w for w, f in segments)
    assert integrated_water == pytest.approx(flow * 1200.0)


def test_split_interval_both_edges_mid_interval():
    """An on-interval fully nested inside the forecast interval: 3 sub-segments."""
    ts_prev = pd.Timestamp("2026-07-03 08:00", tz="Europe/Berlin")
    ts_next = pd.Timestamp("2026-07-03 09:00", tz="Europe/Berlin")
    on_ts = ts_prev + pd.Timedelta(minutes=15)
    off_ts = ts_prev + pd.Timedelta(minutes=45)
    flow = 1.0e-6

    segments = SoilPredictor._split_interval([(on_ts, off_ts)], ts_prev, ts_next, flow)

    assert segments == [(900.0, 0.0), (1800.0, flow), (900.0, 0.0)]
    total_seconds = sum(w for w, _ in segments)
    assert total_seconds == pytest.approx(3600.0)
    integrated_water = sum(f * w for w, f in segments)
    assert integrated_water == pytest.approx(flow * 1800.0)


def test_split_interval_empty_schedule_is_behavior_identity_guard():
    """All-0min schedule: a single segment at zero flow, covering the whole interval.

    This is the exact call the walk loop makes when flow_schedule is None/empty --
    one walk_window(flow_m3s=0.0, window_s=elapsed_s) per forecast interval, byte-
    for-byte today's zero-flow roll (User Story 11).
    """
    ts_prev = pd.Timestamp("2026-07-03 08:00", tz="Europe/Berlin")
    ts_next = pd.Timestamp("2026-07-03 09:00", tz="Europe/Berlin")
    elapsed_s = (ts_next - ts_prev).total_seconds()

    segments = SoilPredictor._split_interval([], ts_prev, ts_next, flow_m3s=5.0e-6)

    assert segments == [(elapsed_s, 0.0)]
