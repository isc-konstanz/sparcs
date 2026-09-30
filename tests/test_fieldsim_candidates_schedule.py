# -*- coding: utf-8 -*-
"""
tests.test_fieldsim_candidates_schedule
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The pure candidate-schedule building blocks: drip-flow derivation, the
window-schedule builder (including its DST-correct window-start resolver)
and the ``split_interval`` edge splitter.
"""

import datetime

import pytest

import pandas as pd
from sparcs.components.agriculture.simulation.core.candidates import (
    WateringWindow,
    build_flow_schedule,
    derive_flow_m3s,
    resolve_window_start,
    split_interval,
)

_TZ = "Europe/Berlin"


# --- Drip-flow derivation ---------------------------------------------------


def test_drip_flow_derivation_matches_hand_computation_kob_layout():
    """Real kob layout: 31 nozzles x 1.0 L/h over a 12.6 m drip line."""
    flow_lpm = 31 * 1.0 / 60.0
    expected_flow_m3s = flow_lpm / (60_000.0 * 12.6)

    flow_m3s = derive_flow_m3s(nozzle_count=31, nozzle_flow_lph=1.0, total_drip_line_length_m=12.6)

    assert flow_lpm == pytest.approx(31 / 60)
    assert flow_m3s == pytest.approx(expected_flow_m3s)
    assert flow_m3s == pytest.approx((31 / 60) / (60_000.0 * 12.6))


def test_drip_flow_derivation_no_live_sim_dependency():
    """Pure function: only takes the three layout numbers, nothing from a live sim."""
    flow_m3s = derive_flow_m3s(nozzle_count=10, nozzle_flow_lph=2.0, total_drip_line_length_m=5.0)
    flow_lpm = 10 * 2.0 / 60.0
    assert flow_m3s == pytest.approx(flow_lpm / (60_000.0 * 5.0))


# --- Window schedule builder -------------------------------------------------


def test_schedule_builder_on_off_edges():
    horizon_start = pd.Timestamp("2026-07-03 07:00", tz=_TZ)
    horizon_end = horizon_start + pd.Timedelta(hours=24)
    windows = [WateringWindow(start=datetime.time(8, 0))]
    durations = [pd.Timedelta(minutes=30)]

    schedule = build_flow_schedule(
        windows, durations, flow_m3s=1.0, horizon_start=horizon_start, horizon_end=horizon_end
    )

    assert len(schedule) == 1
    on_ts, off_ts = schedule[0]
    assert on_ts == pd.Timestamp("2026-07-03 08:00", tz=_TZ)
    assert off_ts == pd.Timestamp("2026-07-03 08:30", tz=_TZ)


def test_schedule_builder_clamps_to_horizon_end():
    """A window whose start+duration exceeds the horizon clamps to horizon_end."""
    horizon_start = pd.Timestamp("2026-07-03 07:00", tz=_TZ)
    horizon_end = horizon_start + pd.Timedelta(hours=24)
    windows = [WateringWindow(start=datetime.time(23, 0))]
    durations = [pd.Timedelta(hours=10)]  # would end at 09:00 next day, past horizon_end (07:00)

    schedule = build_flow_schedule(
        windows, durations, flow_m3s=1.0, horizon_start=horizon_start, horizon_end=horizon_end
    )

    assert len(schedule) == 1
    on_ts, off_ts = schedule[0]
    assert on_ts == pd.Timestamp("2026-07-03 23:00", tz=_TZ)
    assert off_ts == horizon_end


def test_schedule_builder_zero_duration_window_contributes_no_interval():
    horizon_start = pd.Timestamp("2026-07-03 07:00", tz=_TZ)
    horizon_end = horizon_start + pd.Timedelta(hours=24)
    windows = [WateringWindow(start=datetime.time(8, 0)), WateringWindow(start=datetime.time(18, 0))]
    durations = [pd.Timedelta(0), pd.Timedelta(minutes=30)]

    schedule = build_flow_schedule(
        windows, durations, flow_m3s=1.0, horizon_start=horizon_start, horizon_end=horizon_end
    )

    assert len(schedule) == 1
    on_ts, off_ts = schedule[0]
    assert on_ts == pd.Timestamp("2026-07-03 18:00", tz=_TZ)
    assert off_ts == pd.Timestamp("2026-07-03 18:30", tz=_TZ)


def test_schedule_builder_all_zero_durations_yields_empty_schedule():
    horizon_start = pd.Timestamp("2026-07-03 07:00", tz=_TZ)
    horizon_end = horizon_start + pd.Timedelta(hours=24)
    windows = [WateringWindow(start=datetime.time(8, 0)), WateringWindow(start=datetime.time(18, 0))]
    durations = [pd.Timedelta(0), pd.Timedelta(0)]

    schedule = build_flow_schedule(
        windows, durations, flow_m3s=1.0, horizon_start=horizon_start, horizon_end=horizon_end
    )

    assert schedule == []


# --- resolve_window_start: DST correctness -----------------------------------


def test_resolve_window_start_is_dst_correct_on_spring_forward():
    """A window rolled onto the next calendar day must keep its local clock time
    across the spring-forward night, not shift by the fixed 24h that a
    Timedelta(days=1) add would apply."""
    # 2026-03-29 is the German spring-forward day (02:00 -> 03:00, +01:00 -> +02:00).
    horizon_start = pd.Timestamp("2026-03-28 20:00", tz=_TZ)  # evening before, +01:00
    start = datetime.time(18, 0)  # 18:00 has already passed -> rolls to the next day

    on_ts = resolve_window_start(start, horizon_start)

    assert on_ts == pd.Timestamp("2026-03-29 18:00", tz=_TZ)
    assert on_ts.hour == 18  # the fixed-24h bug produced 19:00 here
    assert on_ts.utcoffset() == pd.Timedelta(hours=2)


def test_resolve_window_start_is_dst_correct_on_fall_back():
    """Symmetric fall-back case: the fixed-24h add landed an hour early."""
    # 2026-10-25 is the German fall-back day (03:00 -> 02:00, +02:00 -> +01:00).
    horizon_start = pd.Timestamp("2026-10-24 20:00", tz=_TZ)  # evening before, +02:00
    start = datetime.time(18, 0)

    on_ts = resolve_window_start(start, horizon_start)

    assert on_ts == pd.Timestamp("2026-10-25 18:00", tz=_TZ)
    assert on_ts.hour == 18
    assert on_ts.utcoffset() == pd.Timedelta(hours=1)


def test_resolve_window_start_same_day_unchanged():
    """The same-day (non-rollover) path is unaffected."""
    horizon_start = pd.Timestamp("2026-07-03 06:00", tz=_TZ)
    assert resolve_window_start(datetime.time(8, 0), horizon_start) == pd.Timestamp("2026-07-03 08:00", tz=_TZ)


def test_build_flow_schedule_dst_window_integrates_at_correct_local_time():
    """build_flow_schedule routes through the DST-correct resolver: an evening
    window that rolls across spring-forward starts at 18:00 local, not 19:00."""
    horizon_start = pd.Timestamp("2026-03-28 20:00", tz=_TZ)
    horizon_end = horizon_start + pd.Timedelta("24h")
    window = WateringWindow(start=datetime.time(18, 0))

    intervals = build_flow_schedule([window], [pd.Timedelta("30min")], 1.0e-5, horizon_start, horizon_end)

    assert len(intervals) == 1
    on_ts, off_ts = intervals[0]
    assert on_ts == pd.Timestamp("2026-03-29 18:00", tz=_TZ)
    assert off_ts == pd.Timestamp("2026-03-29 18:30", tz=_TZ)


# --- split_interval edge-splitting -------------------------------------------


def test_split_interval_on_edge_matches_forecast_boundary():
    """On-interval [08:00, 08:30] inside an hourly forecast interval [08:00, 09:00]."""
    ts_prev = pd.Timestamp("2026-07-03 08:00", tz=_TZ)
    ts_next = pd.Timestamp("2026-07-03 09:00", tz=_TZ)
    on_intervals = [(ts_prev, ts_prev + pd.Timedelta(minutes=30))]
    flow = 2.5e-6

    segments = split_interval(on_intervals, ts_prev, ts_next, flow)

    assert segments == [(1800.0, flow), (1800.0, 0.0)]
    assert sum(w for w, _ in segments) == pytest.approx((ts_next - ts_prev).total_seconds())
    assert sum(f * w for w, f in segments) == pytest.approx(flow * 1800.0)


def test_split_interval_off_edge_mid_interval():
    """An on-interval that starts before ts_prev and ends mid-interval."""
    ts_prev = pd.Timestamp("2026-07-03 08:00", tz=_TZ)
    ts_next = pd.Timestamp("2026-07-03 09:00", tz=_TZ)
    on_intervals = [(ts_prev - pd.Timedelta(minutes=10), ts_prev + pd.Timedelta(minutes=20))]
    flow = 3.0e-6

    segments = split_interval(on_intervals, ts_prev, ts_next, flow)

    assert segments == [(1200.0, flow), (2400.0, 0.0)]
    assert sum(f * w for w, f in segments) == pytest.approx(flow * 1200.0)


def test_split_interval_both_edges_mid_interval():
    """An on-interval fully nested inside the forecast interval: 3 sub-segments."""
    ts_prev = pd.Timestamp("2026-07-03 08:00", tz=_TZ)
    ts_next = pd.Timestamp("2026-07-03 09:00", tz=_TZ)
    on_ts = ts_prev + pd.Timedelta(minutes=15)
    off_ts = ts_prev + pd.Timedelta(minutes=45)
    flow = 1.0e-6

    segments = split_interval([(on_ts, off_ts)], ts_prev, ts_next, flow)

    assert segments == [(900.0, 0.0), (1800.0, flow), (900.0, 0.0)]
    assert sum(w for w, _ in segments) == pytest.approx(3600.0)
    assert sum(f * w for w, f in segments) == pytest.approx(flow * 1800.0)


def test_split_interval_empty_schedule_is_behavior_identity_guard():
    """All-0min schedule: a single segment at zero flow, covering the whole interval --
    byte-for-byte the zero-flow roll's one walk_window(flow_m3s=0.0) per interval."""
    ts_prev = pd.Timestamp("2026-07-03 08:00", tz=_TZ)
    ts_next = pd.Timestamp("2026-07-03 09:00", tz=_TZ)
    elapsed_s = (ts_next - ts_prev).total_seconds()

    assert split_interval([], ts_prev, ts_next, flow_m3s=5.0e-6) == [(elapsed_s, 0.0)]
