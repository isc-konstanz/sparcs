# -*- coding: utf-8 -*-
"""
tests.test_fieldsim_planner_frames
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The planner's frame builders for ``agri_field_forecast``, ``agri_soil_forecast`` and
``agri_field_forecast_irrigation``. Pure: no engine, no mesh.
"""

import datetime

import pytest
from conftest import td
from sqlalchemy.types import TIMESTAMP

import numpy as np
import pandas as pd
import pytz
from lories.connectors.sql.columns import DatetimeColumn
from lories.core.errors import ResourceError
from sparcs.components.agriculture.simulation.core.candidates import WateringWindow
from sparcs.components.agriculture.simulation.core.planner import (
    build_detail_frame,
    build_header_frame,
    build_irrigation_frame,
)

_TZ = "Europe/Berlin"
_RUN_TS = pd.Timestamp("2026-07-03 01:00", tz=_TZ)
_WEATHER_TS = pd.Timestamp("2026-07-03 00:00", tz="UTC")
_PROBE_IDS = ["root_20", "root_40"]


def _windows(*starts: datetime.time) -> list:
    return [WateringWindow(start=start) for start in starts]


def _two_windows() -> list:
    return _windows(datetime.time(8, 0), datetime.time(18, 0))


def _synthetic_ladder():
    """A 2-window fill-order-style ladder: 3 candidates."""
    return [(td(0), td(0)), (td(30), td(0)), (td(60), td(0))]


def _header(ladder, chosen, run_ts=_RUN_TS, weather_ts=_WEATHER_TS, max_windows=4, windows=None):
    return build_header_frame(ladder, chosen, run_ts, weather_ts, windows or _two_windows(), max_windows)


# --- build_header_frame ------------------------------------------------------


def test_header_frame_has_expected_columns():
    frame = _header(_synthetic_ladder(), _synthetic_ladder()[1])

    assert set(frame.columns) == {
        "forecast_id",
        "w0_min",
        "w1_min",
        "w2_min",
        "w3_min",
        "w0_start",
        "w1_start",
        "w2_start",
        "w3_start",
        "is_recommended",
        "total_min",
        "weather_creation",
    }


def test_header_frame_row_count_is_one_per_candidate():
    ladder = _synthetic_ladder()
    assert len(_header(ladder, ladder[0])) == len(ladder)


def test_header_frame_indexed_at_run_timestamp_for_every_row():
    """Every candidate's header row is indexed at the single RUN timestamp --
    forecast_id, not a second timestamp channel, is the header's PK partner."""
    ladder = _synthetic_ladder()

    frame = _header(ladder, ladder[0])

    assert (frame.index == _RUN_TS).all()


def test_header_frame_unconfigured_windows_are_null_not_sentinel():
    """Ladder has 2 active windows but max_windows=4: w2/w3 columns are NULL, no -1.0 sentinel."""
    ladder = _synthetic_ladder()

    frame = _header(ladder, ladder[0])

    assert frame["w2_min"].isna().all()
    assert frame["w3_min"].isna().all()
    assert frame["w2_start"].isna().all()
    assert frame["w3_start"].isna().all()
    assert not frame["w0_min"].isna().any()
    assert not frame["w1_min"].isna().any()
    assert not (frame["w0_min"] == -1.0).any()


def test_header_frame_carries_window_start_clock_times():
    """w{i}_start persists the configured window's clock time."""
    ladder = _synthetic_ladder()

    frame = _header(ladder, ladder[0])

    assert (frame["w0_start"] == "08:00").all()
    assert (frame["w1_start"] == "18:00").all()


def test_header_frame_is_recommended_true_exactly_once_per_run():
    ladder = _synthetic_ladder()

    frame = _header(ladder, ladder[1])

    assert frame["is_recommended"].sum() == 1
    recommended_row = frame[frame["is_recommended"]]
    assert recommended_row["w0_min"].iloc[0] == 30.0
    assert recommended_row["w1_min"].iloc[0] == 0.0


def test_header_frame_weather_creation_is_constant_across_rows():
    """weather_creation carries the weather forecast issue time this run used --
    distinct from the run timestamp (the index)."""
    ladder = _synthetic_ladder()
    weather_creation = pd.Timestamp("2026-07-02 23:00", tz="UTC")

    frame = _header(ladder, ladder[0], weather_ts=weather_creation)

    assert (frame["weather_creation"] == weather_creation).all()


def test_header_frame_forecast_id_is_deterministic_ladder_position():
    """forecast_id is the candidate's position in the ladder, stable across calls with the same ladder."""
    ladder = _synthetic_ladder()

    frame_1 = _header(ladder, ladder[0])
    frame_2 = _header(ladder, ladder[2], run_ts=pd.Timestamp("2026-07-04 01:00", tz=_TZ))

    assert dict(zip(frame_1["w0_min"], frame_1["forecast_id"])) == dict(zip(frame_2["w0_min"], frame_2["forecast_id"]))
    assert sorted(frame_1["forecast_id"]) == [0, 1, 2]


def test_header_frame_empty_ladder_returns_empty_frame_with_columns():
    frame = _header([], ())

    assert frame.empty
    assert "w0_min" in frame.columns


def _weather_creation_column() -> DatetimeColumn:
    return DatetimeColumn("weather_creation", TIMESTAMP, timezone=pytz.UTC)


def test_header_frame_weather_creation_passes_the_sql_datetime_column():
    ladder = _synthetic_ladder()

    validated = _weather_creation_column().validate(_header(ladder, ladder[0])["weather_creation"])

    assert (validated == _WEATHER_TS.tz_convert(None)).all()


def test_sql_datetime_column_rejects_a_null_weather_creation_column():
    """The header column as it reached the logger before the issue time was passed through."""
    nulls = pd.Series([None] * 3, index=pd.DatetimeIndex([_RUN_TS] * 3, name="timestamp"), name="weather_creation")

    with pytest.raises(ResourceError):
        _weather_creation_column().validate(nulls)


# --- build_detail_frame (pure, per-probe LONG shape) -------------------------


def _synthetic_ladder_trajectories():
    """3 candidates x 3 timestamps x 2 probes."""
    timestamps = [pd.Timestamp("2026-07-03 08:00", tz=_TZ) + pd.Timedelta(hours=h) for h in range(3)]
    ladder = _synthetic_ladder()
    trajectories = {
        ladder[0]: (timestamps, {"root_20": [0.9, 0.85, 0.8], "root_40": [0.9, 0.9, 0.89]}),
        ladder[1]: (timestamps, {"root_20": [0.9, 0.92, 0.91], "root_40": [0.9, 0.9, 0.9]}),
        ladder[2]: (timestamps, {"root_20": [0.9, 0.95, 0.95], "root_40": [0.9, 0.91, 0.91]}),
    }
    return ladder, trajectories, timestamps


def _detail(ladder, trajectories, run_ts=_RUN_TS):
    return build_detail_frame(ladder, trajectories, run_ts, _PROBE_IDS)


def test_detail_frame_has_expected_columns():
    """Each probe gets its own timestamp_creation/forecast_id twins as per-probe PK partners,
    since each probe carries its own soil_id surrogate attributes."""
    ladder, trajectories, _ = _synthetic_ladder_trajectories()

    frame = _detail(ladder, trajectories)

    assert set(frame.columns) == {
        "traj_root_20",
        "traj_root_20_timestamp_creation",
        "traj_root_20_forecast_id",
        "traj_root_40",
        "traj_root_40_timestamp_creation",
        "traj_root_40_forecast_id",
    }


def test_detail_frame_row_count_is_candidates_times_timestamps_times_probes():
    """The frame is long, not wide with probes packed as columns."""
    ladder, trajectories, timestamps = _synthetic_ladder_trajectories()

    frame = _detail(ladder, trajectories)

    assert len(frame) == len(ladder) * len(timestamps) * 2


def test_detail_frame_is_long_one_probes_full_triplet_populated_per_row():
    """The other probe's triplet is NaN on each row, so the direct write's per-attribute-set
    grouping can split rows back out per probe by its surrogate soil_id/field_id."""
    ladder, trajectories, _ = _synthetic_ladder_trajectories()

    frame = _detail(ladder, trajectories)

    root_20_cols = ["traj_root_20", "traj_root_20_timestamp_creation", "traj_root_20_forecast_id"]
    root_40_cols = ["traj_root_40", "traj_root_40_timestamp_creation", "traj_root_40_forecast_id"]

    root_20_present = frame[root_20_cols].notna().all(axis="columns")
    root_20_absent = frame[root_20_cols].isna().all(axis="columns")
    root_40_present = frame[root_40_cols].notna().all(axis="columns")
    root_40_absent = frame[root_40_cols].isna().all(axis="columns")

    assert (root_20_present | root_20_absent).all()
    assert (root_40_present | root_40_absent).all()
    assert not (root_20_present & root_40_present).any()
    assert not (root_20_absent & root_40_absent).any()


def test_detail_frame_all_candidates_present():
    ladder, trajectories, _ = _synthetic_ladder_trajectories()

    frame = _detail(ladder, trajectories)

    for forecast_id_col in ("traj_root_20_forecast_id", "traj_root_40_forecast_id"):
        present = frame[frame[forecast_id_col].notna()]
        assert set(present[forecast_id_col]) == {0, 1, 2}
        for forecast_id in (0, 1, 2):
            assert (present[forecast_id_col] == forecast_id).sum() == 3  # 3 timestamps


def test_header_and_detail_frames_agree_on_forecast_id_per_candidate():
    """forecast_id is the join key readers use to pull the recommended candidate's trajectory
    out of ``agri_soil_forecast``."""
    ladder, trajectories, _ = _synthetic_ladder_trajectories()

    header = _header(ladder, ladder[1])
    detail = _detail(ladder, trajectories)

    for position, candidate in enumerate(ladder):
        header_row = header[header["forecast_id"] == position]
        assert len(header_row) == 1
        assert header_row["w0_min"].iloc[0] == candidate[0].total_seconds() / 60.0

        detail_rows = detail[detail["traj_root_20_forecast_id"] == position].sort_index()
        _, probe_values = trajectories[candidate]
        assert list(detail_rows["traj_root_20"]) == probe_values["root_20"]


def test_detail_frame_timestamp_creation_is_run_time_not_weather_issue_time():
    """Every probe's timestamp_creation twin must be the RUN time, so two runs
    sharing one weather issue do not collide."""
    ladder, trajectories, _ = _synthetic_ladder_trajectories()

    frame = _detail(ladder, trajectories)

    for creation_col in ("traj_root_20_timestamp_creation", "traj_root_40_timestamp_creation"):
        assert (frame[creation_col].dropna() == _RUN_TS).all()


def test_detail_frame_two_runs_same_weather_issue_get_distinct_run_timestamps():
    """Distinct run timestamps keep a later run from upserting over an earlier one."""
    ladder, trajectories, _ = _synthetic_ladder_trajectories()
    run_1 = pd.Timestamp("2026-07-03 01:00", tz=_TZ)
    run_2 = pd.Timestamp("2026-07-04 01:00", tz=_TZ)

    frame_1 = _detail(ladder, trajectories, run_ts=run_1)
    frame_2 = _detail(ladder, trajectories, run_ts=run_2)

    for creation_col in ("traj_root_20_timestamp_creation", "traj_root_40_timestamp_creation"):
        assert (frame_1[creation_col].dropna() == run_1).all()
        assert (frame_2[creation_col].dropna() == run_2).all()
    assert run_1 != run_2


def test_detail_frame_carries_per_probe_values_as_is():
    """A pure pass-through: in production these are already signed matric
    potential (negative hPa); the frame does not re-convert."""
    ladder, trajectories, _ = _synthetic_ladder_trajectories()

    frame = _detail(ladder, trajectories)

    rows = frame[(frame["traj_root_20_forecast_id"] == 0) & frame["traj_root_20"].notna()]
    np.testing.assert_allclose(sorted(rows["traj_root_20"]), sorted([0.9, 0.85, 0.8]))


def test_detail_frame_empty_ladder_trajectories_returns_empty_frame_with_columns():
    frame = _detail([], {})

    assert frame.empty
    assert "traj_root_20" in frame.columns


# --- build_irrigation_frame --------------------------------------------------


_FLOW_M3S = 1.0e-5


def _irrigation(candidate, horizon_start, horizon_end, run_ts, windows=None):
    return build_irrigation_frame(candidate, windows or _two_windows(), _FLOW_M3S, horizon_start, horizon_end, run_ts)


def test_irrigation_frame_two_windows_yields_four_edge_rows():
    """Morning 08:00/30min + evening 18:00/1h -> two on/off interval pairs ->
    exactly 4 edge rows, the SAME run timestamp on every row."""
    horizon_start = pd.Timestamp("2026-07-03 01:00", tz=_TZ)
    horizon_end = horizon_start + pd.Timedelta(hours=24)

    frame = _irrigation((td(30), td(60)), horizon_start, horizon_end, horizon_start)

    assert len(frame) == 4
    assert list(frame.index) == [
        pd.Timestamp("2026-07-03 08:00", tz=_TZ),
        pd.Timestamp("2026-07-03 08:30", tz=_TZ),
        pd.Timestamp("2026-07-03 18:00", tz=_TZ),
        pd.Timestamp("2026-07-03 19:00", tz=_TZ),
    ]
    assert list(frame["irrigation_state"]) == [True, False, True, False]
    assert (frame["irrigation_timestamp_creation"] == horizon_start).all()


def test_irrigation_frame_zero_duration_candidate_yields_zero_rows():
    """A do-nothing (all-0min) candidate has no state transition to record --
    zero rows, not a marker row; the run stays visible via its header row."""
    horizon_start = pd.Timestamp("2026-07-03 01:00", tz=_TZ)

    frame = _irrigation((td(0), td(0)), horizon_start, horizon_start + pd.Timedelta(hours=24), horizon_start)

    assert frame.empty
    assert "irrigation_state" in frame.columns
    assert "irrigation_timestamp_creation" in frame.columns


def test_irrigation_frame_horizon_clamped_off_edge_emits_closing_false():
    """A window whose duration would run past the horizon must still close
    (never dangle True): the clamped off edge is emitted as the closing row."""
    horizon_start = pd.Timestamp("2026-07-03 07:00", tz=_TZ)
    horizon_end = horizon_start + pd.Timedelta(hours=24)  # 2026-07-04 07:00

    frame = _irrigation(
        (td(10 * 60),), horizon_start, horizon_end, horizon_start, windows=_windows(datetime.time(23, 0))
    )

    assert len(frame) == 2
    assert list(frame.index) == [pd.Timestamp("2026-07-03 23:00", tz=_TZ), horizon_end]
    assert list(frame["irrigation_state"]) == [True, False]


def test_irrigation_frame_empty_schedule_returns_empty_frame_with_columns():
    horizon_start = pd.Timestamp("2026-07-03 01:00", tz=_TZ)

    frame = _irrigation((), horizon_start, horizon_start + pd.Timedelta(hours=24), horizon_start, windows=[])

    assert frame.empty
    assert "irrigation_state" in frame.columns


def test_irrigation_frame_touching_windows_merge_into_one_interval():
    """Window A's off_ts equals window B's on_ts: one interval of 2 edge rows, no (False, True)
    pair on the shared timestamp."""
    horizon_start = pd.Timestamp("2026-07-03 07:00", tz=_TZ)
    candidate = (td(600), td(30))  # window A: 08:00 + 10h == window B's 18:00 on_ts

    frame = _irrigation(candidate, horizon_start, horizon_start + pd.Timedelta(hours=24), horizon_start)

    assert len(frame) == 2
    assert not frame.index.duplicated().any()
    assert list(frame.index) == [
        pd.Timestamp("2026-07-03 08:00", tz=_TZ),
        pd.Timestamp("2026-07-03 18:30", tz=_TZ),
    ]
    assert list(frame["irrigation_state"]) == [True, False]


def test_irrigation_frame_overlapping_windows_merge_into_one_interval():
    """Window A's interval extends past window B's on_ts."""
    horizon_start = pd.Timestamp("2026-07-03 07:00", tz=_TZ)
    candidate = (td(11 * 60), td(30))  # window A: 08:00 + 11h = 19:00, past 18:00

    frame = _irrigation(candidate, horizon_start, horizon_start + pd.Timedelta(hours=24), horizon_start)

    assert len(frame) == 2
    assert not frame.index.duplicated().any()
    assert list(frame.index) == [
        pd.Timestamp("2026-07-03 08:00", tz=_TZ),
        pd.Timestamp("2026-07-03 19:00", tz=_TZ),
    ]
    assert list(frame["irrigation_state"]) == [True, False]


def test_irrigation_frame_short_horizon_drops_degenerate_window_but_keeps_others():
    """A horizon ending at or before a window's on_ts gives on_ts >= off_ts: that window adds no rows."""
    horizon_start = pd.Timestamp("2026-07-03 07:00", tz=_TZ)
    horizon_end = pd.Timestamp("2026-07-03 12:00", tz=_TZ)  # cuts off window B (18:00)

    frame = _irrigation((td(30), td(30)), horizon_start, horizon_end, horizon_start)

    assert len(frame) == 2
    assert list(frame.index) == [
        pd.Timestamp("2026-07-03 08:00", tz=_TZ),
        pd.Timestamp("2026-07-03 08:30", tz=_TZ),
    ]
    assert list(frame["irrigation_state"]) == [True, False]
