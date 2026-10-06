# -*- coding: utf-8 -*-
"""
tests.test_fieldsim_minute_grid
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The live tick's weather on whole minutes: values for the period before their timestamp held over it (hourly
rates for at most that hour), point values interpolated, and nothing past the last record of a covering column.
"""

import pytest

import pandas as pd
from sparcs.components.agriculture.simulation.core.minute_grid import to_minute_grid


def _t(time: str) -> pd.Timestamp:
    return pd.Timestamp(f"2026-10-05 {time}", tz="UTC")


def _frame(**columns: dict) -> pd.DataFrame:
    return pd.DataFrame({key: pd.Series({_t(t): v for t, v in values.items()}) for key, values in columns.items()})


def test_hourly_rain_rate_is_held_over_the_hour_before_its_timestamp():
    frame = _frame(precipitation={"08:00": 0.0, "09:00": 6.0})

    grid = to_minute_grid(frame, _t("08:00"), _t("09:00"), cover=())

    assert len(grid) == 60
    assert grid["precipitation"].tolist() == [6.0] * 60


def test_period_values_are_held_over_the_hour_and_point_values_interpolated():
    frame = _frame(ghi={"08:00": 100.0, "09:00": 400.0}, temp_air={"08:00": 10.0, "09:00": 16.0})

    grid = to_minute_grid(frame, _t("08:00"), _t("09:00"), cover=())

    assert grid["ghi"].tolist() == [400.0] * 60
    assert grid["temp_air"].tolist() == pytest.approx([10.0 + 0.1 * minute for minute in range(1, 61)])


def test_grid_ends_at_the_last_record_of_a_covering_column():
    frame = _frame(
        precipitation={"08:00": 0.0, "09:00": 1.2},
        temp_air={"08:00": 10.0, "09:00": 11.0, "10:00": 12.0},
    )

    grid = to_minute_grid(frame, _t("08:00"), _t("10:00"), cover=("precipitation", "temp_air"))

    assert grid.index[-1] == _t("09:00")


def test_minute_data_passes_through_unchanged():
    minutes = pd.date_range(_t("08:00"), _t("08:05"), freq="1min")
    frame = pd.DataFrame(
        {
            "ghi": [100.0, 110.0, 120.0, 130.0, 140.0, 150.0],
            "temp_air": [10.0, 10.5, 11.0, 11.5, 12.0, 12.5],
            "precipitation": [0.0, 0.2, 0.0, 0.1, 0.0, 0.3],
        },
        index=minutes,
    )

    grid = to_minute_grid(frame, _t("08:00"), _t("08:05"), cover=("ghi", "temp_air", "precipitation"))

    pd.testing.assert_frame_equal(grid, frame.iloc[1:], check_freq=False)


def test_rain_after_a_missing_hour_is_not_smeared_into_it():
    frame = _frame(precipitation={"08:00": 0.0, "10:00": 2.4})

    grid = to_minute_grid(frame, _t("08:00"), _t("10:00"), cover=())

    assert grid["precipitation"][: _t("09:00")].sum() == 0.0
    assert grid["precipitation"][_t("09:01") :].tolist() == [2.4] * 60


def test_records_out_of_order_give_the_same_grid():
    frame = _frame(precipitation={"09:00": 6.0, "08:00": 0.0})

    grid = to_minute_grid(frame, _t("08:00"), _t("09:00"), cover=())

    assert grid["precipitation"].tolist() == [6.0] * 60
