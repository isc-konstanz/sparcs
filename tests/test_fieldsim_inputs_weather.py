# -*- coding: utf-8 -*-
"""
tests.test_fieldsim_inputs_weather
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``ChannelInputs`` reads weather through the connector, never a logger, with columns named by channel key,
trimmed to ``(start, end]``; a chunk missing a required column is dropped with a WARNING naming it.
"""

import datetime as dt
import logging
from types import SimpleNamespace

import pandas as pd
from sparcs.components.agriculture.simulation import components
from sparcs.components.agriculture.simulation.core.config import SoilConfig

COMPONENTS_LOGGER = "sparcs.components.agriculture.simulation.components"


class _RecordingData:
    """``Component.data`` stand-in recording whether a span was read through the
    connector (``read``) or a logger (``read_logged``)."""

    def __init__(self, frame: pd.DataFrame = None):
        self.frame = frame if frame is not None else pd.DataFrame()
        self.calls: list[tuple] = []

    def read(self, channels, start=None, end=None, unique=False):
        self.calls.append(("read", start, end, unique))
        return self.frame.copy()

    def read_logged(self, *args, **kwargs):
        self.calls.append(("read_logged",))
        return pd.DataFrame()


def _inputs(data, *, required=()) -> components.ChannelInputs:
    field = SimpleNamespace(
        name="fieldsim_test",
        data=data,
        setup=SimpleNamespace(soil=SoilConfig.from_dict({"mesh": {}})),
        _weather_channels=object(),
        _required_weather_keys=required,
    )
    return components.ChannelInputs(field)


def test_weather_span_reads_the_connector_not_a_logger():
    """Weather already lives in its source (kob_tracker station / Brightsky), so
    the tick reads the span from the connector; it must not go through a logger.
    The read reaches an hour back and on to the next full hour, so the hourly
    records around the span are there to fill its minutes."""
    data = _RecordingData()
    inputs = _inputs(data)

    start = dt.datetime(2026, 7, 12, 10, tzinfo=dt.timezone.utc)
    end = dt.datetime(2026, 7, 12, 11, 20, tzinfo=dt.timezone.utc)
    inputs.weather(start, end)

    assert data.calls == [
        ("read", pd.Timestamp("2026-07-12 09:00Z"), pd.Timestamp("2026-07-12 12:00Z"), False),
    ]


def test_weather_frame_valid_flags_a_missing_required_column():
    inputs = _inputs(_RecordingData(), required=("ghi", "temp_air"))
    frame = pd.DataFrame({"ghi": [1.0]}, index=pd.DatetimeIndex([pd.Timestamp("2026-07-12 10:00", tz="UTC")]))

    assert inputs._weather_frame_valid(frame) is False
    frame["temp_air"] = 20.0
    assert inputs._weather_frame_valid(frame) is True


def test_invalid_chunk_names_the_missing_column_at_warning_once(caplog):
    """The dropped chunk is the weather-stall cause; the missing column must
    reach the operator, and a persistent outage must not spam."""
    index = pd.date_range("2026-07-12 10:15", periods=2, freq="15min", tz="UTC")
    data = _RecordingData(pd.DataFrame({"ghi": [1.0, 2.0]}, index=index))  # temp_air missing
    inputs = _inputs(data, required=("ghi", "temp_air"))
    start = dt.datetime(2026, 7, 12, 10, tzinfo=dt.timezone.utc)
    end = dt.datetime(2026, 7, 12, 12, tzinfo=dt.timezone.utc)

    with caplog.at_level(logging.WARNING, logger=COMPONENTS_LOGGER):
        assert inputs.weather(start, end).empty
        assert inputs.weather(start, end).empty

    warnings = [r for r in caplog.records if "temp_air" in r.getMessage()]
    assert len(warnings) == 1
    assert inputs._last_invalid_weather_columns == ["temp_air"]


def test_hourly_rain_over_station_minutes_lands_on_whole_minutes_up_to_the_last_rain_hour():
    station = pd.date_range("2026-07-12 08:00:30", "2026-07-12 09:20:30", freq="1min", tz="UTC")
    rain = pd.Series({pd.Timestamp("2026-07-12 08:00Z"): 0.0, pd.Timestamp("2026-07-12 09:00Z"): 1.2})
    frame = pd.DataFrame({"ghi": 300.0, "temp_air": 20.0}, index=station).join(
        rain.rename("precipitation"), how="outer"
    )
    inputs = _inputs(_RecordingData(frame), required=("ghi", "temp_air", "precipitation"))

    weather = inputs.weather(pd.Timestamp("2026-07-12 08:00Z"), pd.Timestamp("2026-07-12 09:20Z"))

    assert weather.index.equals(pd.date_range("2026-07-12 08:01", "2026-07-12 09:00", freq="1min", tz="UTC"))
    assert not weather[["ghi", "temp_air"]].isna().any().any()
    assert weather["precipitation"].tolist() == [1.2] * 60
