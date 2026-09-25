# -*- coding: utf-8 -*-
"""tests.test_fieldsim_inputs_weather
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``ChannelInputs``' weather span: the read goes through the connector (never a
logger), the frame is renamed and trimmed to ``(start, end]``, and a chunk
missing a required column is dropped with the column named at WARNING.
"""

import datetime as dt
import logging
from types import SimpleNamespace

import pandas as pd
from sparcs.components.agriculture.fieldsim import components
from sparcs.components.agriculture.fieldsim.core.config import SoilConfig
from sparcs.components.agriculture.fieldsim.runtime.ports import InputKey

COMPONENTS_LOGGER = "sparcs.components.agriculture.fieldsim.components"


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


def _inputs(data, *, rename=None, required=()) -> components.ChannelInputs:
    field = SimpleNamespace(
        name="fieldsim_test",
        data=data,
        setup=SimpleNamespace(soil=SoilConfig.from_dict({"mesh": {}})),
        _weather_channels=object(),
        _evapo_rename=rename or {},
        _required_weather_keys=required,
    )
    return components.ChannelInputs(field)


def test_weather_span_reads_the_connector_not_a_logger():
    """Weather already lives in its source (kob_tracker station / Brightsky), so
    the tick reads the span from the connector; it must not go through a logger."""
    data = _RecordingData()
    inputs = _inputs(data)

    start = dt.datetime(2026, 7, 12, 10, tzinfo=dt.timezone.utc)
    end = dt.datetime(2026, 7, 12, 11, tzinfo=dt.timezone.utc)
    inputs.read(InputKey.WEATHER, start, end)

    assert data.calls == [("read", start, end, True)]


def test_trim_span_renames_and_keeps_the_half_open_interval():
    index = pd.date_range("2026-07-12 10:00", periods=5, freq="15min", tz="UTC")
    frame = pd.DataFrame({"weather.ghi": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=index)
    inputs = _inputs(_RecordingData(), rename={"weather.ghi": "ghi"})

    trimmed = inputs._trim_span(frame, index[0], pd.Timestamp("2026-07-12 10:45", tz="UTC"))

    assert list(trimmed.columns) == ["ghi"]
    assert trimmed.index[0] == pd.Timestamp("2026-07-12 10:15", tz="UTC")  # start row excluded
    assert trimmed.index[-1] == pd.Timestamp("2026-07-12 10:45", tz="UTC")  # end row kept


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
        assert inputs.read(InputKey.WEATHER, start, end).empty
        assert inputs.read(InputKey.WEATHER, start, end).empty

    warnings = [r for r in caplog.records if "temp_air" in r.getMessage()]
    assert len(warnings) == 1
    assert inputs._last_invalid_weather_columns == ["temp_air"]
