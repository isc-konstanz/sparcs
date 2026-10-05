# -*- coding: utf-8 -*-
"""
tests.test_fieldsim_inputs_irrigation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``ChannelInputs`` reads metered flow, falling back to on/off state times the drip design flow, and never fabricates one.
``FieldRunner`` aligns the samples onto the weather timesteps (backward fill, NULL = not watering).
"""

import datetime as dt
from types import SimpleNamespace

import pytest

import numpy as np
import pandas as pd
from lories.data import Channels
from sparcs.components.agriculture.simulation import components
from sparcs.components.agriculture.simulation.core.candidates import derive_flow_m3s
from sparcs.components.agriculture.simulation.core.config import SoilConfig
from sparcs.components.agriculture.simulation.core.pde import design_flow_lpm
from sparcs.components.agriculture.simulation.runtime.ports import IRRIGATION_LOOKBACK
from sparcs.components.agriculture.simulation.runtime.runner import FieldRunner


def _index(start="2026-05-01 10:00", periods=4, freq="15min"):
    return pd.date_range(start, periods=periods, freq=freq, tz="UTC")


def _series(points: dict) -> pd.Series:
    index = pd.DatetimeIndex([pd.Timestamp(ts, tz="UTC") for ts in points])
    return pd.Series(list(points.values()), index=index)


def _frame(points: dict) -> pd.DataFrame:
    return _series(points).to_frame("v")


class _FakeData:
    """``Component.data`` stand-in: the weather channels by identity, every other
    read by the single channel object it was handed."""

    def __init__(self, weather_channels, weather_frame):
        self.weather_channels = weather_channels
        self.weather_frame = weather_frame
        self.by_channel: dict = {}
        self.reads: list[tuple] = []

    def bind(self, name: str, channel, frame: pd.DataFrame) -> None:
        self.by_channel[id(channel)] = (name, frame)

    def read(self, channels, start=None, end=None, unique=True) -> pd.DataFrame:
        if channels is self.weather_channels:
            self.reads.append(("weather", start, end))
            return self.weather_frame.copy()
        chans = list(channels)
        if len(chans) == 1 and id(chans[0]) in self.by_channel:
            name, frame = self.by_channel[id(chans[0])]
            self.reads.append((name, start, end))
            return frame.copy()
        return pd.DataFrame()


_FLOW = SimpleNamespace(id="irrigation_flow")
_STATE = SimpleNamespace(id="irrigation_state")


def _inputs(*, drip=None, flow=None, state=None, weather_index=None) -> components.ChannelInputs:
    """A ``ChannelInputs`` over a fake field; ``flow``/``state`` are the frames
    the respective channel reads back (``None`` = channel not wired)."""
    index = weather_index if weather_index is not None else _index()
    weather_channels = Channels([])
    data = _FakeData(weather_channels, pd.DataFrame({"ghi": 100.0}, index=index))
    soil = SoilConfig.from_dict({"mesh": {}, **({"drip": drip} if drip else {})})
    if drip is None:
        soil.drip.explicit = False
    field = SimpleNamespace(
        name="fieldsim_test",
        data=data,
        setup=SimpleNamespace(soil=soil),
        _weather_channels=weather_channels,
        _required_weather_keys=(),
        _irrigation_flow_channel=_FLOW if flow is not None else None,
        _irrigation_state_channel=_STATE if state is not None else None,
    )
    if flow is not None:
        data.bind("irrigation_flow", _FLOW, flow)
    if state is not None:
        data.bind("irrigation_state", _STATE, state)
    return components.ChannelInputs(field)


def _read_names(inputs) -> list[str]:
    return [name for name, _, _ in inputs.field.data.reads]


# --------------------------------------------------------------------------- flow arithmetic


def test_design_flow_lpm_from_layout():
    assert design_flow_lpm(32, 1.0) == pytest.approx(32 / 60.0)
    assert design_flow_lpm(1, 1.0) == pytest.approx(1 / 60.0)


def test_derive_flow_m3s_normalizes_per_metre_of_drip_line():
    for nc, nf, length in [(1, 1.0, 1.0), (32, 1.0, 0.4), (16, 2.0, 12.6)]:
        expected = nc * nf / 60.0 / (60_000.0 * length)
        assert derive_flow_m3s(nc, nf, length) == pytest.approx(expected)


# --------------------------------------------------------------------------- FieldRunner._align_flow


def test_align_backward_fills_between_flow_rows():
    """Each weather timestep gets the most recent flow value at or before it."""
    aligned = FieldRunner._align_flow(_series({"2026-05-01 10:00": 50.0, "2026-05-01 10:20": 0.0}), _index())
    assert list(aligned) == [50.0, 50.0, 0.0, 0.0]


def test_align_null_flow_reads_as_zero():
    """A NULL/NaN flow row -> 0.0 from that row on, never NaN into the PDE source."""
    aligned = FieldRunner._align_flow(_series({"2026-05-01 10:00": 50.0, "2026-05-01 10:20": np.nan}), _index())
    assert list(aligned) == [50.0, 50.0, 0.0, 0.0]
    assert not aligned.isna().any()


def test_align_leading_gap_reads_as_zero():
    """No flow row at or before the first weather timestep -> not watering."""
    aligned = FieldRunner._align_flow(_series({"2026-05-01 10:20": 30.0}), _index())
    assert list(aligned) == [0.0, 0.0, 30.0, 30.0]


def test_align_empty_history_is_all_zero():
    assert list(FieldRunner._align_flow(pd.Series(dtype=float), _index())) == [0.0, 0.0, 0.0, 0.0]


def test_align_sub_tick_window_covers_only_its_timesteps():
    """A 20-minute watering window inside an hour waters exactly the timesteps it
    covers, not the whole hour (the distortion the latched scalar had)."""
    index = _index(periods=12, freq="5min")  # 10:00 .. 10:55
    aligned = FieldRunner._align_flow(_series({"2026-05-01 10:10": 40.0, "2026-05-01 10:30": 0.0}), index)
    watered = aligned[aligned > 0.0]
    assert list(watered.index) == list(pd.date_range("2026-05-01 10:10", "2026-05-01 10:25", freq="5min", tz="UTC"))


def test_align_replays_history_not_current_value():
    """Catch-up over a gap uses the flow that was logged then; a burst logged
    after the span cannot leak backwards."""
    aligned = FieldRunner._align_flow(_series({"2026-05-01 10:00": 0.0, "2026-05-01 11:30": 99.0}), _index())
    assert list(aligned) == [0.0, 0.0, 0.0, 0.0]


def test_align_coerces_the_state_series_to_float():
    aligned = FieldRunner._align_flow(_series({"2026-05-01 10:00": True, "2026-05-01 10:20": False}), _index())
    assert list(aligned) == [1.0, 1.0, 0.0, 0.0]
    assert aligned.dtype == float


def test_align_unsorted_duplicate_samples_keep_the_last_per_instant():
    samples = pd.Series(
        [0.0, 5.0, 7.0],
        index=pd.DatetimeIndex(["2026-05-01 10:20", "2026-05-01 10:00", "2026-05-01 10:00"], tz="UTC"),
    )
    assert list(FieldRunner._align_flow(samples, _index())) == [7.0, 7.0, 0.0, 0.0]


# --------------------------------------------------------------------------- raw reads


def test_read_state_span_is_raw_samples_as_float():
    index = _index()  # 10:00, 10:15, 10:30, 10:45
    inputs = _inputs(state=_frame({"2026-05-01 10:00": True, "2026-05-01 10:20": False}))

    samples = inputs._read_state_span(index[0], index[-1])

    assert list(samples) == [1.0, 0.0]
    assert samples.dtype == float


def test_raw_reads_reach_back_one_lookback_before_the_chunk():
    index = _index()
    inputs = _inputs(flow=_frame({"2026-05-01 10:00": 1.0}))

    inputs.irrigation(index[0], index[-1])

    assert inputs.field.data.reads == [("irrigation_flow", index[0] - IRRIGATION_LOOKBACK, index[-1])]


def test_measured_flow_none_when_channel_unwired():
    index = _index()
    assert _inputs()._read_measured_flow(index[0], index[-1]) is None


def test_measured_flow_none_when_meter_reported_nothing():
    index = _index()
    assert _inputs(flow=pd.DataFrame())._read_measured_flow(index[0], index[-1]) is None


def test_measured_flow_none_when_all_rows_nan():
    index = _index()
    inputs = _inputs(flow=_frame({"2026-05-01 10:00": np.nan}))
    assert inputs._read_measured_flow(index[0], index[-1]) is None


def test_measured_flow_zero_is_alive_not_none():
    """A meter reporting 0 is 'not watering', not 'dead' -> a real (zero) series."""
    index = _index()
    inputs = _inputs(flow=_frame({"2026-05-01 10:00": 0.0}))

    measured = inputs._read_measured_flow(index[0], index[-1])

    assert measured is not None
    assert list(measured) == [0.0]


# --------------------------------------------------------------------------- the fallback chain


def _read_irrigation(inputs, index) -> pd.Series:
    start = (index[0] - pd.Timedelta(minutes=1)).to_pydatetime()
    return FieldRunner._align_flow(inputs.irrigation(start, index[-1].to_pydatetime()), index)


def test_flow_lpm_prefers_measured_over_state():
    """Meter reporting -> its value is used and the state channel is never read."""
    index = _index()
    inputs = _inputs(
        drip={"nozzle_count": 30, "nozzle_flow_lph": 1.0},
        flow=_frame({"2026-05-01 10:00": 50.0}),
        state=_frame({"2026-05-01 10:00": True}),
    )

    assert list(_read_irrigation(inputs, index)) == [50.0, 50.0, 50.0, 50.0]
    assert _read_names(inputs) == ["irrigation_flow"]  # state never consulted


def test_flow_lpm_falls_back_to_state_when_meter_dead():
    """Meter reports nothing this span -> state x design-flow drives the sim."""
    index = _index()
    inputs = _inputs(
        drip={"nozzle_count": 30, "nozzle_flow_lph": 1.0},  # 0.5 l/min
        flow=pd.DataFrame(),  # wired but silent (broken meter)
        state=_frame({"2026-05-01 10:00": True, "2026-05-01 10:20": False}),
    )

    assert list(_read_irrigation(inputs, index)) == [0.5, 0.5, 0.0, 0.0]  # state 1/1/0/0 x 0.5
    assert _read_names(inputs) == ["irrigation_flow", "irrigation_state"]


def test_flow_lpm_uses_state_when_flow_channel_unwired():
    index = _index()
    inputs = _inputs(
        drip={"nozzle_count": 120, "nozzle_flow_lph": 1.0},  # 2.0 l/min
        state=_frame({"2026-05-01 10:00": True}),
    )

    assert list(_read_irrigation(inputs, index)) == [2.0, 2.0, 2.0, 2.0]


def test_flow_lpm_does_not_synthesize_from_state_without_explicit_drip():
    """Without an explicit [soil_simulation.drip] a meter gap reads 0, not the placeholder design flow
    from the state channel."""
    index = _index()
    inputs = _inputs(
        flow=pd.DataFrame(),  # wired but silent (broken meter)
        state=_frame({"2026-05-01 10:00": True}),
    )

    assert list(_read_irrigation(inputs, index)) == [0.0, 0.0, 0.0, 0.0]
    assert "irrigation_state" not in _read_names(inputs)  # state never consulted


def test_flow_lpm_zero_when_neither_wired():
    """Rain-fed field (no irrigation input at all): 0 l/min, no read attempted."""
    index = _index()
    inputs = _inputs()

    assert list(_read_irrigation(inputs, index)) == [0.0, 0.0, 0.0, 0.0]
    assert inputs.field.data.reads == []


def test_irrigation_never_reads_the_weather():
    """The runner already holds the chunk's weather; the irrigation read is the
    raw series only, empty when nothing is wired."""
    start = dt.datetime(2026, 5, 1, 10, tzinfo=dt.timezone.utc)
    end = dt.datetime(2026, 5, 1, 11, tzinfo=dt.timezone.utc)
    inputs = _inputs(flow=_frame({"2026-05-01 10:00": 1.0}))

    samples = inputs.irrigation(start, end)

    assert "weather" not in _read_names(inputs)
    assert list(samples) == [1.0]
    assert _inputs().irrigation(start, end).empty
