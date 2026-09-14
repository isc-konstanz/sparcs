# -*- coding: utf-8 -*-
"""sparcs.tests.test_field_simulation_irrigation_state
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The live sim's irrigation input with the physical flow meter unavailable: an
on/off STATE channel times a drip-derived design flow, behind a fallback chain
(measured flow when the meter reports, else state x design-flow), and a loud
startup raise when a configured irrigation has neither input wired.

As in the sibling field-simulation tests, instances come from ``object.__new__``
so no Component/PDE stack is built; ``self.data`` is stubbed and channels are
duck-typed ``SimpleNamespace`` objects (``Channels([...])`` only needs ``.id``).
"""

import types

import pytest

import numpy as np
import pandas as pd

_base = pytest.importorskip("sparcs.components.agriculture.simulation.base")
FieldSimulation = _base.FieldSimulation

from lories.core import ConfigurationUnavailableError  # noqa: E402
from sparcs.components.agriculture.simulation._soil import design_flow_lpm  # noqa: E402


def _index(start="2026-05-01 10:00", periods=4, freq="15min"):
    return pd.date_range(start, periods=periods, freq=freq, tz="UTC")


def _frame(points: dict[str, float]) -> pd.DataFrame:
    index = pd.DatetimeIndex([pd.Timestamp(ts, tz="UTC") for ts in points])
    return pd.DataFrame({"v": list(points.values())}, index=index)


class _MultiData:
    """Stub ``self.data``: return a preset frame per channel id and record reads."""

    def __init__(self, frames: dict[str, pd.DataFrame]):
        self._frames = frames
        self.reads: list[str] = []

    def read(self, channels, start=None, end=None, unique=False):
        cid = next(iter(channels)).id
        self.reads.append(cid)
        return self._frames.get(cid, pd.DataFrame())


def _sim(**attrs) -> FieldSimulation:
    sim = object.__new__(FieldSimulation)
    sim._name = "test_field_simulation"
    sim._irrigation_flow_channel = None
    sim._irrigation_state_channel = None
    sim._design_flow_lpm = 0.0
    sim._drip_explicit = False
    for k, v in attrs.items():
        setattr(sim, k, v)
    return sim


_FLOW = types.SimpleNamespace(id="irrigation_flow")
_STATE = types.SimpleNamespace(id="irrigation_state")


# --- design_flow_lpm helper + _derive_flow_m3s regression ---------------------


def test_design_flow_lpm_from_layout():
    assert design_flow_lpm(32, 1.0) == pytest.approx(32 / 60.0)
    assert design_flow_lpm(1, 1.0) == pytest.approx(1 / 60.0)


def test_derive_flow_m3s_unchanged_by_refactor():
    sp = pytest.importorskip("sparcs.components.agriculture.simulation.soil_predictor")
    for nc, nf, length in [(1, 1.0, 1.0), (32, 1.0, 0.4), (16, 2.0, 12.6)]:
        expected = nc * nf / 60.0 / (60_000.0 * length)
        assert sp.SoilPredictor._derive_flow_m3s(nc, nf, length) == pytest.approx(expected)


# --- _read_state_span ---------------------------------------------------------


def test_read_state_span_without_channel_is_zero_series():
    sim = _sim()
    index = _index()
    assert list(sim._read_state_span(index[0], index[-1], index)) == [0.0, 0.0, 0.0, 0.0]


def test_read_state_span_backfills_and_coerces_to_float():
    index = _index()  # 10:00, 10:15, 10:30, 10:45
    frame = _frame({"2026-05-01 10:00": True, "2026-05-01 10:20": False})
    sim = _sim(_irrigation_state_channel=_STATE, _Component__data=_MultiData({"irrigation_state": frame}))
    aligned = sim._read_state_span(index[0], index[-1], index)
    assert list(aligned) == [1.0, 1.0, 0.0, 0.0]
    assert aligned.dtype == float


def test_read_state_span_null_reads_as_off():
    index = _index()
    frame = _frame({"2026-05-01 10:00": 1.0, "2026-05-01 10:20": np.nan})
    sim = _sim(_irrigation_state_channel=_STATE, _Component__data=_MultiData({"irrigation_state": frame}))
    aligned = sim._read_state_span(index[0], index[-1], index)
    assert list(aligned) == [1.0, 1.0, 0.0, 0.0]
    assert not aligned.isna().any()


# --- _read_measured_flow (meter-alive detection) ------------------------------


def test_measured_flow_none_when_channel_unwired():
    sim = _sim()
    index = _index()
    assert sim._read_measured_flow(index[0], index[-1], index) is None


def test_measured_flow_none_when_meter_reported_nothing():
    index = _index()
    sim = _sim(_irrigation_flow_channel=_FLOW, _Component__data=_MultiData({}))
    assert sim._read_measured_flow(index[0], index[-1], index) is None


def test_measured_flow_none_when_all_rows_nan():
    index = _index()
    sim = _sim(
        _irrigation_flow_channel=_FLOW,
        _Component__data=_MultiData({"irrigation_flow": _frame({"2026-05-01 10:00": np.nan})}),
    )
    assert sim._read_measured_flow(index[0], index[-1], index) is None


def test_measured_flow_zero_is_alive_not_none():
    """A meter reporting 0 is 'not watering', not 'dead' -> a real (zero) series."""
    index = _index()
    sim = _sim(
        _irrigation_flow_channel=_FLOW,
        _Component__data=_MultiData({"irrigation_flow": _frame({"2026-05-01 10:00": 0.0})}),
    )
    measured = sim._read_measured_flow(index[0], index[-1], index)
    assert measured is not None
    assert list(measured) == [0.0, 0.0, 0.0, 0.0]


# --- _irrigation_flow_lpm fallback chain --------------------------------------


def test_flow_lpm_prefers_measured_over_state():
    """Meter reporting -> its value is used and the state channel is never read."""
    index = _index()
    data = _MultiData(
        {
            "irrigation_flow": _frame({"2026-05-01 10:00": 50.0}),
            "irrigation_state": _frame({"2026-05-01 10:00": True}),
        }
    )
    sim = _sim(
        _irrigation_flow_channel=_FLOW,
        _irrigation_state_channel=_STATE,
        _design_flow_lpm=0.5,
        _Component__data=data,
    )
    result = sim._irrigation_flow_lpm(index[0], index[-1], index)
    assert list(result) == [50.0, 50.0, 50.0, 50.0]
    assert data.reads == ["irrigation_flow"]  # state never consulted


def test_flow_lpm_falls_back_to_state_when_meter_dead():
    """Meter reports nothing this span -> state x design-flow drives the sim."""
    index = _index()
    data = _MultiData({"irrigation_state": _frame({"2026-05-01 10:00": True, "2026-05-01 10:20": False})})
    sim = _sim(
        _irrigation_flow_channel=_FLOW,  # wired but silent (broken meter)
        _irrigation_state_channel=_STATE,
        _design_flow_lpm=0.5,
        _drip_explicit=True,
        _Component__data=data,
    )
    result = sim._irrigation_flow_lpm(index[0], index[-1], index)
    assert list(result) == [0.5, 0.5, 0.0, 0.0]  # state 1/1/0/0 x 0.5
    assert data.reads == ["irrigation_flow", "irrigation_state"]


def test_flow_lpm_uses_state_when_flow_channel_unwired():
    index = _index()
    data = _MultiData({"irrigation_state": _frame({"2026-05-01 10:00": True})})
    sim = _sim(
        _irrigation_state_channel=_STATE,
        _design_flow_lpm=2.0,
        _drip_explicit=True,
        _Component__data=data,
    )
    result = sim._irrigation_flow_lpm(index[0], index[-1], index)
    assert list(result) == [2.0, 2.0, 2.0, 2.0]


def test_flow_lpm_does_not_synthesize_from_state_without_explicit_drip():
    """A flow-primary field that also has a state channel but NO explicit
    [soil_simulation.drip] must NOT leak the placeholder design flow on a meter
    gap -- it reads 0 (not watering), never a fabricated forcing."""
    index = _index()
    data = _MultiData({"irrigation_state": _frame({"2026-05-01 10:00": True})})
    sim = _sim(
        _irrigation_flow_channel=_FLOW,  # wired but silent (broken meter)
        _irrigation_state_channel=_STATE,
        _design_flow_lpm=0.0167,  # placeholder default (1 nozzle x 1 l/h)
        _drip_explicit=False,
        _Component__data=data,
    )
    result = sim._irrigation_flow_lpm(index[0], index[-1], index)
    assert list(result) == [0.0, 0.0, 0.0, 0.0]
    assert "irrigation_state" not in data.reads  # state never consulted


def test_flow_lpm_zero_when_neither_wired():
    """Rain-fed field (no irrigation input at all): 0 l/min, no read attempted."""
    index = _index()
    data = _MultiData({})
    sim = _sim(_Component__data=data)
    result = sim._irrigation_flow_lpm(index[0], index[-1], index)
    assert list(result) == [0.0, 0.0, 0.0, 0.0]
    assert data.reads == []


# --- _validate_irrigation_input (startup raise) -------------------------------


def _wired(connector: bool):
    return types.SimpleNamespace(has_connector=lambda: connector)


def test_validate_noop_when_no_irrigation_component():
    """Rain-fed field: no [irrigation] block -> 0 l/min is deliberate, not a raise."""
    _sim(irrigation=None)._validate_irrigation_input()  # must not raise


def test_validate_passes_with_wired_flow():
    _sim(irrigation=object(), _irrigation_flow_channel=_wired(True))._validate_irrigation_input()


def test_validate_passes_with_wired_state_and_explicit_drip():
    _sim(
        irrigation=object(),
        _irrigation_state_channel=_wired(True),
        _drip_explicit=True,
    )._validate_irrigation_input()


def test_validate_raises_when_state_wired_but_drip_not_explicit():
    sim = _sim(irrigation=object(), _irrigation_state_channel=_wired(True), _drip_explicit=False)
    with pytest.raises(ConfigurationUnavailableError, match="soil_simulation.drip"):
        sim._validate_irrigation_input()


def test_validate_raises_when_flow_channel_has_no_connector():
    sim = _sim(irrigation=object(), _irrigation_flow_channel=_wired(False))
    with pytest.raises(ConfigurationUnavailableError):
        sim._validate_irrigation_input()


def test_validate_raises_when_nothing_wired():
    sim = _sim(irrigation=object())
    with pytest.raises(ConfigurationUnavailableError, match="no usable input"):
        sim._validate_irrigation_input()
