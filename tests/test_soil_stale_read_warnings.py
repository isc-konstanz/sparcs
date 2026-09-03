# -*- coding: utf-8 -*-
"""sparcs.tests.test_soil_stale_read_warnings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Issue 23 (W2.5): lories converts connector errors into EMPTY frames, so a
broken tensiometer or forecast source reads as permanent, benign no-data
(error-empty masquerading as no-data-empty -- the brightsky incident class).
LOGGING ONLY -- skip/continue semantics, guard order, and return values are
unchanged on both paths.

Anchor side: ``load_anchor_history`` keeps a per-sensor "last non-empty read
at" WALL-CLOCK timestamp; when its age exceeds the sensor's [anchor] staleness
bound it WARNs once (latched until a non-empty read recovers the sensor).
Wall-clock, never the ``(start, end)`` chunk args -- catch-up replays day
chunks with historical bounds, which must not false-trigger.

Forecast side: ``_fetch_forecast`` records WHY it returned None on a private
cause attr (empty cache vs rows-outside-horizon -- distinguishable messages);
``predict()``'s skip block WARNs that cause once per unclaimed boundary
(``_last_boundary_run`` deliberately stays unclaimed during an outage, so it
cannot be the latch; a separate boundary latch rate-limits the WARN).
"""

import logging
from types import SimpleNamespace

import pytest

import pandas as pd

soil = pytest.importorskip("sparcs.components.agriculture.simulation.soil")
SoilSimulation = soil.SoilSimulation

soil_predictor = pytest.importorskip("sparcs.components.agriculture.simulation.soil_predictor")
SoilPredictor = soil_predictor.SoilPredictor

_TZ = "Europe/Berlin"


# --- anchor history: wall-clock staleness warning ----------------------------


class _AnchorData:
    """``_anchor_data[key]`` stand-in whose ranged read returns a fixed frame."""

    def __init__(self, frame: pd.DataFrame):
        self.frame = frame

    def read(self, channels, start=None, end=None, unique=True):
        return self.frame


def _empty_read():
    return _AnchorData(pd.DataFrame())


def _tension_read(at: pd.Timestamp, value: float = 300.0):
    return _AnchorData(pd.DataFrame({"tension": [value]}, index=pd.DatetimeIndex([at])))


def _anchor_sim(data: _AnchorData, staleness: str = "6h") -> SoilSimulation:
    sim = object.__new__(SoilSimulation)
    sim._name = "test_soil_simulation"
    sim._anchor_cfg = SimpleNamespace(
        enabled=True,
        sensor_staleness=lambda key: pd.Timedelta(staleness),
    )
    sim._anchor_sensors = [SimpleNamespace(key="s30")]
    sim._anchor_data = {"s30": data}
    sim._anchor_channels = {"s30": SimpleNamespace(key="s30")}
    sim._anchor_history = {}
    sim._last_anchored = {}
    return sim


_START = pd.Timestamp("2026-07-12 10:00", tz="UTC")
_END = pd.Timestamp("2026-07-12 11:00", tz="UTC")


def _stale_warnings(caplog):
    return [r for r in caplog.records if "no readings for" in r.getMessage()]


def test_stale_sensor_warns_once_naming_sensor_and_bound(caplog):
    sim = _anchor_sim(_empty_read())
    sim._anchor_last_read = {"s30": pd.Timestamp.now(tz="UTC") - pd.Timedelta("7h")}

    with caplog.at_level(logging.WARNING):
        sim.load_anchor_history(_START, _END)  # stale: warns
        sim.load_anchor_history(_START, _END)  # still stale: latched, silent

    warnings = _stale_warnings(caplog)
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "s30" in message
    assert warnings[0].levelno == logging.WARNING


def test_recovery_clears_latch_and_later_staleness_rewarns(caplog):
    sim = _anchor_sim(_empty_read())
    sim._anchor_last_read = {"s30": pd.Timestamp.now(tz="UTC") - pd.Timedelta("7h")}

    with caplog.at_level(logging.WARNING):
        sim.load_anchor_history(_START, _END)  # stale: warn #1
        sim._anchor_data["s30"] = _tension_read(_END)
        sim.load_anchor_history(_START, _END)  # non-empty: recovers, clears the latch
        sim._anchor_data["s30"] = _empty_read()
        sim._anchor_last_read["s30"] = pd.Timestamp.now(tz="UTC") - pd.Timedelta("7h")
        sim.load_anchor_history(_START, _END)  # stale again: warn #2

    assert len(_stale_warnings(caplog)) == 2


def test_historical_chunk_bounds_do_not_false_trigger(caplog):
    """Catch-up replays day chunks with historical (start, end); staleness is
    wall-clock-keyed, so a fresh sensor reading an old chunk must stay silent."""
    sim = _anchor_sim(_empty_read())
    sim._anchor_last_read = {"s30": pd.Timestamp.now(tz="UTC")}
    month_ago = pd.Timestamp.now(tz="UTC") - pd.Timedelta("30D")

    with caplog.at_level(logging.WARNING):
        sim.load_anchor_history(month_ago, month_ago + pd.Timedelta("1h"))

    assert _stale_warnings(caplog) == []


def test_first_call_seeds_and_stays_silent(caplog):
    """A dead-from-birth sensor: the first call seeds its wall-clock timestamp
    (age 0) and must not warn yet -- the warning comes once staleness elapses."""
    sim = _anchor_sim(_empty_read())

    with caplog.at_level(logging.WARNING):
        sim.load_anchor_history(_START, _END)

    assert _stale_warnings(caplog) == []
    assert "s30" in sim._anchor_last_read


# --- forecast: empty-cause recorded and warned once per boundary -------------


def _forecast_predictor(to_frame) -> SoilPredictor:
    p = object.__new__(SoilPredictor)
    p._name = "test_predictor"
    p._horizon = pd.Timedelta("24h")
    forecast_sub = SimpleNamespace(
        is_enabled=lambda: True,
        data=SimpleNamespace(to_frame=to_frame),
    )
    p._Registrator__context = SimpleNamespace(weather=SimpleNamespace(forecast=forecast_sub))
    return p


def test_fetch_forecast_records_empty_cache_cause():
    p = _forecast_predictor(lambda unique=False: pd.DataFrame())
    now = pd.Timestamp("2026-07-12 10:00", tz=_TZ)

    assert p._fetch_forecast(now) is None
    assert p._forecast_empty_cause is not None
    assert "empty" in p._forecast_empty_cause


def test_fetch_forecast_records_horizon_gap_cause_distinctly():
    stale_idx = pd.DatetimeIndex([pd.Timestamp("2026-07-10 00:00", tz=_TZ)])
    p = _forecast_predictor(lambda unique=False: pd.DataFrame({"f": [1.0]}, index=stale_idx))
    now = pd.Timestamp("2026-07-12 10:00", tz=_TZ)

    assert p._fetch_forecast(now) is None
    gap_cause = p._forecast_empty_cause
    assert gap_cause is not None

    empty = _forecast_predictor(lambda unique=False: pd.DataFrame())
    assert empty._fetch_forecast(now) is None
    assert empty._forecast_empty_cause != gap_cause  # the two failure shapes are distinguishable


def test_fetch_forecast_disabled_records_no_cause():
    p = object.__new__(SoilPredictor)
    p._name = "test_predictor"
    p._horizon = pd.Timedelta("24h")
    forecast_sub = SimpleNamespace(is_enabled=lambda: False, data=None)
    p._Registrator__context = SimpleNamespace(weather=SimpleNamespace(forecast=forecast_sub))

    assert p._fetch_forecast(pd.Timestamp("2026-07-12 10:00", tz=_TZ)) is None
    assert p._forecast_empty_cause is None


def _skip_predictor() -> SoilPredictor:
    """A bare predictor driven through predict() exactly to the missing-forecast
    skip (test_soil_predictor_predict_wiring precedent); _fetch_forecast is a
    stub that records the cause like the real empty-cache site does."""
    p = object.__new__(SoilPredictor)
    p._name = "test_predictor"
    p._Registrator__context = SimpleNamespace(location=SimpleNamespace(timezone=_TZ))
    p._interval_min = 60
    p._offset_min = 0
    p._last_boundary_run = None
    p._last_predicted_key = None
    p._horizon = pd.Timedelta("24h")

    def _empty_fetch(now):
        p._forecast_empty_cause = "forecast cache is empty (stub)"
        return None

    p._fetch_forecast = _empty_fetch
    return p


def _boundary_warnings(caplog):
    return [r for r in caplog.records if "stalled" in r.getMessage()]


def test_empty_forecast_warns_once_per_boundary_and_keeps_retrying(caplog):
    p = _skip_predictor()

    with caplog.at_level(logging.INFO):
        p.predict(pd.Timestamp("2026-07-12 10:10", tz=_TZ), forecast_creation=None)  # warns
        p.predict(pd.Timestamp("2026-07-12 10:20", tz=_TZ), forecast_creation=None)  # same boundary: silent
        p.predict(pd.Timestamp("2026-07-12 11:10", tz=_TZ), forecast_creation=None)  # next boundary: warns

    warnings = _boundary_warnings(caplog)
    assert len(warnings) == 2
    assert all(r.levelno == logging.WARNING for r in warnings)
    assert "forecast cache is empty (stub)" in warnings[0].getMessage()
    # the skip/retry contract is untouched: the boundary is never claimed
    assert p._last_boundary_run is None
    # the pre-existing INFO skip stays alongside the new WARNING
    assert [r for r in caplog.records if "predict skipped: no forecast rows" in r.getMessage()]


def test_skip_without_recorded_cause_stays_info_only(caplog):
    """A None forecast with NO recorded cause (e.g. forecast disabled, or a
    wholesale-stubbed _fetch_forecast as in the wiring tests): the skip stays
    INFO-only -- the stalled WARNING needs a cause."""
    p = _skip_predictor()
    p._fetch_forecast = lambda now: None  # never sets _forecast_empty_cause

    with caplog.at_level(logging.INFO):
        p.predict(pd.Timestamp("2026-07-12 10:10", tz=_TZ), forecast_creation=None)

    assert _boundary_warnings(caplog) == []
    assert [r for r in caplog.records if "predict skipped: no forecast rows" in r.getMessage()]
