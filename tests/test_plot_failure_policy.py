# -*- coding: utf-8 -*-
"""sparcs.tests.test_plot_failure_policy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Issue 27 (W2.9): three divergent policies for the same render failure --
SoilSimulation and GroundShading disabled plotting permanently on the FIRST
failure (one transient matplotlib error killed plotting until restart on a
single log line), the predictor retried forever. All three now share
``plot_style.count_render_failure``: log + skip this tick's rendering, disable
permanently only after ``PLOT_DISABLE_AFTER`` (3) CONSECUTIVE failures with an
ERROR announcing it, reset on any successful render. Strike counts surface on
a per-component in-memory ``plot_strikes`` channel (logger off); the channel
write is a guarded no-op when data is absent (bare test instances drive the
real render paths).

The predictor's tick contract stays pinned (test_soil_predictor_image_table):
a failing tick still returns None and never sets predict_plot -- only the
retry-forever half changed (deliberate: disable after N failing ticks).
"""

import logging

import pytest

import numpy as np
import pandas as pd

plot_style = pytest.importorskip("sparcs.components.agriculture.simulation.plot_style")
soil = pytest.importorskip("sparcs.components.agriculture.simulation.soil")
SoilSimulation = soil.SoilSimulation

soil_predictor = pytest.importorskip("sparcs.components.agriculture.simulation.soil_predictor")
SoilPredictor = soil_predictor.SoilPredictor

ground_shading = pytest.importorskip("sparcs.components.agriculture.simulation.ground_shading")
GroundShading = ground_shading.GroundShading

_TZ = "Europe/Berlin"


# --- the shared helper --------------------------------------------------------


def test_count_render_failure_disables_at_n(caplog):
    log = logging.getLogger("test.plot_policy")
    strikes = 0
    with caplog.at_level(logging.ERROR, logger="test.plot_policy"):
        try:
            raise RuntimeError("boom")
        except RuntimeError:
            strikes, disable1 = plot_style.count_render_failure(log, "c", strikes)
            assert (strikes, disable1) == (1, False)
        try:
            raise RuntimeError("boom")
        except RuntimeError:
            strikes, disable2 = plot_style.count_render_failure(log, "c", strikes)
            assert (strikes, disable2) == (2, False)
        try:
            raise RuntimeError("boom")
        except RuntimeError:
            strikes, disable3 = plot_style.count_render_failure(log, "c", strikes)
            assert (strikes, disable3) == (3, True)

    assert all(r.levelno == logging.ERROR for r in caplog.records)
    disable_records = [r for r in caplog.records if "disabling" in r.getMessage()]
    assert len(disable_records) == 1  # only the Nth announces the disable


# --- SoilSimulation: N-strikes via _render_progress_safe ----------------------


def _bare_sim(render):
    sim = object.__new__(SoilSimulation)
    sim._name = "test_soil_simulation"
    sim._plot_config = plot_style.PlotConfig(interval=pd.Timedelta(0))
    sim._last_plot_simtime = None
    sim._render_progress = render
    return sim


def test_sim_disables_after_three_consecutive_failures(caplog):
    def _boom(sim_t):
        raise RuntimeError("render failed")

    sim = _bare_sim(_boom)
    t = pd.Timestamp("2026-07-12 10:00", tz="UTC")

    with caplog.at_level(logging.ERROR):
        sim._render_progress_safe(t)
        assert sim._plot_config is not None  # strike 1: still enabled
        sim._render_progress_safe(t)
        assert sim._plot_config is not None  # strike 2: still enabled
        sim._render_progress_safe(t)

    assert sim._plot_config is None  # strike 3: disabled
    assert [r for r in caplog.records if "disabling" in r.getMessage()]


def test_sim_success_resets_the_strikes(caplog):
    calls = {"n": 0}

    def _flaky(sim_t):
        calls["n"] += 1
        if calls["n"] != 3:
            raise RuntimeError("render failed")

    sim = _bare_sim(_flaky)
    t = pd.Timestamp("2026-07-12 10:00", tz="UTC")

    with caplog.at_level(logging.ERROR):
        sim._render_progress_safe(t)  # failure #1
        sim._render_progress_safe(t)  # failure #2
        sim._render_progress_safe(t)  # success: resets
        sim._render_progress_safe(t)  # failure: strike 1 again

    assert sim._plot_config is not None  # never reached 3 consecutive
    assert sim._plot_strikes == 1
    assert not [r for r in caplog.records if "disabling" in r.getMessage()]


def test_sim_strike_channel_records_counts_and_reset(monkeypatch):
    class _Chan:
        def __init__(self):
            self.values = []

        def set(self, ts, value):
            self.values.append(value)

    class _Data:
        def __init__(self):
            self.chan = _Chan()

        def __getitem__(self, key):
            assert key == "plot_strikes"
            return self.chan

    data = _Data()
    monkeypatch.setattr(SoilSimulation, "data", property(lambda self: data))
    calls = {"n": 0}

    def _flaky(sim_t):
        calls["n"] += 1
        if calls["n"] != 3:
            raise RuntimeError("render failed")

    sim = _bare_sim(_flaky)
    t = pd.Timestamp("2026-07-12 10:00", tz="UTC")
    sim._render_progress_safe(t)
    sim._render_progress_safe(t)
    sim._render_progress_safe(t)  # success -> reset to 0

    assert data.chan.values == [1.0, 2.0, 0.0]


# --- GroundShading: same policy through _capture_progress ---------------------


def test_shading_disables_after_three_consecutive_failures(caplog):
    gs = object.__new__(GroundShading)
    gs._name = "test_ground_shading"
    gs._plot_config = plot_style.PlotConfig(interval=pd.Timedelta(0))
    gs._last_plot_ts = None

    def _boom(ts, ground, pv_rows, sun_state):
        raise RuntimeError("render failed")

    gs._render_progress = _boom
    t = pd.Timestamp("2026-07-12 10:00", tz="UTC")

    with caplog.at_level(logging.ERROR):
        gs._capture_progress(t, [], [], (0.0, 0.0, None))
        assert gs._plot_config is not None
        gs._capture_progress(t, [], [], (0.0, 0.0, None))
        assert gs._plot_config is not None
        gs._capture_progress(t, [], [], (0.0, 0.0, None))

    assert gs._plot_config is None
    assert [r for r in caplog.records if "disabling" in r.getMessage()]


def test_shading_success_resets_the_strikes(caplog):
    gs = object.__new__(GroundShading)
    gs._name = "test_ground_shading"
    gs._plot_config = plot_style.PlotConfig(interval=pd.Timedelta(0))
    gs._last_plot_ts = None
    calls = {"n": 0}

    def _flaky(ts, ground, pv_rows, sun_state):
        calls["n"] += 1
        if calls["n"] != 3:
            raise RuntimeError("render failed")

    gs._render_progress = _flaky

    t = pd.Timestamp("2026-07-12 10:00", tz="UTC")
    with caplog.at_level(logging.ERROR):
        gs._capture_progress(t, [], [], (0.0, 0.0, None))  # failure #1
        gs._capture_progress(t, [], [], (0.0, 0.0, None))  # failure #2
        gs._capture_progress(t, [], [], (0.0, 0.0, None))  # success: resets
        gs._capture_progress(t, [], [], (0.0, 0.0, None))  # failure: strike 1 again

    assert gs._plot_config is not None
    assert gs._plot_strikes == 1
    assert not [r for r in caplog.records if "disabling" in r.getMessage()]


# --- SoilPredictor: tick contract preserved + disable after N failing ticks ---


class _RecordingChannel:
    def __init__(self, key):
        self.key = key
        self.calls = []

    def set(self, ts, value):
        self.calls.append((ts, value))


class _FakeData:
    """Known keys record; unknown keys raise (exercises the guarded strike write)."""

    def __init__(self, keys):
        self._channels = {k: _RecordingChannel(k) for k in keys}

    def __getitem__(self, key):
        return self._channels[key]


def _publish_fixture(monkeypatch, render):
    predictor = object.__new__(SoilPredictor)
    predictor._name = "test_predictor"
    predictor._channel_keys = {}
    predictor._save_state = False
    predictor._plot_config = plot_style.PlotConfig(interval=pd.Timedelta("6h"))

    fake_data = _FakeData(
        [SoilPredictor._TIMESTAMP_CREATION_KEY, SoilPredictor._PLOT_CHANNEL_KEY]
        + [c.key for c in soil_predictor._DIAGNOSTIC_CONSTANTS]
    )
    monkeypatch.setattr(SoilPredictor, "data", property(lambda self: fake_data))
    monkeypatch.setattr(SoilPredictor, "_render_snapshot_png", render)
    return predictor, fake_data


def _publish_once(predictor):
    index = pd.date_range("2026-07-03 02:00", periods=2, freq="6h", tz=_TZ)
    snapshots = {index[0]: (np.zeros(3), None), index[1]: (np.zeros(3), None)}
    diagnostics = {c.key: [float("nan"), float("nan")] for c in soil_predictor._DIAGNOSTIC_CONSTANTS}
    return predictor._publish_results([], [], list(index), snapshots, diagnostics, index[0])


def test_predictor_failing_tick_returns_none_and_disables_after_three(monkeypatch, caplog):
    def _boom(self, arr, t, **_k):
        raise RuntimeError("render failed")

    predictor, fake_data = _publish_fixture(monkeypatch, _boom)

    with caplog.at_level(logging.ERROR):
        assert _publish_once(predictor) is None  # failing tick 1
        assert predictor._plot_config is not None
        assert _publish_once(predictor) is None  # failing tick 2
        assert _publish_once(predictor) is None  # failing tick 3 -> disable

    assert predictor._plot_config is None
    assert fake_data[SoilPredictor._PLOT_CHANNEL_KEY].calls == []  # pinned tick contract
    assert [r for r in caplog.records if "disabling" in r.getMessage()]


def test_predictor_successful_tick_resets_strikes(monkeypatch):
    calls = {"n": 0}

    def _flaky(self, arr, t, **_k):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise RuntimeError("render failed")
        return b"png"

    predictor, fake_data = _publish_fixture(monkeypatch, _flaky)

    assert _publish_once(predictor) is None  # failing tick (strike 1; render aborts the tick)
    result = _publish_once(predictor)  # renders fine (call #2 raised? no: n=2 raises)

    # call ordering: tick 1 -> n=1 raises (strike 1); tick 2 -> n=2 raises (strike 2);
    # tick 3 -> n=3,4 succeed -> reset
    assert result is None
    assert predictor._plot_strikes == 2
    assert _publish_once(predictor) is not None
    assert predictor._plot_strikes == 0
    assert predictor._plot_config is not None


def test_predictor_register_plot_strike_channel(monkeypatch):
    added = []

    class _Add:
        def add(self, key, **kwargs):
            added.append((key, kwargs))

    predictor = object.__new__(SoilPredictor)
    predictor._name = "test_predictor"
    monkeypatch.setattr(SoilPredictor, "data", property(lambda self: _Add()))

    predictor._register_plot_strike_channel()

    assert [k for k, _ in added] == ["plot_strikes"]
    assert added[0][1]["logger"] == {"enabled": False}
    assert added[0][1]["aggregate"] == "last"
