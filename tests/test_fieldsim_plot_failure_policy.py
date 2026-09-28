# -*- coding: utf-8 -*-
"""
tests.test_fieldsim_plot_failure_policy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The progress-image render policy at the ``ChannelOutputs`` seams, for the
shading frame (``chain``) and the soil frame (``steps``) alike: a failing
render is logged with its traceback and skipped, ``disable_after_failures``
consecutive failures switch the child's plotting off for the rest of the
process with one ERROR announcing it, a successful render resets the count,
every count reaches the child's in-memory ``plot_strikes`` channel, a failing
render never writes a frame, a failing image write is a failed render, a
failing ``plot_strikes`` write never escapes, and ``[plot] interval`` collapses
chunks to one frame per interval. Fake renderers stand in for matplotlib; the
real ones run in ``test_fieldsim_plots_wiring``.
"""

import logging
from types import SimpleNamespace

import pytest

import numpy as np
import pandas as pd
from sparcs.components.agriculture.fieldsim import components
from sparcs.components.agriculture.fieldsim.components import ChannelOutputs
from sparcs.components.agriculture.fieldsim.core.config import PlotConfig
from sparcs.components.agriculture.fieldsim.core.plots import ShadingEnvelope
from sparcs.components.agriculture.fieldsim.core.state import ChainResult, SoilState, StepResult

T0 = pd.Timestamp("2026-07-12 10:00", tz="UTC")


class _Channel:
    def __init__(self):
        self.calls = []

    def set(self, ts, value):
        self.calls.append((ts, value))


class _BrokenChannel(_Channel):
    def set(self, ts, value):
        raise RuntimeError("channel write failed")


class _Child:
    """A ``GroundShading``/``SoilSimulation`` with plotting on and, at interval 0, every frame due."""

    def __init__(self, name: str, image_key: str):
        self.name = name
        self.plot_config = PlotConfig.from_dict({"interval": "0min"})
        self._last_plot_ts = None
        self._plot_strikes = 0
        self.data = {image_key: _Channel(), "plot_strikes": _Channel(), "shading_factor": _Channel()}
        self.image_key = image_key
        self.image = self.data[image_key]
        self.strikes = self.data["plot_strikes"]


def _field():
    return SimpleNamespace(
        setup=SimpleNamespace(location=None, soil=SimpleNamespace(mesh=SimpleNamespace(width=3.0, height=1.5))),
        simulation=SimpleNamespace(engine=SimpleNamespace(top_segment_names=[], mesh=object())),
    )


def _chain_result(ts) -> ChainResult:
    return ChainResult(
        shading=pd.DataFrame({"shading_factor": [1.0]}, index=[ts]),
        evapotranspiration=pd.DataFrame(),
        envelope=ShadingEnvelope(x_half=5.25, y_min=-2.0, y_max=1.0),
    )


def _step_result(ts) -> StepResult:
    state = SoilState(se=np.full(3, 0.5), se_old=np.full(3, 0.5), surface_h={}, at=ts.to_pydatetime())
    return StepResult(state=state, diagnostics={})


class _Seam:
    """One image kind: ``render_with`` installs the fake renderer, ``emit`` drives a chunk."""

    def __init__(self, kind: str, monkeypatch):
        self.kind = kind
        self.monkeypatch = monkeypatch
        if kind == "shading":
            self.child = _Child("ground_shading", "shading_progress_image")
            self.outputs = ChannelOutputs(_field(), shading=self.child, et=None, soil=None, predictor=None)
        else:
            self.child = _Child("soil_simulation", "soil_progress_image")
            self.outputs = ChannelOutputs(_field(), shading=None, et=None, soil=self.child, predictor=None)

    def render_with(self, render) -> None:
        name = "render_shading_png" if self.kind == "shading" else "render_rel_sat_png"
        self.monkeypatch.setattr(components.plots, name, render)

    def emit(self, ts) -> None:
        if self.kind == "shading":
            self.outputs.chain(ts, _chain_result(ts))
        else:
            self.outputs.steps([_step_result(ts)])


@pytest.fixture(params=["shading", "soil"])
def seam(request, monkeypatch) -> _Seam:
    return _Seam(request.param, monkeypatch)


def _boom(*args, **kwargs):
    raise RuntimeError("render failed")


def _flaky(succeed_on: int):
    """Raises on every call but the ``succeed_on``-th."""
    calls = {"n": 0}

    def render(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] != succeed_on:
            raise RuntimeError("render failed")
        return b"png"

    return render


def _counting():
    calls = {"n": 0}

    def render(*args, **kwargs):
        calls["n"] += 1
        return b"png"

    return render, calls


def _disabling(caplog) -> list:
    return [r for r in caplog.records if "disabling" in r.getMessage()]


# --------------------------------------------------------------------------- strikes


def test_disables_after_three_consecutive_failures(seam, caplog):
    seam.render_with(_boom)

    with caplog.at_level(logging.ERROR):
        seam.emit(T0)
        assert seam.child.plot_config is not None  # strike 1: still enabled
        seam.emit(T0)
        assert seam.child.plot_config is not None  # strike 2: still enabled
        seam.emit(T0)

    assert seam.child.plot_config is None  # strike 3: disabled
    assert len(_disabling(caplog)) == 1  # only the Nth announces the disable
    assert seam.child.image.calls == []


def test_success_resets_the_strikes(seam, caplog):
    seam.render_with(_flaky(succeed_on=3))

    with caplog.at_level(logging.ERROR):
        seam.emit(T0)  # failure #1
        seam.emit(T0)  # failure #2
        seam.emit(T0)  # success: resets
        seam.emit(T0)  # failure: strike 1 again

    assert seam.child.plot_config is not None  # never reached 3 consecutive
    assert seam.child._plot_strikes == 1
    assert not _disabling(caplog)


def test_strike_channel_records_counts_and_reset(seam):
    seam.render_with(_flaky(succeed_on=3))

    seam.emit(T0)
    seam.emit(T0)
    seam.emit(T0)  # success -> reset to 0

    assert [value for _, value in seam.child.strikes.calls] == [1.0, 2.0, 0.0]


def test_a_failing_render_never_writes_a_frame(seam):
    seam.render_with(_flaky(succeed_on=2))

    seam.emit(T0)
    assert seam.child.image.calls == []
    seam.emit(T0)
    assert seam.child.image.calls == [(T0, b"png")]


def test_every_failure_is_an_error_with_the_traceback(seam, caplog):
    seam.render_with(_boom)

    with caplog.at_level(logging.ERROR):
        seam.emit(T0)

    [record] = [r for r in caplog.records if "render failed" in r.getMessage()]
    assert record.levelno == logging.ERROR
    assert record.exc_info is not None
    assert record.getMessage().startswith(f"{seam.child.name} progress image")


def test_disable_after_failures_is_the_childs_own_knob(seam, caplog):
    seam.child.plot_config = PlotConfig.from_dict({"interval": "0min", "disable_after_failures": 1})
    seam.render_with(_boom)

    with caplog.at_level(logging.ERROR):
        seam.emit(T0)

    assert seam.child.plot_config is None
    assert len(_disabling(caplog)) == 1


def test_a_disabled_child_renders_nothing(seam):
    render, calls = _counting()
    seam.render_with(render)
    seam.child.plot_config = None

    seam.emit(T0)

    assert calls["n"] == 0
    assert seam.child.image.calls == []
    assert seam.child.strikes.calls == []


def test_soil_rows_of_one_chunk_strike_out_and_the_rest_are_skipped(monkeypatch):
    seam = _Seam("soil", monkeypatch)
    calls = {"n": 0}

    def render(*args, **kwargs):
        calls["n"] += 1
        raise RuntimeError("render failed")

    seam.render_with(render)

    seam.outputs.steps([_step_result(T0 + pd.Timedelta(minutes=i)) for i in range(5)])

    assert calls["n"] == 3
    assert seam.child.plot_config is None
    assert seam.child.image.calls == []


# --------------------------------------------------------------------------- channel writes


def test_an_image_write_failure_is_a_render_failure(seam, caplog):
    render, calls = _counting()
    seam.render_with(render)
    seam.child.data[seam.child.image_key] = _BrokenChannel()

    with caplog.at_level(logging.ERROR):
        seam.emit(T0)
        assert seam.child.plot_config is not None
        seam.emit(T0)
        seam.emit(T0)

    assert calls["n"] == 3
    assert seam.child.plot_config is None
    assert [value for _, value in seam.child.strikes.calls] == [1.0, 2.0, 3.0]
    assert len(_disabling(caplog)) == 1


def test_a_strike_channel_failure_never_escapes(seam):
    seam.child.data["plot_strikes"] = _BrokenChannel()
    seam.render_with(_flaky(succeed_on=2))

    seam.emit(T0)
    assert seam.child._plot_strikes == 1
    seam.emit(T0)
    assert seam.child._plot_strikes == 0
    assert seam.child.image.calls == [(T0, b"png")]


# --------------------------------------------------------------------------- cadence


def test_two_chunks_within_the_interval_render_once(seam):
    seam.child.plot_config = PlotConfig.from_dict({"interval": "1h"})
    render, calls = _counting()
    seam.render_with(render)

    seam.emit(T0)
    seam.emit(T0 + pd.Timedelta(minutes=30))

    assert calls["n"] == 1
    assert seam.child.image.calls == [(T0, b"png")]
    assert seam.child._last_plot_ts == T0

    seam.emit(T0 + pd.Timedelta(hours=1))

    assert calls["n"] == 2
    assert seam.child._last_plot_ts == T0 + pd.Timedelta(hours=1)


def test_a_failed_frame_counts_toward_the_cadence(seam):
    """The failed slot is skipped, not retried on the next row; it comes back after the interval."""
    seam.child.plot_config = PlotConfig.from_dict({"interval": "1h"})
    calls = {"n": 0}

    def render(*args, **kwargs):
        calls["n"] += 1
        raise RuntimeError("render failed")

    seam.render_with(render)

    seam.emit(T0)
    seam.emit(T0 + pd.Timedelta(minutes=30))
    assert calls["n"] == 1
    seam.emit(T0 + pd.Timedelta(hours=1))
    assert calls["n"] == 2
    assert seam.child._plot_strikes == 2
