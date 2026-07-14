# -*- coding: utf-8 -*-
"""sparcs.tests.test_soil_plot_interval
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The end-of-window progress render must honour ``[plot] interval``. The live
tick drives ``simulate_loop``, which calls ``advance`` once per (minute-
resolution) weather row; a previously unconditional final render therefore
emitted one frame per minute regardless of the interval. These tests pin the
interval gate (``_render_progress_if_due``) directly, without a PDE stack:
instances come from ``object.__new__`` and only ``_render_progress`` is stubbed
so the real ``_render_progress_safe`` still advances ``_last_plot_simtime``.
"""

import types

import pytest

import pandas as pd

_soil = pytest.importorskip("sparcs.components.agriculture.simulation.soil")
SoilSimulation = _soil.SoilSimulation


def _sim(interval: str = "1h", *, plot_progress: bool = True) -> SoilSimulation:
    sim = object.__new__(SoilSimulation)
    sim._plot_progress = plot_progress
    sim._plot_config = types.SimpleNamespace(interval=pd.Timedelta(interval))
    sim._last_plot_simtime = None
    sim._rendered = []
    sim._render_progress = sim._rendered.append  # stub the heavy PNG render
    return sim


def test_minutely_advances_render_once_per_interval():
    """simulate_loop's per-row advance calls must collapse to one render/hour."""
    sim = _sim("1h")
    start = pd.Timestamp("2026-07-12 10:00", tz="UTC")
    for i in range(121):  # 10:00 .. 12:00 inclusive, one call per minute
        sim._render_progress_if_due(start + pd.Timedelta(minutes=i))

    assert sim._rendered == [
        pd.Timestamp("2026-07-12 10:00", tz="UTC"),  # first frame always fires
        pd.Timestamp("2026-07-12 11:00", tz="UTC"),
        pd.Timestamp("2026-07-12 12:00", tz="UTC"),
    ]


def test_first_frame_always_renders():
    sim = _sim("1h")
    sim._render_progress_if_due(pd.Timestamp("2026-07-12 10:30", tz="UTC"))
    assert sim._rendered == [pd.Timestamp("2026-07-12 10:30", tz="UTC")]


def test_render_skipped_before_interval_elapses():
    sim = _sim("1h")
    sim._render_progress_if_due(pd.Timestamp("2026-07-12 10:00", tz="UTC"))
    sim._render_progress_if_due(pd.Timestamp("2026-07-12 10:59", tz="UTC"))
    assert sim._rendered == [pd.Timestamp("2026-07-12 10:00", tz="UTC")]


def test_no_render_when_plot_progress_disabled():
    sim = _sim("1h", plot_progress=False)
    sim._render_progress_if_due(pd.Timestamp("2026-07-12 10:00", tz="UTC"))
    assert sim._rendered == []


def test_sub_interval_value_allows_finer_cadence():
    """A smaller interval renders more often -- the knob actually moves."""
    sim = _sim("5min")
    start = pd.Timestamp("2026-07-12 10:00", tz="UTC")
    for i in range(11):  # 10:00 .. 10:10, one call per minute
        sim._render_progress_if_due(start + pd.Timedelta(minutes=i))

    assert sim._rendered == [
        pd.Timestamp("2026-07-12 10:00", tz="UTC"),
        pd.Timestamp("2026-07-12 10:05", tz="UTC"),
        pd.Timestamp("2026-07-12 10:10", tz="UTC"),
    ]
