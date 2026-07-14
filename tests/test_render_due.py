# -*- coding: utf-8 -*-
"""sparcs.tests.test_render_due
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The shared plot-cadence seam: the pure ``render_due`` predicate (adopted by the
soil, ground-shading, and predictor throttle sites) and the unified
``PlotConfig`` whose per-component ``default_interval`` preserves the soil
(``5min``) vs ground-shading (``1h``) cadence defaults.
"""

import pandas as pd
from sparcs.components.agriculture.simulation import plot_style


class _FakeConfigs:
    """Minimal Configurations stand-in: returns the caller's default when a key
    is absent, the stored value otherwise (enough for PlotConfig parsing)."""

    def __init__(self, values=None):
        self._values = values or {}

    def get(self, key, default=None):
        return self._values.get(key, default)

    def get_bool(self, key, default=False):
        return bool(self._values.get(key, default))


# render_due -- pure predicate


def test_render_due_first_frame_when_last_is_none():
    now = pd.Timestamp("2026-07-14 10:00", tz="UTC")
    assert plot_style.render_due(None, now, pd.Timedelta("1h")) is True


def test_render_due_true_exactly_at_interval_boundary():
    t0 = pd.Timestamp("2026-07-14 10:00", tz="UTC")
    interval = pd.Timedelta("1h")
    assert plot_style.render_due(t0, t0 + interval, interval) is True


def test_render_due_false_before_interval_elapses():
    t0 = pd.Timestamp("2026-07-14 10:00", tz="UTC")
    interval = pd.Timedelta("1h")
    assert plot_style.render_due(t0, t0 + interval / 2, interval) is False


def test_render_due_true_well_past_interval():
    t0 = pd.Timestamp("2026-07-14 10:00", tz="UTC")
    interval = pd.Timedelta("1h")
    assert plot_style.render_due(t0, t0 + 2 * interval, interval) is True


# PlotConfig -- per-component defaults + parse


def test_plotconfig_soil_default_interval():
    cfg = plot_style.PlotConfig(_FakeConfigs(), default_dir="d", default_interval="5min")
    assert cfg.interval == pd.Timedelta("5min")


def test_plotconfig_ground_shading_default_interval():
    cfg = plot_style.PlotConfig(_FakeConfigs(), default_dir="d", default_interval="1h")
    assert cfg.interval == pd.Timedelta("1h")


def test_plotconfig_numeric_interval_is_seconds():
    cfg = plot_style.PlotConfig(_FakeConfigs({"interval": 30}), default_dir="d", default_interval="1h")
    assert cfg.interval == pd.Timedelta(seconds=30)


def test_plotconfig_string_interval_overrides_default():
    cfg = plot_style.PlotConfig(_FakeConfigs({"interval": "15min"}), default_dir="d", default_interval="1h")
    assert cfg.interval == pd.Timedelta("15min")


def test_plotconfig_dir_and_flag_defaults():
    cfg = plot_style.PlotConfig(_FakeConfigs(), default_dir="out/soil", default_interval="1h")
    assert cfg.dir == "out/soil"
    assert (cfg.live, cfg.save, cfg.show) == (True, False, False)
