# -*- coding: utf-8 -*-
"""sparcs.tests.test_render_due
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The shared plot-cadence seam: the pure ``render_due`` predicate (adopted by the
soil, ground-shading, and predictor throttle sites) and ``load_plot_config``,
which reads a ``[plot]`` block into a ``PlotConfig`` (or ``None`` when
``enabled = false``). The per-component ``default_interval`` preserves the soil
(``5min``) vs ground-shading / predictor (``1h``) cadence defaults; PNGs persist
only to a DB blob, so there are no filesystem knobs.
"""

import pandas as pd
from sparcs.components.agriculture.simulation import plot_style


class _FakeConfigs:
    """Minimal Configurations stand-in: ``get``/``get_bool`` fall back to the
    caller's default; ``get_member`` returns a nested ``_FakeConfigs`` (an empty
    one when the member is absent, mirroring ``ensure_exists=True``)."""

    def __init__(self, values=None, members=None):
        self._values = values or {}
        self._members = members or {}

    def get(self, key, default=None):
        return self._values.get(key, default)

    def get_bool(self, key, default=False):
        return bool(self._values.get(key, default))

    def get_member(self, key, defaults=None, ensure_exists=False):
        return self._members.get(key, _FakeConfigs())


def _with_plot(**plot_values) -> _FakeConfigs:
    return _FakeConfigs(members={"plot": _FakeConfigs(plot_values)})


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


# load_plot_config -- [plot] gate + per-component interval defaults


def test_load_plot_config_soil_default_interval():
    cfg = plot_style.load_plot_config(_FakeConfigs(), default_interval="5min")
    assert cfg is not None
    assert cfg.interval == pd.Timedelta("5min")


def test_load_plot_config_ground_shading_default_interval():
    cfg = plot_style.load_plot_config(_FakeConfigs(), default_interval="1h")
    assert cfg is not None
    assert cfg.interval == pd.Timedelta("1h")


def test_load_plot_config_disabled_returns_none():
    cfg = plot_style.load_plot_config(_with_plot(enabled=False), default_interval="1h")
    assert cfg is None


def test_load_plot_config_enabled_default_true():
    # No [plot] block at all -> plotting on, code default interval.
    cfg = plot_style.load_plot_config(_FakeConfigs(), default_interval="1h")
    assert cfg is not None


def test_load_plot_config_interval_overrides_default():
    cfg = plot_style.load_plot_config(_with_plot(interval="15min"), default_interval="1h")
    assert cfg is not None
    assert cfg.interval == pd.Timedelta("15min")


def test_load_plot_config_numeric_interval_is_seconds():
    cfg = plot_style.load_plot_config(_with_plot(interval=30), default_interval="1h")
    assert cfg is not None
    assert cfg.interval == pd.Timedelta(seconds=30)
