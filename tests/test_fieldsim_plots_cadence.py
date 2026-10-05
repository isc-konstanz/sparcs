# -*- coding: utf-8 -*-
"""
tests.test_fieldsim_plots_cadence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``render_due`` collapses per-minute forcing rows to one frame per ``[plot] interval``.
The first frame always fires; a ``None`` config renders nothing.
"""

import pandas as pd
from sparcs.components.agriculture.simulation.core.config import PlotConfig
from sparcs.components.agriculture.simulation.core.plots import render_due


def _rendered(timestamps, config) -> list:
    """The frames a caller gating on ``render_due`` would emit."""
    last = None
    out = []
    for ts in timestamps:
        if render_due(last, ts, config):
            out.append(ts)
            last = ts
    return out


def _minutely(start: str, count: int) -> list:
    t0 = pd.Timestamp(start, tz="UTC")
    return [t0 + pd.Timedelta(minutes=i) for i in range(count)]


def test_render_due_first_frame_when_last_is_none():
    now = pd.Timestamp("2026-07-14 10:00", tz="UTC")
    assert render_due(None, now, PlotConfig.from_dict({"interval": "1h"})) is True


def test_render_due_true_exactly_at_interval_boundary():
    config = PlotConfig.from_dict({"interval": "1h"})
    t0 = pd.Timestamp("2026-07-14 10:00", tz="UTC")
    assert render_due(t0, t0 + config.interval, config) is True


def test_render_due_false_before_interval_elapses():
    config = PlotConfig.from_dict({"interval": "1h"})
    t0 = pd.Timestamp("2026-07-14 10:00", tz="UTC")
    assert render_due(t0, t0 + config.interval / 2, config) is False


def test_render_due_true_well_past_interval():
    config = PlotConfig.from_dict({"interval": "1h"})
    t0 = pd.Timestamp("2026-07-14 10:00", tz="UTC")
    assert render_due(t0, t0 + 2 * config.interval, config) is True


def test_minutely_rows_render_once_per_interval():
    config = PlotConfig.from_dict({"interval": "1h"})

    assert _rendered(_minutely("2026-07-12 10:00", 121), config) == [
        pd.Timestamp("2026-07-12 10:00", tz="UTC"),  # first frame always fires
        pd.Timestamp("2026-07-12 11:00", tz="UTC"),
        pd.Timestamp("2026-07-12 12:00", tz="UTC"),
    ]


def test_no_render_when_plotting_disabled():
    assert _rendered(_minutely("2026-07-12 10:00", 121), None) == []


def test_sub_interval_value_allows_finer_cadence():
    """A smaller interval renders more often."""
    config = PlotConfig.from_dict({"interval": "5min"})

    assert _rendered(_minutely("2026-07-12 10:00", 11), config) == [
        pd.Timestamp("2026-07-12 10:00", tz="UTC"),
        pd.Timestamp("2026-07-12 10:05", tz="UTC"),
        pd.Timestamp("2026-07-12 10:10", tz="UTC"),
    ]
