# -*- coding: utf-8 -*-
"""sparcs.tests.test_plot_style_title
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The shared progress-plot title renders the timestamp in the site timezone with
a colon-form offset (``+HH:MM``) when a ``tz`` is given, assuming UTC for naive
timestamps, and carries no ``mode`` suffix any more.
"""

import pandas as pd
from sparcs.components.agriculture.simulation import plot_style


def test_naive_timestamp_has_no_offset():
    ts = pd.Timestamp("2026-07-14 08:46")
    assert plot_style.format_progress_title("Ground shading", ts) == "Ground shading — 2026-07-14 08:46"


def test_utc_timestamp_localized_to_summer_offset():
    ts = pd.Timestamp("2026-07-14 08:46", tz="UTC")
    title = plot_style.format_progress_title("Ground shading", ts, tz="Europe/Berlin")
    assert title == "Ground shading — 2026-07-14 10:46 +02:00"


def test_utc_timestamp_localized_to_winter_offset():
    # DST off in January: Europe/Berlin is +01:00.
    ts = pd.Timestamp("2026-01-14 08:46", tz="UTC")
    title = plot_style.format_progress_title("Relative saturation", ts, tz="Europe/Berlin")
    assert title == "Relative saturation — 2026-01-14 09:46 +01:00"


def test_naive_timestamp_is_assumed_utc_when_tz_given():
    ts = pd.Timestamp("2026-07-14 08:46")  # naive → assumed UTC
    title = plot_style.format_progress_title("Relative saturation", ts, tz="Europe/Berlin")
    assert title == "Relative saturation — 2026-07-14 10:46 +02:00"
