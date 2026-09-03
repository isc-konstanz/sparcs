# -*- coding: utf-8 -*-
"""Regression: irrigation must stop when the logged flow stops.

A NULL flow row means "not watering". Before the fix the bench loaded the flow
without zero-filling, and ``_build_flux_rates`` sampled it with ``Series.asof``,
which skips NaN and latched the last nonzero burst forward across the gap -- so
the model kept irrigating after watering had physically stopped (water "kept
coming in"). See ``_irrigation_series_from_frame``.

``soil_tuning`` pulls in the Dash UI stack at import time; ``importorskip``
keeps this out of environments that lack it (the full check runs on the box).
"""

import pytest

import numpy as np
import pandas as pd

soil_tuning = pytest.importorskip("soil_tuning")


def test_null_flow_reads_as_zero():
    """NULL/NaN flow -> 0 L/min (not left as NaN for asof to skip)."""
    ts = pd.date_range("2026-05-01 10:00", periods=5, freq="1min")
    flow_df = pd.DataFrame({"flow": [0.0, 50.0, 50.0, np.nan, np.nan]}, index=ts)

    series = soil_tuning._irrigation_series_from_frame(flow_df)

    assert list(series.values) == [0.0, 50.0, 50.0, 0.0, 0.0]


def test_irrigation_stops_after_watering_stops():
    """The source must drop to zero after a logged stop, not latch the burst."""
    ts = pd.date_range("2026-05-01 10:00", periods=5, freq="1min")
    flow = soil_tuning._irrigation_series_from_frame(
        pd.DataFrame({"flow": [0.0, 50.0, 50.0, np.nan, np.nan]}, index=ts)
    )

    during = soil_tuning._build_flux_rates(ts[1], 60.0, pd.DataFrame(), {}, flow)
    assert during.flow_m3s == pytest.approx(50.0 / 60_000.0)

    after_stop = soil_tuning._build_flux_rates(ts[4], 60.0, pd.DataFrame(), {}, flow)
    assert after_stop.flow_m3s == 0.0
