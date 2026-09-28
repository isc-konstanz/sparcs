# -*- coding: utf-8 -*-
"""
tests.test_fieldsim_shading_zero_irradiance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Regression pins for the zero-irradiance NaN stall (copperhead 2026-07-23): the
Perez transposition is undefined at DHI=0, so a sun-up twilight row with
dni=dhi=0 makes every ground surface's qinc NaN and one such row poisoned the
whole chunk's per-segment GHI mean (NaN is rejected at VALID state in lories
and killed the tick). Lightless rows never reach pvfactors, a stray non-finite
qinc never poisons the segment means, and an all-non-finite segment still
yields a finite value.
"""

import pytest

import numpy as np
import pandas as pd
from lories.components.weather import Weather
from sparcs.components.agriculture.fieldsim.core.shading import ShadingConfig, ShadingModel

_SEGMENT_RANGES = {"seg": (-1.0, 1.0)}


def _weather_frame(rows):
    """Build a shading input frame from (ts, zenith, azimuth, dni, dhi) rows."""
    return pd.DataFrame(
        {
            "solar_zenith": [r[1] for r in rows],
            "solar_azimuth": [r[2] for r in rows],
            Weather.DNI: [r[3] for r in rows],
            Weather.DHI: [r[4] for r in rows],
        },
        index=pd.to_datetime([r[0] for r in rows]),
    )


def _model(mode: str = "free_field") -> ShadingModel:
    config = ShadingConfig.from_dict({"mode": mode, "surface_tilt": 20.0, "surface_azimuth": 90.0})
    config.derive(bay_width=3.4, segment_ranges=_SEGMENT_RANGES)
    return ShadingModel(config)


def test_build_input_drops_lightless_rows():
    df = _weather_frame(
        [
            ("2026-07-22 12:00:00+00:00", 30.0, 180.0, 600.0, 120.0),
            # Sun geometrically up (zenith < 89) but the feed reports zero
            # radiation -- the incident twilight shape.
            ("2026-07-22 19:30:00+00:00", 88.8, 300.0, 0.0, 0.0),
        ]
    )
    out = _model()._build_pvfactors_input(df)
    assert len(out) == 1
    assert out.index[0] == df.index[0]


def test_build_input_keeps_rows_with_any_light():
    df = _weather_frame(
        [
            ("2026-07-22 19:30:00+00:00", 88.8, 300.0, 0.0, 5.0),
            ("2026-07-22 19:45:00+00:00", 88.9, 301.0, 2.0, 0.0),
        ]
    )
    assert len(_model()._build_pvfactors_input(df)) == 2


def test_build_input_all_lightless_returns_empty():
    df = _weather_frame([("2026-07-22 19:30:00+00:00", 88.8, 300.0, 0.0, 0.0)])
    assert _model()._build_pvfactors_input(df).empty


def test_aggregate_skips_non_finite_qinc():
    # Timestep 0: healthy ground with qinc 100; timestep 1: NaN qinc.
    ground_ok = [((-10.0, 0.0), (10.0, 0.0), {"qinc": 100.0})]
    ground_nan = [((-10.0, 0.0), (10.0, 0.0), {"qinc": float("nan")})]
    ghi_open = np.array([500.0, 0.0])

    seg_factors, seg_ghi = _model()._aggregate_per_segment([ground_ok, ground_nan], ghi_open)

    assert np.isfinite(seg_ghi["seg"])
    assert seg_ghi["seg"] == pytest.approx(100.0)
    assert np.isfinite(seg_factors["seg"])


def test_aggregate_all_non_finite_yields_finite_placeholder():
    """NaN would raise ResourceError in lories' Channel.set at VALID state and
    stall the tick; the published placeholder must be finite."""
    ground_nan = [((-10.0, 0.0), (10.0, 0.0), {"qinc": float("nan")})]

    seg_factors, seg_ghi = _model()._aggregate_per_segment([ground_nan], np.array([500.0]))

    assert seg_ghi["seg"] == 0.0
    assert seg_factors["seg"] == 1.0
