# -*- coding: utf-8 -*-
"""
tests.test_ground_shading_zero_irradiance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Regression tests for the zero-irradiance NaN stall (copperhead 2026-07-23).

The Perez transposition is undefined at DHI=0 (sky-clearness epsilon is 0/0),
pvlib returns NaN there, and solarfactors' ``poa_sky_diffuse == 0`` luminance
guard misses NaN -- so a sun-up twilight row with dni=dhi=0 makes every ground
surface's qinc NaN. One such row then poisoned the whole day-chunk's
per-segment GHI mean, and the NaN "placeholder" write raised ResourceError in
lories (NaN is rejected at VALID state), killing the tick and stalling the
frontier on the same chunk forever.

Three layers are pinned here: lightless rows never reach pvfactors, a stray
non-finite qinc never poisons the segment means, and the missing-value
placeholder published to SEG_GHI is 0.0 (finite), never NaN.
"""

from types import SimpleNamespace

import pytest

import numpy as np
import pandas as pd

gs = pytest.importorskip("sparcs.components.agriculture.simulation.ground_shading")


def _weather_frame(rows):
    """Build a ground-shading input frame from (ts, zenith, azimuth, dni, dhi) rows."""
    from lories.components.weather import Weather

    idx = pd.to_datetime([r[0] for r in rows])
    return pd.DataFrame(
        {
            "solar_zenith": [r[1] for r in rows],
            "solar_azimuth": [r[2] for r in rows],
            Weather.DNI: [r[3] for r in rows],
            Weather.DHI: [r[4] for r in rows],
        },
        index=idx,
    )


def _as_is_stub():
    """Minimal instance stub for the fixed-tilt _build_pvfactors_input path."""
    return SimpleNamespace(
        _albedo=0.25,
        _mode=gs.MODE_AS_IS,
        _surface_tilt=20.0,
        _surface_azimuth=90.0,
    )


def test_build_input_drops_lightless_rows():
    df = _weather_frame(
        [
            ("2026-07-22 12:00:00+00:00", 30.0, 180.0, 600.0, 120.0),
            # Sun geometrically up (zenith < 89) but the feed reports zero
            # radiation -- the incident twilight shape.
            ("2026-07-22 19:30:00+00:00", 88.8, 300.0, 0.0, 0.0),
        ]
    )
    out = gs.GroundShading._build_pvfactors_input(_as_is_stub(), df)
    assert len(out) == 1
    assert out.index[0] == df.index[0]


def test_build_input_keeps_rows_with_any_light():
    df = _weather_frame(
        [
            ("2026-07-22 19:30:00+00:00", 88.8, 300.0, 0.0, 5.0),
            ("2026-07-22 19:45:00+00:00", 88.9, 301.0, 2.0, 0.0),
        ]
    )
    out = gs.GroundShading._build_pvfactors_input(_as_is_stub(), df)
    assert len(out) == 2


def test_build_input_all_lightless_returns_empty():
    df = _weather_frame([("2026-07-22 19:30:00+00:00", 88.8, 300.0, 0.0, 0.0)])
    out = gs.GroundShading._build_pvfactors_input(_as_is_stub(), df)
    assert out.empty


def test_aggregate_skips_non_finite_qinc():
    stub = SimpleNamespace(_segment_ranges={"seg": (-1.0, 1.0)})
    # Timestep 0: healthy ground with qinc 100; timestep 1: NaN qinc.
    ground_ok = [((-10.0, 0.0), (10.0, 0.0), {"qinc": 100.0})]
    ground_nan = [((-10.0, 0.0), (10.0, 0.0), {"qinc": float("nan")})]
    ghi_open = np.array([500.0, 0.0])

    seg_factors, seg_ghi = gs.GroundShading._aggregate_per_segment(stub, [ground_ok, ground_nan], ghi_open)

    assert np.isfinite(seg_ghi["seg"])
    assert seg_ghi["seg"] == pytest.approx(100.0)
    assert np.isfinite(seg_factors["seg"])


def test_publish_placeholder_is_finite_zero():
    published = {}

    def set_segment_values(channel, ts, mapping):
        published.update(mapping)

    stub = SimpleNamespace(
        name="Ground Shading",
        context=SimpleNamespace(SEG_GHI="seg_ghi", set_segment_values=set_segment_values),
    )
    ts = pd.Timestamp("2026-07-23 00:00:00+00:00")
    gs.GroundShading._publish_per_segment_ghi(stub, ts, {"a": 42.0, "b": float("nan"), "c": None})

    assert published["a"] == pytest.approx(42.0)
    # NaN would raise ResourceError in lories' Channel.set at VALID state and
    # stall the tick; the placeholder must be finite.
    assert published["b"] == 0.0
    assert published["c"] == 0.0
    assert all(np.isfinite(v) for v in published.values())
