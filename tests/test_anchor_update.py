# -*- coding: utf-8 -*-
"""Unit tests for the anchor orchestration layer (the freshness gate + update).

Exercises ``anchor_update`` -- the shared entry both backends call -- with a real
retention model, a synthetic two-cell grid, and a stub ``read_tension`` closure,
so no FiPy mesh or component tree is built. Covers the event-driven cadence the
PRD pins (.scratch/soil-sensor-anchoring/PRD.md): fresh-only, strictly-newer,
within staleness.
"""

import numpy as np
import pandas as pd
from sparcs.components.agriculture.simulation._anchor import (
    AnchorConfig,
    AnchorSensor,
    SensorOverrides,
    anchor_update,
)
from sparcs.components.agriculture.soil.models import Genuchten

MODEL = Genuchten(theta_r=0.0, theta_s=0.43, alpha=0.02, n=1.14, k_s=1e-5)
# Two cells on the bay-center axis at 30 cm and 60 cm depth (mesh y -0.3 / -0.6).
CELLS = np.array([[1.5, 1.5], [-0.3, -0.6]])
SE_MIN, SE_MAX = 1e-6, 0.999
WIDTH_M = 3.0
NOW = pd.Timestamp("2026-05-01 12:00")
# 30 cm sensor -> mesh (1.5, -0.3) -> nearest cell 0.
S30 = AnchorSensor(key="s30", x_offset_cm=0.0, depth_cm=30.0)
S60 = AnchorSensor(key="s60", x_offset_cm=0.0, depth_cm=60.0)


def _cfg(staleness="1h", sensors=None):
    return AnchorConfig(
        enabled=True,
        sigma_sys=0.1,
        sigma_meas_pf=0.1,
        r_horizontal=0.5,
        r_vertical=0.5,
        staleness=pd.Timedelta(staleness),
        sensors=sensors if sensors is not None else {"s30": None, "s60": None},
    )


def _reader(table):
    """table: key -> (timestamp, tension_hpa). Missing keys read as (None, nan)."""

    def read(sensor):
        return table.get(sensor.key, (None, float("nan")))

    return read


def _run(table, sensors=(S30,), last=None, cfg=None):
    return anchor_update(
        np.array([0.6, 0.6]),
        CELLS,
        sensors,
        _reader(table),
        NOW,
        cfg or _cfg(),
        MODEL,
        WIDTH_M,
        last if last is not None else {},
        SE_MIN,
        SE_MAX,
    )


def test_fresh_reading_anchors_and_reports():
    res = _run({"s30": (NOW, 300.0)})
    assert res is not None
    assert set(res.anchored_at) == {"s30"} and res.anchored_at["s30"] == NOW
    assert not np.isclose(res.se_new[0], 0.6)  # cell at the sensor was pulled
    assert np.all(res.se_new >= SE_MIN) and np.all(res.se_new <= SE_MAX)
    assert np.isclose(res.innovations["s30"], MODEL.se_from_psi(300.0) - 0.6)


def test_stale_reading_is_skipped():
    res = _run({"s30": (NOW - pd.Timedelta("2h"), 300.0)})  # staleness is 1h
    assert res is None


def test_reading_not_newer_than_last_is_skipped():
    res = _run({"s30": (NOW, 300.0)}, last={"s30": NOW})  # not strictly newer
    assert res is None


def test_reading_newer_than_last_anchors():
    res = _run({"s30": (NOW, 300.0)}, last={"s30": NOW - pd.Timedelta("10min")})
    assert res is not None and res.anchored_at["s30"] == NOW


def test_nan_and_missing_readings_are_skipped():
    assert _run({"s30": (NOW, float("nan"))}) is None
    assert _run({"s30": (None, 300.0)}) is None
    assert _run({}) is None  # no reading at all


def test_only_fresh_sensor_of_several_is_anchored():
    table = {"s30": (NOW, 300.0), "s60": (NOW - pd.Timedelta("3h"), 200.0)}
    res = _run(table, sensors=(S30, S60))
    assert set(res.anchored_at) == {"s30"}  # s60 stale, dropped


def test_last_anchored_is_not_mutated():
    last = {"s30": NOW - pd.Timedelta("10min")}
    snapshot = dict(last)
    _run({"s30": (NOW, 300.0)}, last=last)
    assert last == snapshot  # caller merges anchored_at itself, only on commit


def test_per_sensor_pf_override_changes_trust():
    """A tighter per-sensor pF std pulls the field harder (lower variance)."""
    base = _run({"s30": (NOW, 300.0)}, cfg=_cfg(sensors={"s30": None}))
    tight = _run({"s30": (NOW, 300.0)}, cfg=_cfg(sensors={"s30": SensorOverrides(sigma_meas_pf=0.01)}))
    # Both pull toward the same reading; the tighter sensor moves cell 0 further.
    pulled_base = abs(base.se_new[0] - 0.6)
    pulled_tight = abs(tight.se_new[0] - 0.6)
    assert pulled_tight > pulled_base


def test_per_sensor_radii_override_changes_reach():
    """A per-sensor vertical reach lets a 30 cm sensor touch the 60 cm cell (or not)."""
    # Cells sit 30 cm apart in depth; the sensor is at cell 0 (30 cm).
    narrow = _run({"s30": (NOW, 300.0)}, cfg=_cfg(sensors={"s30": SensorOverrides(r_vertical=0.1)}))
    wide = _run({"s30": (NOW, 300.0)}, cfg=_cfg(sensors={"s30": SensorOverrides(r_vertical=0.5)}))
    assert np.isclose(narrow.se_new[1], 0.6)  # 30 cm below is outside r_v = 0.1
    assert not np.isclose(wide.se_new[1], 0.6)  # inside r_v = 0.5 -> pulled


def test_per_sensor_staleness_override_gates_independently():
    """A sensor with its own longer staleness anchors where the global would drop it."""
    reading = {"s30": (NOW - pd.Timedelta("3h"), 300.0)}  # 3h old
    dropped = _run(reading, cfg=_cfg(staleness="1h", sensors={"s30": None}))
    kept = _run(reading, cfg=_cfg(staleness="1h", sensors={"s30": SensorOverrides(staleness=pd.Timedelta("6h"))}))
    assert dropped is None  # 3h > global 1h
    assert kept is not None and kept.anchored_at["s30"] == NOW - pd.Timedelta("3h")
