# -*- coding: utf-8 -*-
"""Regression: the soil_tuning offline anchor backend mirrors the live path.

Exercises ``_anchor_replay_step`` -- the bench's per-step hook -- with a fake PDE
(no FiPy solve) and the worker globals monkeypatched, so the freshness/allowlist/
staleness gates and the set_state write-back are checked without a mesh or logged
data. ``soil_tuning`` pulls in the Dash stack at import; ``importorskip`` keeps
this out of environments that lack it (the full bench runs on the box).
"""

import types

import pytest

import numpy as np
import pandas as pd

soil_tuning = pytest.importorskip("soil_tuning")

from sparcs.components.agriculture.simulation._anchor import AnchorConfig, AnchorSensor  # noqa: E402
from sparcs.components.agriculture.soil.models import Genuchten  # noqa: E402

NOW = pd.Timestamp("2026-05-01 12:00")


class _FakeVar:
    def __init__(self, arr):
        self.value = np.array(arr, dtype=float)


class _FakePDE:
    """Minimal stand-in exposing only what _anchor_replay_step touches."""

    def __init__(self):
        self.rel_sat = _FakeVar([0.6, 0.6])
        self.mesh = types.SimpleNamespace(cellCenters=np.array([[1.5, 1.5], [-0.3, -0.6]]))
        self.soil_model = Genuchten(theta_r=0.0, theta_s=0.43, alpha=0.02, n=1.14, k_s=1e-5)
        self.committed = None

    def set_state(self, arr, update_old=True):
        self.rel_sat.value = np.array(arr, dtype=float)
        self.committed = (np.array(arr, dtype=float), update_old)


def _cfg(staleness="6h", sensors=None):
    return AnchorConfig(
        enabled=True,
        sigma_sys=0.1,
        sigma_meas_pf=0.15,
        r_horizontal=0.5,
        r_vertical=0.5,
        staleness=pd.Timedelta(staleness),
        sensors=sensors if sensors is not None else {"s30": None},
    )


@pytest.fixture
def world(monkeypatch):
    monkeypatch.setattr(soil_tuning, "_W_MESH_CONFIG", types.SimpleNamespace(width=3.0))
    monkeypatch.setattr(soil_tuning, "_W_ANCHOR_SENSORS", [AnchorSensor("s30", 0.0, 30.0)])
    monkeypatch.setattr(soil_tuning, "_W_ANCHOR_CFG", _cfg())
    # Measured tension in the negative matric convention (as _sensor_tension_series emits).
    monkeypatch.setattr(
        soil_tuning,
        "_W_ANCHOR_HISTORY",
        {"s30": pd.Series([-300.0, -310.0], index=[NOW - pd.Timedelta("1h"), NOW - pd.Timedelta("5min")])},
    )


def test_fresh_reading_pulls_field_and_commits(world):
    pde = _FakePDE()
    last = {}
    soil_tuning._anchor_replay_step(pde, NOW, last)
    assert pde.committed is not None and pde.committed[1] is True  # set_state(update_old=True)
    assert not np.isclose(pde.rel_sat.value[0], 0.6)  # cell at the sensor was pulled
    assert last["s30"] == NOW - pd.Timedelta("5min")


def test_not_newer_than_last_is_skipped(world):
    pde = _FakePDE()
    soil_tuning._anchor_replay_step(pde, NOW, {"s30": NOW})
    assert pde.committed is None


def test_sensor_outside_allowlist_is_skipped(world, monkeypatch):
    monkeypatch.setattr(soil_tuning, "_W_ANCHOR_CFG", _cfg(sensors={"other": None}))
    pde = _FakePDE()
    soil_tuning._anchor_replay_step(pde, NOW, {})
    assert pde.committed is None


def test_stale_history_is_skipped(world, monkeypatch):
    monkeypatch.setattr(soil_tuning, "_W_ANCHOR_CFG", _cfg(staleness="10min"))
    monkeypatch.setattr(
        soil_tuning, "_W_ANCHOR_HISTORY", {"s30": pd.Series([-300.0], index=[NOW - pd.Timedelta("2h")])}
    )
    pde = _FakePDE()
    soil_tuning._anchor_replay_step(pde, NOW, {})
    assert pde.committed is None
