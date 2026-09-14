# -*- coding: utf-8 -*-
"""
tests.test_soil_tuning_irrigation_fallback
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The bench replay must water when the field did: with no metered flow in the
logger (unlogged or broken meter), ``_load_history`` falls back to the live
tick's ``FieldSimulation._irrigation_flow_lpm`` chain (meter, else on/off
state x design flow) aligned on the forcing index.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import pandas as pd

soil_tuning = pytest.importorskip("soil_tuning")


class _Data(dict):
    """Minimal DataAccess stand-in: item access + a canned ``from_logger``."""

    def __init__(self, frame: pd.DataFrame, **channels):
        super().__init__(channels)
        self._frame = frame

    def from_logger(self, *args, **kwargs) -> pd.DataFrame:
        return self._frame


def _fixture(index: pd.DatetimeIndex, logged_flow: pd.DataFrame, fallback: pd.Series):
    weather = pd.DataFrame({"ghi": 100.0, "temp_air": 20.0}, index=index)
    et_data = pd.DataFrame({"evapotranspiration": 0.0}, index=index)
    calls: dict = {}

    def _flow_lpm(start, end, idx):
        calls["args"] = (start, end, idx)
        return fallback

    field_sim = SimpleNamespace(
        weather=SimpleNamespace(data=_Data(weather)),
        _run_chain=lambda df, publish=False: (et_data, {}),
        irrigation=SimpleNamespace(data=_Data(logged_flow, flow="flow-channel")),
        _irrigation_flow_lpm=_flow_lpm,
        context=SimpleNamespace(components={}),
    )
    soil_sim = SimpleNamespace(data=_Data(pd.DataFrame(), simulation_state="state-channel"))
    return field_sim, soil_sim, calls


def test_empty_logger_flow_uses_live_fallback_chain(monkeypatch):
    monkeypatch.setattr(soil_tuning.Irrigation, "FLOW", "flow", raising=False)
    monkeypatch.setattr(soil_tuning.SoilSimulation, "SIMULATION_STATE", "simulation_state", raising=False)
    index = pd.date_range("2026-06-09", periods=6, freq="10min", tz="UTC")
    fallback = pd.Series([0.0, 0.53, 0.53, 0.53, 0.0, 0.0], index=index)
    field_sim, soil_sim, calls = _fixture(index, pd.DataFrame(), fallback)

    _, _, irrigation, _, _, _, _ = soil_tuning._load_history(soil_sim, field_sim, index[0], index[-1])

    pd.testing.assert_series_equal(irrigation, fallback)
    assert calls["args"][2].equals(index)


def test_logged_flow_wins_over_fallback(monkeypatch):
    monkeypatch.setattr(soil_tuning.Irrigation, "FLOW", "flow", raising=False)
    monkeypatch.setattr(soil_tuning.SoilSimulation, "SIMULATION_STATE", "simulation_state", raising=False)
    index = pd.date_range("2026-06-09", periods=4, freq="10min", tz="UTC")
    logged = pd.DataFrame({"flow": [0.0, 0.4, None, 0.0]}, index=index)
    fallback = pd.Series(0.53, index=index)
    field_sim, soil_sim, calls = _fixture(index, logged, fallback)

    _, _, irrigation, _, _, _, _ = soil_tuning._load_history(soil_sim, field_sim, index[0], index[-1])

    assert "args" not in calls
    assert irrigation.tolist() == [0.0, 0.4, 0.0, 0.0]


def test_all_zero_fallback_keeps_zero_series(monkeypatch):
    monkeypatch.setattr(soil_tuning.Irrigation, "FLOW", "flow", raising=False)
    monkeypatch.setattr(soil_tuning.SoilSimulation, "SIMULATION_STATE", "simulation_state", raising=False)
    index = pd.date_range("2026-06-09", periods=3, freq="10min", tz="UTC")
    field_sim, soil_sim, _ = _fixture(index, pd.DataFrame(), pd.Series(0.0, index=index))

    _, _, irrigation, _, _, _, _ = soil_tuning._load_history(soil_sim, field_sim, index[0], index[-1])

    assert irrigation.empty or not (irrigation != 0.0).any()
