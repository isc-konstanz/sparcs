# -*- coding: utf-8 -*-
"""Regression: ``SoilSimulation._read_live_tension`` must gate on
``Channel.is_valid()`` -- a method. The pre-fix attribute test
(``not channel.is_valid``) was always falsy on a bound method, so a
never-written channel fell through to ``float(None)`` and crashed the
anchor step inside ``advance()``.
"""

import math
import types

import pandas as pd
from sparcs.components.agriculture.simulation._anchor import AnchorSensor
from sparcs.components.agriculture.simulation.soil import SoilSimulation


def _fake_sim(channels: dict) -> types.SimpleNamespace:
    return types.SimpleNamespace(_anchor_channels=channels)


def _sensor(key: str = "s1") -> AnchorSensor:
    return AnchorSensor(key=key, x_offset_cm=0.0, depth_cm=20.0)


def test_invalid_channel_reads_as_missing():
    channel = types.SimpleNamespace(is_valid=lambda: False, timestamp=pd.NaT, value=None)
    ts, tension = SoilSimulation._read_live_tension(_fake_sim({"s1": channel}), _sensor())
    assert ts is None
    assert math.isnan(tension)


def test_unknown_sensor_reads_as_missing():
    ts, tension = SoilSimulation._read_live_tension(_fake_sim({}), _sensor())
    assert ts is None
    assert math.isnan(tension)


def test_valid_channel_returns_timestamp_and_value():
    now = pd.Timestamp("2026-07-08 12:00", tz="UTC")
    channel = types.SimpleNamespace(is_valid=lambda: True, timestamp=now, value=-123.0)
    ts, tension = SoilSimulation._read_live_tension(_fake_sim({"s1": channel}), _sensor())
    assert ts == now
    assert tension == -123.0
