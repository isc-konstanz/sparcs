# -*- coding: utf-8 -*-
"""Unit tests for the [anchor] config parser (the live wiring's only pure seam).

Locks the safety-critical default -- anchoring OFF unless explicitly enabled --
and the allowlist parsing. Importing soil.py pulls the FiPy stack, so the full
integration (discovery, advance() hook) is box-verified; this just pins the parse.
"""

import pandas as pd
from sparcs.components.agriculture.simulation.soil import _parse_anchor_config


class _Cfg:
    """Stub Configurations exposing only get / get_bool."""

    def __init__(self, d):
        self.d = d

    def get(self, key, default=None):
        return self.d.get(key, default)

    def get_bool(self, key, default=False):
        return bool(self.d.get(key, default))


def test_anchor_is_off_by_default():
    cfg = _parse_anchor_config(_Cfg({}))
    assert cfg.enabled is False
    assert cfg.sensors == {}
    assert cfg.staleness == pd.Timedelta("6h")


def test_allowlist_and_overrides_parse():
    cfg = _parse_anchor_config(
        _Cfg({"enabled": True, "sensors": ["soil_3", "soil_4"], "sigma_sys": 0.1, "r_vertical": 0.15})
    )
    assert cfg.enabled is True
    assert set(cfg.sensors) == {"soil_3", "soil_4"}
    assert cfg.sigma_sys == 0.1 and cfg.r_vertical == 0.15
    # No per-sensor override yet -> the global pF std applies.
    assert cfg.sensor_sigma("soil_3") == cfg.sigma_meas_pf


def test_sensors_accepts_comma_string():
    cfg = _parse_anchor_config(_Cfg({"enabled": True, "sensors": "soil_3, soil_4"}))
    assert set(cfg.sensors) == {"soil_3", "soil_4"}
