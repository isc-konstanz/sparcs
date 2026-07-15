# -*- coding: utf-8 -*-
"""Unit tests for the [anchor] config parser (the live wiring's only pure seam).

Locks the safety-critical default -- anchoring OFF unless explicitly enabled --
and the allowlist parsing. Importing soil.py pulls the FiPy stack, so the full
integration (discovery, advance() hook) is box-verified; this just pins the parse.
"""

import pandas as pd
from sparcs.components.agriculture.simulation._anchor import SensorOverrides
from sparcs.components.agriculture.simulation.soil import _parse_anchor_config


class _Cfg:
    """Stub Configurations exposing get / get_bool and the sub-section accessors.

    ``members`` maps a section name to its sub-config; for ``[anchor.sensors.X]``
    the ``sensors`` member is a dict of ``{sensor_key: _Cfg(overrides)}`` so
    ``get_member("sensors").items()`` mirrors the real nested-section iteration.
    """

    def __init__(self, d, members=None):
        self.d = d
        self._members = members or {}

    def get(self, key, default=None):
        return self.d.get(key, default)

    def get_bool(self, key, default=False):
        return bool(self.d.get(key, default))

    def has_member(self, key):
        return key in self._members

    def get_member(self, key, defaults=None, ensure_exists=False):
        return self._members[key]


def test_anchor_is_off_by_default():
    cfg = _parse_anchor_config(_Cfg({}))
    assert cfg.enabled is False
    assert cfg.sensors == {}
    assert cfg.staleness == pd.Timedelta("6h")


def test_min_tension_floor_default_and_override():
    assert _parse_anchor_config(_Cfg({})).min_tension_hpa == 1.0  # dead-sensor floor on by default
    assert _parse_anchor_config(_Cfg({"min_tension_hpa": 0.0})).min_tension_hpa == 0.0


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
    assert cfg.sensors["soil_3"] is None  # bare list -> all-inherit entries


def test_bare_list_sensor_inherits_every_global():
    cfg = _parse_anchor_config(_Cfg({"enabled": True, "sensors": ["soil_3"]}))
    assert cfg.sensor_sigma("soil_3") == cfg.sigma_meas_pf
    assert cfg.sensor_staleness("soil_3") == cfg.staleness
    assert cfg.sensor_radii("soil_3") == (cfg.r_horizontal, cfg.r_vertical)


def test_per_sensor_subsections_parse_and_inherit():
    cfg = _parse_anchor_config(
        _Cfg(
            {"enabled": True, "sigma_meas_pf": 0.15, "r_horizontal": 0.6, "r_vertical": 0.3, "staleness": "6h"},
            members={
                "sensors": {
                    # soil_3 overrides trust + vertical reach; inherits the rest.
                    "soil_3": _Cfg({"sigma_meas_pf": 0.05, "r_vertical": 0.15}),
                    # soil_4 overrides staleness only.
                    "soil_4": _Cfg({"staleness": "12h"}),
                }
            },
        )
    )
    assert set(cfg.sensors) == {"soil_3", "soil_4"}
    assert isinstance(cfg.sensors["soil_3"], SensorOverrides)

    # soil_3: overridden fields win, omitted fields fall back to the globals.
    assert cfg.sensor_sigma("soil_3") == 0.05
    assert cfg.sensor_radii("soil_3") == (0.6, 0.15)
    assert cfg.sensor_staleness("soil_3") == cfg.staleness

    # soil_4: only staleness overridden; trust and radii inherit.
    assert cfg.sensor_staleness("soil_4") == pd.Timedelta("12h")
    assert cfg.sensor_sigma("soil_4") == 0.15
    assert cfg.sensor_radii("soil_4") == (0.6, 0.3)
