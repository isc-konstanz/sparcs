# -*- coding: utf-8 -*-
"""
tests.test_fieldsim_assimilator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Assimilator`` over a synthetic two-cell field: ``parse_anchor_config``
parity with the live parser, ``ingest``/``update`` blending fresh tensiometer
readings into a ``SoilState``, staleness and disabled/no-sensor no-ops, and
equivalence with the live ``simulation._anchor.anchor_update`` on the same
inputs. Pure numpy; no FiPy, no Gmsh.
"""

import datetime as dt
import types

import pytest

import numpy as np
import pandas as pd
from sparcs.components.agriculture.fieldsim.core.anchor import AnchorConfig, AnchorSensor
from sparcs.components.agriculture.fieldsim.core.assimilator import Assimilator, parse_anchor_config
from sparcs.components.agriculture.fieldsim.core.state import SoilState

UTC = dt.timezone.utc
T0 = dt.datetime(2026, 9, 25, 12, 0, tzinfo=UTC)


class _Cfg:
    """Stub Configurations exposing get / get_bool and the sub-section accessors,
    mirroring ``tests/test_anchor_config_parse.py``'s live-parser fixture."""

    def __init__(self, d, members=None):
        self.d = d
        self._members = members or {}

    def __bool__(self):
        return True

    def get(self, key, default=None):
        return self.d.get(key, default)

    def get_bool(self, key, default=False):
        return bool(self.d.get(key, default))

    def has_member(self, key):
        return key in self._members

    def get_member(self, key, defaults=None, ensure_exists=False):
        return self._members[key]


class _FakeModel:
    """Simple monotone Se<->psi inverse pair, plus a constant retention slope."""

    def se_from_psi(self, psi_hpa: float) -> float:
        return float(np.clip(1.0 + 0.002 * psi_hpa, 0.01, 0.99))

    def psi_from_se(self, se: float) -> float:
        return (float(se) - 1.0) / 0.002

    def dh_dse(self, se: float) -> float:
        return 500.0


def _fake_engine(cell_centers, width=3.0, se_bounds=(0.01, 0.99)):
    return types.SimpleNamespace(
        cell_centers=np.asarray(cell_centers, dtype=float),
        model=_FakeModel(),
        mesh_config=types.SimpleNamespace(width=width),
        se_bounds=se_bounds,
    )


def _cfg(**overrides) -> AnchorConfig:
    values = dict(
        enabled=True,
        sigma_sys=0.1,
        sigma_meas_pf=0.15,
        r_horizontal=0.5,
        r_vertical=0.5,
        staleness=pd.Timedelta("6h"),
        sensors={"s1": None},
    )
    values.update(overrides)
    return AnchorConfig(**values)


# --------------------------------------------------------------------------- parse_anchor_config


def test_parse_anchor_config_off_by_default_on_empty_dict():
    cfg = parse_anchor_config({})
    assert cfg.enabled is False
    assert cfg.sensors == {}
    assert cfg.staleness == pd.Timedelta("6h")


def test_parse_anchor_config_off_by_default_on_none():
    cfg = parse_anchor_config(None)
    assert cfg.enabled is False
    assert cfg.sensors == {}


def test_parse_anchor_config_allowlist_and_overrides():
    cfg = parse_anchor_config(
        _Cfg({"enabled": True, "sensors": ["soil_3", "soil_4"], "sigma_sys": 0.1, "r_vertical": 0.15})
    )
    assert cfg.enabled is True
    assert set(cfg.sensors) == {"soil_3", "soil_4"}
    assert cfg.sigma_sys == 0.1 and cfg.r_vertical == 0.15
    assert cfg.sensor_sigma("soil_3") == cfg.sigma_meas_pf


def test_parse_anchor_config_sensors_accepts_comma_string():
    cfg = parse_anchor_config(_Cfg({"enabled": True, "sensors": "soil_3, soil_4"}))
    assert set(cfg.sensors) == {"soil_3", "soil_4"}
    assert cfg.sensors["soil_3"] is None


def test_parse_anchor_config_per_sensor_subsections_parse_and_inherit():
    cfg = parse_anchor_config(
        _Cfg(
            {"enabled": True, "sigma_meas_pf": 0.15, "r_horizontal": 0.6, "r_vertical": 0.3, "staleness": "6h"},
            members={
                "sensors": {
                    "soil_3": _Cfg({"sigma_meas_pf": 0.05, "r_vertical": 0.15}),
                    "soil_4": _Cfg({"staleness": "12h"}),
                }
            },
        )
    )
    assert set(cfg.sensors) == {"soil_3", "soil_4"}
    assert cfg.sensor_sigma("soil_3") == 0.05
    assert cfg.sensor_radii("soil_3") == (0.6, 0.15)
    assert cfg.sensor_staleness("soil_4") == pd.Timedelta("12h")
    assert cfg.sensor_sigma("soil_4") == 0.15


# --------------------------------------------------------------------------- Assimilator.update


def _near_far_setup():
    # Sensor at bay-centered x_offset 0, depth 30 cm, width 3.0 m -> mesh (1.5, -0.3).
    # Cell 0 sits at the sensor; cell 1 is 1.5 m away in x, outside r_h = 0.5.
    engine = _fake_engine(cell_centers=[[1.5, 0.0], [-0.3, -0.3]])
    sensor = AnchorSensor(key="s1", x_offset_cm=0.0, depth_cm=30.0)
    state = SoilState(se=np.array([0.5, 0.5]), se_old=np.array([0.5, 0.5]), surface_h={"top": 0.0}, at=T0)
    return engine, sensor, state


def test_update_moves_nearer_cell_toward_observation_and_sets_last_anchored():
    engine, sensor, state = _near_far_setup()
    assimilator = Assimilator(_cfg(), engine)
    assimilator.set_sensors([sensor])
    # tension -50 hPa -> se_meas = 1 + 0.002*(-50) = 0.9, above the field's 0.5.
    assimilator.ingest({"s1": pd.Series([-50.0], index=[T0])})

    new_state = assimilator.update(state, T0)

    assert new_state is not state
    assert new_state.se[0] > 0.5
    assert new_state.se[1] == pytest.approx(0.5)
    assert np.array_equal(new_state.se, new_state.se_old)
    assert new_state.surface_h == {"top": 0.0}
    assert new_state.at == T0
    assert assimilator.last_anchored["s1"] == T0
    assert assimilator.last_result is not None


def test_update_stale_reading_is_a_noop():
    engine, sensor, state = _near_far_setup()
    assimilator = Assimilator(_cfg(staleness=pd.Timedelta("6h")), engine)
    assimilator.set_sensors([sensor])
    old_ts = T0 - dt.timedelta(hours=1)
    assimilator.ingest({"s1": pd.Series([-50.0], index=[old_ts])})

    now = T0 + dt.timedelta(hours=10)  # now - old_ts = 11h > 6h staleness
    new_state = assimilator.update(state, now)

    assert new_state is state
    assert assimilator.last_anchored == {}


def test_update_disabled_config_is_a_noop():
    engine, sensor, state = _near_far_setup()
    assimilator = Assimilator(_cfg(enabled=False), engine)
    assimilator.set_sensors([sensor])
    assimilator.ingest({"s1": pd.Series([-50.0], index=[T0])})

    assert assimilator.enabled is False
    assert assimilator.update(state, T0) is state


def test_update_no_sensors_is_a_noop():
    engine, _, state = _near_far_setup()
    assimilator = Assimilator(_cfg(), engine)  # no set_sensors call

    assert assimilator.enabled is False
    assert assimilator.update(state, T0) is state


# --------------------------------------------------------------------------- ingest


def test_ingest_empty_series_warns_once_and_keeps_previous(caplog):
    engine, sensor, _ = _near_far_setup()
    assimilator = Assimilator(_cfg(), engine)
    assimilator.set_sensors([sensor])
    assimilator.ingest({"s1": pd.Series([-50.0], index=[T0])})
    kept = assimilator.history["s1"]

    with caplog.at_level("WARNING"):
        assimilator.ingest({"s1": pd.Series([], dtype=float)})
        assimilator.ingest({"s1": pd.Series([], dtype=float)})

    assert assimilator.history["s1"] is kept  # unchanged
    warnings = [r for r in caplog.records if r.levelname == "WARNING" and "s1" in r.getMessage()]
    assert len(warnings) == 1  # latched, warned only once

    # A fresh non-empty read clears the latch; the next empty read warns again.
    with caplog.at_level("WARNING"):
        assimilator.ingest({"s1": pd.Series([-40.0], index=[T0])})
        assimilator.ingest({"s1": pd.Series([], dtype=float)})
    assert len([r for r in caplog.records if r.levelname == "WARNING" and "s1" in r.getMessage()]) == 2


# --------------------------------------------------------------------------- equivalence with the live module


def test_matches_live_anchor_update():
    live = pytest.importorskip("sparcs.components.agriculture.fieldsim.core.anchor")

    engine, sensor, state = _near_far_setup()
    cfg = _cfg()
    assimilator = Assimilator(cfg, engine)
    assimilator.set_sensors([sensor])
    history = {"s1": pd.Series([-50.0], index=[T0])}
    assimilator.ingest(history)

    got = assimilator.update(state, T0)

    live_result = live.anchor_update(
        state.se,
        engine.cell_centers,
        [sensor],
        lambda s: live.latest_reading_at(history.get(s.key), T0),
        T0,
        cfg,
        engine.model,
        engine.mesh_config.width,
        {},
        engine.se_bounds[0],
        engine.se_bounds[1],
    )

    assert live_result is not None
    assert np.array_equal(got.se, live_result.se_new)
