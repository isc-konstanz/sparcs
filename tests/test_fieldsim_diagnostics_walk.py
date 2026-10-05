# -*- coding: utf-8 -*-
"""
tests.test_fieldsim_diagnostics_walk
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The adaptive walk's ``skipped_s`` (seconds held at ``dt_min``) and ``retries`` (substep rollbacks) reach the
diagnostics every step, 0.0 when nothing happened; only a non-zero ``skipped_s`` logs an ERROR.
"""

import datetime as dt
import logging
import types

import numpy as np
from sparcs.components.agriculture.simulation.core.engine import SoilEngine
from sparcs.components.agriculture.simulation.core.pde import WalkResult
from sparcs.components.agriculture.simulation.core.state import Forcing, SoilState

UTC = dt.timezone.utc

_WINDOW_S = 600.0
_AT = dt.datetime(2026, 7, 16, 10, 0, tzinfo=UTC)


def _engine(walk: WalkResult) -> SoilEngine:
    """An engine over a stub core: ``walk_window`` returns a fixed result, the
    state read-back is constant."""
    se = np.full(3, 0.5)
    pde = types.SimpleNamespace(
        soil_model=None,
        segment_face_len={"WateringTopSegment": 1.0, "GroundBottomSegment": 1.0},
        top_segment_names=[],
        rain_face_len=1.0,
        rel_sat=types.SimpleNamespace(value=se, _old=types.SimpleNamespace(value=se)),
        surface_h={},
        load_state_blob=lambda blob: None,
        total_water=lambda: 0.0,
        surface_water=lambda: 0.0,
        bottom_drainage_estimate=lambda: 0.0,
        walk_window=lambda **kwargs: walk,
    )
    return SoilEngine(types.SimpleNamespace(mesh=None), types.SimpleNamespace(), pde)


def _advance(walk: WalkResult):
    engine = _engine(walk)
    state = SoilState(se=np.full(3, 0.5), se_old=np.full(3, 0.5), surface_h={}, at=_AT)
    return engine.advance(state, Forcing(at=_AT, dt_s=_WINDOW_S))


def test_skipped_s_reported_and_logged_as_error(caplog):
    with caplog.at_level(logging.WARNING):
        result = _advance(WalkResult(skipped_s=125.0))

    assert result.diagnostics["skipped_s"] == 125.0
    held_records = [r for r in caplog.records if "held state through" in r.getMessage()]
    assert len(held_records) == 1
    assert held_records[0].levelno == logging.ERROR
    assert "125.0s of a 600.0s window" in held_records[0].getMessage()


def test_skipped_s_zero_reported_silently(caplog):
    with caplog.at_level(logging.WARNING):
        result = _advance(WalkResult(skipped_s=0.0))

    assert result.diagnostics["skipped_s"] == 0.0
    assert not [r for r in caplog.records if "held state through" in r.getMessage()]


def test_retries_reported_without_escalation(caplog):
    with caplog.at_level(logging.DEBUG):
        result = _advance(WalkResult(retries=3))

    assert result.diagnostics["retries"] == 3.0
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


def test_retries_zero_reported(caplog):
    with caplog.at_level(logging.DEBUG):
        result = _advance(WalkResult(retries=0))

    assert result.diagnostics["retries"] == 0.0
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
