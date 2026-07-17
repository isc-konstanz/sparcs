# -*- coding: utf-8 -*-
"""sparcs.tests.test_soil_walk_retries_diagnostics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Issue 22 (W2.4): adaptive-walk retries (``WalkResult.retries`` -- substep
rollbacks at reduced dt) used to be DEBUG-logged and dropped, so a solver
grinding through rollbacks every tick was invisible in the persisted data.
``advance()`` now threads ``walk_result.retries`` into
``_record_diagnostics``, which writes it to the ``retries`` channel every tick
(0.0 when clean -- the B7 dashboarding convention). Log levels are unchanged:
retries keep their DEBUG log; ERROR stays reserved for ``skipped_s``.

Mirrors test_soil_skipped_s_diagnostics.py's shape (bare ``object.__new__``
sim, stub ``_pde.walk_window``, recording ``self.data``).
"""

import logging
import types

import pytest

import pandas as pd

soil = pytest.importorskip("sparcs.components.agriculture.simulation.soil")
SoilSimulation = soil.SoilSimulation

_soil_core = pytest.importorskip("sparcs.components.agriculture.simulation._soil")
FluxRates = _soil_core.FluxRates
ClipDiagnostics = _soil_core.ClipDiagnostics
WalkResult = _soil_core.WalkResult


class _RecordingChannel:
    def __init__(self, channel_id: str):
        self.id = channel_id
        self.calls: list[tuple] = []

    def set(self, timestamp, value) -> None:
        self.calls.append((timestamp, value))


class _RecordingData:
    """``self.data`` stand-in: auto-creates a recording channel per key (these
    tests bypass ``configure()``, so there is no ``add()`` call to seed the map)."""

    def __init__(self):
        self._channels: dict = {}

    def __getitem__(self, key: str) -> _RecordingChannel:
        return self._channels.setdefault(key, _RecordingChannel(key))


def _bare_sim(monkeypatch, *, retries: int):
    sim = object.__new__(SoilSimulation)
    sim._name = "test_soil_simulation"
    sim._probes = []  # _sample_probes short-circuits on empty, no _soil_model needed
    sim._pde = types.SimpleNamespace(
        walk_window=lambda **kwargs: WalkResult(retries=retries),
        segment_face_len={"WateringTopSegment": 1.0, "GroundBottomSegment": 1.0},
        top_segment_names=[],
        rain_face_len=1.0,
        bottom_drainage_estimate=lambda: 0.0,
    )
    fake = _RecordingData()
    # Component.data is a read-only class property; patch the CLASS for the bare
    # instance (monkeypatch auto-restores).
    monkeypatch.setattr(SoilSimulation, "data", property(lambda self: fake))
    return sim, fake


def _walk_and_record(sim, now: pd.Timestamp) -> dict[str, float]:
    """Reproduce ``advance()``'s ``_walk`` -> ``_record_diagnostics`` thread."""
    rates = FluxRates(seg_evap={}, seg_transp={}, flow_m3s=0.0, rain_flux=0.0)
    clip_total = ClipDiagnostics()
    walk_result = sim._walk(
        rates=rates,
        window_s=600.0,
        clip_total=clip_total,
        sim_t0=now - pd.Timedelta(seconds=600),
        plot_interval=None,
    )
    return sim._record_diagnostics(rates, now, 0.0, 600.0, clip_total, walk_result.skipped_s, walk_result.retries)


def test_retries_persist_channel_and_keep_debug_log_level(monkeypatch, caplog):
    sim, fake = _bare_sim(monkeypatch, retries=3)
    now = pd.Timestamp("2026-07-16 10:00", tz="UTC")

    with caplog.at_level(logging.DEBUG):
        diagnostics = _walk_and_record(sim, now)

    assert fake["retries"].calls == [(now, 3.0)]
    assert diagnostics["retries"] == 3.0
    retry_records = [r for r in caplog.records if "retry(s)" in r.getMessage()]
    assert len(retry_records) == 1
    assert retry_records[0].levelno == logging.DEBUG  # never escalated; the channel is the signal


def test_retries_zero_sets_channel_silently(monkeypatch, caplog):
    sim, fake = _bare_sim(monkeypatch, retries=0)
    now = pd.Timestamp("2026-07-16 10:00", tz="UTC")

    with caplog.at_level(logging.DEBUG):
        diagnostics = _walk_and_record(sim, now)

    assert fake["retries"].calls == [(now, 0.0)]
    assert diagnostics["retries"] == 0.0
    assert not [r for r in caplog.records if "retry(s)" in r.getMessage()]
