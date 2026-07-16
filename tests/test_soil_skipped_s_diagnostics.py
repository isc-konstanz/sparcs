# -*- coding: utf-8 -*-
"""sparcs.tests.test_soil_skipped_s_diagnostics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Issue 10 (B7): seconds skipped at ``dt_min`` (``WalkResult.skipped_s``) must be
threaded from ``_walk`` into a persisted ``skipped_s`` diagnostics channel with
ERROR-level escalation -- accept+mark, no hold/retry (Q2 re-decided). ``_walk``
now returns the skipped seconds for the window instead of ``None``;
``advance()`` passes that value into ``_record_diagnostics``, which writes it
to the ``skipped_s`` channel every tick (0.0 when nothing was skipped -- a
column that only appears on failure cannot be dashboarded).

Instances are bare ``object.__new__`` with a stubbed ``_pde`` (``walk_window``
for ``_walk``; the ``_compute_diagnostics``-required attributes per
``test_soil_diagnostics_rename.py``'s precedent) and a recording ``self.data``
(``test_soil_probe_tension.py``'s precedent).
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
    """``self.data`` stand-in: auto-creates a recording channel per key for
    ``__getitem__`` (these tests bypass ``configure()``, so there is no
    ``add()`` call to seed the channel map)."""

    def __init__(self):
        self._channels: dict = {}

    def __getitem__(self, key: str) -> _RecordingChannel:
        return self._channels.setdefault(key, _RecordingChannel(key))


def _bare_sim(monkeypatch, *, skipped_s: float):
    """A bare ``SoilSimulation`` with a stub PDE: ``walk_window`` returns a
    fixed ``WalkResult`` (for ``_walk``)."""
    sim = object.__new__(SoilSimulation)
    sim._name = "test_soil_simulation"
    sim._probes = []  # _sample_probes short-circuits on empty, no _soil_model needed
    sim._pde = types.SimpleNamespace(
        walk_window=lambda **kwargs: WalkResult(skipped_s=skipped_s),
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
    return sim._record_diagnostics(rates, now, 0.0, 600.0, clip_total, walk_result.skipped_s)


def test_skipped_s_persists_channel_and_logs_error(monkeypatch, caplog):
    sim, fake = _bare_sim(monkeypatch, skipped_s=125.0)
    now = pd.Timestamp("2026-07-16 10:00", tz="UTC")

    with caplog.at_level(logging.WARNING):
        diagnostics = _walk_and_record(sim, now)

    assert fake["skipped_s"].calls == [(now, 125.0)]
    assert diagnostics["skipped_s"] == 125.0
    held_records = [r for r in caplog.records if "held state through" in r.getMessage()]
    assert len(held_records) == 1
    assert held_records[0].levelno == logging.ERROR
    assert "125.0s of a 600.0s window" in held_records[0].getMessage()


def test_skipped_s_zero_sets_channel_silently_no_error(monkeypatch, caplog):
    sim, fake = _bare_sim(monkeypatch, skipped_s=0.0)
    now = pd.Timestamp("2026-07-16 10:00", tz="UTC")

    with caplog.at_level(logging.WARNING):
        diagnostics = _walk_and_record(sim, now)

    assert fake["skipped_s"].calls == [(now, 0.0)]
    assert diagnostics["skipped_s"] == 0.0
    assert not [r for r in caplog.records if "held state through" in r.getMessage()]
