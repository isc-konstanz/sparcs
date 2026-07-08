# -*- coding: utf-8 -*-
"""sparcs.tests.test_soil_probe_tension
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Unit tests for the live-simulation publish boundary: ``SoilSimulation`` now
registers its probe channels in water tension (``unit="hPa"``) and converts the
sampled relative saturation Se to tension with the retention model at publish
time (``_register_probe`` / ``_sample_probes``). The PDE core keeps returning Se
(that path -- anchoring, total_water -- is not exercised here).

Importing ``soil`` pulls the full lories + soil (FiPy/Gmsh) stack; ``importorskip``
keeps this out of environments that lack it (the full check runs on the box). The
methods under test touch only ``self._probes`` / ``self._pde`` / ``self.data``, so
they run against a bare ``object.__new__`` instance with a stub PDE and a recording
``self.data`` -- no Component/PDE bootstrap, matching the predictor test pattern.
"""

import types

import pytest

import pandas as pd

soil = pytest.importorskip("sparcs.components.agriculture.simulation.soil")
SoilSimulation = soil.SoilSimulation

soil_models = pytest.importorskip("sparcs.components.agriculture.soil.models")
Genuchten = soil_models.Genuchten

_MODEL = Genuchten(theta_r=0.05, theta_s=0.43, alpha=0.08, n=1.6, k_s=1.0e-4)


class _RecordingChannel:
    def __init__(self, channel_id: str):
        self.id = channel_id
        self.calls: list[tuple] = []

    def set(self, timestamp, value) -> None:
        self.calls.append((timestamp, value))


class _RecordingData:
    """``self.data`` stand-in: captures ``add(...)`` kwargs and auto-creates a
    recording channel per key for ``__getitem__``."""

    def __init__(self):
        self.added: list[tuple] = []
        self._channels: dict = {}

    def add(self, channel_id, **kwargs) -> None:
        self.added.append((channel_id, kwargs))
        self._channels[channel_id] = _RecordingChannel(channel_id)

    def __getitem__(self, key: str) -> _RecordingChannel:
        return self._channels.setdefault(key, _RecordingChannel(key))


def _bare_sim(monkeypatch, probes, sample_by_id):
    """A bare ``SoilSimulation`` with a stub PDE (``sample`` reads ``sample_by_id``
    per probe, ``soil_model`` drives the conversion) and a recording ``self.data``."""
    sim = object.__new__(SoilSimulation)
    sim._name = "test_soil_simulation"
    sim._probes = probes
    sim._pde = types.SimpleNamespace(
        sample=lambda probe: sample_by_id[probe.channel_id],
        soil_model=_MODEL,
    )
    fake = _RecordingData()
    # Component.data is a read-only class property; patch the CLASS for the bare
    # instance (monkeypatch auto-restores).
    monkeypatch.setattr(SoilSimulation, "data", property(lambda self: fake))
    return sim, fake


def _probe(channel_id: str, name: str):
    return types.SimpleNamespace(channel_id=channel_id, name=name)


def test_register_probe_uses_hpa_unit(monkeypatch):
    sim, fake = _bare_sim(monkeypatch, probes=[], sample_by_id={})

    sim._register_probe(_probe("soil_30cm", "Soil 30cm"))

    assert len(fake.added) == 1
    channel_id, kwargs = fake.added[0]
    assert channel_id == "soil_30cm"
    assert kwargs["unit"] == "hPa"
    assert kwargs["type"] is float


def test_sample_probes_publishes_tension_not_se(monkeypatch):
    """A known Se is published as the signed matric potential ``psi_from_se(Se)``
    (negative hPa, matching the DB / tensiometer), not the raw Se in [0, 1]."""
    probe = _probe("soil_30cm", "Soil 30cm")
    sim, fake = _bare_sim(monkeypatch, probes=[probe], sample_by_id={"soil_30cm": 0.3})

    sim._sample_probes(pd.Timestamp("2026-07-03 00:00", tz="Europe/Berlin"))

    calls = fake["soil_30cm"].calls
    assert len(calls) == 1
    published = calls[0][1]
    assert published == pytest.approx(float(_MODEL.psi_from_se(0.3)))
    assert published < -1.0  # signed hPa, out of the [0, 1] saturation range


def test_sample_probes_drier_probe_yields_larger_tension(monkeypatch):
    """A drier probe (lower Se) must publish a MORE NEGATIVE matric potential
    (larger tension magnitude) -- the sign contract."""
    dry = _probe("soil_30cm", "Soil 30cm")  # Se 0.3 (drier)
    wet = _probe("soil_60cm", "Soil 60cm")  # Se 0.8 (wetter)
    sim, fake = _bare_sim(
        monkeypatch,
        probes=[dry, wet],
        sample_by_id={"soil_30cm": 0.3, "soil_60cm": 0.8},
    )

    sim._sample_probes(pd.Timestamp("2026-07-03 00:00", tz="Europe/Berlin"))

    dry_tension = fake["soil_30cm"].calls[0][1]
    wet_tension = fake["soil_60cm"].calls[0][1]
    assert dry_tension < wet_tension < 0.0
