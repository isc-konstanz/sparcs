# -*- coding: utf-8 -*-
"""
tests.test_fieldsim_engine_tension
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The probe publish boundary is water tension, not relative saturation:
``SoilEngine.tension_at`` converts the sampled Se with the retention model and
returns a plain signed-negative ``float`` in hPa, and the probe channel is
registered in ``hPa`` on ``agri_soil_simulation.water_tension``.
"""

import types

import pytest

import numpy as np
from sparcs.components.agriculture.fieldsim.components import SoilSimulation
from sparcs.components.agriculture.fieldsim.core.engine import SoilEngine
from sparcs.components.agriculture.soil.models import Genuchten

_MODEL = Genuchten(theta_r=0.05, theta_s=0.43, alpha=0.08, n=1.6, k_s=1.0e-4)


def _engine(sample_by_id: dict, model=_MODEL) -> SoilEngine:
    pde = types.SimpleNamespace(
        soil_model=model,
        sample=lambda probe: sample_by_id[probe.channel_id],
        load_state_blob=lambda blob: None,
    )
    return SoilEngine(types.SimpleNamespace(mesh=None), types.SimpleNamespace(), pde)


def _probe(channel_id: str, name: str):
    return types.SimpleNamespace(channel_id=channel_id, name=name)


def _state():
    return types.SimpleNamespace(to_blob=lambda: b"")


def test_tension_at_returns_matric_potential_not_se():
    """A known Se comes back as ``psi_from_se(Se)`` (negative hPa, matching the
    DB / tensiometer), not the raw Se in [0, 1]."""
    engine = _engine({"soil_30cm": 0.3})

    tension = engine.tension_at(_state(), _probe("soil_30cm", "Soil 30cm"))

    assert tension == pytest.approx(float(_MODEL.psi_from_se(0.3)))
    assert tension < -1.0  # signed hPa, out of the [0, 1] saturation range


def test_drier_probe_yields_larger_tension():
    """A drier probe (lower Se) must give a MORE NEGATIVE matric potential
    (larger tension magnitude) -- the sign contract."""
    engine = _engine({"soil_30cm": 0.3, "soil_60cm": 0.8})
    state = _state()

    dry_tension = engine.tension_at(state, _probe("soil_30cm", "Soil 30cm"))
    wet_tension = engine.tension_at(state, _probe("soil_60cm", "Soil 60cm"))

    assert dry_tension < wet_tension < 0.0


def test_tension_at_returns_a_plain_float():
    # np.float64 subclasses float, so also exclude np.floating: the seam's
    # promise is a PLAIN float on the publish path.
    model = types.SimpleNamespace(psi_from_se=lambda se: np.float64(-500.0))
    engine = _engine({"soil_30cm": 0.5}, model=model)

    tension = engine.tension_at(_state(), _probe("soil_30cm", "Soil 30cm"))

    assert isinstance(tension, float) and not isinstance(tension, np.floating)
    assert tension == -500.0


def test_register_probe_uses_hpa_unit_and_the_soil_simulation_column(monkeypatch):
    """Without the table/column override the probe channel inherits the
    component's default table (keyed by field_id only) and N probes
    upsert-clobber one shared column."""
    added: list[tuple] = []
    data = types.SimpleNamespace(add=lambda channel_id, **kwargs: added.append((channel_id, kwargs)))
    monkeypatch.setattr(SoilSimulation, "data", property(lambda self: data))
    component = object.__new__(SoilSimulation)

    component.register_probe(_probe("soil_30cm", "Soil 30cm"))

    assert len(added) == 1
    channel_id, kwargs = added[0]
    assert channel_id == "soil_30cm"
    assert kwargs["unit"] == "hPa"
    assert kwargs["type"] is float
    assert kwargs["logger"]["table"] == "agri_soil_simulation"
    assert kwargs["logger"]["column"] == "water_tension"
