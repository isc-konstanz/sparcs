# -*- coding: utf-8 -*-
"""The whole-field irrigation flow must be normalized to per-metre-of-row.

The production flow meter measures the whole-field total [l/min], while the 2D
soil mesh is one bay cross-section representing 1 m of a single row.
``_compute_flux_rates`` therefore divides the metered flow by
``total_drip_line_length_m`` (n_rows * row_length); the default of 1.0 reads the
metered value as already per-metre. A warn-once guard flags strip fluxes that
can only come from mis-scaled flow values or a wrong drip-line length.

Importing ``SoilSimulation`` pulls the full lories + FiPy stack; ``importorskip``
keeps this out of environments that lack it (the full check runs on the box).
``_compute_flux_rates`` only touches the irrigation flow on the context, the
drip-line length, and the mesh config, so a bare instance built with
``object.__new__`` exercises it without the Component machinery; ``name`` and
``context`` are read-only properties, so their backing attributes (``_name``,
name-mangled ``_Registrator__context``) are set directly.
"""

import logging
from types import SimpleNamespace

import pytest

import pandas as pd

soil = pytest.importorskip("sparcs.components.agriculture.simulation.soil")
SoilSimulation = soil.SoilSimulation

WATERING_WIDTH = 0.05


def _bare_sim(flow_lpm: float, drip_line_length_m: float) -> "SoilSimulation":
    sim = object.__new__(SoilSimulation)
    sim._name = "soil_test"
    sim._Registrator__context = SimpleNamespace(_irrigation_flow_lpm=flow_lpm)
    sim._total_drip_line_length_m = drip_line_length_m
    sim._mesh_config = SimpleNamespace(watering_width=WATERING_WIDTH)
    sim._strip_flux_warned = False
    return sim


def _flow_m3s(sim: "SoilSimulation") -> float:
    et_data = pd.DataFrame(index=pd.to_datetime(["2026-05-01 10:00"]))
    return SoilSimulation._compute_flux_rates(sim, et_data, {}, 3600.0).flow_m3s


def test_default_length_keeps_per_metre_reading():
    """L=1 (default): the metered value is read as already per-metre, /60000 only."""
    sim = _bare_sim(flow_lpm=60.0, drip_line_length_m=1.0)
    assert _flow_m3s(sim) == pytest.approx(60.0 / 60_000.0)


def test_whole_field_flow_divided_by_drip_line_length():
    """A whole-field meter reading is spread over the total drip-line length."""
    sim = _bare_sim(flow_lpm=60.0, drip_line_length_m=1000.0)
    assert _flow_m3s(sim) == pytest.approx(60.0 / 60_000.0 / 1000.0)


def test_absurd_strip_flux_warns_once(caplog):
    """A strip flux far beyond drip rates warns once, not on every callback."""
    # 60 l/min over 1 m of line -> 72000 mm/h over the 0.05 m strip.
    sim = _bare_sim(flow_lpm=60.0, drip_line_length_m=1.0)
    with caplog.at_level(logging.WARNING):
        _flow_m3s(sim)
        _flow_m3s(sim)
    warnings = [r for r in caplog.records if "strip flux" in r.getMessage()]
    assert len(warnings) == 1
    assert "total_drip_line_length_m" in warnings[0].getMessage()


def test_sane_strip_flux_stays_silent(caplog):
    """A drip-scale flux (~20 mm/h over the strip) does not trigger the guard."""
    # 16.7 l/min whole-field over 1000 m of line -> ~1 l/h per metre -> ~20 mm/h.
    sim = _bare_sim(flow_lpm=16.7, drip_line_length_m=1000.0)
    with caplog.at_level(logging.WARNING):
        _flow_m3s(sim)
    assert not [r for r in caplog.records if "strip flux" in r.getMessage()]
    assert not sim._strip_flux_warned
