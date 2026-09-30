# -*- coding: utf-8 -*-
"""
tests.test_fieldsim_diagnostics_keys
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The water-balance diagnostic Constants keep their short keys under
``context="water"``: the bare key is the channel key and the
``agri_field_simulation`` SQL column, the registry id stays ``water_*``-unique.
``SoilEngine`` computes the same literal keys.
"""

import types

import pytest

from sparcs.components.agriculture.simulation.components import SoilSimulation
from sparcs.components.agriculture.simulation.core.engine import SoilEngine
from sparcs.components.agriculture.simulation.core.pde import ClipDiagnostics, FluxRates


@pytest.mark.parametrize(
    "constant, expected_key, expected_id",
    [
        (SoilSimulation.WATER_TOP_IN, "top_in", "water_top_in"),
        (SoilSimulation.WATER_TOP_OUT, "top_out", "water_top_out"),
        (SoilSimulation.WATER_BOTTOM, "bottom_out", "water_bottom_out"),
        (SoilSimulation.WATER_TRANSP, "transpiration", "water_transpiration"),
        (SoilSimulation.WATER_RUNOFF, "runoff", "water_runoff"),
        (SoilSimulation.WATER_DEMAND_UNMET, "demand_unmet", "water_demand_unmet"),
        (SoilSimulation.WATER_BALANCE_RESIDUAL, "balance_residual", "water_balance_residual"),
        (SoilSimulation.WATER_ANCHOR, "anchor", "water_anchor"),
        (SoilSimulation.WALK_SKIPPED_S, "skipped_s", "water_skipped_s"),
        (SoilSimulation.WEATHER_STALL, "weather_stall", "water_weather_stall"),
        (SoilSimulation.TICK_FAILURES, "tick_failures", "water_tick_failures"),
        (SoilSimulation.WALK_RETRIES, "retries", "water_retries"),
    ],
)
def test_diagnostic_constants_use_short_keys_with_water_registry_id(constant, expected_key, expected_id):
    assert constant.key == expected_key
    assert constant.id == expected_id


def test_top_in_display_name_mentions_irrigation_and_rain():
    name = SoilSimulation.WATER_TOP_IN.name.lower()
    assert "irrigation" in name
    assert "rain" in name


def test_compute_diagnostics_returns_short_keys():
    pde = types.SimpleNamespace(
        soil_model=None,
        segment_face_len={"WateringTopSegment": 1.0, "GroundBottomSegment": 1.0},
        top_segment_names=[],
        rain_face_len=1.0,
        bottom_drainage_estimate=lambda: 0.0,
    )
    engine = SoilEngine(types.SimpleNamespace(mesh=None), types.SimpleNamespace(), pde)
    rates = FluxRates(seg_evap={}, seg_transp={}, flow_m3s=0.0, rain_flux=0.0)

    diagnostics = engine._compute_diagnostics(rates, delta_storage=0.0, elapsed_s=60.0, clip=ClipDiagnostics())

    assert set(diagnostics) == {
        "top_in",
        "top_out",
        "bottom_out",
        "transpiration",
        "runoff",
        "demand_unmet",
        "balance_residual",
    }
