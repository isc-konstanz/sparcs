# -*- coding: utf-8 -*-
"""sparcs.tests.test_soil_diagnostics_rename
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Issue 02(b): the eight water-balance diagnostic Constants move to short keys
under ``context="water"`` (house pattern, cf. ``context="pv"`` in
``solar/system.py``) -- the registry id stays ``water_*``-unique, the bare key
becomes the channel key and the ``agri_field_simulation`` SQL column.
``_compute_diagnostics``'s literal dict keys move in the same commit.

Issue 02(c): the two blob channels (``SIMULATION_STATE`` / ``SOIL_PROGRESS_IMAGE``)
get ``logger.column`` overrides so their side tables use the PRD's short column
names (``state`` / ``image``) instead of key-derived ones.

Importing ``soil``/``_soil`` pulls the full lories + soil (FiPy/Gmsh) stack;
``importorskip`` keeps this out of environments that lack it (the full check
runs on the box).
"""

import types

import pytest

soil = pytest.importorskip("sparcs.components.agriculture.simulation.soil")
SoilSimulation = soil.SoilSimulation

_soil_core = pytest.importorskip("sparcs.components.agriculture.simulation._soil")
FluxRates = _soil_core.FluxRates
ClipDiagnostics = _soil_core.ClipDiagnostics


# --- (b) Constant rename: short key, "water_*" registry id -------------------


@pytest.mark.parametrize(
    "constant, expected_key, expected_id",
    [
        (SoilSimulation.WATER_TOP_IN, "top_in", "water_top_in"),
        (SoilSimulation.WATER_TOP_OUT, "top_out", "water_top_out"),
        # id = f"{context}_{key}" = "water_bottom_out" -- NOT the old bare key
        # "water_bottom" (only "water_*"-style is required, not the exact string).
        (SoilSimulation.WATER_BOTTOM, "bottom_out", "water_bottom_out"),
        (SoilSimulation.WATER_TRANSP, "transpiration", "water_transpiration"),
        (SoilSimulation.WATER_RUNOFF, "runoff", "water_runoff"),
        (SoilSimulation.WATER_DEMAND_UNMET, "demand_unmet", "water_demand_unmet"),
        (SoilSimulation.WATER_BALANCE_RESIDUAL, "balance_residual", "water_balance_residual"),
        (SoilSimulation.WATER_ANCHOR, "anchor", "water_anchor"),
        # B7: skipped-at-dt_min seconds; bare key is the SQL column, context
        # namespaces the global Constant registry like every sibling above.
        (SoilSimulation.WALK_SKIPPED_S, "skipped_s", "water_skipped_s"),
        # W2.1: consecutive weather-stall ticks preceding the committing tick.
        (SoilSimulation.WEATHER_STALL, "weather_stall", "water_weather_stall"),
        # W2.2: consecutive tick failures preceding the committing tick.
        (SoilSimulation.TICK_FAILURES, "tick_failures", "water_tick_failures"),
        # W2.4: adaptive-walk substep rollbacks for the window.
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


# --- _compute_diagnostics: literal dict keys follow the rename ---------------


def _bare_soil_base():
    """A bare instance with just the ``_pde``-backed properties
    ``_compute_diagnostics`` reads (``_segment_face_len``/``_top_segment_names``/
    ``_rain_face_len``); pure computation, no channel writes, so no FiPy mesh
    is needed."""
    sim = object.__new__(SoilSimulation)
    sim._pde = types.SimpleNamespace(
        segment_face_len={"WateringTopSegment": 1.0, "GroundBottomSegment": 1.0},
        top_segment_names=[],
        rain_face_len=1.0,
        bottom_drainage_estimate=lambda: 0.0,
    )
    return sim


def test_compute_diagnostics_returns_short_keys():
    sim = _bare_soil_base()
    rates = FluxRates(seg_evap={}, seg_transp={}, flow_m3s=0.0, rain_flux=0.0)
    clip = ClipDiagnostics()

    diagnostics = sim._compute_diagnostics(rates, delta_storage=0.0, elapsed_s=60.0, clip=clip)

    assert set(diagnostics) == {
        "top_in",
        "top_out",
        "bottom_out",
        "transpiration",
        "runoff",
        "demand_unmet",
        "balance_residual",
    }


# --- (c) blob channel columns -------------------------------------------------


class _RecordingData:
    def __init__(self):
        self.added: list[tuple] = []

    def add(self, channel_id, **kwargs) -> None:
        self.added.append((channel_id, kwargs))


def test_register_state_channel_sets_state_column(monkeypatch):
    sim = object.__new__(SoilSimulation)
    fake = _RecordingData()
    monkeypatch.setattr(SoilSimulation, "data", property(lambda self: fake))

    sim._register_state_channel()

    channel_id, kwargs = fake.added[0]
    assert channel_id == "simulation_state"
    assert kwargs["logger"]["column"] == "state"


def test_register_progress_image_channel_sets_image_column(monkeypatch):
    sim = object.__new__(SoilSimulation)
    fake = _RecordingData()
    monkeypatch.setattr(SoilSimulation, "data", property(lambda self: fake))

    sim._register_progress_image_channel()

    channel_id, kwargs = fake.added[0]
    assert channel_id == "soil_progress_image"
    assert kwargs["logger"]["column"] == "image"
