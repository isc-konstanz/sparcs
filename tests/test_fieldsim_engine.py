# -*- coding: utf-8 -*-
"""sparcs.tests.test_fieldsim_engine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``SoilEngine`` over the live ``SoilPDECore``. Heavy (Gmsh + FiPy): marked slow.
"""

import datetime as dt

import pytest

import numpy as np

pytestmark = pytest.mark.slow

from sparcs.components.agriculture.fieldsim.core.config import SoilConfig  # noqa: E402
from sparcs.components.agriculture.fieldsim.core.engine import SoilEngine  # noqa: E402
from sparcs.components.agriculture.fieldsim.core.pde import FluxRates, SoilPDECore  # noqa: E402
from sparcs.components.agriculture.fieldsim.core.state import Forcing, SoilState  # noqa: E402

UTC = dt.timezone.utc

_MESH_KW = {
    "dl": 0.2,
    "width": 3.0,
    "height": 1.5,
    "plant_width": 1.0,
    "plant_height": 0.5,
    "watering_width": 0.5,
    "d_x": 0.5,
}


def _soil_config(tmp_path, filename: str, **pde_values) -> SoilConfig:
    soil = SoilConfig.from_dict(
        {
            "mesh": {**_MESH_KW, "filename": str(tmp_path / filename)},
            "pde": {"dt": "30s", "dt_min": "1s", **pde_values},
        }
    )
    soil.mesh.derive(bay_width=3.0)
    return soil


@pytest.fixture
def engine(tmp_path) -> SoilEngine:
    soil = _soil_config(tmp_path, "soil_test.msh")
    return SoilEngine.build(soil, rel_sat_name="Se_engine_test")


def test_build_rejects_undrived_mesh_width():
    mesh_kw = {k: v for k, v in _MESH_KW.items() if k != "width"}
    soil = SoilConfig.from_dict({"mesh": mesh_kw, "pde": {"dt": "30s", "dt_min": "1s"}})
    assert soil.mesh.width is None
    with pytest.raises(ValueError, match="mesh.width"):
        SoilEngine.build(soil)


def test_initial_state_shape_and_bounds(engine):
    at = dt.datetime(2026, 1, 1, tzinfo=UTC)
    state = engine.initial_state(at)
    n = int(engine.pde.rel_sat.value.shape[0])

    assert state.se.shape == (n,)
    assert state.se_old.shape == (n,)
    assert np.array_equal(state.se, state.se_old)
    assert state.at == at
    assert state.ponding_m == 0.0

    lo, hi = engine.se_bounds
    assert np.all(state.se >= lo) and np.all(state.se <= hi)


def test_cold_start_s_follows_the_pde_config(tmp_path):
    default = SoilEngine.build(_soil_config(tmp_path, "soil_cold_default.msh"), rel_sat_name="Se_cold_default")
    assert default.cold_start_s == 3 * 3600.0

    hydrostatic = SoilEngine.build(
        _soil_config(tmp_path, "soil_cold_hydro.msh", ic_water_table_depth=1.0), rel_sat_name="Se_cold_hydro"
    )
    assert hydrostatic.cold_start_s == 0.0

    explicit = SoilEngine.build(
        _soil_config(tmp_path, "soil_cold_explicit.msh", cold_start="30min"), rel_sat_name="Se_cold_explicit"
    )
    assert explicit.cold_start_s == 1800.0


def test_advance_rain_increases_storage(engine):
    at = dt.datetime(2026, 1, 1, tzinfo=UTC)
    initial = engine.initial_state(at)
    water_before = engine.diagnostics(initial)["water_total"]

    forcing = Forcing(at=at, dt_s=3600.0, rain_flux=1.0e-4)
    result = engine.advance(initial, forcing)

    assert not result.cancelled
    assert result.state.at == forcing.end
    assert result.diagnostics["water_total"] > water_before
    assert result.diagnostics["delta_storage"] > 0.0
    assert result.diagnostics["walk_ok"] == 1.0

    expected_keys = {
        "top_out",
        "transpiration",
        "top_in",
        "bottom_out",
        "runoff",
        "demand_unmet",
        "balance_residual",
        "water_total",
        "surface_water",
        "delta_storage",
        "skipped_s",
        "retries",
        "walk_ok",
    }
    assert expected_keys <= result.diagnostics.keys()
    for key in expected_keys:
        assert np.isfinite(result.diagnostics[key]), key


def test_advance_matches_live_core_directly(tmp_path):
    soil = _soil_config(tmp_path, "soil_equiv.msh")
    engine = SoilEngine.build(soil, rel_sat_name="Se_equiv_engine")

    at = dt.datetime(2026, 1, 1, tzinfo=UTC)
    initial = engine.initial_state(at)
    forcing = Forcing(at=at, dt_s=3600.0, rain_flux=1.0e-5, flow_m3s=2.0e-6)

    result = engine.advance(initial, forcing)

    core = SoilPDECore(engine.mesh_config, engine.ode, rel_sat_name="Se_equiv_core")
    core.load_state_blob(initial.to_blob())
    rates = FluxRates(seg_evap={}, seg_transp={}, flow_m3s=forcing.flow_m3s, rain_flux=forcing.rain_flux)
    walk = core.walk_window(rates=rates, window_s=forcing.dt_s, accept_at_dt_min=True)

    assert walk.ok
    assert np.allclose(result.state.se, np.asarray(core.rel_sat.value), rtol=0, atol=1e-12)
    assert result.state.surface_h.keys() == core.surface_h.keys()
    for name, value in result.state.surface_h.items():
        assert value == pytest.approx(core.surface_h[name], abs=1e-12)


def test_state_blob_round_trip(engine):
    at = dt.datetime(2026, 1, 1, tzinfo=UTC)
    initial = engine.initial_state(at)

    from_initial_blob = SoilState.from_blob(initial.to_blob(), initial.at)
    assert np.array_equal(from_initial_blob.se, initial.se)
    assert np.array_equal(from_initial_blob.se_old, initial.se_old)
    assert from_initial_blob.surface_h == initial.surface_h

    forcing = Forcing(at=at, dt_s=1800.0, rain_flux=5.0e-6)
    result = engine.advance(initial, forcing)

    from_core_blob = SoilState.from_blob(engine.pde.save_state_blob(), result.state.at)
    assert np.array_equal(from_core_blob.se, result.state.se)
    assert np.array_equal(from_core_blob.se_old, result.state.se_old)
    assert from_core_blob.surface_h == result.state.surface_h


def test_cancel_holds_input_state_then_resumes(engine):
    at = dt.datetime(2026, 1, 1, tzinfo=UTC)
    initial = engine.initial_state(at)
    forcing = Forcing(at=at, dt_s=3600.0, rain_flux=1.0e-5)

    cancelled = engine.advance(initial, forcing, cancel=lambda: True)
    assert cancelled.cancelled is True
    assert cancelled.state is initial
    assert cancelled.diagnostics == {}

    resumed = engine.advance(initial, forcing, cancel=None)
    assert not resumed.cancelled
    assert resumed.state.at == forcing.end
    assert resumed.diagnostics["walk_ok"] == 1.0


def test_tension_at_probe(tmp_path):
    soil = SoilConfig.from_dict(
        {
            "mesh": {**_MESH_KW, "filename": str(tmp_path / "soil_probe.msh")},
            "pde": {"dt": "30s", "dt_min": "1s"},
            "probes": {"points": {"strip": {"x_offset": 0.0, "depth": 30.0}}},
        }
    )
    soil.mesh.derive(bay_width=3.0)
    engine = SoilEngine.build(soil, rel_sat_name="Se_probe_test")

    probes_block = soil.configs.get_member("probes", defaults={}, ensure_exists=True)
    probes = engine.probes(probes_block)
    assert len(probes) == 1

    at = dt.datetime(2026, 1, 1, tzinfo=UTC)
    state = engine.initial_state(at)
    tension = engine.tension_at(state, probes[0])

    assert np.isfinite(tension)
    assert tension < 0.0


def test_probes_empty_block_returns_empty_list(engine):
    assert engine.probes(None) == []
    assert engine.probes({}) == []
