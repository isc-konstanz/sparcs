# -*- coding: utf-8 -*-
"""
tests.test_fieldsim_engine
~~~~~~~~~~~~~~~~~~~~~~~~~~

``SoilEngine`` over the live ``SoilPDECore``. Heavy (Gmsh + FiPy): marked slow.
"""

import datetime as dt
import io
import types

import pytest
from conftest import MESH_KW

import numpy as np

pytestmark = pytest.mark.slow

from sparcs.components.agriculture.fieldsim.core.anchor import AnchorSensor  # noqa: E402
from sparcs.components.agriculture.fieldsim.core.config import SoilConfig  # noqa: E402
from sparcs.components.agriculture.fieldsim.core.engine import SoilEngine  # noqa: E402
from sparcs.components.agriculture.fieldsim.core.pde import FluxRates, SoilPDECore  # noqa: E402
from sparcs.components.agriculture.fieldsim.core.simulation import Simulation  # noqa: E402
from sparcs.components.agriculture.fieldsim.core.state import Forcing, SoilState  # noqa: E402

UTC = dt.timezone.utc


def _soil_config(tmp_path, filename: str, **pde_values) -> SoilConfig:
    soil = SoilConfig.from_dict(
        {
            "mesh": {**MESH_KW, "filename": str(tmp_path / filename)},
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
    mesh_kw = {k: v for k, v in MESH_KW.items() if k != "width"}
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


def test_core_blob_is_the_state_blob(engine):
    """One codec: the core persists exactly the bytes ``SoilState.to_blob`` writes."""
    state = engine.initial_state(dt.datetime(2026, 1, 1, tzinfo=UTC))
    engine.load(state)

    assert engine.pde.save_state_blob() == state.to_blob()


def test_a_blob_that_needs_pickle_is_refused(engine):
    n = int(engine.pde.rel_sat.value.shape[0])
    buf = io.BytesIO()
    np.savez(
        buf, rel_sat=np.full(n, 0.5), surface_names=np.array(["WateringTopSegment"], dtype=object), surface_h=[0.0]
    )

    with pytest.raises(ValueError, match="allow_pickle"):
        SoilState.from_blob(buf.getvalue(), dt.datetime(2026, 1, 1, tzinfo=UTC))
    with pytest.raises(ValueError, match="allow_pickle"):
        engine.pde.load_state_blob(buf.getvalue())


def test_a_state_from_another_mesh_decodes_but_is_refused_at_resume(engine):
    n = int(engine.pde.rel_sat.value.shape[0])
    at = dt.datetime(2026, 1, 1, tzinfo=UTC)
    blob = SoilState(np.full(n + 2, 0.5), np.full(n + 2, 0.5), {}, at).to_blob()
    stale = SoilState.from_blob(blob, at)
    simulation = Simulation(None, engine, chain=None, assimilator=None)

    with pytest.raises(ValueError, match="cells"):
        simulation.resume(stale)
    assert simulation.state is None


def test_probe_from_sensor_takes_an_anchor_sensor(engine):
    probe = engine.probe_from_sensor(AnchorSensor(key="bay1_30cm", x_offset_cm=0.0, depth_cm=30.0))

    assert probe.channel_id == "bay1_30cm"
    assert len(probe.cell_indices) == 1


def test_load_marks_the_state_current():
    """A loaded state is not loaded again for the samples that follow."""
    loads: list = []
    pde = types.SimpleNamespace(soil_model=types.SimpleNamespace(psi_from_se=lambda se: -se), sample=lambda probe: 0.5)
    pde.load_state_blob = loads.append
    engine = SoilEngine(types.SimpleNamespace(mesh=None), types.SimpleNamespace(), pde)
    state = SoilState(np.full(3, 0.5), np.full(3, 0.5), {}, dt.datetime(2026, 1, 1, tzinfo=UTC))

    engine.load(state)
    engine.tension_at(state, object())
    engine.tension_at(state, object())

    assert len(loads) == 1


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
            "mesh": {**MESH_KW, "filename": str(tmp_path / "soil_probe.msh")},
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
