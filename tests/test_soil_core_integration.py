# -*- coding: utf-8 -*-
"""Regression tests for the hardened SoilPDECore solve / walk_window path.

Background: the soil-tuning parameter sweep (sparcs/soil_tuning.py) showed
that high-rain forcing produces near-singular linear systems that crashed
the previous direct-LU solve (uncatchable C-abort) and could push NaN /
out-of-band Se into the committed state. These tests pin the hardened
behaviour: GMRES never hard-crashes, non-finite states are never committed,
and Se stays in [SE_MIN, SE_MAX].

Heavy (builds a real Gmsh mesh and runs FiPy): marked slow.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.slow

from lories import Configurations  # noqa: E402

from sparcs.components.agriculture.simulation._soil import (  # noqa: E402
    SE_MAX,
    SE_MIN,
    FluxRates,
    MeshConfig,
    PDEConfig,
    SoilPDECore,
    SolveResult,
    ensure_mesh,
)


def _configs(tmp_dir: str, **values) -> Configurations:
    return Configurations.load(
        "test.conf", conf_dir=tmp_dir, require=False, **values,
    )


@pytest.fixture(scope="module")
def pde(tmp_path_factory) -> SoilPDECore:
    tmp = tmp_path_factory.mktemp("soil_core")
    mesh_config = MeshConfig(_configs(
        str(tmp),
        filename=str(tmp / "soil_test.msh"),
        dl=0.2,
        width=3.0,
        height=1.5,
        plant_width=1.0,
        plant_height=0.5,
        watering_width=0.5,
        d_x=0.5,
    ))
    ode_config = PDEConfig(_configs(
        str(tmp),
        dt="50s",
        dt_min="1s",
    ))
    ensure_mesh(mesh_config)
    return SoilPDECore(mesh_config, ode_config, rel_sat_name="Se_test")


def _extreme_rain() -> FluxRates:
    # ~360 mm/h — far beyond any infiltration capacity; the regime that
    # used to segfault the LU solve and trip the clipper hardest.
    return FluxRates(
        seg_evap={}, seg_transp={}, flow_m3s=0.0, rain_flux=0.1,
    )


def test_extreme_rain_walk_stays_finite_and_in_band(pde):
    result = pde.walk_window(
        rates=_extreme_rain(), window_s=600.0, accept_at_dt_min=True,
    )
    se = np.asarray(pde.rel_sat.value)
    assert result.ok
    assert np.all(np.isfinite(se))
    assert np.all(se >= SE_MIN) and np.all(se <= SE_MAX)
    se_old = np.asarray(pde.rel_sat._old.value)
    assert np.all(se_old >= SE_MIN) and np.all(se_old <= SE_MAX)


def test_walk_cancel_aborts(pde):
    result = pde.walk_window(
        rates=_extreme_rain(), window_s=600.0,
        accept_at_dt_min=False, cancel=lambda: True,
    )
    assert not result.ok
    assert result.cancelled
    assert result.reason == "cancelled"


def test_walk_strict_fails_at_dt_min_and_holds_state(pde, monkeypatch):
    before = pde.snapshot()
    monkeypatch.setattr(
        pde, "solve",
        lambda dt, **kw: SolveResult(
            residual=float("inf"), converged=False, sweeps=25, finite=False,
        ),
    )
    result = pde.walk_window(
        rates=_extreme_rain(), window_s=120.0, accept_at_dt_min=False,
    )
    assert not result.ok
    assert "dt_min" in result.reason
    assert np.allclose(np.asarray(pde.rel_sat.value), before)


def test_walk_accept_skips_poisoned_substeps(pde, monkeypatch):
    before = pde.snapshot()
    monkeypatch.setattr(
        pde, "solve",
        lambda dt, **kw: SolveResult(
            residual=float("inf"), converged=False, sweeps=25, finite=False,
        ),
    )
    result = pde.walk_window(
        rates=_extreme_rain(), window_s=120.0, accept_at_dt_min=True,
    )
    # Walk covers the window without ever committing the poisoned state.
    assert result.ok
    assert result.skipped_s == pytest.approx(120.0)
    assert np.allclose(np.asarray(pde.rel_sat.value), before)


def test_solve_uses_gmres_not_lu(pde):
    from fipy.solvers import LinearGMRESSolver

    assert isinstance(pde._solver, LinearGMRESSolver)
