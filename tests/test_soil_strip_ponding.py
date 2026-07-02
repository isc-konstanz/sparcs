# -*- coding: utf-8 -*-
"""Watering-strip ponding + implicit irrigation intake (SoilPDECore).

Background: the drip emitter is a concentrated source on one narrow surface
strip. The old path injected the flow explicitly and clipped it against the
per-step saturation headroom, booking the clipped water as permanently lost
``top_rejected`` "runoff" — at physically correct fluxes the loss reached
57–100 % and depended on dt (smaller dt lost more). These tests pin the
hardened behaviour: irrigation water is offered through a linearized implicit
source (the solver throttles intake near saturation), un-infiltrated water
ponds on the strip and re-offers on later substeps, the pond bookkeeping is
exact, and rollbacks / skipped substeps never mutate the ponds.

Heavy (builds a real Gmsh mesh and runs FiPy): marked slow.
"""

import pytest

import numpy as np

pytestmark = pytest.mark.slow

from lories import Configurations  # noqa: E402
from sparcs.components.agriculture.simulation._soil import (  # noqa: E402
    RHO_W,
    FluxRates,
    MeshConfig,
    PDEConfig,
    SoilPDECore,
    SolveResult,
    ensure_mesh,
)

WATERING = "WateringTopSegment"


def _configs(tmp_dir: str, **values) -> Configurations:
    return Configurations.load(
        "test.conf",
        conf_dir=tmp_dir,
        require=False,
        **values,
    )


def _build_core(tmp_path, dt: str = "30s") -> SoilPDECore:
    mesh_config = MeshConfig(
        _configs(
            str(tmp_path),
            filename=str(tmp_path / "soil_test.msh"),
            dl=0.2,
            width=3.0,
            height=1.5,
            plant_width=1.0,
            plant_height=0.5,
            watering_width=0.5,
            d_x=0.5,
        )
    )
    ode_config = PDEConfig(
        _configs(
            str(tmp_path),
            dt=dt,
            dt_min="1s",
        )
    )
    ensure_mesh(mesh_config)
    return SoilPDECore(mesh_config, ode_config, rel_sat_name="Se_test")


def _irrigation(flow_m3s: float) -> FluxRates:
    return FluxRates(seg_evap={}, seg_transp={}, flow_m3s=flow_m3s, rain_flux=0.0)


def _pond_mass(core: SoilPDECore) -> float:
    return core.surface_h[WATERING] * core.segment_face_len[WATERING] * RHO_W


# ~20 mm/h over the 0.5 m strip — drip-scale, well under default k_s (360 mm/h).
MODERATE_FLOW = 20.0e-3 / 3600.0 * 0.5
# ~2000 mm/h over the strip — far beyond intake; must pond and overflow, not vanish.
EXTREME_FLOW = 2000.0e-3 / 3600.0 * 0.5


def test_moderate_irrigation_fully_infiltrates(tmp_path):
    core = _build_core(tmp_path)
    result = core.walk_window(rates=_irrigation(MODERATE_FLOW), window_s=1800.0)
    injected = MODERATE_FLOW * RHO_W * 1800.0

    assert result.ok
    assert result.clip.top_rejected == pytest.approx(0.0, abs=1e-12)
    assert result.clip.ponding_overflow == pytest.approx(0.0, abs=1e-12)
    # The pond may carry a residual, but essentially everything infiltrated.
    assert _pond_mass(core) < 0.02 * injected


def test_irrigation_mass_reaches_the_soil(tmp_path):
    """Soil storage gain vs a zero-flow control run equals the water that left
    the pond system — irrigation is not silently created or destroyed."""
    control = _build_core(tmp_path)
    control.walk_window(rates=_irrigation(0.0), window_s=1800.0)
    dw_control = control.total_water()

    core = _build_core(tmp_path)
    result = core.walk_window(rates=_irrigation(MODERATE_FLOW), window_s=1800.0)
    injected = MODERATE_FLOW * RHO_W * 1800.0
    delivered = injected - _pond_mass(core) - result.clip.ponding_overflow

    # The tolerance absorbs the pre-existing background drift of the solver
    # (per-step SE_MIN/SE_MAX safety clips create/destroy small amounts of
    # water); the old inject-then-clip path lost tens of percent here.
    assert core.total_water() - dw_control == pytest.approx(delivered, rel=0.12)


def test_extreme_irrigation_ponds_and_overflows_honestly(tmp_path):
    core = _build_core(tmp_path)
    h_max_m = core.ode_config.ponding.h_max_mm / 1000.0
    result = core.walk_window(rates=_irrigation(EXTREME_FLOW), window_s=600.0)
    injected = EXTREME_FLOW * RHO_W * 600.0
    delivered = injected - _pond_mass(core) - result.clip.ponding_overflow

    # Nothing is booked as clip loss; the pond is bounded; the excess is
    # explicit overflow; whatever was delivered is non-negative.
    assert result.clip.top_rejected == pytest.approx(0.0, abs=1e-12)
    assert core.surface_h[WATERING] <= h_max_m + 1e-12
    assert result.clip.ponding_overflow > 0.0
    assert delivered >= 0.0
    se = np.asarray(core.rel_sat.value)
    assert np.all(np.isfinite(se))


def test_ponded_water_reapplies_after_pulse_ends(tmp_path):
    core = _build_core(tmp_path)
    core.walk_window(rates=_irrigation(EXTREME_FLOW), window_s=120.0)
    pond_after_pulse = _pond_mass(core)
    assert pond_after_pulse > 0.0

    core.walk_window(rates=_irrigation(0.0), window_s=1800.0)
    assert _pond_mass(core) < 0.5 * pond_after_pulse


def test_cumulative_infiltration_is_dt_stable(tmp_path):
    """Soil + pond storage after the same irrigation window agrees across dt
    (the old inject-then-clip path diverged by tens of percent)."""
    gained = {}
    for dt in (10.0, 120.0):
        core = _build_core(tmp_path, dt=f"{dt:g}s")
        w0 = core.total_water()
        result = core.walk_window(rates=_irrigation(MODERATE_FLOW), window_s=1800.0)
        assert result.ok
        gained[dt] = core.total_water() - w0 + _pond_mass(core) + result.clip.ponding_overflow
    assert gained[10.0] == pytest.approx(gained[120.0], rel=0.05)


def test_rollback_and_skip_never_mutate_the_ponds(tmp_path, monkeypatch):
    core = _build_core(tmp_path)
    core.ode_config.ponding.enabled = True
    ponds_before = dict(core.surface_h)
    monkeypatch.setattr(
        core,
        "solve",
        lambda dt, **kw: SolveResult(
            residual=float("inf"),
            converged=False,
            sweeps=25,
            finite=False,
        ),
    )
    rates = FluxRates(seg_evap={}, seg_transp={}, flow_m3s=MODERATE_FLOW, rain_flux=0.01)
    result = core.walk_window(rates=rates, window_s=120.0, accept_at_dt_min=True)

    # Every substep was skipped: no forcing entered the system, so the ponds
    # must be untouched (the old code accumulated rain into the buckets even
    # for rolled-back and skipped substeps).
    assert result.skipped_s == pytest.approx(120.0)
    assert core.surface_h == ponds_before


def test_watering_h_max_config(tmp_path):
    from sparcs.components.agriculture.simulation._soil import PondingConfig

    inherits = PondingConfig(_configs(str(tmp_path), h_max_mm=7.0))
    assert inherits.watering_h_max_mm == 7.0

    explicit = PondingConfig(_configs(str(tmp_path), h_max_mm=7.0, watering_h_max_mm=50.0))
    assert explicit.h_max_mm == 7.0
    assert explicit.watering_h_max_mm == 50.0


def test_state_blob_roundtrip_and_pre_pond_compat(tmp_path):
    core = _build_core(tmp_path)
    core.walk_window(rates=_irrigation(EXTREME_FLOW), window_s=120.0)
    assert core.surface_h[WATERING] > 0.0

    blob = core.save_state_blob()
    fresh = _build_core(tmp_path)
    fresh.load_state_blob(blob)
    assert fresh.surface_h == core.surface_h

    # Blobs written before the watering pond existed lack its key.
    core.surface_h.pop(WATERING)
    old_blob = core.save_state_blob()
    fresh = _build_core(tmp_path)
    fresh.load_state_blob(old_blob)
    assert fresh.surface_h[WATERING] == 0.0
