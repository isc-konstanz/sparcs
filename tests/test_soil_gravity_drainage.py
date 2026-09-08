# -*- coding: utf-8 -*-
"""
tests.test_soil_gravity_drainage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Gravity drains a uniformly wet column through the bottom face at ~K(Se) and
must not feed K(Se_top) back in through the surface (FiPy's divergence sums
exterior faces; before the fix the two cancelled and wet soil never drained).
"""

import pytest

import numpy as np

pytestmark = pytest.mark.slow

from sparcs.components.agriculture.simulation._soil import RHO_W, FluxRates  # noqa: E402

IC_SE = 0.9
WINDOW_S = 600.0


def test_uniform_wet_column_drains_at_bottom_conductivity(pde_core_factory):
    pde = pde_core_factory("soil_gravity", ic_se=IC_SE)
    before = pde.total_water()
    top_cells = np.unique(np.concatenate([pde.segment_cells[n] for n in pde.top_segment_names]))

    result = pde.walk_window(
        rates=FluxRates(seg_evap={}, seg_transp={}, flow_m3s=0.0, rain_flux=0.0),
        window_s=WINDOW_S,
        accept_at_dt_min=False,
    )
    assert result.ok

    k_ic = float(np.asarray(pde.soil_model.k_from_se(IC_SE)))
    expected = k_ic * pde.segment_face_len["GroundBottomSegment"] * WINDOW_S * RHO_W
    drained = before - pde.total_water()
    assert 0.5 * expected < drained < 1.5 * expected
    assert np.all(np.asarray(pde.rel_sat.value)[top_cells] < IC_SE - 1e-3)
