# -*- coding: utf-8 -*-
"""Regression: Genuchten conductivity/diffusivity stay finite and real at the
Se domain boundaries. Before the internal clipping (mirroring BrooksCorey's),
``dh_dse(0)`` was NaN, ``dh_dse(1.0)`` raised ZeroDivisionError on floats, and
``k_from_se`` of an out-of-domain Se returned a complex number -- all reachable
from an unvalidated ``[pde] ic_se`` of 0/1 on the first PDE assembly.
"""

import pytest

import numpy as np
from sparcs.components.agriculture.soil.models import Genuchten, create_soil_model

VG_PARAMS = {"theta_r": 0.05, "theta_s": 0.45, "alpha": 0.02, "n": 1.5, "k_s": 1e-6}


def test_genuchten_k_finite_and_real_at_boundaries():
    vg = Genuchten(**VG_PARAMS)
    se = np.array([-0.01, 0.0, 0.5, 1.0, 1.01])
    k = np.asarray(vg.k_from_se(se))
    assert not np.iscomplexobj(k)
    assert np.all(np.isfinite(k))
    assert np.all(k >= 0)
    assert k[3] == pytest.approx(VG_PARAMS["k_s"])


def test_genuchten_dh_dse_finite_at_boundaries():
    vg = Genuchten(**VG_PARAMS)
    slope = np.asarray(vg.dh_dse(np.array([0.0, 0.5, 1.0])))
    assert np.all(np.isfinite(slope))
    assert np.all(slope > 0)


def test_genuchten_dh_dse_accepts_plain_floats_at_bounds():
    vg = Genuchten(**VG_PARAMS)
    assert np.isfinite(vg.dh_dse(0.0))
    assert np.isfinite(vg.dh_dse(1.0))


def test_create_soil_model_reports_missing_alpha_and_n():
    with pytest.raises(ValueError, match="alpha"):
        create_soil_model(None, theta_r=0.05, theta_s=0.45, k_s=1e-6, n=1.5)
    with pytest.raises(ValueError, match="'n'"):
        create_soil_model(None, theta_r=0.05, theta_s=0.45, k_s=1e-6, alpha=0.02)
