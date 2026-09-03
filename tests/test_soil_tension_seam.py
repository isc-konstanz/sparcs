# -*- coding: utf-8 -*-
"""sparcs.tests.test_soil_tension_seam
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Shape-coercion pins for ``SoilBase._tension_from_se`` -- the single
component-publish Se -> tension seam (issue 38): scalar in -> ``float``
out, sequence/ndarray in -> ``list[float]`` out, values exactly what
``psi_from_se`` produced. The seam owns coercion only; the signed-negative
convention is enforced in ``SoilModel.psi_from_se`` and pinned by
``test_psi_from_se_dry_se_yields_negative_matric_potential``.

Both publishing components inherit the seam from ``SoilBase``, so the pins
run against bare ``object.__new__`` instances of each with the same
``SimpleNamespace`` ``_pde`` stub the tension suites already use.
"""

import pytest

import numpy as np

soil_module = pytest.importorskip("sparcs.components.agriculture.simulation.soil")

from types import SimpleNamespace  # noqa: E402

from sparcs.components.agriculture.simulation.soil_predictor import SoilPredictor  # noqa: E402

SoilSimulation = soil_module.SoilSimulation


def _bare(component_cls):
    instance = object.__new__(component_cls)
    instance._pde = SimpleNamespace(
        soil_model=SimpleNamespace(psi_from_se=lambda values: np.asarray(values, dtype=float) * -2.0)
    )
    return instance


@pytest.mark.parametrize("component_cls", [SoilSimulation, SoilPredictor])
def test_scalar_in_float_out(component_cls):
    result = _bare(component_cls)._tension_from_se(0.5)
    # np.float64 subclasses float, so also exclude np.floating: the seam's
    # promise is a PLAIN float on the publish path.
    assert isinstance(result, float) and not isinstance(result, np.floating)
    assert result == -1.0


@pytest.mark.parametrize("component_cls", [SoilSimulation, SoilPredictor])
def test_sequence_in_list_of_floats_out(component_cls):
    result = _bare(component_cls)._tension_from_se([0.5, 0.25, 0.0])
    assert isinstance(result, list)
    assert all(isinstance(v, float) and not isinstance(v, np.floating) for v in result)
    assert result == [-1.0, -0.5, 0.0]


def test_ndarray_in_list_out():
    result = _bare(SoilPredictor)._tension_from_se(np.array([1.0, 0.5]))
    assert result == [-2.0, -1.0]
