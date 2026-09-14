# -*- coding: utf-8 -*-
"""sparcs.tests.test_predictor_candidates_aliases
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Module-identity pins for the ``_predictor_candidates`` extraction:
``SoilPredictor`` keeps every one of the 11 pinned ``_x`` static-method names
as a delegating ``staticmethod`` alias onto the matching, glossary-correct
function in ``_predictor_candidates``, and re-exports ``WateringWindow``
from the same module. Attribute access on a class unwraps ``staticmethod``
to the plain underlying function object (true since ``staticmethod`` has
always defined ``__get__``), so ``SoilPredictor._x is _predictor_candidates.y``
is a valid identity check -- it catches copy-instead-of-move drift and
duplicate definitions, not just behavioral equivalence.

Importing ``soil_predictor`` pulls the full lories + soil (FiPy/Gmsh) stack
via ``_soil.py``; ``importorskip`` keeps this out of environments that lack
it, matching the other ``soil_predictor`` unit tests. ``_predictor_candidates``
is imported PLAINLY on purpose: its absence must be a hard failure (the
extraction landing is exactly what this file pins), never a silent skip.
No FiPy-heavy fixtures are needed here -- this is pure identity, not
behavior -- so nothing is marked slow.
"""

import pytest

soil_predictor = pytest.importorskip("sparcs.components.agriculture.simulation.soil_predictor")

from sparcs.components.agriculture.simulation import _predictor_candidates  # noqa: E402

SoilPredictor = soil_predictor.SoilPredictor


@pytest.mark.parametrize(
    ("pinned_name", "module_name"),
    [
        ("_build_ladder", "build_candidate_grid"),
        ("_check_combo_cap", "check_candidate_cap"),
        ("_score_candidate", "score_candidate"),
        ("_select", "select_candidate"),
        ("_total_minutes", "total_minutes"),
        ("_build_flow_schedule", "build_flow_schedule"),
        ("_split_interval", "split_interval"),
        ("_resolve_window_start", "resolve_window_start"),
        ("_derive_flow_m3s", "derive_flow_m3s"),
        ("_current_boundary", "current_boundary"),
        ("_resolve_ode_config", "resolve_ode_config"),
    ],
)
def test_soil_predictor_alias_is_predictor_candidates_function(pinned_name, module_name):
    pinned = getattr(SoilPredictor, pinned_name)
    target = getattr(_predictor_candidates, module_name)
    assert pinned is target


def test_watering_window_is_reexported_from_predictor_candidates():
    assert soil_predictor.WateringWindow is _predictor_candidates.WateringWindow
