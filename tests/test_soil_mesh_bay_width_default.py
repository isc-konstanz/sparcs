"""sparcs.tests.test_soil_mesh_bay_width_default
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

B6: ``SoilSimulation`` used to re-parse the ``[soil_simulation.mesh]`` block
FieldSimulation already parsed eagerly, with its OWN fallback
(``_DEFAULT_BAY_WIDTH`` = 10.0) diverging from the field-level ``bay_width``
default (3.5) used everywhere else. ``_resolve_mesh_config`` (soil.py) now
reuses the context's already-parsed ``MeshConfig`` when present, and shares
ONE fallback constant for the standalone (no parent) case.

Importing ``soil.py`` pulls the full FiPy stack (see test_anchor_config_parse.py);
this exercises the pure resolution seam without building a mesh.
"""

import types

from lories import Configurations
from sparcs.components.agriculture.simulation._soil import _DEFAULT_BAY_WIDTH, MeshConfig
from sparcs.components.agriculture.simulation.soil import _resolve_mesh_config


def _configs(tmp_dir: str, **mesh_values) -> Configurations:
    """A SoilSimulation-level config: no top-level values, one [mesh] member."""
    return Configurations.load("test.conf", conf_dir=tmp_dir, require=False, mesh=mesh_values)


def test_standalone_default_matches_shared_constant(tmp_path):
    """No parent mesh_config, no context bay_width, no explicit [mesh] width ->
    falls back to the SAME 3.5 the field-level bay_width default uses (the
    orphaned 10.0 fallback is gone)."""
    assert _DEFAULT_BAY_WIDTH == 3.5
    context = types.SimpleNamespace(mesh_config=None, bay_width=None)
    mesh = _resolve_mesh_config(context, _configs(str(tmp_path)))
    assert mesh.width == 3.5


def test_in_context_reuses_parents_instance(tmp_path):
    """When the parent FieldSimulation already parsed a MeshConfig, the child
    reuses that EXACT instance instead of parsing [mesh] again -- an explicit
    field bay_width (not 3.5) still resolves correctly, and the child's own
    [mesh] block is never consulted for width."""
    parent_mesh = MeshConfig(_configs(str(tmp_path), width=4.0), bay_width=4.0)
    context = types.SimpleNamespace(mesh_config=parent_mesh, bay_width=4.0)
    mesh = _resolve_mesh_config(context, _configs(str(tmp_path)))
    assert mesh is parent_mesh
    assert mesh.width == 4.0


def test_standalone_context_bay_width_passes_through(tmp_path):
    """No parent mesh_config, but the context carries a real bay_width -> that
    value (not the shared fallback) becomes the width default."""
    context = types.SimpleNamespace(mesh_config=None, bay_width=4.0)
    mesh = _resolve_mesh_config(context, _configs(str(tmp_path)))
    assert mesh.width == 4.0


def test_explicit_width_wins_over_standalone_default(tmp_path):
    """A standalone config's own explicit [mesh] width still wins over the
    shared bay_width fallback."""
    context = types.SimpleNamespace(mesh_config=None, bay_width=None)
    mesh = _resolve_mesh_config(context, _configs(str(tmp_path), width=7.0))
    assert mesh.width == 7.0
