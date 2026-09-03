# -*- coding: utf-8 -*-
"""sparcs.tests.test_anchor_runtime_extraction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Module-identity pins for the ``_anchor_runtime`` extraction --
``soil._walk_components`` / ``soil._parse_anchor_config`` must BE the
module-level functions in ``_anchor_runtime`` (re-exports, catching
copy-instead-of-move drift) -- plus the FiPy-isolation guard: ``_anchor.py``
must never import ``fipy`` or ``_soil`` (that is the whole reason the
runtime lives in its own module instead of ``_anchor.py``). The guard
parses ``_anchor.py``'s imports statically: a runtime ``sys.modules`` check
would be polluted by the package having imported the FiPy stack already.

``soil`` pulls the full lories + FiPy/Gmsh stack; ``importorskip`` keeps
this file out of environments that lack it. ``_anchor_runtime`` is imported
PLAINLY on purpose: its absence must be a hard failure (the extraction
landing is exactly what this file pins), never a silent skip.
"""

import ast
import pathlib

import pytest

soil = pytest.importorskip("sparcs.components.agriculture.simulation.soil")

from sparcs.components.agriculture.simulation import _anchor, _anchor_runtime  # noqa: E402


@pytest.mark.parametrize(
    ("soil_name", "runtime_name"),
    [
        ("_walk_components", "_walk_components"),
        ("_parse_anchor_config", "_parse_anchor_config"),
    ],
)
def test_soil_reexport_is_runtime_function(soil_name, runtime_name):
    assert getattr(soil, soil_name) is getattr(_anchor_runtime, runtime_name)


def test_anchor_module_stays_fipy_free():
    tree = ast.parse(pathlib.Path(_anchor.__file__).read_text(encoding="utf-8"))
    imported: set = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported.add(node.module.split(".")[0])
            else:
                # `from . import x` -- the imported names are the modules.
                imported.update(alias.name for alias in node.names)
    assert "fipy" not in imported, "_anchor.py must stay FiPy-free (import isolation)"
    assert "_soil" not in imported, "_anchor.py must not pull _soil (it imports FiPy)"
