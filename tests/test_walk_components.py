# -*- coding: utf-8 -*-
"""sparcs.tests.test_walk_components
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Issue 26 (W2.8): ``_walk_components`` swallowed child-iteration failures twice,
ending in a bare ``pass`` (the only pass-only handler in the subpackage) -- a
component whose children raise on iteration silently vanished from sensor-probe
discovery, invisibly feeding the anchor-discovery failure mode. The first catch
is now narrowed to (AttributeError, TypeError) ("not a mapping / no .values")
and the last-resort catch logs ONE WARNING naming the component whose subtree
is dropped. The identical duplicate in soil_tuning.py gets the same change
(no shared-helper extraction -- Wave 3 owns structural moves).
"""

import logging
from types import SimpleNamespace

import pytest

soil = pytest.importorskip("sparcs.components.agriculture.simulation.soil")
soil_tuning = pytest.importorskip("soil_tuning")

WALKERS = [
    pytest.param(soil._walk_components, id="soil"),
    pytest.param(soil_tuning._walk_components, id="soil_tuning"),
]


class _BrokenChildren:
    """No .values() (AttributeError -> first catch) and raising iteration
    (-> last-resort catch logs)."""

    def __iter__(self):
        raise RuntimeError("broken container")


class _ExplodingValues:
    """.values() itself raises OUTSIDE (AttributeError, TypeError): the
    narrowed first catch must let it propagate, never mask it as a
    shape-mismatch and fall back."""

    def values(self):
        raise RuntimeError("corrupt registry")

    def __iter__(self):
        raise AssertionError("fallback iteration must not be attempted")


class _ValuesNotCallable:
    """.values is not callable (TypeError -> first catch); iteration works."""

    values = None

    def __init__(self, children):
        self._children = children

    def __iter__(self):
        return iter(self._children)


@pytest.mark.parametrize("walk", WALKERS)
def test_dict_shaped_children_walk_normally_no_log(walk, caplog):
    leaf = SimpleNamespace(key="leaf", components=None)
    root = SimpleNamespace(key="root", components={"leaf": leaf})

    with caplog.at_level(logging.WARNING):
        out = walk(root)

    assert out == [root, leaf]
    assert caplog.records == []


@pytest.mark.parametrize("walk", WALKERS)
def test_valuesless_children_fall_back_to_plain_iteration(walk):
    """A list-shaped container has no .values() (AttributeError, first catch
    narrowed) -- the fallback still walks the children."""
    leaf = SimpleNamespace(key="leaf", components=None)
    root = SimpleNamespace(key="root", components=[leaf])

    out = walk(root)

    assert out == [root, leaf]


@pytest.mark.parametrize("walk", WALKERS)
def test_genuinely_raising_values_propagates(walk):
    """THE discriminator for the narrowed first catch: a .values() that raises
    outside (AttributeError, TypeError) is a real failure, not a container
    shape mismatch -- it must propagate to the caller (soil.py: contained by
    validate_sensor_probes' strict/log-and-continue split), never be silently
    downgraded to the fallback. A broad `except Exception` here fails this test."""
    root = SimpleNamespace(key="root", components=_ExplodingValues())

    with pytest.raises(RuntimeError, match="corrupt registry"):
        walk(root)


@pytest.mark.parametrize("walk", WALKERS)
def test_non_callable_values_falls_back_to_plain_iteration(walk):
    """The TypeError arm of the first catch: `.values` exists but is not
    callable -- a shape mismatch, so the fallback iteration still walks."""
    leaf = SimpleNamespace(key="leaf", components=None)
    root = SimpleNamespace(key="root", components=_ValuesNotCallable([leaf]))

    out = walk(root)

    assert out == [root, leaf]


@pytest.mark.parametrize("walk", WALKERS)
def test_unwalkable_children_log_one_warning_and_walk_continues(walk, caplog):
    broken = SimpleNamespace(key="broken_component", components=_BrokenChildren())
    healthy_leaf = SimpleNamespace(key="healthy_leaf", components=None)
    healthy = SimpleNamespace(key="healthy", components={"leaf": healthy_leaf})
    root = SimpleNamespace(key="root", components={"broken": broken, "healthy": healthy})

    with caplog.at_level(logging.WARNING):
        out = walk(root)

    # the broken component itself is still in the walk; only its SUBTREE is dropped
    assert broken in out
    assert healthy in out and healthy_leaf in out
    warnings = [r for r in caplog.records if "skipping its subtree" in r.getMessage()]
    assert len(warnings) == 1
    assert "broken_component" in warnings[0].getMessage()


@pytest.mark.parametrize("walk", WALKERS)
def test_unwalkable_container_without_key_is_named_by_repr(walk, caplog):
    broken = SimpleNamespace(components=_BrokenChildren())  # no .key attr

    with caplog.at_level(logging.WARNING):
        walk(broken)

    warnings = [r for r in caplog.records if "skipping its subtree" in r.getMessage()]
    assert len(warnings) == 1
    assert "namespace" in warnings[0].getMessage()  # repr(SimpleNamespace(...))
