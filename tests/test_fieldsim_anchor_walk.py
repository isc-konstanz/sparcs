# -*- coding: utf-8 -*-
"""
tests.test_fieldsim_anchor_walk
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``_walk_components`` feeds sensor-probe discovery: a container shape it does
not recognise falls back to plain iteration, a genuinely raising ``.values()``
propagates, and an unwalkable subtree is dropped with ONE named WARNING
instead of vanishing silently. ``soil_tuning`` carries its own duplicate.
"""

import logging
from types import SimpleNamespace

import pytest

from sparcs.components.agriculture.fieldsim.anchor_runtime import _walk_components

soil_tuning = pytest.importorskip("soil_tuning")

WALKERS = [
    pytest.param(_walk_components, id="fieldsim"),
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
    """A .values() raising outside (AttributeError, TypeError) is a real failure,
    not a container shape mismatch: it must propagate, never be downgraded to
    the fallback. A broad `except Exception` here fails this test."""
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
