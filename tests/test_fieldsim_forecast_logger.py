# -*- coding: utf-8 -*-
"""
tests.test_fieldsim_forecast_logger
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``SoilPredictor.activate`` refuses a ``logger`` id that resolves to nothing or to a connector without ``write()``.
Unset is a no-op; a bare id resolves up the predictor's id path, so a root-level connector serves a nested one.
"""

from types import SimpleNamespace

import pytest

from lories.core import ConfigurationUnavailableError
from sparcs.components.agriculture.simulation import components

_PREDICTOR_ID = "agri.field_1.field_simulation.soil_predictor"


def _bare(logger_id, context=None, monkeypatch=None, **attrs) -> components.SoilPredictor:
    p = object.__new__(components.SoilPredictor)
    p._name = "test_predictor"
    p._id = _PREDICTOR_ID
    p._logger_id = logger_id
    if monkeypatch is not None:
        connectors = SimpleNamespace(context=dict(context or {}), **attrs)
        monkeypatch.setattr(components.SoilPredictor, "connectors", property(lambda self: connectors))
    return p


class _Writable:
    def write(self, frame):
        pass


def test_validate_raises_when_nothing_resolves(monkeypatch):
    p = _bare("db", monkeypatch=monkeypatch)
    with pytest.raises(ConfigurationUnavailableError, match="db"):
        p.tables().validate_logger_connector()


def test_validate_raises_when_resolved_object_has_no_write(monkeypatch):
    p = _bare("db", {"db": object()}, monkeypatch=monkeypatch)
    with pytest.raises(ConfigurationUnavailableError, match="write"):
        p.tables().validate_logger_connector()


def test_validate_noop_when_logger_not_configured():
    p = _bare(None)
    p.tables().validate_logger_connector()  # a resolution attempt would need the unset connectors


def test_validate_passes_with_a_writable_connector(monkeypatch):
    p = _bare("db", {"db": _Writable()}, monkeypatch=monkeypatch)
    p.tables().validate_logger_connector()


@pytest.mark.parametrize(
    "declared_id",
    ["mariadb", "agri.mariadb", f"{_PREDICTOR_ID}.mariadb"],
    ids=["root", "ancestor", "own"],
)
def test_a_bare_id_resolves_up_the_predictor_path(monkeypatch, declared_id):
    connector = _Writable()
    p = _bare("mariadb", {declared_id: connector, "other.mariadb": object()}, monkeypatch=monkeypatch)

    assert p.tables().resolve_logger_connector("mariadb") is connector


def test_the_innermost_declaration_wins(monkeypatch):
    inner = _Writable()
    p = _bare("mariadb", {"mariadb": _Writable(), "agri.field_1.mariadb": inner}, monkeypatch=monkeypatch)

    assert p.tables().resolve_logger_connector("mariadb") is inner


def test_the_component_scoped_attribute_stays_a_fallback(monkeypatch):
    connector = _Writable()
    p = _bare("db", monkeypatch=monkeypatch, db=connector)

    assert p.tables().resolve_logger_connector("db") is connector


def test_activate_runs_super_then_validator(monkeypatch):
    order = []
    monkeypatch.setattr(components.ChannelNamespace, "activate", lambda self: order.append("super"), raising=False)
    p = _bare("db")
    p.tables = lambda: SimpleNamespace(validate_logger_connector=lambda: order.append("validate"))
    p.activate()
    assert order == ["super", "validate"]
