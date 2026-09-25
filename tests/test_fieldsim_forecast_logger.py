# -*- coding: utf-8 -*-
"""sparcs.tests.test_fieldsim_forecast_logger
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``SoilPredictor.activate`` refuses a ``logger`` id that resolves to nothing or to a
connector without ``write()``; unset stays a no-op.
"""

from types import SimpleNamespace

import pytest

from lories.core import ConfigurationUnavailableError
from sparcs.components.agriculture.fieldsim import components


def _bare(logger_id, connector=None, monkeypatch=None) -> components.SoilPredictor:
    p = object.__new__(components.SoilPredictor)
    p._name = "test_predictor"
    p._logger_id = logger_id
    p._logger_connector_from_channel = lambda: None
    if monkeypatch is not None:
        monkeypatch.setattr(
            components.SoilPredictor,
            "connectors",
            property(lambda self: SimpleNamespace(db=connector) if connector is not None else SimpleNamespace()),
        )
    return p


class _Writable:
    def write(self, frame):
        pass


def test_validate_raises_when_nothing_resolves(monkeypatch):
    p = _bare("db", connector=None, monkeypatch=monkeypatch)
    with pytest.raises(ConfigurationUnavailableError, match="db"):
        p.tables().validate_logger_connector()


def test_validate_raises_when_resolved_object_has_no_write(monkeypatch):
    p = _bare("db", connector=object(), monkeypatch=monkeypatch)
    with pytest.raises(ConfigurationUnavailableError, match="write"):
        p.tables().validate_logger_connector()


def test_validate_noop_when_logger_not_configured():
    p = _bare(None)
    p._logger_connector_from_channel = lambda: (_ for _ in ()).throw(AssertionError("must not resolve"))
    p.tables().validate_logger_connector()


def test_validate_passes_with_a_writable_connector(monkeypatch):
    p = _bare("db", connector=_Writable(), monkeypatch=monkeypatch)
    p.tables().validate_logger_connector()


def test_validate_passes_via_the_channel_resolution_path(monkeypatch):
    p = _bare("mariadb", connector=None, monkeypatch=monkeypatch)
    p._logger_connector_from_channel = lambda: _Writable()
    p.tables().validate_logger_connector()


def test_activate_runs_super_then_validator(monkeypatch):
    order = []
    monkeypatch.setattr(components.ChannelNamespace, "activate", lambda self: order.append("super"), raising=False)
    p = _bare("db")
    p.tables = lambda: SimpleNamespace(validate_logger_connector=lambda: order.append("validate"))
    p.activate()
    assert order == ["super", "validate"]
