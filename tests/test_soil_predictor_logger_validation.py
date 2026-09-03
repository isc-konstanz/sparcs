# -*- coding: utf-8 -*-
"""sparcs.tests.test_soil_predictor_logger_validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Issue 25 (W2.7): a misconfigured ``[soil_predictor] logger =`` id used to
degrade at tick time -- "connector not found; skipping" per table write, per
tick, forever. It is a pure wiring error, so ``SoilPredictor.activate()`` now
validates it once (``_validate_logger_connector``, mirroring
``_validate_irrigation_input``): connectors connect BEFORE components activate
in lories, so a failed resolution at activate IS "id does not exist" and
raises ``ConfigurationUnavailableError``. All write-time behavior (the
resolution ladder, warn-and-skip) is byte-for-byte unchanged -- only the
never-resolvable case became fail-fast.

Tests drive ``_validate_logger_connector`` directly on bare ``object.__new__``
instances (the irrigation-precedent style): ``Component.activate`` has an
is_configured guard that bare instances cannot satisfy.
"""

from types import SimpleNamespace

import pytest

lories_core = pytest.importorskip("lories.core")
ConfigurationUnavailableError = lories_core.ConfigurationUnavailableError

soil_predictor = pytest.importorskip("sparcs.components.agriculture.simulation.soil_predictor")
SoilPredictor = soil_predictor.SoilPredictor

_soil = pytest.importorskip("sparcs.components.agriculture.simulation._soil")
SoilBase = _soil.SoilBase


def _bare(logger_id, connector=None, monkeypatch=None) -> SoilPredictor:
    p = object.__new__(SoilPredictor)
    p._name = "test_predictor"
    p._logger_id = logger_id
    p._logger_connector_from_channel = lambda: None  # force the id-based ladder
    if monkeypatch is not None:
        monkeypatch.setattr(
            SoilPredictor,
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
        p._validate_logger_connector()


def test_validate_raises_when_resolved_object_has_no_write(monkeypatch):
    p = _bare("db", connector=object(), monkeypatch=monkeypatch)
    with pytest.raises(ConfigurationUnavailableError, match="write"):
        p._validate_logger_connector()


def test_validate_noop_when_logger_not_configured():
    p = object.__new__(SoilPredictor)
    p._name = "test_predictor"
    p._logger_id = None
    p._validate_logger_connector()  # must not raise, must not resolve anything


def test_validate_passes_with_a_writable_connector(monkeypatch):
    p = _bare("db", connector=_Writable(), monkeypatch=monkeypatch)
    p._validate_logger_connector()  # must not raise


def test_validate_passes_via_the_channel_resolution_path(monkeypatch):
    """The common rig shape: a root-level connector unreachable via the
    component-scoped id lookup but already bound by the header channel."""
    p = _bare("mariadb", connector=None, monkeypatch=monkeypatch)
    p._logger_connector_from_channel = lambda: _Writable()
    p._validate_logger_connector()  # must not raise


def test_activate_runs_super_then_validator(monkeypatch):
    """Structural pin: the new activate() override calls super().activate()
    first, then the validator. SoilBase gains a recorder activate() for the
    test (it defines none; monkeypatch adds + auto-removes it) so the
    otherwise-silent no-op super().activate() becomes observable -- bare
    object.__new__ instances never pass the Activator metaclass, so the
    class-level override is called directly."""
    order = []
    monkeypatch.setattr(SoilBase, "activate", lambda self: order.append("super"), raising=False)
    p = object.__new__(SoilPredictor)
    p._name = "test_predictor"
    p._logger_id = None
    p._validate_logger_connector = lambda: order.append("validate")

    p.activate()

    assert order == ["super", "validate"]
