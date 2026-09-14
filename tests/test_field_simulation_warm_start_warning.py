# -*- coding: utf-8 -*-
"""sparcs.tests.test_field_simulation_warm_start_warning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Issue 02(d): a configure/activate-time WARNING when ``SIMULATION_STATE`` has a
logger but no read-side connector -- today the restore silently never
registers (``base.py``: the listener registration was already gated on
``has_connector()``, but nothing warned about the write-only case).
``_check_state_channel_warm_start`` is the extracted, directly-testable guard:
it warns and returns whether the caller should register the restore listener.

Importing ``base`` pulls the full lories + soil (FiPy/Gmsh) stack via
``soil.py``; ``importorskip`` keeps this out of environments that lack it (the
full check runs on the box). The method only touches its ``soil_data`` argument
and ``self.name``, so a bare ``object.__new__`` instance exercises it without a
Component bootstrap (the same pattern ``test_field_simulation_irrigation.py``
uses).
"""

import types

import pytest

base = pytest.importorskip("sparcs.components.agriculture.simulation.base")
FieldSimulation = base.FieldSimulation


def _soil_data(*, has_logger: bool, has_connector: bool):
    channel = types.SimpleNamespace(
        has_logger=lambda *ids: has_logger,
        has_connector=lambda id=None: has_connector,
    )
    return types.SimpleNamespace(simulation_state=channel)


def test_write_only_state_channel_warns_and_skips_registration(caplog):
    sim = object.__new__(FieldSimulation)
    sim._name = "test_field_simulation"
    soil_data = _soil_data(has_logger=True, has_connector=False)

    with caplog.at_level("WARNING"):
        should_register = FieldSimulation._check_state_channel_warm_start(sim, soil_data)

    assert should_register is False
    assert any("read-side connector" in message for message in caplog.messages)


def test_no_logger_state_channel_warns_no_persistence(caplog):
    sim = object.__new__(FieldSimulation)
    sim._name = "test_field_simulation"
    soil_data = _soil_data(has_logger=False, has_connector=False)

    with caplog.at_level("WARNING"):
        should_register = FieldSimulation._check_state_channel_warm_start(sim, soil_data)

    assert should_register is False
    assert any("no logger" in message.lower() for message in caplog.messages)


def test_fully_wired_state_channel_registers_without_warning(caplog):
    sim = object.__new__(FieldSimulation)
    sim._name = "test_field_simulation"
    soil_data = _soil_data(has_logger=True, has_connector=True)

    with caplog.at_level("WARNING"):
        should_register = FieldSimulation._check_state_channel_warm_start(sim, soil_data)

    assert should_register is True
    assert caplog.messages == []
