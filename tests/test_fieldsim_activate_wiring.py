# -*- coding: utf-8 -*-
"""tests.test_fieldsim_activate_wiring
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

What the field wires up before the ticker starts: the field-level ``[plot]``
block cascades into every child as its default, a configured irrigation with
no usable input raises, and a ``SIMULATION_STATE`` channel that cannot
round-trip warns about the warm start it will not do. Bare ``object.__new__``
instances exercise the guards; they touch only their arguments and ``self``.
"""

from types import SimpleNamespace

import pytest

from lories import Component, Configurations
from lories.core import ConfigurationUnavailableError
from sparcs.components.agriculture.fieldsim.components import ChannelNamespace, FieldSimulation, SoilSimulation


def _field(**attrs) -> FieldSimulation:
    field = object.__new__(FieldSimulation)
    field._name = "test_field_simulation"
    field.irrigation = None
    field._irrigation_flow_channel = None
    field._irrigation_state_channel = None
    field.setup = SimpleNamespace(soil=SimpleNamespace(drip=SimpleNamespace(explicit=False)))
    for key, value in attrs.items():
        setattr(field, key, value)
    return field


def _wired(connector: bool):
    return SimpleNamespace(has_connector=lambda: connector)


def _explicit_drip(explicit: bool):
    return SimpleNamespace(soil=SimpleNamespace(drip=SimpleNamespace(explicit=explicit)))


# --------------------------------------------------------------------------- plot cascade


def _child_block(tmp_path, child_type: str, **values) -> Configurations:
    """Reproduce ``FieldSimulation.configure``'s child-config path."""
    field = Configurations.load("field.conf", conf_dir=str(tmp_path), require=False, **values)
    defaults = Component._build_defaults(field, includes=["model", "plot"], strict=True)
    return field.get_member(child_type, defaults=defaults)


def test_field_plot_enabled_cascades_to_a_child_without_its_own_block(tmp_path):
    block = _child_block(tmp_path, "soil_simulation", plot={"enabled": False}, soil_simulation={})

    assert ChannelNamespace._plot_enabled(block) is False


def test_child_plot_block_overrides_the_field_default(tmp_path):
    block = _child_block(
        tmp_path, "ground_shading", plot={"enabled": False}, ground_shading={"plot": {"enabled": True}}
    )

    assert ChannelNamespace._plot_enabled(block) is True


def test_no_field_plot_block_leaves_plotting_on(tmp_path):
    block = _child_block(tmp_path, "soil_simulation", soil_simulation={})

    assert ChannelNamespace._plot_enabled(block) is True


# --------------------------------------------------------------------------- irrigation input


def test_validate_noop_when_no_irrigation_component():
    """Rain-fed field: no [irrigation] block -> 0 l/min is deliberate, not a raise."""
    _field()._validate_irrigation_input()  # must not raise


def test_validate_passes_with_wired_flow():
    _field(irrigation=object(), _irrigation_flow_channel=_wired(True))._validate_irrigation_input()


def test_validate_passes_with_wired_state_and_explicit_drip():
    _field(
        irrigation=object(),
        _irrigation_state_channel=_wired(True),
        setup=_explicit_drip(True),
    )._validate_irrigation_input()


def test_validate_raises_when_state_wired_but_drip_not_explicit():
    field = _field(irrigation=object(), _irrigation_state_channel=_wired(True))
    with pytest.raises(ConfigurationUnavailableError, match="soil_simulation.drip"):
        field._validate_irrigation_input()


def test_validate_raises_when_flow_channel_has_no_connector():
    field = _field(irrigation=object(), _irrigation_flow_channel=_wired(False))
    with pytest.raises(ConfigurationUnavailableError):
        field._validate_irrigation_input()


def test_validate_raises_when_nothing_wired():
    field = _field(irrigation=object())
    with pytest.raises(ConfigurationUnavailableError, match="no usable input"):
        field._validate_irrigation_input()


# --------------------------------------------------------------------------- warm start


def _soil_data(*, has_logger: bool, has_connector: bool) -> dict:
    channel = SimpleNamespace(
        has_logger=lambda *ids: has_logger,
        has_connector=lambda id=None: has_connector,
    )
    return {SoilSimulation.SIMULATION_STATE: channel}


def test_write_only_state_channel_warns_and_skips_registration(caplog):
    with caplog.at_level("WARNING"):
        should_register = _field()._check_state_channel_warm_start(_soil_data(has_logger=True, has_connector=False))

    assert should_register is False
    assert any("read-side connector" in message for message in caplog.messages)


def test_no_logger_state_channel_warns_no_persistence(caplog):
    with caplog.at_level("WARNING"):
        should_register = _field()._check_state_channel_warm_start(_soil_data(has_logger=False, has_connector=False))

    assert should_register is False
    assert any("no logger" in message.lower() for message in caplog.messages)


def test_fully_wired_state_channel_registers_without_warning(caplog):
    with caplog.at_level("WARNING"):
        should_register = _field()._check_state_channel_warm_start(_soil_data(has_logger=True, has_connector=True))

    assert should_register is True
    assert caplog.messages == []
