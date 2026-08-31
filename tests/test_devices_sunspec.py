# -*- coding: utf-8 -*-
"""
sparcs.tests.test_devices_sunspec
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Unit tests for the SunSpec protocol layer: ``SunSpecInverter`` and ``SunSpecMeter``
bind the vocabulary their device base declares. The device vocabulary itself is
tested in ``test_electrical_components.py``; what is asserted here is only what the
mixin contributes -- point names, model ids, unit id, instance, connector -- plus the
three irregular cases the standard forces (the immediate-controls model, model 201's
line-to-line point names, and the meter/inverter reactive-power spelling).

``device`` and ``instance`` are different axes and are asserted separately: ``device``
selects the Modbus unit, ``instance`` selects a repeated model within one unit.
"""

from __future__ import annotations

import pytest

from lories.components.context import registry
from lories.core import Constant
from lories.core.configs import ConfigurationError
from sparcs.components import EnergyMeter, SunSpecBinding, SunSpecInverter, SunSpecMeter
from sparcs.components.electrical import ACComponent

INVERTER_UNIT = 126
METER_UNIT = 127


@pytest.fixture
def configure_device(configure_component):
    """``configure_component`` with the required SunSpec unit id supplied."""

    def _configure(component_class, toml_text: str = "", device: int = INVERTER_UNIT) -> dict:
        return configure_component(component_class, f"device = {device}\n{toml_text}")

    return _configure


def test_component_types_registered():
    assert registry.has_type("sunspec_inverter")
    assert registry.has_type("sunspec_meter")


# ---------------------------------------------------------------- inverter


def test_inverter_binds_its_vocabulary(configure_device):
    channels = configure_device(SunSpecInverter, "phases = true\nquality = true\n")

    power = channels["power"]
    assert power["point"] == "W"
    assert power["model"] == 103
    assert power["instance"] == 1
    assert power["connector"] == "sunspec"
    assert channels["energy"]["point"] == "WH"
    assert channels["state"]["point"] == "St"
    assert channels["temperature"]["point"] == "TmpCab"
    assert channels["dc_power"]["point"] == "DCW"
    assert channels["l1_current"]["point"] == "AphA"
    assert channels["l3_voltage"]["point"] == "PhVphC"
    assert channels["reactive_power"]["point"] == "VAr"


def test_inverter_model_and_instance(configure_device):
    channels = configure_device(SunSpecInverter, "model = 101\ninstance = 2\n")

    assert all(channel["model"] == 101 for channel in channels.values())
    assert all(channel["instance"] == 2 for channel in channels.values())


def test_inverter_control_channels_target_the_controls_model(configure_device):
    channels = configure_device(SunSpecInverter, "instance = 2\ncontrol = true\n")

    # Control points live in the immediate-controls model, not the inverter model,
    # but inherit the component's instance (documented device-topology assumption)
    assert channels["power_limit"]["model"] == SunSpecInverter.CONTROLS_MODEL
    assert channels["power_limit"]["point"] == "WMaxLimPct"
    assert channels["power_limit"]["instance"] == 2
    assert channels["power_limit_enabled"]["model"] == SunSpecInverter.CONTROLS_MODEL
    assert channels["power_limit_enabled"]["point"] == "WMaxLim_Ena"
    assert channels["power"]["model"] == 103


def test_inverter_rejects_invalid_model(configure_device):
    with pytest.raises(ConfigurationError):
        configure_device(SunSpecInverter, "model = 104\n")


# ---------------------------------------------------------------- meter


def test_meter_binds_its_vocabulary(configure_device):
    channels = configure_device(SunSpecMeter, "quality = true\n")

    assert channels["power"]["model"] == 203
    assert channels["voltage"]["point"] == "PhV"
    assert channels["energy_import"]["point"] == "TotWhImp"
    assert channels["energy_export"]["point"] == "TotWhExp"
    assert channels["apparent_power"]["point"] == "VA"
    assert channels["power_factor"]["point"] == "PF"
    # The meter models spell the reactive-power point "VAR", unlike the inverter's "VAr"
    assert channels["reactive_power"]["point"] == "VAR"


def test_meter_per_phase_points(configure_device):
    channels = configure_device(SunSpecMeter, "instance = 3\nphases = true\nquality = true\n")

    assert channels["l1_power"]["point"] == "WphA"
    assert channels["l2_current"]["point"] == "AphB"
    assert channels["l3_power_factor"]["point"] == "PFphC"
    assert all(channel["instance"] == 3 for channel in channels.values())


def test_meter_line_to_line_points_diverge_on_model_201(configure_device):
    # Model 201 alone names line-to-line voltage points PPVph*, all others PhVph*
    single = configure_device(SunSpecMeter, "model = 201\nphases = true\n")
    assert single["l12_voltage"]["point"] == "PPVphAB"
    assert single["l31_voltage"]["point"] == "PPVphCA"

    three = configure_device(SunSpecMeter, "model = 203\nphases = true\n")
    assert three["l12_voltage"]["point"] == "PhVphAB"
    assert three["l31_voltage"]["point"] == "PhVphCA"


def test_meter_direction_channels_stay_unbound(configure_device):
    channels = configure_device(SunSpecMeter, "directions = true\n")

    for key in ("power_import", "power_export"):
        channel = channels[key]
        assert "point" not in channel
        assert "connector" not in channel
        assert "model" not in channel
        assert "instance" not in channel
    # Point-carrying channels still get the full SunSpec binding
    assert channels["power"]["connector"] == "sunspec"
    assert channels["power"]["model"] == 203


def test_meter_rejects_invalid_model(configure_device):
    with pytest.raises(ConfigurationError):
        configure_device(SunSpecMeter, "model = 205\n")


# ---------------------------------------------------------------- wiring


def test_connector_defaults_to_sunspec_and_is_overridable(configure_device):
    default = configure_device(SunSpecMeter, "")
    assert default["power"]["connector"] == "sunspec"

    named = configure_device(SunSpecMeter, 'connector = "grid_gateway"\n')
    assert named["power"]["connector"] == "grid_gateway"


def test_device_unit_id_is_bound_to_every_channel(configure_device):
    channels = configure_device(SunSpecInverter, "phases = true\ncontrol = true\n")

    assert all(channel["device"] == INVERTER_UNIT for channel in channels.values())
    # Control points sit in another model but still on the same unit
    assert channels["power_limit"]["device"] == INVERTER_UNIT


def test_devices_on_one_connector_address_different_units(configure_device):
    inverter = configure_device(SunSpecInverter, "", device=INVERTER_UNIT)
    meter = configure_device(SunSpecMeter, "", device=METER_UNIT)

    # Two components behind one gateway: same connector, different unit ids
    assert inverter["power"]["connector"] == meter["power"]["connector"] == "sunspec"
    assert inverter["power"]["device"] == INVERTER_UNIT
    assert meter["power"]["device"] == METER_UNIT


def test_device_unit_id_is_required(configure_component):
    with pytest.raises(ConfigurationError):
        configure_component(SunSpecMeter, "model = 203\n")


def test_bound_constants_are_declared_by_the_device():
    # The protocol layer contributes point names, never vocabulary: every constant
    # it binds must already be declared by the device base or its device class
    declared = {
        value for cls in (ACComponent, EnergyMeter) for value in vars(cls).values() if isinstance(value, Constant)
    }
    assert set(SunSpecMeter.POINTS) <= declared


def test_mixin_requires_a_bindable_component_base():
    with pytest.raises(TypeError):

        class Orphan(SunSpecBinding):
            pass


def test_mixin_must_precede_the_device_in_the_mro():
    with pytest.raises(TypeError):

        class Backwards(EnergyMeter, SunSpecBinding):
            pass
