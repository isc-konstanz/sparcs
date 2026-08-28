# -*- coding: utf-8 -*-
"""
sparcs.tests.test_devices_sunspec
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Unit tests for the SunSpec protocol layer: ``SunSpecInverter`` and ``SunSpecMeter``
bind the vocabulary their device base declares. The device vocabulary itself is
tested in ``test_electrical_components.py``; what is asserted here is only what the
mixin contributes -- point names, model ids, instance, connector -- plus the three
irregular cases the standard forces (the immediate-controls model, model 201's
line-to-line point names, and the meter/inverter reactive-power spelling).
"""

from __future__ import annotations

import pytest

from lories.components.context import registry
from lories.core import Constant
from lories.core.configs import ConfigurationError
from sparcs.components import EnergyMeter, SunSpecBinding, SunSpecInverter, SunSpecMeter
from sparcs.components.electrical import ACComponent


def test_component_types_registered():
    assert registry.has_type("sunspec_inverter")
    assert registry.has_type("sunspec_meter")


# ---------------------------------------------------------------- inverter


def test_inverter_binds_its_vocabulary(configure_component):
    channels = configure_component(SunSpecInverter, "phases = true\nquality = true\n")

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


def test_inverter_model_and_instance(configure_component):
    channels = configure_component(SunSpecInverter, "model = 101\ninstance = 2\n")

    assert all(channel["model"] == 101 for channel in channels.values())
    assert all(channel["instance"] == 2 for channel in channels.values())


def test_inverter_control_channels_target_the_controls_model(configure_component):
    channels = configure_component(SunSpecInverter, "instance = 2\ncontrol = true\n")

    # Control points live in the immediate-controls model, not the inverter model,
    # but inherit the component's instance (documented device-topology assumption)
    assert channels["power_limit"]["model"] == SunSpecInverter.CONTROLS_MODEL
    assert channels["power_limit"]["point"] == "WMaxLimPct"
    assert channels["power_limit"]["instance"] == 2
    assert channels["power_limit_enabled"]["model"] == SunSpecInverter.CONTROLS_MODEL
    assert channels["power_limit_enabled"]["point"] == "WMaxLim_Ena"
    assert channels["power"]["model"] == 103


def test_inverter_rejects_invalid_model(configure_component):
    with pytest.raises(ConfigurationError):
        configure_component(SunSpecInverter, "model = 104\n")


# ---------------------------------------------------------------- meter


def test_meter_binds_its_vocabulary(configure_component):
    channels = configure_component(SunSpecMeter, "quality = true\n")

    assert channels["power"]["model"] == 203
    assert channels["voltage"]["point"] == "PhV"
    assert channels["energy_import"]["point"] == "TotWhImp"
    assert channels["energy_export"]["point"] == "TotWhExp"
    assert channels["apparent_power"]["point"] == "VA"
    assert channels["power_factor"]["point"] == "PF"
    # The meter models spell the reactive-power point "VAR", unlike the inverter's "VAr"
    assert channels["reactive_power"]["point"] == "VAR"


def test_meter_per_phase_points(configure_component):
    channels = configure_component(SunSpecMeter, "instance = 3\nphases = true\nquality = true\n")

    assert channels["l1_power"]["point"] == "WphA"
    assert channels["l2_current"]["point"] == "AphB"
    assert channels["l3_power_factor"]["point"] == "PFphC"
    assert all(channel["instance"] == 3 for channel in channels.values())


def test_meter_line_to_line_points_diverge_on_model_201(configure_component):
    # Model 201 alone names line-to-line voltage points PPVph*, all others PhVph*
    single = configure_component(SunSpecMeter, "model = 201\nphases = true\n")
    assert single["l12_voltage"]["point"] == "PPVphAB"
    assert single["l31_voltage"]["point"] == "PPVphCA"

    three = configure_component(SunSpecMeter, "model = 203\nphases = true\n")
    assert three["l12_voltage"]["point"] == "PhVphAB"
    assert three["l31_voltage"]["point"] == "PhVphCA"


def test_meter_direction_channels_stay_unbound(configure_component):
    channels = configure_component(SunSpecMeter, "directions = true\n")

    for key in ("power_import", "power_export"):
        channel = channels[key]
        assert "point" not in channel
        assert "connector" not in channel
        assert "model" not in channel
        assert "instance" not in channel
    # Point-carrying channels still get the full SunSpec binding
    assert channels["power"]["connector"] == "sunspec"
    assert channels["power"]["model"] == 203


def test_meter_rejects_invalid_model(configure_component):
    with pytest.raises(ConfigurationError):
        configure_component(SunSpecMeter, "model = 205\n")


# ---------------------------------------------------------------- wiring


def test_connector_defaults_to_sunspec_and_is_overridable(configure_component):
    default = configure_component(SunSpecMeter, "")
    assert default["power"]["connector"] == "sunspec"

    named = configure_component(SunSpecMeter, 'connector = "grid_gateway"\n')
    assert named["power"]["connector"] == "grid_gateway"


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
