# -*- coding: utf-8 -*-
"""
sparcs.tests.test_electrical_components
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Unit tests for the protocol-free device layer: the ``ElectricalDevice`` vocabulary
and the flag-gated channel groups ``EnergyMeter`` and ``SolarInverter`` add on top.
Nothing here may know about SunSpec -- the load-bearing assertions are the negative
ones, which pin that these classes declare channels carrying no ``point``, ``model``,
``instance`` or ``connector``. Bindings are tested in ``test_sunspec_components.py``.
"""

from __future__ import annotations

from lories.components.context import registry
from sparcs.components import ElectricalDevice, EnergyMeter
from sparcs.components.solar import SolarInverter

BINDING_KEYS = ("point", "model", "instance", "connector")


def test_component_types_registered():
    assert registry.has_type("meter")
    assert registry.has_type("pv_inverter")
    assert registry.has_type("solar_inverter")


def test_inverter_is_not_registered_as_inverter():
    # '[inverter]' in a PV system config already means that system's pvlib inverter
    # parameters, and both would otherwise take a 'model' key with incompatible meanings
    assert not registry.has_type("inverter")


def test_electrical_device_is_not_registered():
    # A base, not a device to wire in TOML
    assert all(registry.from_type(t).type is not ElectricalDevice for t in registry.get_types())


# ---------------------------------------------------------------- meter


def test_meter_default_channels(configure_component):
    channels = configure_component(EnergyMeter, "")

    assert set(channels) == {"power", "current", "voltage", "frequency", "energy_import", "energy_export"}
    power = channels["power"]
    assert power["name"] == "Device 1 Power"
    assert power["column"] == "dev1_power"
    assert power["unit"] == "W"
    assert power["aggregate"] == "mean"
    assert channels["energy_import"]["aggregate"] == "last"
    assert channels["energy_export"]["column"] == "dev1_energy_export"


def test_meter_flag_gated_channels(configure_component):
    channels = configure_component(EnergyMeter, "phases = true\n")

    assert len(channels) == 18
    assert channels["l1_power"]["name"] == "Device 1 Phase 1 Power"
    assert channels["l12_voltage"]["column"] == "dev1_l12_voltage"
    assert "power_factor" not in channels
    assert "power_import" not in channels


def test_meter_per_phase_power_factors_need_both_flags(configure_component):
    quality_only = configure_component(EnergyMeter, "quality = true\n")
    assert set(quality_only) >= {"apparent_power", "reactive_power", "power_factor"}
    assert "l1_power_factor" not in quality_only

    both = configure_component(EnergyMeter, "quality = true\nphases = true\n")
    assert both["l3_power_factor"]["name"] == "Device 1 Phase 3 Power Factor"


def test_meter_direction_channels(configure_component):
    channels = configure_component(EnergyMeter, "directions = true\n")

    # No device reports these: they are declared for a processor or another
    # component to fill from the signed power
    assert channels["power_import"]["column"] == "dev1_power_import"
    assert channels["power_export"]["unit"] == "W"


def test_meter_channels_are_protocol_free(configure_component):
    channels = configure_component(EnergyMeter, "phases = true\nquality = true\ndirections = true\n")

    assert len(channels) == 26
    for channel in channels.values():
        for key in BINDING_KEYS:
            assert key not in channel, f"{channel['key']} leaked '{key}'"


# ---------------------------------------------------------------- inverter


def test_inverter_default_channels(configure_component):
    channels = configure_component(SolarInverter, "")

    # The DC group is on by default
    assert set(channels) == {
        "power",
        "current",
        "frequency",
        "energy",
        "state",
        "temperature",
        "dc_power",
        "dc_voltage",
        "dc_current",
    }
    assert channels["power"]["name"] == "Device 1 Power"
    assert channels["energy"]["aggregate"] == "last"
    assert channels["state"]["type"] is int
    assert channels["dc_power"]["column"] == "dev1_dc_power"


def test_inverter_flag_gated_channels(configure_component):
    channels = configure_component(SolarInverter, "dc = false\nphases = true\nquality = true\ncontrol = true\n")

    assert "dc_power" not in channels
    assert channels["l1_current"]["name"] == "Device 1 Phase 1 Current"
    assert channels["reactive_power"]["unit"] == "var"
    assert channels["power_limit"]["aggregate"] == "last"
    assert channels["power_limit_enabled"]["type"] is int


def test_inverter_channels_are_protocol_free(configure_component):
    channels = configure_component(SolarInverter, "phases = true\nquality = true\ncontrol = true\n")

    assert len(channels) == 20
    for channel in channels.values():
        for key in BINDING_KEYS:
            assert key not in channel, f"{channel['key']} leaked '{key}'"


# ---------------------------------------------------------------- shared vocabulary


def test_shared_quantities_are_declared_once():
    # Constants are globally unique on 'context_key', so a quantity both devices bind
    # is declared on the base and is the SAME object on either subclass
    assert EnergyMeter.POWER is SolarInverter.POWER
    assert EnergyMeter.POWER is ElectricalDevice.POWER
    assert EnergyMeter.VOLTAGE_L1 is SolarInverter.VOLTAGE_L1
    assert EnergyMeter.POWER_FACTOR is SolarInverter.POWER_FACTOR


def test_device_specific_quantities_stay_on_their_device():
    assert not hasattr(SolarInverter, "ENERGY_IMPORT")
    assert not hasattr(SolarInverter, "VOLTAGE")
    assert not hasattr(EnergyMeter, "POWER_DC")
    assert not hasattr(EnergyMeter, "POWER_LIMIT")
