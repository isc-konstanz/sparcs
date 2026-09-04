# -*- coding: utf-8 -*-
"""
sparcs.tests.test_devices_openems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tests for the OpenEMS devices on a real, headless application whose ``openems_edge``
connector reports a canned discovery listing shaped like the Edge's ``meter0``: every
bound channel carries the OpenEMS component id, the channel name and the milli factor
that brings mA / mV / mHz to the vocabulary's units, the meter's ``meter_type`` decides
whether the powers are sign-flipped into the load reference, and an address the device
does not offer or a unit that contradicts the binding fails at configure.

Nothing here talks to an Edge: ``discover()`` is monkeypatched, and the connector is only
configured, never connected.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from lories.connectors import unavailable
from lories.connectors.openems import ChannelInfo, OpenEMSConnector
from lories.core.configs import ConfigurationError
from sparcs.components import EnergyMeter, OpenEMSInverter, OpenEMSMeter, SolarInverter

# The devices ride on the OpenEMS connector's dependency: without it the classes exist but
# are marked unavailable, so there is no connector to configure them against. `importorskip`
# cannot see that, because lories mocks an absent optional dependency away at import time.
if "openems" in unavailable():
    pytest.skip(f"openems connector unavailable: {unavailable()['openems']}", allow_module_level=True)

_SETTINGS_CONF = """
name = "oemsdev"
action = "run"

[interface]
enabled = false
"""

_SYSTEM_CONF = """
key = "sys"
name = "OpenEMS Device Test System"

[connectors.oe]
type = "openems_edge"

[connectors.virt]
type = "virtual"
"""

_METER_CONF = """
type = "openems_meter"
name = "Grid Meter"
device = "meter0"
connector = "oe"
meter_type = "grid"
phases = true
quality = true
directions = true
"""

_INVERTER_CONF = """
type = "openems_inverter"
name = "PV Inverter"
device = "pvInverter0"
connector = "oe"
phases = true
quality = true
control = true
"""

# The units the real Edge reports for its meter natures: powers in W / var, currents in
# mA, voltages in mV, the frequency in mHz, and the energies as cumulated Wh counters.
_UNITS = {
    "ActivePower": "W",
    "ActivePowerL1": "W",
    "ActivePowerL2": "W",
    "ActivePowerL3": "W",
    "ReactivePower": "var",
    "Current": "mA",
    "CurrentL1": "mA",
    "CurrentL2": "mA",
    "CurrentL3": "mA",
    "Voltage": "mV",
    "VoltageL1": "mV",
    "VoltageL2": "mV",
    "VoltageL3": "mV",
    "Frequency": "mHz",
    "ActiveConsumptionEnergy": "Wh_Σ",
    "ActiveProductionEnergy": "Wh_Σ",
}


def _listing(component: str, **overrides: str):
    units = dict(_UNITS)
    units.update(overrides)
    return {
        f"{component}/{channel}": ChannelInfo(
            address=f"{component}/{channel}",
            component=component,
            channel=channel,
            type="INTEGER",
            unit=unit,
            access_mode="RO",
        )
        for channel, unit in units.items()
    }


@pytest.fixture
def load_device(tmp_path, monkeypatch):
    """Write a headless project into ``tmp_path`` and load it; returns the bound device."""
    lories = pytest.importorskip("lories")

    def _load(device_class, device_conf: str, listing=None):
        if listing is None:
            listing = {**_listing("meter0"), **_listing("pvInverter0")}
        monkeypatch.setattr(OpenEMSConnector, "discover", lambda self, refresh=False: listing)

        conf_dir = tmp_path / "conf"
        conf_dir.mkdir(exist_ok=True)
        (conf_dir / "settings.conf").write_text(_SETTINGS_CONF)
        (conf_dir / "system.conf").write_text(_SYSTEM_CONF)
        (conf_dir / "oemsdev.conf").write_text(device_conf)
        monkeypatch.chdir(tmp_path)

        app = lories.load("oemsdev")
        (device,) = [c for c in app.components.values() if isinstance(c, device_class)]
        return device

    return _load


@pytest.fixture
def load_meter(load_device):
    def _load(device_conf: str = _METER_CONF, listing=None):
        return load_device(OpenEMSMeter, device_conf, listing)

    return _load


# ------------------------------------------------------------------- meter


def test_meter_binds_every_point_to_its_component_and_channel(load_meter):
    device = load_meter()

    for constant, entry in OpenEMSMeter.POINTS.items():
        channel = device.data[constant]
        name = entry[0] if isinstance(entry, tuple) else entry
        assert channel.get("component") == "meter0", constant
        assert channel.get("channel") == name, constant
        assert channel.connector.id == "sys.oe", constant


def test_meter_scales_the_milli_units_into_the_vocabulary(load_meter):
    device = load_meter()

    for key in ("current", "l1_current", "voltage", "l3_voltage", "frequency"):
        assert device.data[key].get("scale") == 0.001, key


@pytest.mark.parametrize(
    "meter_type,expected",
    [
        # OpenEMS signs these positive on import already, which is the load reference
        ("grid", None),
        ("consumption_metered", None),
        # ... and these positive on production, which is its opposite
        ("production", -1),
        ("production_and_consumption", -1),
    ],
)
def test_meter_type_decides_the_sign_of_the_powers(load_meter, meter_type, expected):
    device = load_meter(_METER_CONF.replace('meter_type = "grid"', f'meter_type = "{meter_type}"'))

    for key in ("power", "l1_power", "l2_power", "l3_power", "reactive_power"):
        assert device.data[key].get("scale") == expected, key
    # The counters are only-positive in OpenEMS for every meter type
    assert device.data[EnergyMeter.ENERGY_IMPORT].get("scale") is None
    assert device.data[EnergyMeter.ENERGY_EXPORT].get("scale") is None
    # A flip does not disturb the unit factors either
    assert device.data[EnergyMeter.CURRENT].get("scale") == 0.001


def test_energies_map_to_the_consumption_and_production_counters(load_meter):
    device = load_meter()

    assert device.data[EnergyMeter.ENERGY_IMPORT].get("channel") == "ActiveConsumptionEnergy"
    assert device.data[EnergyMeter.ENERGY_EXPORT].get("channel") == "ActiveProductionEnergy"


def test_meter_constants_without_an_openems_counterpart_stay_unbound(load_meter):
    device = load_meter()

    for key in (
        "apparent_power",
        "power_factor",
        "l1_power_factor",
        "l12_voltage",
        "power_import",
        "power_export",
    ):
        assert not device.data[key].has_connector(), key


@pytest.mark.parametrize("meter_type", ["", 'meter_type = "grid_"\n', 'meter_type = "battery"\n'])
def test_missing_or_invalid_meter_type_lists_the_valid_names(load_meter, meter_type):
    conf = _METER_CONF.replace('meter_type = "grid"\n', meter_type)

    with pytest.raises(ConfigurationError) as excinfo:
        load_meter(conf)

    message = str(excinfo.value)
    assert "meter_type" in message
    for name in OpenEMSMeter.METER_TYPES:
        assert name in message


def test_non_string_meter_type_fails_naming_the_key(load_meter):
    with pytest.raises(ConfigurationError, match="meter_type"):
        load_meter(_METER_CONF.replace('meter_type = "grid"', "meter_type = 3"))


def test_connector_of_another_family_is_rejected_naming_the_class(load_meter):
    # The seam's family check fires before the binding is validated against a listing
    with pytest.raises(ConfigurationError, match="VirtualConnector"):
        load_meter(_METER_CONF.replace('connector = "oe"', 'connector = "virt"'))


def test_meter_type_is_case_insensitive(load_meter):
    device = load_meter(_METER_CONF.replace('meter_type = "grid"', 'meter_type = "PRODUCTION"'))

    assert device.meter_type == "production"
    assert device.data[EnergyMeter.POWER].get("scale") == -1


def test_address_the_device_does_not_offer_is_named(load_meter):
    listing = {**_listing("meter0"), **_listing("pvInverter0")}
    listing.pop("meter0/Current")

    with pytest.raises(ConfigurationError, match="meter0/Current"):
        load_meter(listing=listing)


def test_unit_without_the_milli_prefix_is_a_mismatch(load_meter):
    listing = {**_listing("meter0", Current="A"), **_listing("pvInverter0")}

    with pytest.raises(ConfigurationError) as excinfo:
        load_meter(listing=listing)

    message = str(excinfo.value)
    assert "'current'" in message
    assert "'A'" in message
    assert "'mA'" in message


# ---------------------------------------------------------------- inverter


def test_inverter_binds_the_production_counter_and_the_ac_quantities(load_device):
    device = load_device(OpenEMSInverter, _INVERTER_CONF)

    assert device.data[SolarInverter.ENERGY].get("channel") == "ActiveProductionEnergy"
    assert device.data[SolarInverter.ENERGY].get("component") == "pvInverter0"
    assert device.data[SolarInverter.POWER].get("channel") == "ActivePower"
    assert device.data[SolarInverter.CURRENT].get("scale") == 0.001
    assert device.data[SolarInverter.VOLTAGE_L2].get("scale") == 0.001
    assert device.data[SolarInverter.FREQUENCY].get("scale") == 0.001


def test_inverter_keeps_the_generator_reference(load_device):
    device = load_device(OpenEMSInverter, _INVERTER_CONF)

    assert device.data[SolarInverter.POWER].get("scale") is None
    assert device.data[SolarInverter.POWER_REACTIVE].get("scale") is None


def test_inverter_requires_the_openems_component_id(load_device):
    with pytest.raises(ConfigurationError, match="device"):
        load_device(OpenEMSInverter, _INVERTER_CONF.replace('device = "pvInverter0"\n', ""))


def test_inverter_requires_a_connector(load_device):
    with pytest.raises(ConfigurationError, match="requires a connector of type"):
        load_device(OpenEMSInverter, _INVERTER_CONF.replace('connector = "oe"\n', ""))


def test_inverter_leaves_dc_state_and_control_unbound(load_device):
    device = load_device(OpenEMSInverter, _INVERTER_CONF)

    for key in ("dc_power", "dc_voltage", "state", "temperature", "power_limit", "apparent_power"):
        assert not device.data[key].has_connector(), key


def test_registration_degrades_without_websocket_client():
    # The mixin lives next to its connector, so a missing 'websocket-client' marks both
    # unavailable -- importing sparcs must still work, the type must just refuse to build.
    code = (
        "import sys; sys.modules['websocket'] = None; "
        "import sparcs; from lories.components.context import registry; "
        "print(registry.from_type('openems_meter').available, registry.from_type('openems_inverter').available)"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip().endswith("False False")
