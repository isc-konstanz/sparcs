# -*- coding: utf-8 -*-
"""
sparcs.tests.test_devices_mahle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tests for the Mahle chargeBIG device: the fixed OPC UA node names it binds, the virtual
channels it leaves unbound, and the station children addressed by physical id. The project
is loaded headless with a ``virtual`` connector standing in for the OPC UA server (``connector``
overrides the ``opcua`` family default), so nothing here needs the ``opcua`` package; a
subprocess check pins that importing ``sparcs.components`` does not need it either.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from sparcs.components import ChargeBig, EnergyMeter
from sparcs.components.devices.mahle import ChargeBigStation

_SETTINGS_CONF = """
name = "chargebig_test"
action = "run"

[interface]
enabled = false
"""

_SYSTEM_CONF = """
key = "sys"
name = "chargeBIG Test System"

[connectors.virt]
type = "virtual"
"""

_CHARGE_BIG_CONF = """
name = "Charge Big"
type = "charge_big"
connector = "virt"

[stations]
count = 3

[stations.mapping]
1 = 15
2 = 14
"""

# main's channel set (2026-08 charge-big line) minus the four renames of the devices PRD, decision 5
EXPECTED_KEYS = {
    "setpoint",
    "setpoint_max",
    "setpoint_power",
    "current",
    "power",
    "reactive_power",
    "l1_power",
    "l2_power",
    "l3_power",
    "l1_active_energy",
    "l2_active_energy",
    "l3_active_energy",
    "l1_current",
    "l2_current",
    "l3_current",
    "l1_cos_phi",
    "l2_cos_phi",
    "l3_cos_phi",
}


@pytest.fixture
def charge_big(tmp_path, monkeypatch):
    lories = pytest.importorskip("lories")
    conf_dir = tmp_path / "conf"
    conf_dir.mkdir()
    (conf_dir / "settings.conf").write_text(_SETTINGS_CONF)
    (conf_dir / "system.conf").write_text(_SYSTEM_CONF)
    (conf_dir / "charge_big.conf").write_text(_CHARGE_BIG_CONF)
    monkeypatch.chdir(tmp_path)

    app = lories.load("chargebig_test")
    (device,) = [c for c in app.components.values() if isinstance(c, ChargeBig)]
    return device


def test_declares_exactly_the_known_channel_set(charge_big):
    assert {channel.key for channel in charge_big.data.values()} == EXPECTED_KEYS


def test_bound_channels_carry_node_name_and_connector(charge_big):
    for constant, address in ChargeBig.ADDRESSES.items():
        channel = charge_big.data[constant]
        assert channel.address == address, constant
        assert channel.has_connector(), constant
        assert channel.connector.id == "sys.virt"
    assert charge_big.data[EnergyMeter.POWER_L1].address == "Zähler_Leistung_Phase1"
    assert charge_big.data[ChargeBig.ENERGY_L1].aggregate == "last"


def test_virtual_channels_stay_unbound(charge_big):
    for constant in ChargeBig.VIRTUAL:
        channel = charge_big.data[constant]
        assert not channel.has_connector(), constant
        assert not hasattr(channel, "address"), constant


def test_stations_follow_the_mapping_and_share_the_connector(charge_big):
    stations = sorted(
        (c for c in charge_big.components.values() if isinstance(c, ChargeBigStation)), key=lambda c: c.id
    )
    assert [s.key for s in stations] == ["station_0", "station_1", "station_2"]
    # station index -> physical id: mapped ones are 1-based in TOML, unmapped ones keep their index
    assert [s.station_id for s in stations] == [14, 13, 2]

    first = stations[0]
    assert first.data[ChargeBigStation.STATE].address == "Ladepunkt_14_Status"
    assert first.data[ChargeBigStation.LIMIT].address == "Ladepunkt_14_Grenzwert"
    assert first.data[ChargeBigStation.STATE].station_id == 0
    assert first.data[ChargeBigStation.STATE].connector.id == "sys.virt"
    assert len(first.connectors) == 0


def test_components_import_without_the_opcua_package():
    code = (
        "import sys; sys.modules['opcua'] = None; "
        "import sparcs.components; print(sparcs.components.ChargeBig.CONNECTOR)"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip().endswith("opcua")
