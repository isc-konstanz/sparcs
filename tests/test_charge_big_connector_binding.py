# -*- coding: utf-8 -*-
"""
tests.test_charge_big_connector_binding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The chargeBIG component references its OPC UA connector by id; the framework builds
the connector from the ``[connectors.<id>]`` section of the component configuration.
No network access: configuring an OPC UA connector only constructs the client object.
"""

import sys
from argparse import ArgumentParser

import pytest

SYSTEM_CONF = """
key = "system"
name = "System"

[location]
latitude = 47.67
longitude = 9.15
timezone = "Europe/Berlin"
"""

CHARGE_BIG_CONF = """
name = "Charge Big"
type = "charge_big"
connector = "opcua"

[connectors.opcua]
host = "10.1.20.12"
settings = "ns=1"

[stations]
count = 2

[stations.mapping]
1 = 15
2 = 14
"""


def _load(tmp_path, monkeypatch, charge_big_conf: str):
    conf = tmp_path / "conf"
    conf.mkdir()
    (conf / "system.conf").write_text(SYSTEM_CONF, encoding="utf-8")
    (conf / "charge_big.conf").write_text(charge_big_conf, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["probe", "-c", str(conf), "-d", str(tmp_path), "run"])

    import sparcs

    return sparcs.load("charge_big_binding_test", parser=ArgumentParser())


@pytest.fixture
def app(tmp_path, monkeypatch):
    return _load(tmp_path, monkeypatch, CHARGE_BIG_CONF)


def _opcua_connectors(app):
    from lories.connectors.opcua import OpcUaConnector

    return [c for c in app.connectors.values() if isinstance(c, OpcUaConnector)]


def test_connector_is_built_from_the_connectors_section(app):
    connectors = _opcua_connectors(app)
    assert [c.id for c in connectors] == ["system.charge_big.opcua"]
    connector = connectors[0]
    assert connector._host == "10.1.20.12"
    assert connector._settings == ["ns=1"]


def test_park_and_station_channels_bind_to_the_connector(app):
    connector = _opcua_connectors(app)[0]
    bound = {}
    for component in {id(c): c for c in app.components.values()}.values():
        for channel in component.data.values():
            if channel.get("address", None) is not None:
                assert channel.connector.id == connector.id, channel.id
                assert channel.connector.enabled, channel.id
                bound[channel.id] = channel.get("address")

    park = {k: v for k, v in bound.items() if ".station_" not in k}
    stations = {k: v for k, v in bound.items() if ".station_" in k}
    assert len(park) == 13
    assert park["system.charge_big.setpoint"] == "Sollwert_aktiv"
    assert stations == {
        "system.charge_big.station_0.state": "Ladepunkt_14_Status",
        "system.charge_big.station_0.limit": "Ladepunkt_14_Grenzwert",
        "system.charge_big.station_1.state": "Ladepunkt_13_Status",
        "system.charge_big.station_1.limit": "Ladepunkt_13_Grenzwert",
    }


def test_missing_connector_key_is_a_configuration_error(tmp_path, monkeypatch):
    from lories.core import ConfigurationError

    conf = CHARGE_BIG_CONF.replace('connector = "opcua"\n', "")
    with pytest.raises(ConfigurationError, match="connector"):
        _load(tmp_path, monkeypatch, conf)
