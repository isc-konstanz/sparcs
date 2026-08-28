# -*- coding: utf-8 -*-
"""
sparcs.tests.test_binding
~~~~~~~~~~~~~~~~~~~~~~~~~

Tests for the connector rules of ``BindableComponent`` on a real, headless application
loaded from a temporary project: a bare connector id resolves upward to a shared gateway,
a locally declared ``[connectors.<id>]`` receives the family defaults with the config's
own keys winning, ``connector = "<id>"`` on the device overrides the family default, and
an id that resolves nowhere fails at configure instead of leaving the channel silently
unbound. The channel vocabulary itself is covered by ``test_electrical_components.py``.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from lories.components import register_component_type
from lories.core import Constant
from lories.core.configs import ConfigurationError
from lories.typing import Configurations
from sparcs.components import BindableComponent

_SETTINGS_CONF = """
name = "bindtest"
action = "run"

[interface]
enabled = false
"""

_SYSTEM_CONF = """
key = "sys"
name = "Binding Test System"
"""

_VIRTUAL = 'type = "virtual"'


@register_component_type("bindtest")
class BindTestDevice(BindableComponent):
    CONNECTOR = "virt"

    READING = Constant(float, "reading", "Reading", "W", context="bindtest")
    NOTE = Constant(float, "note", "Note", "", context="bindtest")

    def _connector_defaults(self) -> Dict[str, Any]:
        return {"family": "bindtest-default", "shared": "from-class"}

    def _bind(self, constant: Constant) -> Dict[str, Any]:
        if constant is BindTestDevice.READING:
            return {"connector": self._connector_id, "address": "reg1"}
        return {}

    def _add_channels(self, configs: Configurations) -> None:
        self._add_channel(BindTestDevice.READING)
        self._add_channel(BindTestDevice.NOTE)


@pytest.fixture
def load_project(tmp_path, monkeypatch):
    """Write a headless project into ``tmp_path`` and load it; returns the ``bindtest`` component."""
    lories = pytest.importorskip("lories")

    def _load(device_conf: str, system_extra: str = ""):
        conf_dir = tmp_path / "conf"
        conf_dir.mkdir(exist_ok=True)
        (conf_dir / "settings.conf").write_text(_SETTINGS_CONF)
        (conf_dir / "system.conf").write_text(_SYSTEM_CONF + system_extra)
        (conf_dir / "bindtest.conf").write_text('type = "bindtest"\nname = "Bind Test"\n' + device_conf)
        monkeypatch.chdir(tmp_path)

        app = lories.load("bindtest")
        devices = [c for c in app.components.values() if isinstance(c, BindTestDevice)]
        assert len(devices) == 1, [c.id for c in app.components.values()]
        return devices[0]

    return _load


def test_bare_id_resolves_to_the_shared_upstream_connector(load_project):
    device = load_project("", system_extra=f"\n[connectors.virt]\n{_VIRTUAL}\n")

    reading = device.data[BindTestDevice.READING]
    assert reading.has_connector()
    assert reading.connector.id == "sys.virt"
    # Nothing is created on the device when it declares no connector of its own
    assert len(device.connectors) == 0
    # Channels the binding did not claim stay unbound without complaint
    assert not device.data[BindTestDevice.NOTE].has_connector()


def test_local_connector_receives_family_defaults_and_toml_wins(load_project):
    device = load_project(f'\n[connectors.virt]\n{_VIRTUAL}\nshared = "from-toml"\n')

    assert device.data[BindTestDevice.READING].connector.id == "sys.bindtest.virt"
    (connector,) = device.connectors.values()
    assert connector.configs.get("family") == "bindtest-default"
    assert connector.configs.get("shared") == "from-toml"


def test_connector_key_overrides_the_family_default(load_project):
    device = load_project('connector = "gateway"\n', system_extra=f"\n[connectors.gateway]\n{_VIRTUAL}\n")

    assert device.data[BindTestDevice.READING].connector.id == "sys.gateway"


def test_unresolved_connector_fails_at_configure(load_project):
    with pytest.raises(ConfigurationError, match="virt"):
        load_project("")
