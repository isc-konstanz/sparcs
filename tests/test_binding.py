# -*- coding: utf-8 -*-
"""
sparcs.tests.test_binding
~~~~~~~~~~~~~~~~~~~~~~~~~

Tests that the sparcs vocabulary sits on the lories binding seam: `ACComponent` and its
subclasses derive from the very ``BindableComponent`` lories exports, declare no binding
family of their own, and therefore leave ``connector`` optional -- a bare vocabulary
component loads with every channel unbound, and a single channel wired in TOML resolves
its connector without the component knowing about it. The seam's own connector rules are
covered in lories (`tests/test_components_binding.py`); the channel vocabulary itself in
``test_electrical_components.py``.
"""

from __future__ import annotations

import pytest

from lories.components.binding import BindableComponent as LoriesBindableComponent
from sparcs.components import ACComponent, BindableComponent, EnergyMeter, SolarInverter
from sparcs.components.vehicle import EVSE

_SETTINGS_CONF = """
name = "bindtest"
action = "run"

[interface]
enabled = false
"""

_SYSTEM_CONF = """
key = "sys"
name = "Binding Test System"

[connectors.virt]
type = "virtual"
"""


@pytest.fixture
def load_meter(tmp_path, monkeypatch):
    """Write a headless project into ``tmp_path`` and load it; returns the ``meter`` component."""
    lories = pytest.importorskip("lories")

    def _load(device_conf: str = ""):
        conf_dir = tmp_path / "conf"
        conf_dir.mkdir(exist_ok=True)
        (conf_dir / "settings.conf").write_text(_SETTINGS_CONF)
        (conf_dir / "system.conf").write_text(_SYSTEM_CONF)
        (conf_dir / "meter.conf").write_text('type = "meter"\nname = "Grid Meter"\n' + device_conf)
        monkeypatch.chdir(tmp_path)

        app = lories.load("bindtest")
        (device,) = [c for c in app.components.values() if isinstance(c, EnergyMeter)]
        return device

    return _load


def test_vocabulary_sits_on_the_lories_seam():
    # sparcs re-exports the seam, it does not own a copy of it
    assert BindableComponent is LoriesBindableComponent
    for cls in (ACComponent, EnergyMeter, SolarInverter, EVSE):
        assert issubclass(cls, LoriesBindableComponent), cls.__name__


def test_vocabulary_declares_no_binding_family():
    # A protocol-free component names no connector types, so `connector` stays optional
    for cls in (ACComponent, EnergyMeter, SolarInverter, EVSE):
        assert cls.CONNECTOR_TYPES == (), cls.__name__


def test_unwired_vocabulary_loads_with_every_channel_unbound(load_meter):
    device = load_meter()

    assert len(device.data) > 0
    assert not any(channel.has_connector() for channel in device.data.values())


def test_toml_wires_a_single_vocabulary_channel(load_meter):
    device = load_meter('\n[data.channels.power]\nconnector = "virt"\naddress = "reg1"\n')

    power = device.data[EnergyMeter.POWER]
    assert power.has_connector()
    assert power.connector.id == "sys.virt"
    assert power.address == "reg1"
    # The remaining vocabulary is untouched by one channel's wiring
    assert not device.data[EnergyMeter.CURRENT].has_connector()
