# -*- coding: utf-8 -*-
"""sparcs.tests.conftest
~~~~~~~~~~~~~~~~~~~~~~~~

Shared fixtures: the electrical-component ``configure()`` driver shared by
``test_electrical_components.py`` and ``test_sunspec_components.py``,
which must agree on the harness to be comparable.

Deliberately minimal: the other ``object.__new__`` helpers are each file's
pin mechanism and stay local to their files.
"""

import pytest


@pytest.fixture
def configure_component(monkeypatch, tmp_path):
    """Drive a component's ``configure()`` and return the channels it declared.

    Full component construction needs an application context, so this runs
    ``configure()`` on a ``__new__``-bypassed instance with the lories base
    ``configure()`` no-op'ed and id/name/key/data patched. The recorded
    ``data.add`` calls are the assertion surface: the whole inheritance
    chain still runs, only lories' own ``Component.configure`` is stubbed.
    """
    from lories.components import Component
    from lories.core.configs.configurations import Configurations

    def _configure(component_class, toml_text: str) -> dict:
        (tmp_path / "test.conf").write_text(toml_text)
        configs = Configurations.load("test.conf", data_dir=str(tmp_path), flat=True)

        calls = []

        class _Data:
            def add(self, **channel):
                calls.append(channel)

        monkeypatch.setattr(Component, "configure", lambda self, c: None)
        monkeypatch.setattr(component_class, "id", property(lambda self: "sys.dev1"), raising=False)
        monkeypatch.setattr(component_class, "name", property(lambda self: "Device 1"), raising=False)
        monkeypatch.setattr(component_class, "key", property(lambda self: "dev1"), raising=False)
        monkeypatch.setattr(component_class, "data", property(lambda self: _Data()), raising=False)

        component = component_class.__new__(component_class)
        component.configure(configs)
        return {channel["key"]: channel for channel in calls}

    return _configure
