# -*- coding: utf-8 -*-
"""
sparcs.components.binding
~~~~~~~~~~~~~~~~~~~~~~~~~


"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from lories.components import Component
from lories.core import ConfigurationError, Constant
from lories.typing import Configurations


class BindableComponent(Component):
    """
    Base for components that declare their channels from Constants and let a subclass wire them
    to a protocol. `_add_channels` declares the channels through `_add_channel`, which merges
    whatever `_bind` returns for a constant (point names, addresses, the connector id) into the
    channel config and leaves the channel unbound when `_bind` returns nothing.

    Connector wiring follows four rules. The connector id is the device's `connector` key,
    defaulting to `CONNECTOR`, the family default a binding layer sets. If the device's own
    config declares `[connectors.<id>]`, the family defaults from `_connector_defaults` are merged
    into that block with the config's own keys winning. If it does not, the id resolves upward
    through the parents, so several devices can share one gateway; nothing is created. Once the
    connectors are loaded, every channel a binding claimed must resolve its connector, otherwise
    configuring fails and names the id.
    """

    CONNECTOR: Optional[str] = None

    _connector_id: Optional[str] = None
    _bound: List[Constant]

    def configure(self, configs: Configurations) -> None:
        super().configure(configs)
        self._bound = []
        self._configure_bindings(configs)
        self._configure_connector(configs)
        self._add_channels(configs)

    def _configure_bindings(self, configs: Configurations) -> None:
        """Read the addressing a binding needs, before any channel is added; call `super()` first."""
        connector = configs.get("connector", default=type(self).CONNECTOR)
        self._connector_id = connector if isinstance(connector, str) else type(self).CONNECTOR

    def _configure_connector(self, configs: Configurations) -> None:
        if self._connector_id is None or not configs.has_member("connectors"):
            return
        if configs.get_member("connectors").has_member(self._connector_id):
            self.connectors.add(self._connector_id, **self._connector_defaults())

    # noinspection PyMethodMayBeStatic
    def _connector_defaults(self) -> Dict[str, Any]:
        """Family defaults for a locally declared connector; the config's own keys win."""
        return {}

    # noinspection PyMethodMayBeStatic, PyUnusedLocal
    def _bind(self, constant: Constant) -> Dict[str, Any]:
        """Return the binding for a constant, or an empty dict to leave the channel unbound."""
        return {}

    def _add_channel(self, constant: Constant, **custom: Any) -> None:
        channel = constant.to_dict()
        channel["name"] = f"{self.name} {constant.name}"
        channel.update(self._bind(constant))
        channel.update(custom)
        if channel.get("connector") is not None:
            self._bound.append(constant)
        self.data.add(**channel)

    def _add_channels(self, configs: Configurations) -> None:
        """Declare the component's channels through `_add_channel`."""

    def _on_configure(self, configs: Configurations) -> None:
        super()._on_configure(configs)
        unresolved = [
            str(constant) for constant in getattr(self, "_bound", []) if not self.data[constant].has_connector()
        ]
        if unresolved:
            raise ConfigurationError(
                f"{type(self).__name__} '{self.id}' cannot resolve connector '{self._connector_id}' for channels "
                f"{unresolved}: declare [connectors.{self._connector_id}] in this component's configuration or in "
                f"a parent's, or set 'connector' to an existing id"
            )
