# -*- coding: utf-8 -*-
"""
sparcs.components.openems
~~~~~~~~~~~~~~~~~~~~~~~~~


"""

from __future__ import annotations

import json
import re
from typing import Optional

from lories.components import Component, register_component_type
from lories.core.configs.errors import ConfigurationError
from lories.core.configs.parameters import Parameter
from lories.io.rest import Rest
from lories.typing import Configurations

from sparcs.connectors.openems import OpenEMSBackendConnector, OpenEMSEdgeConnector

_CONNECTOR_CLASSES = {
    "OpenEMSEdge": OpenEMSEdgeConnector,
    "OpenEMSBackend": OpenEMSBackendConnector,
}


@register_component_type("openems")
class OpenEMSComponent(Component):
    """A component that dynamically creates a channel for every OpenEMS
    component/channel discovered via the REST API at configure time.

    The concrete WebSocket connector is selected via the ``type`` key in
    the ``[connector]`` sub-section and is always created internally — no
    separate ``[connectors.openems]`` entry is needed in ``system.conf``.

    Example ``system.conf`` (Edge device)::

        [components.openems]
        type = "openems"
        show_hidden_channels = false

        [components.openems.connector]
        type           = "OpenEMSEdge"
        host           = "localhost"
        username       = "admin"
        password       = "admin"
        timeout        = 10
        rest_port      = 8084
        rest_endpoint  = "rest"
        ws_port        = 8085

    Example ``system.conf`` (Backend)::

        [components.openems.connector]
        type           = "OpenEMSBackend"
        edge_id        = "edge0"
        host           = "backend.example.com"
        ws_port        = 8085

    Configuration keys (all optional with defaults shown):
        show_hidden_channels  = false
    """

    _show_hidden_channels = Parameter(key="show_hidden_channels", type=bool, default=False, desc="Include channels marked as hidden in OpenEMS")

    _show_hidden_channels: bool

    def configure(self, configs: Configurations) -> None:
        super().configure(configs)

        # ----------------------------------------------------------------
        # Build the sub-connector, selecting the concrete class via `type`
        # ----------------------------------------------------------------
        connector_configs = configs.get_member("connector", defaults={})

        connector_type = connector_configs.get("type", default="OpenEMSEdge")
        connector_cls = _CONNECTOR_CLASSES.get(connector_type)
        if connector_cls is None:
            raise ConfigurationError(
                f"Unknown OpenEMS connector type '{connector_type}'. "
                f"Valid types: {list(_CONNECTOR_CLASSES)}"
            )

        connector = connector_cls(
            key="openems",
            name=f"{self.name} OpenEMS",
            context=self,
            configs=connector_configs,
        )
        connector.configure(connector_configs)
        self.connectors.add(connector)

        # ----------------------------------------------------------------
        # REST channel discovery (uses the same connector_configs params)
        # ----------------------------------------------------------------
        rest = Rest(
            host=connector_configs.get("host", default="localhost"),
            port=connector_configs.get_int("rest_port", default=8084),
            username=connector_configs.get("username", default="admin"),
            password=connector_configs.get("password", default="admin"),
            endpoint=connector_configs.get("rest_endpoint", default="rest"),
            timeout=connector_configs.get_int("timeout", default=10),
        )

        raw = None
        for attempt in range(1, 4):
            try:
                raw = json.loads(rest.get_request("channel/.*/.*"))
                break
            except Exception as e:
                self._logger.warning(
                    f"OpenEMS REST channel discovery attempt {attempt}/3 failed: {e}"
                )
        if raw is None:
            self._logger.error("OpenEMS REST channel discovery failed after 3 attempts")
            return

        # Group by component and iterate deterministically
        channels_by_comp: dict = {}
        for ch in raw:
            comp_id, chan_id = ch["address"].split("/", 1)
            channels_by_comp.setdefault(comp_id, []).append(ch)

        available_channels = []
        for comp_id in sorted(channels_by_comp):
            if not self._show_hidden_channels and comp_id.startswith("_"):
                continue

            for ch in sorted(channels_by_comp[comp_id], key=lambda x: x["address"]):
                _, chan_id = ch["address"].split("/", 1)

                if not self._show_hidden_channels and chan_id.startswith("_"):
                    continue

                self.data.add(
                    key=self._make_key(comp_id, chan_id),
                    name=f"{comp_id} / {chan_id}",
                    type=self._map_type(ch.get("type")),
                    address=ch["address"],
                    unit=ch.get("unit") or None,
                    aggregate="last",
                    connector=connector.id,
                )
                available_channels.append(ch["address"])

        # print available channels
        self._logger.info(f"Discovered {len(available_channels)} OpenEMS channels: {available_channels}")

        # # write the list as a text file
        # data = ""
        # for ch in available_channels:
        #     data += f"{ch}\n"
        #
        # with open("openems_channels.txt", "w") as f:
        #     f.write(data)

        pass




    @staticmethod
    def _map_type(openems_type: Optional[str]) -> type:
        """Map an OpenEMS channel type string to a Python type."""
        if openems_type is None:
            return str
        t = openems_type.upper()
        if t in ("INTEGER", "LONG", "SHORT"):
            return int
        if t in ("FLOAT", "DOUBLE"):
            return float
        if t == "BOOLEAN":
            return bool
        return str

    @staticmethod
    def _make_key(comp_id: str, chan_id: str) -> str:
        """Build a valid, lowercase lories channel key from an OpenEMS address."""
        raw = f"{comp_id.lstrip('_')}_{chan_id}"
        return re.sub(r"[^a-zA-Z0-9]+", "_", raw).strip("_").lower()
