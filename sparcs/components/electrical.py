# -*- coding: utf-8 -*-
"""
sparcs.components.electrical
~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""

from __future__ import annotations

from typing import Any, Dict

from lories.components import Component
from lories.core import Constant
from lories.typing import Configurations


class ElectricalDevice(Component):
    """
    Base for AC-connected devices that report electrical quantities. Declares the vocabulary
    shared by every such device and adds it as ordinary, unbound channels, leaving the wiring
    to TOML or to a protocol subclass. Optional channel groups are enabled with the `phases`
    and `quality` flags.

    Subclasses extend the vocabulary by overriding `_add_core_channels`, `_add_phase_channels`
    and `_add_quality_channels` (calling `super()` first), and add their own flag-gated groups
    in `_add_optional_channels`. A protocol subclass reads its device addressing in
    `_configure_bindings` and returns per-channel bindings from `_bind`; this class binds
    nothing, so on its own every channel it declares is left for TOML to wire.
    """

    POWER = Constant(float, "power", "Power", "W", context="device", aggregate="mean")
    CURRENT = Constant(float, "current", "Current", "A", context="device", aggregate="mean")
    FREQUENCY = Constant(float, "frequency", "Frequency", "Hz", context="device", aggregate="mean")

    CURRENT_L1 = Constant(float, "l1_current", "Phase 1 Current", "A", context="device", aggregate="mean")
    CURRENT_L2 = Constant(float, "l2_current", "Phase 2 Current", "A", context="device", aggregate="mean")
    CURRENT_L3 = Constant(float, "l3_current", "Phase 3 Current", "A", context="device", aggregate="mean")
    VOLTAGE_L1 = Constant(float, "l1_voltage", "Phase 1 Voltage", "V", context="device", aggregate="mean")
    VOLTAGE_L2 = Constant(float, "l2_voltage", "Phase 2 Voltage", "V", context="device", aggregate="mean")
    VOLTAGE_L3 = Constant(float, "l3_voltage", "Phase 3 Voltage", "V", context="device", aggregate="mean")

    POWER_APPARENT = Constant(float, "apparent_power", "Apparent Power", "VA", context="device", aggregate="mean")
    POWER_REACTIVE = Constant(float, "reactive_power", "Reactive Power", "var", context="device", aggregate="mean")
    POWER_FACTOR = Constant(float, "power_factor", "Power Factor", "%", context="device", aggregate="mean")

    def configure(self, configs: Configurations) -> None:
        super().configure(configs)
        self._configure_bindings(configs)
        self._add_channels(configs)

    def _configure_bindings(self, configs: Configurations) -> None:
        """Read the device addressing a protocol subclass needs, before any channel is added."""

    # noinspection PyMethodMayBeStatic, PyUnusedLocal
    def _bind(self, constant: Constant) -> Dict[str, Any]:
        """Return the protocol binding for a constant, or an empty dict to leave it unbound."""
        return {}

    def _add_channel(self, constant: Constant, **custom: Any) -> None:
        channel = constant.to_dict()
        channel["name"] = f"{self.name} {constant.name}"
        channel["column"] = f"{self.key}_{constant.key}"
        channel.update(self._bind(constant))
        channel.update(custom)
        self.data.add(**channel)

    def _add_channels(self, configs: Configurations) -> None:
        self._add_core_channels(configs)
        if configs.get_bool("phases", default=False):
            self._add_phase_channels(configs)
        if configs.get_bool("quality", default=False):
            self._add_quality_channels(configs)
        self._add_optional_channels(configs)

    def _add_core_channels(self, configs: Configurations) -> None:
        self._add_channel(ElectricalDevice.POWER)
        self._add_channel(ElectricalDevice.CURRENT)
        self._add_channel(ElectricalDevice.FREQUENCY)

    def _add_phase_channels(self, configs: Configurations) -> None:
        self._add_channel(ElectricalDevice.CURRENT_L1)
        self._add_channel(ElectricalDevice.CURRENT_L2)
        self._add_channel(ElectricalDevice.CURRENT_L3)
        self._add_channel(ElectricalDevice.VOLTAGE_L1)
        self._add_channel(ElectricalDevice.VOLTAGE_L2)
        self._add_channel(ElectricalDevice.VOLTAGE_L3)

    def _add_quality_channels(self, configs: Configurations) -> None:
        self._add_channel(ElectricalDevice.POWER_APPARENT)
        self._add_channel(ElectricalDevice.POWER_REACTIVE)
        self._add_channel(ElectricalDevice.POWER_FACTOR)

    def _add_optional_channels(self, configs: Configurations) -> None:
        """Add the device's own flag-gated channel groups."""
