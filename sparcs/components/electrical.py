# -*- coding: utf-8 -*-
"""
sparcs.components.electrical
~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""

from __future__ import annotations

from lories.components.binding import BindableComponent
from lories.core import Constant
from lories.typing import Configurations


class ACComponent(BindableComponent):
    """
    Base for AC-connected components that report electrical quantities. Declares the vocabulary
    every such device shares and adds it as ordinary channels that stay unbound until TOML or a
    binding layer wires them. Optional channel groups are enabled with the `phases` and `quality`
    flags.

    Subclasses extend the vocabulary by overriding `_add_core_channels`, `_add_phase_channels`
    and `_add_quality_channels` (calling `super()` first) and add their own flag-gated groups in
    `_add_optional_channels`. A device with a fixed point list overrides `_add_channels` instead
    and declares exactly what it can feed.
    """

    POWER = Constant(float, "power", "Power", "W", context="electrical", aggregate="mean")
    CURRENT = Constant(float, "current", "Current", "A", context="electrical", aggregate="mean")
    FREQUENCY = Constant(float, "frequency", "Frequency", "Hz", context="electrical", aggregate="mean")

    CURRENT_L1 = Constant(float, "l1_current", "Phase 1 Current", "A", context="electrical", aggregate="mean")
    CURRENT_L2 = Constant(float, "l2_current", "Phase 2 Current", "A", context="electrical", aggregate="mean")
    CURRENT_L3 = Constant(float, "l3_current", "Phase 3 Current", "A", context="electrical", aggregate="mean")
    VOLTAGE_L1 = Constant(float, "l1_voltage", "Phase 1 Voltage", "V", context="electrical", aggregate="mean")
    VOLTAGE_L2 = Constant(float, "l2_voltage", "Phase 2 Voltage", "V", context="electrical", aggregate="mean")
    VOLTAGE_L3 = Constant(float, "l3_voltage", "Phase 3 Voltage", "V", context="electrical", aggregate="mean")

    POWER_APPARENT = Constant(float, "apparent_power", "Apparent Power", "VA", context="electrical", aggregate="mean")
    POWER_REACTIVE = Constant(float, "reactive_power", "Reactive Power", "var", context="electrical", aggregate="mean")
    POWER_FACTOR = Constant(float, "power_factor", "Power Factor", "%", context="electrical", aggregate="mean")

    def _add_channels(self, configs: Configurations) -> None:
        self._add_core_channels(configs)
        if configs.get_bool("phases", default=False):
            self._add_phase_channels(configs)
        if configs.get_bool("quality", default=False):
            self._add_quality_channels(configs)
        self._add_optional_channels(configs)

    def _add_core_channels(self, configs: Configurations) -> None:
        self._add_channel(ACComponent.POWER)
        self._add_channel(ACComponent.CURRENT)
        self._add_channel(ACComponent.FREQUENCY)

    def _add_phase_channels(self, configs: Configurations) -> None:
        self._add_channel(ACComponent.CURRENT_L1)
        self._add_channel(ACComponent.CURRENT_L2)
        self._add_channel(ACComponent.CURRENT_L3)
        self._add_channel(ACComponent.VOLTAGE_L1)
        self._add_channel(ACComponent.VOLTAGE_L2)
        self._add_channel(ACComponent.VOLTAGE_L3)

    def _add_quality_channels(self, configs: Configurations) -> None:
        self._add_channel(ACComponent.POWER_APPARENT)
        self._add_channel(ACComponent.POWER_REACTIVE)
        self._add_channel(ACComponent.POWER_FACTOR)

    def _add_optional_channels(self, configs: Configurations) -> None:
        """Add the component's own flag-gated channel groups."""
