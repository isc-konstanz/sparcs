# -*- coding: utf-8 -*-
"""
sparcs.components.solar.inverter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""

from __future__ import annotations

from lories.components import register_component_type
from lories.core import Constant
from lories.typing import Configurations
from sparcs.components.electrical import ElectricalDevice


@register_component_type("pv_inverter", "solar_inverter")
class SolarInverter(ElectricalDevice):
    """
    A grid-tied PV inverter. Beyond the AC quantities every electrical device reports, it adds
    its energy yield, operating state and cabinet temperature, and two optional groups: the DC
    input channels behind the `dc` flag, and the curtailment channels behind the `control`
    flag. Per-phase and power-quality groups come from `ElectricalDevice` via the `phases` and
    `quality` flags.

    Registered as `pv_inverter` (alias `solar_inverter`) rather than `inverter`, because
    `[inverter]` inside a PV system config already means that system's pvlib inverter
    parameters, and both would otherwise take a `model` key with incompatible meanings.

    Channels are declared unbound: wire them in TOML, or use `SunSpecInverter` to have the
    SunSpec point names, model id and connector filled in.
    """

    ENERGY = Constant(float, "energy", "Energy Yield", "Wh", context="inverter", aggregate="last")
    STATE = Constant(int, "state", "State", context="inverter", aggregate="last")
    TEMPERATURE = Constant(float, "temperature", "Temperature", "°C", context="inverter", aggregate="mean")

    POWER_DC = Constant(float, "dc_power", "DC Power", "W", context="inverter", aggregate="mean")
    VOLTAGE_DC = Constant(float, "dc_voltage", "DC Voltage", "V", context="inverter", aggregate="mean")
    CURRENT_DC = Constant(float, "dc_current", "DC Current", "A", context="inverter", aggregate="mean")

    POWER_LIMIT = Constant(float, "power_limit", "Power Limit", "%", context="inverter", aggregate="last")
    POWER_LIMIT_ENABLED = Constant(
        int, "power_limit_enabled", "Power Limit Enabled", context="inverter", aggregate="last"
    )

    def _add_core_channels(self, configs: Configurations) -> None:
        super()._add_core_channels(configs)
        self._add_channel(SolarInverter.ENERGY)
        self._add_channel(SolarInverter.STATE)
        self._add_channel(SolarInverter.TEMPERATURE)

    def _add_optional_channels(self, configs: Configurations) -> None:
        if configs.get_bool("dc", default=True):
            self._add_channel(SolarInverter.POWER_DC)
            self._add_channel(SolarInverter.VOLTAGE_DC)
            self._add_channel(SolarInverter.CURRENT_DC)

        if configs.get_bool("control", default=False):
            self._add_channel(SolarInverter.POWER_LIMIT)
            self._add_channel(SolarInverter.POWER_LIMIT_ENABLED)
