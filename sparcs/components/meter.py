# -*- coding: utf-8 -*-
"""
sparcs.components.meter
~~~~~~~~~~~~~~~~~~~~~~~


"""

from __future__ import annotations

from lories.components import register_component_type
from lories.core import Constant
from lories.typing import Configurations
from sparcs.components.electrical import ElectricalDevice


@register_component_type("meter")
class EnergyMeter(ElectricalDevice):
    """
    An AC energy meter: a typed alternative to grouping raw meter channels in a generic
    component. Import/export energies are the device's raw lifetime counters; derive interval
    energies or power with channel processors. Per-phase and power-quality channel groups are
    enabled with the `phases` and `quality` flags (per-phase power factors need both); the
    `directions` flag adds `power_import`/`power_export`, which no meter reports directly and
    which are there for another component or a processor to fill from the signed power.

    Channels are declared unbound: wire them in TOML, or use `SunSpecMeter` to have the
    SunSpec point names, model id and connector filled in.
    """

    VOLTAGE = Constant(float, "voltage", "Voltage", "V", context="meter", aggregate="mean")
    ENERGY_IMPORT = Constant(float, "energy_import", "Imported Energy", "Wh", context="meter", aggregate="last")
    ENERGY_EXPORT = Constant(float, "energy_export", "Exported Energy", "Wh", context="meter", aggregate="last")

    # Not reported by meters: import/export power is derived from the signed power
    # (e.g. by a processor or another component), so these channels stay unbound
    POWER_IMPORT = Constant(float, "power_import", "Imported Power", "W", context="meter", aggregate="mean")
    POWER_EXPORT = Constant(float, "power_export", "Exported Power", "W", context="meter", aggregate="mean")

    POWER_L1 = Constant(float, "l1_power", "Phase 1 Power", "W", context="meter", aggregate="mean")
    POWER_L2 = Constant(float, "l2_power", "Phase 2 Power", "W", context="meter", aggregate="mean")
    POWER_L3 = Constant(float, "l3_power", "Phase 3 Power", "W", context="meter", aggregate="mean")
    VOLTAGE_L12 = Constant(float, "l12_voltage", "Phase 1-2 Voltage", "V", context="meter", aggregate="mean")
    VOLTAGE_L23 = Constant(float, "l23_voltage", "Phase 2-3 Voltage", "V", context="meter", aggregate="mean")
    VOLTAGE_L31 = Constant(float, "l31_voltage", "Phase 3-1 Voltage", "V", context="meter", aggregate="mean")

    POWER_FACTOR_L1 = Constant(float, "l1_power_factor", "Phase 1 Power Factor", "%", context="meter", aggregate="mean")
    POWER_FACTOR_L2 = Constant(float, "l2_power_factor", "Phase 2 Power Factor", "%", context="meter", aggregate="mean")
    POWER_FACTOR_L3 = Constant(float, "l3_power_factor", "Phase 3 Power Factor", "%", context="meter", aggregate="mean")

    def _add_core_channels(self, configs: Configurations) -> None:
        super()._add_core_channels(configs)
        self._add_channel(EnergyMeter.VOLTAGE)
        self._add_channel(EnergyMeter.ENERGY_IMPORT)
        self._add_channel(EnergyMeter.ENERGY_EXPORT)

    def _add_phase_channels(self, configs: Configurations) -> None:
        super()._add_phase_channels(configs)
        self._add_channel(EnergyMeter.POWER_L1)
        self._add_channel(EnergyMeter.POWER_L2)
        self._add_channel(EnergyMeter.POWER_L3)
        self._add_channel(EnergyMeter.VOLTAGE_L12)
        self._add_channel(EnergyMeter.VOLTAGE_L23)
        self._add_channel(EnergyMeter.VOLTAGE_L31)

    def _add_quality_channels(self, configs: Configurations) -> None:
        super()._add_quality_channels(configs)
        if configs.get_bool("phases", default=False):
            self._add_channel(EnergyMeter.POWER_FACTOR_L1)
            self._add_channel(EnergyMeter.POWER_FACTOR_L2)
            self._add_channel(EnergyMeter.POWER_FACTOR_L3)

    def _add_optional_channels(self, configs: Configurations) -> None:
        if configs.get_bool("directions", default=False):
            self._add_channel(EnergyMeter.POWER_IMPORT)
            self._add_channel(EnergyMeter.POWER_EXPORT)
