# -*- coding: utf-8 -*-
"""
    penguin.components.current
    ~~~~~~~~~~~~~~~~~~~~~~~~~~


"""
from loris import Component


class DirectCurrent(Component):
    TYPE: str = "dc"

    POWER_DC: str = "dc_power"
    ENERGY_DC: str = "dc_energy"

    @property
    def type(self) -> str:
        return self.TYPE


class AlternatingCurrent(Component):
    TYPE: str = "ac"

    POWER_ACTIVE: str = "active_power"
    POWER_L1_ACTIVE: str = "l1_active_power"
    POWER_L2_ACTIVE: str = "l2_active_power"
    POWER_L3_ACTIVE: str = "l3_active_power"

    POWER_REACTIVE: str = "reactive_power"
    POWER_L1_REACTIVE: str = "l1_reactive_power"
    POWER_L2_REACTIVE: str = "l2_reactive_power"
    POWER_L3_REACTIVE: str = "l3_reactive_power"

    POWER_APPARENT: str = "apparent_power"
    POWER_L1_APPARENT: str = "l1_apparent_power"
    POWER_L2_APPARENT: str = "l2_apparent_power"
    POWER_L3_APPARENT: str = "l3_apparent_power"

    POWER_IMPORT: str = "import_power"
    POWER_L1_IMPORT: str = "l1_import_power"
    POWER_L2_IMPORT: str = "l2_import_power"
    POWER_L3_IMPORT: str = "l3_import_power"

    POWER_EXPORT: str = "export_power"
    POWER_L1_EXPORT: str = "l1_export_power"
    POWER_L2_EXPORT: str = "l2_export_power"
    POWER_L3_EXPORT: str = "l3_export_power"

    ENERGY_ACTIVE: str = "active_power"
    ENERGY_L1_ACTIVE: str = "l1_active_power"
    ENERGY_L2_ACTIVE: str = "l2_active_power"
    ENERGY_L3_ACTIVE: str = "l3_active_power"

    ENERGY_REACTIVE: str = "reactive_power"
    ENERGY_L1_REACTIVE: str = "l1_reactive_power"
    ENERGY_L2_REACTIVE: str = "l2_reactive_power"
    ENERGY_L3_REACTIVE: str = "l3_reactive_power"

    ENERGY_APPARENT: str = "apparent_power"
    ENERGY_L1_APPARENT: str = "l1_apparent_power"
    ENERGY_L2_APPARENT: str = "l2_apparent_power"
    ENERGY_L3_APPARENT: str = "l3_apparent_power"

    ENERGY_IMPORT: str = "import_power"
    ENERGY_L1_IMPORT: str = "l1_import_power"
    ENERGY_L2_IMPORT: str = "l2_import_power"
    ENERGY_L3_IMPORT: str = "l3_import_power"

    ENERGY_EXPORT: str = "export_power"
    ENERGY_L1_EXPORT: str = "l1_export_power"
    ENERGY_L2_EXPORT: str = "l2_export_power"
    ENERGY_L3_EXPORT: str = "l3_export_power"

    VOLTAGE_L1: str = "l1_voltage"
    VOLTAGE_L2: str = "l2_voltage"
    VOLTAGE_L3: str = "l3_voltage"

    CURRENT_L1: str = "l1_current"
    CURRENT_L2: str = "l2_current"
    CURRENT_L3: str = "l3_current"

    FREQUENCY: str = "frequency"

    @property
    def type(self) -> str:
        return self.TYPE
