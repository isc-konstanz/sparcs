# -*- coding: utf-8 -*-
"""
penguin.components.solar.inverter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""

from lori.core import Constant
from lori.components import Component


class SolarInverter(Component):
    POWER_ACTIVE = Constant(float, "active_power", "Active Power", "W")
    POWER_L1_ACTIVE = Constant(float, "l1_active_power", "Phase 1 Active Power", "W")
    POWER_L2_ACTIVE = Constant(float, "l2_active_power", "Phase 2 Active Power", "W")
    POWER_L3_ACTIVE = Constant(float, "l3_active_power", "Phase 3 Active Power", "W")

    POWER_REACTIVE = Constant(float, "reactive_power", "Reactive Power", "W")
    POWER_L1_REACTIVE = Constant(float, "l1_reactive_power", "Phase 1 Reactive Power", "W")
    POWER_L2_REACTIVE = Constant(float, "l2_reactive_power", "Phase 2 Reactive Power", "W")
    POWER_L3_REACTIVE = Constant(float, "l3_reactive_power", "Phase 3 Reactive Power", "W")

    POWER_APPARENT = Constant(float, "apparent_power", "Apparent Power", "W")
    POWER_L1_APPARENT = Constant(float, "l1_apparent_power", "Phase 1 Apparent Power", "W")
    POWER_L2_APPARENT = Constant(float, "l2_apparent_power", "Phase 2 Apparent Power", "W")
    POWER_L3_APPARENT = Constant(float, "l3_apparent_power", "Phase 3 Apparent Power", "W")

    POWER_IMPORT = Constant(float, "import_power", "Imported Power", "W")
    POWER_L1_IMPORT = Constant(float, "l1_import_power", "Phase 1 Imported Power", "W")
    POWER_L2_IMPORT = Constant(float, "l2_import_power", "Phase 2 Imported Power", "W")
    POWER_L3_IMPORT = Constant(float, "l3_import_power", "Phase 3 Imported Power", "W")

    POWER_EXPORT = Constant(float, "export_power", "Exported Power", "W")
    POWER_L1_EXPORT = Constant(float, "l1_export_power", "Phase 1 Exported Power", "W")
    POWER_L2_EXPORT = Constant(float, "l2_export_power", "Phase 2 Exported Power", "W")
    POWER_L3_EXPORT = Constant(float, "l3_export_power", "Phase 3 Exported Power", "W")

    ENERGY_ACTIVE = Constant(float, "Active Energy", "kWh")
    ENERGY_L1_ACTIVE = Constant(float, "Phase 1 Active Energy", "kWh")
    ENERGY_L2_ACTIVE = Constant(float, "Phase 2 Active Energy", "kWh")
    ENERGY_L3_ACTIVE = Constant(float, "Phase 3 Active Energy", "kWh")

    ENERGY_REACTIVE = Constant(float, "Reactive Energy", "kWh")
    ENERGY_L1_REACTIVE = Constant(float, "Phase 1 Reactive Energy", "kWh")
    ENERGY_L2_REACTIVE = Constant(float, "Phase 2 Reactive Energy", "kWh")
    ENERGY_L3_REACTIVE = Constant(float, "Phase 3 Reactive Energy", "kWh")

    ENERGY_APPARENT = Constant(float, "Total Apparent Energy", "kWh")
    ENERGY_L1_APPARENT = Constant(float, "Phase 1 Apparent Energy", "kWh")
    ENERGY_L2_APPARENT = Constant(float, "Phase 2 Apparent Energy", "kWh")
    ENERGY_L3_APPARENT = Constant(float, "Phase 3 Apparent Energy", "kWh")

    ENERGY_IMPORT = Constant(float, "Imported Energy", "kWh")
    ENERGY_L1_IMPORT = Constant(float, "Phase 1 Imported Energy", "kWh")
    ENERGY_L2_IMPORT = Constant(float, "Phase 2 Imported Energy", "kWh")
    ENERGY_L3_IMPORT = Constant(float, "Phase 3 Imported Energy", "kWh")

    ENERGY_EXPORT = Constant(float, "Exported Energy", "kWh")
    ENERGY_L1_EXPORT = Constant(float, "Phase 1 Exported Energy", "kWh")
    ENERGY_L2_EXPORT = Constant(float, "Phase 2 Exported Energy", "kWh")
    ENERGY_L3_EXPORT = Constant(float, "Phase 3 Exported Energy", "kWh")

    VOLTAGE_L1 = Constant(float, "l1_voltage", "Phase 1 Voltage", "V")
    VOLTAGE_L2 = Constant(float, "l2_voltage", "Phase 2 Voltage", "V")
    VOLTAGE_L3 = Constant(float, "l3_voltage", "Phase 3 Voltage", "V")

    CURRENT_L1 = Constant(float, "l1_current", "Phase 1 Current", "A")
    CURRENT_L2 = Constant(float, "l2_current", "Phase 2 Current", "A")
    CURRENT_L3 = Constant(float, "l3_current", "Phase 3 Current", "A")

    FREQUENCY = Constant(float, "frequency", "Frequency", "Hz")
