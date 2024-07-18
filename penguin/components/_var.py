# -*- coding: utf-8 -*-
"""
    penguin.components._var
    ~~~~~~~~~~~~~~~~~~~~~~~


"""
from penguin.components import (
    AlternatingCurrent,
    DirectCurrent,
    ElectricalEnergyStorage,
    PVSystem,
    ThermalEnergyStorage,
)

STORAGE_POWER = {
    ElectricalEnergyStorage.POWER_CHARGE: "EES Charging Power [W]",
    ElectricalEnergyStorage.POWER_DISCHARGE: "EES Discharging Power [W]"
}

STORAGE_ENERGY = {
    ElectricalEnergyStorage.ENERGY_CHARGE:    "EES Charged Energy [kWh]",
    ElectricalEnergyStorage.ENERGY_DISCHARGE: "EES Discharged Energy [kWh]"
}

PV_POWER = {
    PVSystem.POWER:      "Generated PV Power [W]",
    PVSystem.POWER_CALC: "Calculated PV Power [W]",
    PVSystem.POWER_EXP:  "Exported PV Power [W]"
}

PV_ENERGY = {
    PVSystem.ENERGY:      "Generated PV Energy [kWh]",
    PVSystem.ENERGY_CALC: "Calculated PV Energy [kWh]",
    PVSystem.ENERGY_EXP:  "Exported PV Energy [kWh]"
}

PV = {
    **PV_POWER,
    **PV_ENERGY
}

DC_POWER = {
    DirectCurrent.POWER_DC: "Generated DC Power [W]"
}

DC_ENERGY = {
    DirectCurrent.ENERGY_DC: "Generated DC Energy [kWh]"
}

DC = {
    **DC_POWER,
    **DC_ENERGY
}

STATES = {
    ElectricalEnergyStorage.STATE_OF_CHARGE: "EES State of Charge [%]",
    ThermalEnergyStorage.TEMPERATURE:        "TES Temperature [°C]"
}

AC_POWER = {
    AlternatingCurrent.POWER_ACTIVE: "Total Active Power [W]",
    AlternatingCurrent.POWER_L1_ACTIVE: "Phase 1 Active Power [W]",
    AlternatingCurrent.POWER_L2_ACTIVE: "Phase 2 Active Power [W]",
    AlternatingCurrent.POWER_L3_ACTIVE: "Phase 3 Active Power [W]",
    AlternatingCurrent.POWER_REACTIVE: "Total Reactive Power [W]",
    AlternatingCurrent.POWER_L1_REACTIVE: "Phase 1 Reactive Power [W]",
    AlternatingCurrent.POWER_L2_REACTIVE: "Phase 2 Reactive Power [W]",
    AlternatingCurrent.POWER_L3_REACTIVE: "Phase 3 Reactive Power [W]",
    AlternatingCurrent.POWER_APPARENT: "Total Apparent Power [W]",
    AlternatingCurrent.POWER_L1_APPARENT: "Phase 1 Apparent Power [W]",
    AlternatingCurrent.POWER_L2_APPARENT: "Phase 2 Apparent Power [W]",
    AlternatingCurrent.POWER_L3_APPARENT: "Phase 3 Apparent Power [W]",
    AlternatingCurrent.POWER_IMPORT: "Total Imported Power [W]",
    AlternatingCurrent.POWER_L1_IMPORT: "Phase 1 Imported Power [W]",
    AlternatingCurrent.POWER_L2_IMPORT: "Phase 2 Imported Power [W]",
    AlternatingCurrent.POWER_L3_IMPORT: "Phase 3 Imported Power [W]",
    AlternatingCurrent.POWER_EXPORT: "Total Exported Power [W]",
    AlternatingCurrent.POWER_L1_EXPORT: "Phase 1 Exported Power [W]",
    AlternatingCurrent.POWER_L2_EXPORT: "Phase 2 Exported Power [W]",
    AlternatingCurrent.POWER_L3_EXPORT: "Phase 3 Exported Power [W]"
}

AC_ENERGY = {
    AlternatingCurrent.ENERGY_ACTIVE: "Total Active Energy [kWh]",
    AlternatingCurrent.ENERGY_L1_ACTIVE: "Phase 1 Active Energy [kWh]",
    AlternatingCurrent.ENERGY_L2_ACTIVE: "Phase 2 Active Energy [kWh]",
    AlternatingCurrent.ENERGY_L3_ACTIVE: "Phase 3 Active Energy [kWh]",
    AlternatingCurrent.ENERGY_REACTIVE: "Total Reactive Energy [kWh]",
    AlternatingCurrent.ENERGY_L1_REACTIVE: "Phase 1 Reactive Energy [kWh]",
    AlternatingCurrent.ENERGY_L2_REACTIVE: "Phase 2 Reactive Energy [kWh]",
    AlternatingCurrent.ENERGY_L3_REACTIVE: "Phase 3 Reactive Energy [kWh]",
    AlternatingCurrent.ENERGY_APPARENT: "Total Apparent Energy [kWh]",
    AlternatingCurrent.ENERGY_L1_APPARENT: "Phase 1 Apparent Energy [kWh]",
    AlternatingCurrent.ENERGY_L2_APPARENT: "Phase 2 Apparent Energy [kWh]",
    AlternatingCurrent.ENERGY_L3_APPARENT: "Phase 3 Apparent Energy [kWh]",
    AlternatingCurrent.ENERGY_IMPORT: "Total Imported Energy [kWh]",
    AlternatingCurrent.ENERGY_L1_IMPORT: "Phase 1 Imported Energy [kWh]",
    AlternatingCurrent.ENERGY_L2_IMPORT: "Phase 2 Imported Energy [kWh]",
    AlternatingCurrent.ENERGY_L3_IMPORT: "Phase 3 Imported Energy [kWh]",
    AlternatingCurrent.ENERGY_EXPORT: "Total Exported Energy [kWh]",
    AlternatingCurrent.ENERGY_L1_EXPORT: "Phase 1 Exported Energy [kWh]",
    AlternatingCurrent.ENERGY_L2_EXPORT: "Phase 2 Exported Energy [kWh]",
    AlternatingCurrent.ENERGY_L3_EXPORT: "Phase 3 Exported Energy [kWh]"
}

AC = {
    AlternatingCurrent.VOLTAGE_L1: "Phase 1 Voltage [V]",
    AlternatingCurrent.VOLTAGE_L2: "Phase 2 Voltage [V]",
    AlternatingCurrent.VOLTAGE_L3: "Phase 3 Voltage [V]",
    AlternatingCurrent.CURRENT_L1: "Phase 1 Current [A]",
    AlternatingCurrent.CURRENT_L2: "Phase 2 Current [A]",
    AlternatingCurrent.CURRENT_L3: "Phase 3 Current [A]",
    AlternatingCurrent.FREQUENCY: "Frequency [Hz]"
}

POWER = {
    **STORAGE_POWER,
    **PV_POWER,
    **DC_POWER,
    **AC_POWER
}

ENERGY = {
    **STORAGE_ENERGY,
    **PV_ENERGY,
    **DC_ENERGY,
    **AC_ENERGY
}
