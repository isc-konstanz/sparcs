# -*- coding: utf-8 -*-
"""
sparcs.components.devices.openems.inverter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""

from __future__ import annotations

from lories.components import register_component_type
from lories.connectors.openems import OpenEMSBinding
from sparcs.components.solar.inverter import SolarInverter


@register_component_type("openems_inverter")
class OpenEMSInverter(OpenEMSBinding, SolarInverter):
    """
    A `SolarInverter` read from one OpenEMS PV inverter component. Those implement the same
    `ElectricityMeter` nature the meters do, with the meter type PRODUCTION, so their
    `ActivePower` is already positive while producing -- the generator reference this
    vocabulary uses -- and no entry needs a sign flip. `energy` is the lifetime yield counter
    `ActiveProductionEnergy`.

    The DC group, the operating state, the cabinet temperature and the curtailment channels
    stay unbound: the OpenEMS connector is push-only and has no write path, and the remaining
    quantities have no counterpart on the nature. Apparent power and the power factors have
    none either. Per-phase powers are not bound, because `SolarInverter` declares no per-phase
    power vocabulary; wire them in TOML if a site needs them.
    """

    POINTS = {
        SolarInverter.POWER: "ActivePower",
        SolarInverter.POWER_REACTIVE: "ReactivePower",
        # OpenEMS publishes currents in mA
        SolarInverter.CURRENT: ("Current", 0.001),
        SolarInverter.CURRENT_L1: ("CurrentL1", 0.001),
        SolarInverter.CURRENT_L2: ("CurrentL2", 0.001),
        SolarInverter.CURRENT_L3: ("CurrentL3", 0.001),
        # ... voltages in mV
        SolarInverter.VOLTAGE_L1: ("VoltageL1", 0.001),
        SolarInverter.VOLTAGE_L2: ("VoltageL2", 0.001),
        SolarInverter.VOLTAGE_L3: ("VoltageL3", 0.001),
        # ... and the frequency in mHz
        SolarInverter.FREQUENCY: ("Frequency", 0.001),
        SolarInverter.ENERGY: "ActiveProductionEnergy",
    }
