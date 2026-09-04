# -*- coding: utf-8 -*-
"""
sparcs.components.devices.openems.meter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""

from __future__ import annotations

from typing import Optional

from lories.components import register_component_type
from lories.connectors.openems import OpenEMSBinding
from lories.core import ConfigurationError, Constant
from lories.typing import Configurations
from sparcs.components.meter import EnergyMeter


@register_component_type("openems_meter")
class OpenEMSMeter(OpenEMSBinding, EnergyMeter):
    """
    An `EnergyMeter` read from one OpenEMS `ElectricityMeter` component. The channel names are
    the same for every meter, so wiring a device needs only the OpenEMS component id its
    channels live under (`device`) and the connector -- plus `meter_type`, the OpenEMS
    `MeterType` the component was configured with, because the sign of `ActivePower` is defined
    per type and the Edge does not reliably publish it.

    OpenEMS reports a production meter positive while it produces, the opposite of the load
    reference this vocabulary uses, so `production` and `production_and_consumption` meters get
    `scale = -1` on the active and reactive powers, total and per phase. A `grid` or
    `consumption_metered` meter is already positive on import and is taken as it comes. The
    energies are only-positive counters for every meter type, so `energy_import` and
    `energy_export` map to `ActiveConsumptionEnergy` and `ActiveProductionEnergy` whatever the
    role, and are never flipped.

    Apparent power, the power factors, the line-to-line voltages and `power_import` /
    `power_export` have no OpenEMS counterpart and stay unbound, to be filled by TOML or by a
    processor.
    """

    #: OpenEMS MeterType names, lowercased.
    METER_TYPES = ("grid", "production", "consumption_metered", "production_and_consumption")

    #: The meter types OpenEMS reports positive while producing.
    PRODUCTION_TYPES = ("production", "production_and_consumption")

    #: Channels whose sign follows the active-power reference and is flipped for those types.
    SIGNED = (
        EnergyMeter.POWER,
        EnergyMeter.POWER_L1,
        EnergyMeter.POWER_L2,
        EnergyMeter.POWER_L3,
        EnergyMeter.POWER_REACTIVE,
    )

    POINTS = {
        EnergyMeter.POWER: "ActivePower",
        EnergyMeter.POWER_L1: "ActivePowerL1",
        EnergyMeter.POWER_L2: "ActivePowerL2",
        EnergyMeter.POWER_L3: "ActivePowerL3",
        EnergyMeter.POWER_REACTIVE: "ReactivePower",
        # OpenEMS publishes currents in mA
        EnergyMeter.CURRENT: ("Current", 0.001),
        EnergyMeter.CURRENT_L1: ("CurrentL1", 0.001),
        EnergyMeter.CURRENT_L2: ("CurrentL2", 0.001),
        EnergyMeter.CURRENT_L3: ("CurrentL3", 0.001),
        # ... voltages in mV
        EnergyMeter.VOLTAGE: ("Voltage", 0.001),
        EnergyMeter.VOLTAGE_L1: ("VoltageL1", 0.001),
        EnergyMeter.VOLTAGE_L2: ("VoltageL2", 0.001),
        EnergyMeter.VOLTAGE_L3: ("VoltageL3", 0.001),
        # ... and the frequency in mHz
        EnergyMeter.FREQUENCY: ("Frequency", 0.001),
        # Only-positive counters, mapped the same way for every meter type
        EnergyMeter.ENERGY_IMPORT: "ActiveConsumptionEnergy",
        EnergyMeter.ENERGY_EXPORT: "ActiveProductionEnergy",
    }

    meter_type: str

    def _configure_bindings(self, configs: Configurations) -> None:
        super()._configure_bindings(configs)
        meter_type = configs.get("meter_type", default=None)
        if not isinstance(meter_type, str) or meter_type.lower() not in OpenEMSMeter.METER_TYPES:
            raise ConfigurationError(
                f"{type(self).__name__} '{self.id}' requires the OpenEMS meter type its powers are signed by: "
                f"set 'meter_type' to one of {list(OpenEMSMeter.METER_TYPES)}, got '{meter_type}'"
            )
        self.meter_type = meter_type.lower()

    def _scale(self, constant: Constant) -> Optional[float]:
        scale = super()._scale(constant)
        scale = 1.0 if scale is None else float(scale)
        if self.meter_type in OpenEMSMeter.PRODUCTION_TYPES and constant in OpenEMSMeter.SIGNED:
            scale *= -1
        return None if scale == 1.0 else scale
