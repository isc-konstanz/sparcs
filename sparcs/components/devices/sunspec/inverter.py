# -*- coding: utf-8 -*-
"""
sparcs.components.devices.sunspec.inverter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""

from __future__ import annotations

from lories.components import register_component_type
from lories.connectors.sunspec import SunSpecBinding
from lories.core import Constant
from sparcs.components.solar.inverter import SolarInverter


@register_component_type("sunspec_inverter")
class SunSpecInverter(SunSpecBinding, SolarInverter):
    """
    A `SolarInverter` read over SunSpec. The point names are identical across the three
    inverter models -- 101 single phase, 102 split phase, 103 three phase -- so wiring a device
    needs only its model id. The control channels address the immediate-controls model 123 and
    are writable through the connector; they reuse the metering model's instance, which assumes
    the device indexes both blocks alike. Override `instance` per channel in TOML if it does
    not.
    """

    MODELS = [101, 102, 103]
    DEFAULT_MODEL = 103
    CONTROLS_MODEL = 123

    POINTS = {
        # Model 10x 'W' is the AC output power, positive while producing (generator reference)
        SolarInverter.POWER: "W",
        SolarInverter.CURRENT: "A",
        SolarInverter.FREQUENCY: "Hz",
        SolarInverter.ENERGY: "WH",
        SolarInverter.STATE: "St",
        SolarInverter.TEMPERATURE: "TmpCab",
        SolarInverter.POWER_DC: "DCW",
        SolarInverter.VOLTAGE_DC: "DCV",
        SolarInverter.CURRENT_DC: "DCA",
        SolarInverter.CURRENT_L1: "AphA",
        SolarInverter.CURRENT_L2: "AphB",
        SolarInverter.CURRENT_L3: "AphC",
        SolarInverter.VOLTAGE_L1: "PhVphA",
        SolarInverter.VOLTAGE_L2: "PhVphB",
        SolarInverter.VOLTAGE_L3: "PhVphC",
        SolarInverter.POWER_APPARENT: "VA",
        SolarInverter.POWER_REACTIVE: "VAr",
        SolarInverter.POWER_FACTOR: "PF",
        SolarInverter.POWER_LIMIT: "WMaxLimPct",
        SolarInverter.POWER_LIMIT_ENABLED: "WMaxLim_Ena",
    }

    CONTROLS = (SolarInverter.POWER_LIMIT, SolarInverter.POWER_LIMIT_ENABLED)

    def _model(self, constant: Constant) -> int:
        if constant in SunSpecInverter.CONTROLS:
            return SunSpecInverter.CONTROLS_MODEL
        return super()._model(constant)
