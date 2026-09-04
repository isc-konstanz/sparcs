# -*- coding: utf-8 -*-
"""
sparcs.components.devices.sunspec.meter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""

from __future__ import annotations

from typing import Optional

from lories.components import register_component_type
from lories.connectors.sunspec import SunSpecBinding
from lories.core import Constant
from sparcs.components.meter import EnergyMeter


@register_component_type("sunspec_meter")
class SunSpecMeter(SunSpecBinding, EnergyMeter):
    """
    An `EnergyMeter` read over SunSpec. The point names are identical across the four meter
    models -- 201 single phase, 202 split phase, 203 wye three phase, 204 delta three phase --
    except that model 201 alone names its line-to-line voltage points `PPVph*` instead of
    `PhVph*`. Meters behind one gateway are picked by instance.
    """

    MODELS = [201, 202, 203, 204]
    DEFAULT_MODEL = 203

    LINE_TO_LINE = {
        EnergyMeter.VOLTAGE_L12: "AB",
        EnergyMeter.VOLTAGE_L23: "BC",
        EnergyMeter.VOLTAGE_L31: "CA",
    }

    POINTS = {
        # Model 20x defines no sign for 'W': the CT orientation decides, so a site that is
        # wired against the load reference sets scale = -1 on the channel in TOML
        EnergyMeter.POWER: "W",
        EnergyMeter.CURRENT: "A",
        EnergyMeter.VOLTAGE: "PhV",
        EnergyMeter.FREQUENCY: "Hz",
        EnergyMeter.ENERGY_IMPORT: "TotWhImp",
        EnergyMeter.ENERGY_EXPORT: "TotWhExp",
        EnergyMeter.POWER_L1: "WphA",
        EnergyMeter.POWER_L2: "WphB",
        EnergyMeter.POWER_L3: "WphC",
        EnergyMeter.CURRENT_L1: "AphA",
        EnergyMeter.CURRENT_L2: "AphB",
        EnergyMeter.CURRENT_L3: "AphC",
        EnergyMeter.VOLTAGE_L1: "PhVphA",
        EnergyMeter.VOLTAGE_L2: "PhVphB",
        EnergyMeter.VOLTAGE_L3: "PhVphC",
        EnergyMeter.VOLTAGE_L12: "PhVphAB",
        EnergyMeter.VOLTAGE_L23: "PhVphBC",
        EnergyMeter.VOLTAGE_L31: "PhVphCA",
        EnergyMeter.POWER_APPARENT: "VA",
        EnergyMeter.POWER_REACTIVE: "VAR",
        EnergyMeter.POWER_FACTOR: "PF",
        EnergyMeter.POWER_FACTOR_L1: "PFphA",
        EnergyMeter.POWER_FACTOR_L2: "PFphB",
        EnergyMeter.POWER_FACTOR_L3: "PFphC",
    }

    def _point(self, constant: Constant) -> Optional[str]:
        if self.model == 201 and constant in SunSpecMeter.LINE_TO_LINE:
            return f"PPVph{SunSpecMeter.LINE_TO_LINE[constant]}"
        return super()._point(constant)
