# -*- coding: utf-8 -*-
"""
sparcs.components.storage.thermal
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""

from lories.components import Component, register_component_type
from lories.core import Constant
from lories.typing import Configurations


@register_component_type("tes")
class ThermalEnergyStorage(Component):
    TEMPERATURE = Constant(float, "temp", "TES Temperature", "°C", context="tes")
    TEMPERATURE_DOMESTIC = Constant(float, "temp_dom", "TES Domestic Temperature ", "°C", context="tes")
    TEMPERATURE_HEATING = Constant(float, "temp_ht", "TES Heating Temperature ", "°C", context="tes")

    volume: float
    capacity: float

    def configure(self, configs: Configurations) -> None:
        super().configure(configs)
        self.volume = configs.get_float("volume")

        # For the thermal storage capacity in kWh/K, it will be assumed to be filled with water,
        # resulting in a specific heat capacity of 4.184 J/g*K.
        # TODO: Make tank content and specific heat capacity configurable
        self.capacity = 4.184 * self.volume / 3600

        def add_channel(constant: Constant, **custom) -> None:
            channel = constant.to_dict()
            channel["name"] = constant.name.replace("TES", self.name, 1)
            channel["column"] = constant.id.replace("tes", self.key, 1)
            channel["aggregate"] = "mean"
            channel.update(custom)

        add_channel(ThermalEnergyStorage.TEMPERATURE)
        # add_channel(ThermalEnergyStorage.TEMPERATURE_DOMESTIC)
        # add_channel(ThermalEnergyStorage.TEMPERATURE_HEATING)
