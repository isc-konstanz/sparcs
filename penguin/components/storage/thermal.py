# -*- coding: utf-8 -*-
"""
penguin.components.storage.thermal
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""

from lori.components import Component, register_component_type
from lori.core import Configurations, Constant


@register_component_type("tes")
class ThermalEnergyStorage(Component):
    TEMPERATURE = Constant(float, "tes_temp", "TES Temperature", "°C")
    TEMPERATURE_DOMESTIC = Constant(float, "tes_dom_temp", "TES Domestic Temperature ", "°C")
    TEMPERATURE_HEATING = Constant(float, "tes_ht_temp", "TES Heating Temperature ", "°C")

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
            channel["column"] = constant.key.replace("tes", self.key, 1)
            channel["aggregate"] = "mean"
            channel["connector"] = None
            channel.update(custom)

        add_channel(ThermalEnergyStorage.TEMPERATURE)
