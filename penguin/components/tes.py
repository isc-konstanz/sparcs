# -*- coding: utf-8 -*-
"""
    penguin.components.tes
    ~~~~~~~~~~~~~~~~~~~~~~


"""
from loris import Component, Configurations


class ThermalEnergyStorage(Component):
    TYPE: str = "tes"

    TEMPERATURE: str = "tes_temp"
    TEMPERATURE_HEATING: str = "tes_ht_temp"
    TEMPERATURE_DOMESTIC: str = "tes_dom_temp"

    def configure(self, configs: Configurations) -> None:
        super().configure(configs)
        self.volume = configs.get_float("volume")

        # For the thermal storage capacity in kWh/K, it will be assumed to be filled with water,
        # resulting in a specific heat capacity of 4.184 J/g*K.
        # TODO: Make tank content and specific heat capacity configurable
        self.capacity = 4.184 * self.volume / 3600

    def activate(self) -> None:
        super().activate()

    def deactivate(self) -> None:
        super().deactivate()

    @property
    def type(self) -> str:
        return self.TYPE
