# -*- coding: utf-8 -*-
"""
    penguin.components.vehicle
    ~~~~~~~~~~~~~~~~~~~~~~~~~~


"""
from loris import Component, Configurations


class ElectricVehicle(Component):
    TYPE = "ev"

    def configure(self, configs: Configurations) -> None:
        super().configure(configs)
        self.capacity = configs.get_float("capacity")

    @property
    def type(self) -> str:
        return self.TYPE
