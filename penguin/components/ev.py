# -*- coding: utf-8 -*-
"""
    penguin.components.ev
    ~~~~~~~~~~~~~~~~~~~~~


"""
from loris import Component, Configurations


class ElectricVehicle(Component):
    TYPE = "ev"

    def configure(self, configs: Configurations) -> None:
        super().configure(configs)
        self.capacity = configs.get_float("capacity")

    def activate(self) -> None:
        super().activate()

    def deactivate(self) -> None:
        super().deactivate()

    @property
    def type(self) -> str:
        return self.TYPE
