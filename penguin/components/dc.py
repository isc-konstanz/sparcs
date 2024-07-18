# -*- coding: utf-8 -*-
"""
    penguin.components.dc
    ~~~~~~~~~~~~~~~~~~~~~


"""
from loris import Component, Configurations


class DirectCurrent(Component):
    TYPE: str = "dc"

    POWER_DC: str = "dc_power"
    ENERGY_DC: str = "dc_energy"

    def configure(self, configs: Configurations) -> None:
        super().configure(configs)

    def activate(self) -> None:
        super().activate()

    def deactivate(self) -> None:
        super().deactivate()

    @property
    def type(self) -> str:
        return self.TYPE
