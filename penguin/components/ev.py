# -*- coding: utf-8 -*-
"""
    penguin.components.ev
    ~~~~~~~~~~~~~~~~~~~~~


"""
from loris import Configurations, Component


class ElectricVehicle(Component):
    TYPE = 'ev'

    # noinspection PyProtectedMember
    def __configure__(self, configs: Configurations) -> None:
        super().__configure__(configs)
        self.capacity = configs.get_float('capacity')

    def get_type(self) -> str:
        return self.TYPE
