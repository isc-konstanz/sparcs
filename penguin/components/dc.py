# -*- coding: utf-8 -*-
"""
    penguin.components.dc
    ~~~~~~~~~~~~~~~~~~~~~


"""
from loris import Configurations, Component


class DirectCurrent(Component):
    TYPE: str = 'dc'

    POWER_DC:  str = 'dc_power'
    ENERGY_DC:  str = 'dc_energy'

    # noinspection PyProtectedMember
    def __configure__(self, configs: Configurations) -> None:
        super().__configure__(configs)

    def get_type(self) -> str:
        return self.TYPE
