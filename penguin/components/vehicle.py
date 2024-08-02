# -*- coding: utf-8 -*-
"""
    penguin.components.vehicle
    ~~~~~~~~~~~~~~~~~~~~~~~~~~


"""
from loris.components import Component, register_component_type
from loris.core import Configurations


@register_component_type
class ElectricVehicle(Component):
    TYPE = "ev"

    def configure(self, configs: Configurations) -> None:
        super().configure(configs)
        self.capacity = configs.get_float("capacity")
