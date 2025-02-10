# -*- coding: utf-8 -*-
"""
penguin.components.vehicle
~~~~~~~~~~~~~~~~~~~~~~~~~~


"""

from lori.components import Component, register_component_type
from lori.core import Configurations

TYPE: str = "ev"


@register_component_type(TYPE)
class ElectricVehicle(Component):
    def configure(self, configs: Configurations) -> None:
        super().configure(configs)
        self.capacity = configs.get_float("capacity")
