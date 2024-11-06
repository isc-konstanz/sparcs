# -*- coding: utf-8 -*-
"""
penguin.components.vehicle
~~~~~~~~~~~~~~~~~~~~~~~~~~


"""

from lori.components import Component, register_component_type
from lori.core import Configurations


@register_component_type
class ElectricVehicle(Component):
    TYPE = "ev"

    def configure(self, configs: Configurations) -> None:
        super().configure(configs)
        self.capacity = configs.get_float("capacity")
