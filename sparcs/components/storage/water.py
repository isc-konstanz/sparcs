# -*- coding: utf-8 -*-
"""
sparcs.components.storage.water
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""

from __future__ import annotations

from lories import Constant
from lories.typing import Configurations
from lories.components import Component


class WaterStorage(Component):
    TYPE = "water_storage"

    LEVEL = Constant(float, "level", "Water Storage Level", "%")
    LITERS = Constant(float, "liters", "Water Storage Liters", "l")

    def configure(self, configs: Configurations) -> None:
        super().configure(configs)

        # TODO: Verify if "last" storage level as aggregation is correct
        self.data.add(WaterStorage.LEVEL, aggregate="last")
