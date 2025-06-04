# -*- coding: utf-8 -*-
"""
penguin.components.storage.water
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""

from __future__ import annotations

from lori import Configurations, Constant
from lori.components import Component


class WaterStorage(Component):
    SECTION = "water_storage"

    LEVEL = Constant(float, "level", "Water Storage Level", "%")
    LITERS = Constant(float, "liters", "Water Storage Liters", "l")

    def configure(self, configs: Configurations) -> None:
        super().configure(configs)

        # TODO: Verify if "last" storage level as aggregation is correct
        self.data.add(WaterStorage.LEVEL, aggregate="last")
