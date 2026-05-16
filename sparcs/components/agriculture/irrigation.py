# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.irrigation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""

from __future__ import annotations

from typing import Sequence

from lories import Component, Constant
from lories.typing import Configurations
from sparcs.components.agriculture import SoilMoisture


class Irrigation(Component):
    TYPE = "irrigation"

    STATE = Constant(bool, "state", "Irrigation State", context="irrigation")
    FLOW = Constant(float, "flow", "Irrigation Flow", context="irrigation", unit="l/min")

    soil: Sequence[SoilMoisture]

    def __init__(self, context: Component, configs: Configurations, soil: Sequence[SoilMoisture]) -> None:
        super().__init__(context=context, configs=configs)
        self.soil = soil

    def configure(self, configs: Configurations) -> None:
        super().configure(configs)

        def add_channel(constant: Constant, **custom) -> None:
            channel = constant.to_dict()
            channel["name"] = constant.name.replace("Irrigation", self.name, 1)
            channel["column"] = constant.id.replace("irrigation", self.key, 1)
            channel.update(custom)
            self.data.add(**channel)

        add_channel(Irrigation.STATE, aggregate="max")
        add_channel(Irrigation.FLOW, aggregate="sum")
