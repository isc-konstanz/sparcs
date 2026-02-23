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

    STATE = Constant(bool, "state", "Irrigation State", alias="irrigation_state")
    FLOW = Constant(float, "flow", "Irrigation Flow", alias="irrigation_flow", unit="l/min")

    soil: Sequence[SoilMoisture]

    def __init__(self, context: Component, configs: Configurations, soil: Sequence[SoilMoisture]) -> None:
        super().__init__(context=context, configs=configs)
        self.soil = soil

    def configure(self, configs: Configurations) -> None:
        super().configure(configs)
        self.data.add(Irrigation.STATE, aggregate="max")
        self.data.add(Irrigation.FLOW, aggregate="sum")
