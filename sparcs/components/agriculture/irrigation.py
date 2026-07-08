# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.irrigation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""

from __future__ import annotations

from lories import Component, Constant
from lories.typing import Configurations


class Irrigation(Component):
    TYPE = "irrigation"

    STATE = Constant(bool, "state", "Irrigation State", context="irrigation")
    FLOW = Constant(float, "flow", "Irrigation Flow", unit="l/min", context="irrigation")

    def configure(self, configs: Configurations) -> None:
        super().configure(configs)

        def add_channel(constant: Constant, **custom) -> None:
            channel = constant.to_dict()
            channel["name"] = constant.name.replace("Irrigation", self.name, 1)
            channel.update(custom)
            self.data.add(**channel)

        add_channel(Irrigation.STATE, aggregate="max")
        add_channel(Irrigation.FLOW, aggregate="sum")
