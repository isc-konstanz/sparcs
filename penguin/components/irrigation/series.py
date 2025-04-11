# -*- coding: utf-8 -*-
"""
penguin.components.irrigation.series
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""

from __future__ import annotations

from lori import Component, Configurations, Constant
from penguin.components.irrigation import SoilMoisture


# noinspection SpellCheckingInspection
class IrrigationSeries(Component):
    SECTION = "series"
    INCLUDES = [SoilMoisture.SECTION]

    STATE = Constant(bool, "irrigation_state", "Irrigation state")

    soil: SoilMoisture

    def __init__(self, component: Component, **kwargs) -> None:
        super().__init__(context=component, **kwargs)
        self.soil = SoilMoisture(self)

    def configure(self, configs: Configurations) -> None:
        super().configure(configs)
        self.soil.configure(configs.get_section(SoilMoisture.SECTION, defaults={}))

        def add_channel(constant: Constant, **custom) -> None:
            channel = constant.to_dict()
            channel["name"] = constant.name.replace("Irrigation", self.name, 1)
            channel["column"] = constant.key.replace("irrigation", self.key, 1)
            channel["aggregate"] = "max"
            channel.update(custom)
            self.data.add(**channel)

        add_channel(IrrigationSeries.STATE)

    # noinspection SpellCheckingInspection
    def activate(self) -> None:
        super().activate()
        self.soil.activate()

        # Initialize value for testing
        # FIXME: Remove this
        self.data.irrigation_state.value = False

    def deactivate(self) -> None:
        super().deactivate()
        self.soil.deactivate()
