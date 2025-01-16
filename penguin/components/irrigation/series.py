# -*- coding: utf-8 -*-
"""
penguin.components.irrigation.series
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""

from __future__ import annotations

from lori import Component, Configurations
from penguin.components.irrigation import SoilMoisture


# noinspection SpellCheckingInspection
class IrrigationSeries(Component):
    SECTION = "series"
    INCLUDES = [SoilMoisture.SECTION]

    soil: SoilMoisture

    def __init__(self, component: Component, **kwargs) -> None:
        super().__init__(context=component, **kwargs)
        self.soil = SoilMoisture(self)

    def configure(self, configs: Configurations) -> None:
        super().configure(configs)
        self.soil.configure(configs.get_section(SoilMoisture.SECTION, defaults={}))

        self.data.add("irrigation_state", name="Irrigation state", type=bool)

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
