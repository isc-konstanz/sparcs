# -*- coding: utf-8 -*-
"""
penguin.components.irrigation.soil.moisture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""

from __future__ import annotations

from typing import Optional

import pandas as pd
from lori import ChannelState, Component, Configurations, Constant
from penguin.components.irrigation.soil import Genuchten, SoilModel

DEFAULT_WILTING_POINT: float = 4.2
DEFAULT_FIELD_CAPACITY: float = 1.8


# noinspection SpellCheckingInspection
class SoilMoisture(Component):
    SECTION: str = "soil"

    TEMPERATURE = Constant(float, "temp", "Soil Temperature", "°C")
    WATER_CONTENT = Constant(float, "water_content", "Soil Water Content", "%")
    WATER_TENSION = Constant(float, "water_tension", "Soil Water Tension", "hPa")
    WATER_SUPPLY = Constant(float, "water_supply", "Soil Water Supply Coverage", "%")

    wilting_point: float = DEFAULT_WILTING_POINT
    field_capacity: float = DEFAULT_FIELD_CAPACITY
    water_capacity_available: float

    model: Optional[SoilModel] = None

    def __init__(self, context: Context, configs: Configurations, key="soil", **kwargs) -> None:  # noqa
        super().__init__(context, configs, key=key, **kwargs)
        self.model = None

    def configure(self, configs: Configurations) -> None:
        super().configure(configs)
        self.model = Genuchten(**configs["model"])

        def add_channel(constant: Constant, **custom) -> None:
            channel = constant.to_dict()
            channel["name"] = constant.name.replace("Soil", self.name, 1)
            channel["aggregate"] = "mean"
            channel.update(custom)
            self.data.add(**channel)

        # TODO: Implement validation if water tension is measured directly
        add_channel(SoilMoisture.TEMPERATURE)
        add_channel(SoilMoisture.WATER_CONTENT)
        add_channel(SoilMoisture.WATER_TENSION)

        # As % of plant available water capacity (PAWC)
        add_channel(SoilMoisture.WATER_SUPPLY)

        wilting_point = configs.get("wilting_point", default=SoilMoisture.wilting_point)
        field_capacity = configs.get("field_capacity", default=SoilMoisture.field_capacity)

        self.wilting_point = self.model.water_content(self.model.pf_to_pressure(wilting_point))
        self.field_capacity = self.model.water_content(self.model.pf_to_pressure(field_capacity))
        self.water_capacity_available = self.field_capacity - self.wilting_point

    # noinspection SpellCheckingInspection
    def activate(self) -> None:
        super().activate()

        if not self.data.water_tension.has_connector():
            self.data.register(self._water_content_callback, self.data.water_content)
        self.data.register(self._water_tension_callback, self.data.water_tension)

    def _water_content_callback(self, data: pd.DataFrame) -> None:
        if not data.empty:
            timestamp = data.index[0]
            water_content = data.dropna(axis="columns").mean(axis="columns") / 100
            if len(water_content) == 1:
                water_content = water_content.iloc[0]
            water_tension = self.model.water_tension(water_content)
            self.data[SoilMoisture.WATER_TENSION].set(timestamp, water_tension)
        else:
            self.data[SoilMoisture.WATER_TENSION].state = ChannelState.NOT_AVAILABLE

    def _water_tension_callback(self, data: pd.DataFrame) -> None:
        if not data.empty:
            timestamp = data.index[0]
            water_tension = data.dropna(axis="columns").mean(axis="columns")
            if len(water_tension) == 1:
                water_tension = water_tension.iloc[0]
            water_content = self.model.water_content(water_tension)
            water_supply = (water_content - self.wilting_point) / self.water_capacity_available
            self.data[SoilMoisture.WATER_SUPPLY].set(timestamp, water_supply * 100)
        else:
            self.data[SoilMoisture.WATER_SUPPLY].state = ChannelState.NOT_AVAILABLE
