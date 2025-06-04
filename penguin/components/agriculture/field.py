# -*- coding: utf-8 -*-
"""
penguin.components.agriculture.field
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""

from __future__ import annotations

from statistics import geometric_mean
from typing import Optional, Sequence

import pandas as pd
from lori import Component, Configurations, Constant
from lori.data import ChannelState
from penguin.components.agriculture.irrigation import Irrigation
from penguin.components.agriculture.soil import SoilMoisture


class AgriculturalField(Component):
    INCLUDES = [SoilMoisture.SECTION, Irrigation.SECTION]

    WATER_SUPPLY_MEAN = Constant(float, "water_supply_mean", "Water Supply Coverage", "%")

    irrigation: Optional[Irrigation] = None

    # noinspection PyTypeChecker
    @property
    def soil(self) -> Sequence[SoilMoisture]:
        return self.components.get_all(SoilMoisture)

    def configure(self, configs: Configurations) -> None:
        super().configure(configs)
        self.components.load_from_type(
            SoilMoisture,
            configs,
            SoilMoisture.SECTION,
            key=SoilMoisture.SECTION,
            name=f"{self.name} Soil",
            includes=SoilMoisture.INCLUDES,
        )
        if configs.has_section(Irrigation.SECTION, includes=True):
            defaults = Component._build_defaults(configs, strict=True)
            irrigation = Irrigation(self, configs.get_section(Irrigation.SECTION, defaults=defaults), self.soil)
            self.components.add(irrigation)
        else:
            irrigation = None
        self.irrigation = irrigation

        self.data.add(AgriculturalField.WATER_SUPPLY_MEAN, aggregate="mean", logger={"enabled": False})

    # noinspection SpellCheckingInspection
    def activate(self) -> None:
        super().activate()
        water_supplies = [s.data[SoilMoisture.WATER_SUPPLY] for s in self.soil]
        self.data.register(self._water_supply_callback, water_supplies, how="all", unique=True)

    def _water_supply_callback(self, data: pd.DataFrame) -> None:
        water_supply = data[[c for c in data.columns if SoilMoisture.WATER_SUPPLY in c]]
        if not water_supply.empty:
            water_supply.ffill().dropna(axis="index", how="any", inplace=True)
            water_supply_mean = water_supply.apply(AgriculturalField._water_supply_mean_geometric, axis="columns")
            if len(water_supply_mean) == 1:
                water_supply_mean = water_supply_mean.iloc[0]
            self.data[AgriculturalField.WATER_SUPPLY_MEAN].set(data.index[0], water_supply_mean)
        else:
            self.data[AgriculturalField.WATER_SUPPLY_MEAN].state = ChannelState.NOT_AVAILABLE

    @staticmethod
    def _water_supply_mean_geometric(data: pd.Series) -> float:
        if any(v == 0 for v in data):
            return 0
        return geometric_mean(data)

    def has_irrigation(self) -> bool:
        return self.irrigation is not None and self.irrigation.is_enabled()
