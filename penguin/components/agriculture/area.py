# -*- coding: utf-8 -*-
"""
penguin.components.agriculture.area
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""

from __future__ import annotations

from typing import Optional, Sequence

import pandas as pd
from lori import Configurations
from lori.components import Component, register_component_type
from lori.data import ChannelState
from penguin.components.agriculture.field import AgriculturalField
from penguin.components.storage.water import WaterStorage


# noinspection SpellCheckingInspection
@register_component_type("agri", "agriculture")
class AgriculturalArea(Component):
    INCLUDES = [WaterStorage.SECTION, *AgriculturalField.INCLUDES]

    water_storage: Optional[WaterStorage] = None

    # noinspection PyTypeChecker
    @property
    def fields(self) -> Sequence[AgriculturalField]:
        return self.components.get_all(AgriculturalField)

    def configure(self, configs: Configurations) -> None:
        super().configure(configs)
        defaults = Component._build_defaults(configs, strict=True)
        self.components.load_from_type(
            AgriculturalField,
            configs,
            "fields",
            key="field",
            name=f"{self.name} Field",
            includes=AgriculturalField.INCLUDES,
            defaults=defaults,
        )
        # TODO: Decide if water storage should
        if configs.has_section(WaterStorage.SECTION, includes=True):
            water_storage = WaterStorage(self, configs.get_section(WaterStorage.SECTION, defaults=defaults))
            self.components.add(water_storage)
        else:
            water_storage = None
        self.water_storage = water_storage

        self.data.add(AgriculturalField.WATER_SUPPLY_MEAN, aggregate="mean", logger={"enabled": False})

    # noinspection SpellCheckingInspection
    def activate(self) -> None:
        super().activate()
        water_supplies = [s.data[AgriculturalField.WATER_SUPPLY_MEAN] for s in self.fields]
        self.data.register(self._water_supply_callback, water_supplies, how="all", unique=True)

    # noinspection PyProtectedMember
    def _water_supply_callback(self, data: pd.DataFrame) -> None:
        water_supply = data[[c for c in data.columns if AgriculturalField.WATER_SUPPLY_MEAN in c]]
        if not water_supply.empty:
            water_supply.ffill().dropna(axis="index", how="any", inplace=True)
            water_supply_mean = water_supply.apply(AgriculturalField._water_supply_mean_geometric, axis="columns")
            if len(water_supply_mean) == 1:
                water_supply_mean = water_supply_mean.iloc[0]
            self.data[AgriculturalField.WATER_SUPPLY_MEAN].set(data.index[0], water_supply_mean)
        else:
            self.data[AgriculturalField.WATER_SUPPLY_MEAN].state = ChannelState.NOT_AVAILABLE

    def has_water_storage(self) -> bool:
        return self.water_storage is not None and self.water_storage.is_enabled()
