# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.field
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""

from __future__ import annotations

from statistics import geometric_mean
from typing import Dict, Optional, Sequence

import pandas as pd
from lories import Component, Constant, Location
from lories.components.weather import Weather
from lories.data import Channels, ChannelState
from lories.typing import Configurations
from sparcs.components.agriculture import Evapotranspiration
from sparcs.components.agriculture.irrigation import Irrigation
from sparcs.components.agriculture.soil import SoilMoisture
from sparcs.components.weather import validate_meteo_inputs


class AgriculturalField(Component):
    INCLUDES = [SoilMoisture.TYPE, Irrigation.TYPE, Evapotranspiration.TYPE]

    WATER_SUPPLY_MEAN = Constant(float, "water_supply_mean", "Water Supply Coverage", "%")

    location: Location
    weather: Weather

    irrigation: Optional[Irrigation] = None
    evapotranspiration: Optional[Evapotranspiration] = None
    evapo_rename: Dict[str, str]

    # noinspection PyTypeChecker
    @property
    def soil(self) -> Sequence[SoilMoisture]:
        return self.components.get_all(SoilMoisture)

    def configure(self, configs: Configurations) -> None:
        super().configure(configs)

        self.components.load_from_type(
            SoilMoisture,
            configs,
            SoilMoisture.TYPE,
            key=SoilMoisture.TYPE,
            name=f"{self.name} Soil",
            includes=SoilMoisture.INCLUDES,
        )
        if configs.has_member(Irrigation.TYPE, includes=True):
            defaults = Component._build_defaults(configs, strict=True)
            irrigation = Irrigation(self, configs.get_member(Irrigation.TYPE, defaults=defaults), self.soil)
            self.components.add(irrigation)
        else:
            irrigation = None
        self.irrigation = irrigation

        if configs.has_member(Evapotranspiration.TYPE, includes=True):
            defaults = Component._build_defaults(configs, strict=True)
            evapotranspiration = Evapotranspiration(
                self, configs.get_member(Evapotranspiration.TYPE, defaults=defaults)
            )
            self.components.add(evapotranspiration)
        else:
            evapotranspiration = None
        self.evapotranspiration = evapotranspiration

        for c in Evapotranspiration.REQUIRED_CHANNELS:
            self.data.add(c, aggregate="mean", logger={"enabled": False})

        self.data.add(AgriculturalField.WATER_SUPPLY_MEAN, aggregate="mean", logger={"enabled": False})

    # noinspection SpellCheckingInspection
    def activate(self) -> None:
        super().activate()

        self.location = self.context.context.location
        self.weather = self.context.context.weather

        water_supplies = [s.data[SoilMoisture.WATER_SUPPLY] for s in self.soil]
        self.data.register(self._water_supply_callback, water_supplies, how="all", unique=True)

        evapo_input_channels = Channels(
            [
                *self.data[Evapotranspiration.REQUIRED_CHANNELS],
                *self.weather.data.values(),
            ]
        )
        self.evapo_rename = {c.id: c.key for c in evapo_input_channels}
        self.data.register(
            self._evapotranspiration_callback,
            evapo_input_channels,
            how="any",
            unique=True,
        )

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

    def _evapotranspiration_callback(self, data: pd.DataFrame) -> None:
        data = data.rename(columns=self.evapo_rename)
        data = validate_meteo_inputs(data, self.location)
        data = data[[*Evapotranspiration.REQUIRED_WEATHER_CHANNELS, *Evapotranspiration.REQUIRED_CHANNELS]]

        results = self.evapotranspiration.evaluate(data)
        print(results[:50].to_string())
        pass

    @staticmethod
    def _water_supply_mean_geometric(data: pd.Series) -> float:
        if any(v == 0 for v in data):
            return 0
        return geometric_mean(data)

    def has_irrigation(self) -> bool:
        return self.irrigation is not None and self.irrigation.is_enabled()
