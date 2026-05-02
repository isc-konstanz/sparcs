# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.field
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""

from __future__ import annotations

from statistics import geometric_mean
from typing import Any, Optional, Sequence

import pandas as pd
from lories import Component, Constant
from lories.data import ChannelState
from lories.typing import Configurations, Timestamp
from sparcs.components.agriculture.irrigation import Irrigation
from sparcs.components.agriculture.simulation import FieldSimulation
from sparcs.components.agriculture.soil import SoilMoisture


class AgriculturalField(Component):
    INCLUDES = [
        SoilMoisture.TYPE,
        Irrigation.TYPE,
        FieldSimulation.TYPE,
    ]

    WATER_SUPPLY_MEAN = Constant(float, "water_supply_mean", "Water Supply Coverage", "%")

    irrigation: Optional[Irrigation] = None
    simulation: Optional[FieldSimulation] = None

    # noinspection PyTypeChecker
    @property
    def soil(self) -> Sequence[SoilMoisture]:
        return self.components.get_all(SoilMoisture)

    def configure(self, configs: Configurations) -> None:
        super().configure(configs)
        defaults = Component._build_defaults(configs, strict=True)

        if configs.has_member(Irrigation.TYPE, includes=True):
            irrigation = Irrigation(self, configs.get_member(Irrigation.TYPE, defaults=defaults), self.soil)
            self.components.add(irrigation)
        else:
            irrigation = None
        self.irrigation = irrigation

        # Soil-simulation chain (ground_shading + evapotranspiration +
        # soil_simulation) lives under the FieldSimulation child so the field
        # itself stays a thin coordinator.
        if configs.has_member(FieldSimulation.TYPE, includes=True):
            simulation = FieldSimulation(
                self,
                configs.get_member(FieldSimulation.TYPE, defaults=defaults),
            )
            self.components.add(simulation)
        else:
            simulation = None
        self.simulation = simulation

        if configs.has_member(SoilMoisture.TYPE, includes=True):
            self.components.load_from_type(
                SoilMoisture,
                configs,
                SoilMoisture.TYPE,
                key=SoilMoisture.TYPE,
                name=f"{self.name} Soil",
                includes=SoilMoisture.INCLUDES,
            )

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

    def has_simulation(self) -> bool:
        return self.simulation is not None and self.simulation.is_enabled()

    def simulate(
        self,
        weather: pd.DataFrame,
        start: Timestamp,
        end: Timestamp,
        prior: Optional[pd.DataFrame] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        if not self.has_simulation():
            return pd.DataFrame()
        return self.simulation.simulate(weather, prior=prior, **kwargs)
