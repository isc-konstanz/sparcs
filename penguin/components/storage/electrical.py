# -*- coding: utf-8 -*-
"""
penguin.components.storage.electrical
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""

import pandas as pd
from lori.components import Component, register_component_type
from lori.core import Configurations, Constant


@register_component_type("ees")
class ElectricalEnergyStorage(Component):
    STATE_OF_CHARGE = Constant(float, "ees_soc", "EES State of Charge", "%")

    POWER_CHARGE = Constant(float, "ees_charge_power", "EES Charging Power", "W")

    CYCLES = Constant(int, "ees_cycles", "EES Cycles")

    capacity: float
    efficiency: float

    power_max: float

    grid_power_max: float
    grid_power_min: float

    def configure(self, configs: Configurations) -> None:
        super().configure(configs)
        self.capacity = configs.get_float("capacity")
        self.efficiency = configs.get_float("efficiency")

        self.power_max = configs.get_float("power_max") * 1000

        self.grid_power_max = configs.get_float("grid_power_max", default=0) * 1000
        self.grid_power_min = configs.get_float("grid_power_min", default=self.grid_power_max) * 1000

        def add_channel(constant: Constant, **custom) -> None:
            channel = constant.to_dict()
            channel["name"] = constant.name.replace("EES", self.name, 1)
            channel["column"] = constant.key.replace("ees", self.key, 1)
            channel["aggregate"] = "mean"
            channel["connector"] = None
            channel.update(custom)
            self.data.add(**channel)

        add_channel(ElectricalEnergyStorage.STATE_OF_CHARGE)
        add_channel(ElectricalEnergyStorage.POWER_CHARGE)

    def percent_to_energy(self, percent) -> float:
        return percent * self.capacity / 100

    def energy_to_percent(self, capacity) -> float:
        return capacity / self.capacity * 100

    # noinspection PyUnresolvedReferences
    def infer_soc(self, data: pd.DataFrame, soc: float = 50.0) -> pd.DataFrame:
        from penguin.system import System

        if System.POWER_EL not in data.columns:
            raise ValueError("Unable to infer battery storage state of charge without import/export power")

        columns = [self.STATE_OF_CHARGE, self.POWER_CHARGE]

        soc_max = 100
        soc_min = 0

        results = []
        prior = None
        for index, row in data.iterrows():
            charge_power = 0
            if prior is not None:
                grid_power = row[System.POWER_EL]
                if grid_power > self.grid_power_max:
                    charge_power = self.grid_power_max - grid_power
                elif grid_power < self.grid_power_min:
                    charge_power = self.grid_power_min - grid_power

                if charge_power != 0:
                    hours = (index - prior).total_seconds() / 3600.0

                    discharge_power_max = max(-self.power_max, self.percent_to_energy(soc_min - soc) * 1000.0 / hours)
                    charge_power_max = min(self.power_max, self.percent_to_energy(soc_max - soc) * 1000.0 / hours)
                    if charge_power > charge_power_max:
                        charge_power = charge_power_max
                    elif charge_power < discharge_power_max:
                        charge_power = discharge_power_max

                    soc += self.energy_to_percent(charge_power / 1000.0 * hours)

            prior = index
            results.append([soc, charge_power])
        return pd.DataFrame(results, index=data.index, columns=columns)
