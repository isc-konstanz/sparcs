# -*- coding: utf-8 -*-
"""
    penguin.components.storage
    ~~~~~~~~~~~~~~~~~~~~~~~~~~


"""
import pandas as pd
from loris import Component, Configurations


class ElectricalEnergyStorage(Component):
    TYPE = "ees"

    STATE_OF_CHARGE = "ees_soc"

    POWER_CHARGE = "ees_charge_power"
    POWER_DISCHARGE = "ees_discharge_power"

    ENERGY_CHARGE = "ees_charge_energy"
    ENERGY_DISCHARGE = "ees_discharge_energy"

    def configure(self, configs: Configurations) -> None:
        super().configure(configs)
        self.capacity = configs.get_float("capacity")
        self.efficiency = configs.get_float("efficiency")

        self.power_max = configs.get_float("power_max") * 1000

        self.grid_power_max = configs.get_float("grid_power_max", default=0) * 1000
        self.grid_power_min = configs.get_float("grid_power_min", default=self.grid_power_max) * 1000

    @property
    def type(self) -> str:
        return self.TYPE

    def percent_to_energy(self, percent) -> float:
        return percent * self.capacity / 100

    def energy_to_percent(self, capacity) -> float:
        return capacity / self.capacity * 100

    def infer_soc(self, data: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
        from copy import deepcopy

        from .. import System

        if System.POWER_EL not in data.columns:
            raise ValueError("Unable to infer battery storage state of charge without import/export power")

        columns = [self.STATE_OF_CHARGE, self.POWER_CHARGE, self.POWER_DISCHARGE]

        if not inplace:
            data = deepcopy(data)
        data.loc[data.index[0], columns] = [0, 0, 0]

        for i in range(1, len(data.index)):
            index = data.index[i]
            hours = (index - data.index[i - 1]).total_seconds() / 3600.0

            power = data.loc[index, System.POWER_EL]
            if power > self.grid_power_max:
                power = self.grid_power_max - power
            elif power < self.grid_power_min:
                power = self.grid_power_min - power
            else:
                power = 0
            if power > self.power_max:
                power = self.power_max
            elif power < -self.power_max:
                power = -self.power_max

            soc = data.loc[data.index[i - 1], self.STATE_OF_CHARGE]
            charge_max = self.percent_to_energy(100 - soc)
            discharge_max = self.percent_to_energy(0 - soc)

            energy = power / 1000.0 * hours
            energy = min(charge_max, max(discharge_max, energy))
            power = energy * 1000.0 / hours

            soc += self.energy_to_percent(energy)

            data.loc[index, columns] = [soc, max(0.0, power), max(0.0, -power)]
            data.loc[index, System.POWER_EL] += power

        return data


class ThermalEnergyStorage(Component):
    TYPE: str = "tes"

    TEMPERATURE: str = "tes_temp"
    TEMPERATURE_HEATING: str = "tes_ht_temp"
    TEMPERATURE_DOMESTIC: str = "tes_dom_temp"

    def configure(self, configs: Configurations) -> None:
        super().configure(configs)
        self.volume = configs.get_float("volume")

        # For the thermal storage capacity in kWh/K, it will be assumed to be filled with water,
        # resulting in a specific heat capacity of 4.184 J/g*K.
        # TODO: Make tank content and specific heat capacity configurable
        self.capacity = 4.184 * self.volume / 3600

    @property
    def type(self) -> str:
        return self.TYPE
