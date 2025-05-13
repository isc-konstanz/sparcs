# -*- coding: utf-8 -*-
"""
penguin.components.storage.electrical
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""

from collections.abc import Callable
from typing import Any, Dict

import pandas as pd
from lori.components import Component, register_component_type
from lori.core import Configurations, Constant


@register_component_type("ees")
class ElectricalEnergyStorage(Component):
    STATE_OF_CHARGE = Constant(float, "ees_soc", "EES State of Charge", "%")
    POWER_CHARGE = Constant(float, "ees_charge_power", "EES Charging Power", "W")

    MODES = ["self_consumption", "self_peak_shaving", "peak_shaving"]

    capacity: float
    efficiency: float

    power_max: float

    soc_max: float
    soc_min: float

    mode: str
    _mode_parameters: Dict[str, Any]

    _predict_charge_power: Callable[[float, float, float], float]

    def configure(self, configs: Configurations) -> None:
        super().configure(configs)
        self.capacity = configs.get_float("capacity")
        self.efficiency = configs.get_float("efficiency")

        power_max = configs.get_float("power_max")
        self.power_max = power_max * 1000

        self.soc_max = configs.get_float("soc_max", default=100)
        self.soc_min = configs.get_float("soc_min", default=0)

        mode = configs.get("mode", default="self_consumption")
        if mode not in ElectricalEnergyStorage.MODES:
            raise ValueError(f"Invalid mode '{mode}' for {self.name}.")

        mode_section = configs.get_section(mode, defaults={})

        self.mode = mode
        self._mode_parameters = {
            "charge_power_max": mode_section.get_float("charge_power_max", default=power_max) * 1000,
        }
        if mode == "self_consumption":
            self._mode_parameters["grid_power_target"] = mode_section.get_float("grid_power_target", default=0.0) * 1000
            self._predict_charge_power = self._predict_self_consumption

        elif mode == "self_peak_shaving":
            self._mode_parameters["grid_power_target"] = mode_section.get_float("grid_power_target", default=0.0) * 1000
            self._mode_parameters["grid_power_max"] = mode_section.get_float("grid_power_max") * 1000
            self._mode_parameters["grid_power_min"] = mode_section.get_float("grid_power_min") * 1000
            self._mode_parameters["soc_reserve"] = mode_section.get_float("soc_reserve")
            self._predict_charge_power = self._predict_self_peak_shaving

        elif mode == "peak_shaving":
            self._mode_parameters["grid_power_max"] = mode_section.get_float("grid_power_max") * 1000
            self._mode_parameters["grid_power_min"] = mode_section.get_float("grid_power_min") * 1000
            self._predict_charge_power = self._predict_peak_shaving

        def add_channel(constant: Constant, aggregate: str = "mean", **custom) -> None:
            channel = constant.to_dict()
            channel["name"] = constant.name.replace("EES", self.name, 1)
            channel["column"] = constant.key.replace("ees", self.key, 1)
            channel["aggregate"] = aggregate
            channel["connector"] = None
            channel.update(custom)
            self.data.add(**channel)

        add_channel(ElectricalEnergyStorage.POWER_CHARGE)
        add_channel(ElectricalEnergyStorage.STATE_OF_CHARGE, aggregate="last")

    def percent_to_energy(self, percent) -> float:
        return percent * self.capacity / 100

    def energy_to_percent(self, capacity) -> float:
        return capacity / self.capacity * 100

    # noinspection PyUnresolvedReferences
    def predict(self, data: pd.DataFrame, soc: float = 50.0) -> pd.DataFrame:
        from penguin.system import System

        if System.POWER_EL not in data.columns:
            raise ValueError("Unable to predict battery storage state of charge without import/export power")

        columns = [self.STATE_OF_CHARGE, self.POWER_CHARGE]

        results = []

        prior = None
        for index, row in data.iterrows():
            charge_power = 0
            if prior is not None:
                grid_power = row[System.POWER_EL]
                hours = (index - prior).total_seconds() / 3600.0

                charge_power = self._predict_charge_power(hours, soc, grid_power)
                if charge_power != 0:
                    soc += self.energy_to_percent(charge_power / 1000.0 * hours)

            prior = index
            results.append([soc, charge_power])
        return pd.DataFrame(results, index=data.index, columns=columns)

    def _predict_self_consumption(self, hours: float, soc: float, grid_power: float) -> float:
        grid_power_target = self._mode_parameters["grid_power_target"]

        # Calculate the charge power based on the grid power and the grid power min
        if grid_power > grid_power_target:
            charge_power = grid_power_target - grid_power
        elif grid_power < 0:
            charge_power = abs(grid_power)
        else:
            charge_power = 0.0
        return self._limit_charge_power(hours, soc, charge_power)

    def _predict_self_peak_shaving(self, hours: float, soc: float, grid_power: float) -> float:
        soc_reserve = self._mode_parameters["soc_reserve"]

        # First, check if the grid power is above grid_power_max
        if soc < soc_reserve or grid_power > self._mode_parameters["grid_power_max"]:
            charge_power = self._predict_peak_shaving(hours, soc, grid_power)
            if charge_power > 0:
                charge_power = min(charge_power, self.percent_to_energy(soc_reserve - soc) * 1000.0 / hours)
            return charge_power
        return self._predict_self_consumption(hours, soc, grid_power)

    def _predict_peak_shaving(self, hours: float, soc: float, grid_power: float) -> float:
        grid_power_max = self._mode_parameters["grid_power_max"]
        grid_power_min = self._mode_parameters["grid_power_min"]

        # Calculate the charge power based on the grid power and the grid power limits
        if grid_power > grid_power_max:
            return self._limit_charge_power(hours, soc, grid_power_max - grid_power, hard_max=True)
        elif grid_power < grid_power_min:
            return self._limit_charge_power(hours, soc, grid_power_min - grid_power)
        else:
            return 0.0

    def _limit_charge_power(self, hours: float, soc: float, charge_power: float, hard_max: bool = False) -> float:
        charge_power_max = self.power_max
        if not hard_max:
            charge_power_max = min(charge_power_max, self._mode_parameters["charge_power_max"])
        if charge_power > 0:
            charge_power_max = min(charge_power_max, self.percent_to_energy(self.soc_max - soc) * 1000.0 / hours)
            charge_power = min(charge_power, charge_power_max)
        elif charge_power < 0:
            charge_power_max = max(-charge_power_max, self.percent_to_energy(self.soc_min - soc) * 1000.0 / hours)
            charge_power = max(charge_power, charge_power_max)
        return charge_power
