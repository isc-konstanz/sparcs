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

    MODES = ["peak_shaving", "self_consumption", "self_peak_shaving"]

    capacity: float
    efficiency: float

    charge_power_max: float
    discharge_power_max: float

    soc_max: float
    soc_min: float

    mode: str
    mode_parameters: dict
    mode_function: callable

    def configure(self, configs: Configurations) -> None:
        super().configure(configs)
        self.capacity = configs.get_float("capacity")
        self.efficiency = configs.get_float("efficiency")

        self.charge_power_max = configs.get_float("charge_power_max") * 1000
        self.discharge_power_max = configs.get_float("discharge_power_max") * 1000

        self.soc_max = configs.get_float("soc_max", default=100)
        self.soc_min = configs.get_float("soc_min", default=0)

        mode = configs.get("mode", default="peak_shaving")
        if mode not in ElectricalEnergyStorage.MODES:
            raise ValueError(f"Invalid mode '{mode}' for {self.name}.")

        self.mode = mode

        mode_config = configs.get_section(mode)
        self.mode_parameters = {}
        if mode == "peak_shaving":
            self.mode_parameters["grid_power_max"] = mode_config.get_float("grid_power_max") * 1000
            self.mode_parameters["grid_power_min"] = mode_config.get_float("grid_power_min") * 1000
            self.mode_function = self.peak_shaving
        elif mode == "self_consumption":
            self.mode_parameters["grid_target"] = mode_config.get_float("grid_target") * 1000
            self.mode_function = self.self_consumption
        elif mode == "self_peak_shaving":
            self.mode_parameters["grid_power_max"] = mode_config.get_float("grid_power_max") * 1000
            self.mode_parameters["charge_power_parameter"] = mode_config.get_float("charge_power_parameter") * 1000
            self.mode_parameters["reserve_soc"] = mode_config.get_float("reserve_soc")
            self.mode_parameters["grid_target"] = mode_config.get_float("grid_target") * 1000
            self.mode_parameters["reserve_charge_power"] = mode_config.get_float("reserve_charge_power") * 1000

            #TODO:
            self.mode_parameters["grip_power_min"] = mode_config.get_float("grip_power_min") * 1000

            self.mode_function = self.self_peak_shaving

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

        results = []

        prior = None
        for index, row in data.iterrows():
            charge_power = 0
            if prior is not None:
                grid_power = row[System.POWER_EL]
                hours = (index - prior).total_seconds() / 3600.0
                charge_power = self.mode_function(grid_power, soc, hours)

                if charge_power != 0:
                    soc += self.energy_to_percent(charge_power / 1000.0 * hours)

            prior = index
            results.append([soc, charge_power])
        return pd.DataFrame(results, index=data.index, columns=columns)

    def get_max_charge_discharge_power(self, soc: float, soc_max: float, soc_min: float, hours: float) -> tuple:
        """
        Calculate the maximum charge and discharge power based on SoC.
        """
        charge_power_max = min(self.charge_power_max, self.percent_to_energy(soc_max - soc) * 1000.0 / hours)
        charge_power_min = max(-self.discharge_power_max, self.percent_to_energy(soc_min - soc) * 1000.0 / hours)

        return charge_power_max, charge_power_min

    def peak_shaving(self, grid_power: float, soc: float, hours: float) -> float:
        """
        Apply the peak shaving strategy to the given grid power and state of charge.
        """

        grid_power_max = self.mode_parameters["grid_power_max"]
        grid_power_min = self.mode_parameters["grid_power_min"]

        charge_power = 0
        # Calculate the charge power based on the grid power and the grid power limits
        if grid_power > grid_power_max:
            charge_power = grid_power_max - grid_power
        elif grid_power < grid_power_min:
            charge_power = grid_power_min - grid_power

        # Calculate the maximum remaining charge and discharge power
        charge_power_max, charge_power_min = \
            self.get_max_charge_discharge_power(soc, 100, 0, hours)

        # Limit the charge power to the maximum allowed charge and discharge power
        charge_power = min(charge_power, charge_power_max)
        charge_power = max(charge_power, charge_power_min)

        return charge_power

    def self_consumption(self, grid_power: float, soc: float, hours: float) -> float:
        """
        Apply the self consumption strategy to the given grid power and state of charge.
        """
        grid_target = self.mode_parameters["grid_target"]

        charge_power = 0
        # Calculate the charge power based on the grid power and the grid setpoint
        if grid_power > grid_target:
            charge_power = grid_target - grid_power
        elif grid_power < grid_target:
            charge_power = grid_target - grid_power

        # Calculate the maximum remaining charge and discharge power
        charge_power_max, charge_power_min = \
            self.get_max_charge_discharge_power(soc, 100, 0, hours)

        # Limit the charge power to the maximum allowed charge and discharge power
        charge_power = min(charge_power, charge_power_max)
        charge_power = max(charge_power, charge_power_min)

        return charge_power

    def self_peak_shaving(self, grid_power: float, soc: float, hours: float) -> float:
        """
        Apply the self peak shaving strategy to the given grid power and state of charge.
        """

        """
        die Entscheidung bis wie viel Netzbezug geladen wird sollte imho nicht grid_min sein sondern max(0, min(grid_min, grid_power)), solange soc > reserve_soc ist.
        Mit Parametern kommst du da nciht weit, da du ja immernoch charge_enable_power ~= 100 kW brauchst, wenn soc < reserve_soc
        """
        mode_parameters = self.mode_parameters

        grid_power_max = mode_parameters["grid_power_max"]
        charge_power_parameter = mode_parameters["charge_power_parameter"]
        reserve_soc = mode_parameters["reserve_soc"]
        grid_target = mode_parameters["grid_target"]
        reserve_charge_power = mode_parameters["reserve_charge_power"]
        grip_power_min = mode_parameters["grip_power_min"]

        charge_power = 0
        # First check if the grid power is above  grid_power_max
        if grid_power > grid_power_max:
            charge_power = grid_power_max - grid_power

        # Second check if SoC is below the reserve SoC
        elif soc < reserve_soc:
            # max charge power to reach grid_power_max
            charge_power = grid_power_max - grid_power

            # available power till grid_power_max
            power_max_diff = grid_power_max - grid_power

            # Limit the charge power to the maximum allowed charge
            charge_power = min(charge_power, reserve_charge_power)
            charge_power = min(charge_power, power_max_diff)

        # Else use self consumption
        else:
            if grid_power > grid_target:
                charge_power = grid_target - grid_power
            elif grid_power < grid_target:
                charge_power = grid_target - grid_power

            # TODO: new strategy: AdMin?
            # only charge if grid_power is below grid_power_min
            charge_power = max(0, min(grip_power_min, grid_power))

            # Limit the charge power to the maximum allowed charge and discharge power
            charge_power = min(charge_power, charge_power_parameter)
            charge_power = max(charge_power, -charge_power_parameter)

        # Limit the charge power to the maximum allowed charge and discharge power
        charge_power_max, charge_power_min = \
            self.get_max_charge_discharge_power(soc, 100, 0, hours)
        charge_power = min(charge_power, charge_power_max)
        charge_power = max(charge_power, charge_power_min)

        return charge_power
