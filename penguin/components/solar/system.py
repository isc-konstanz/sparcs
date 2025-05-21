# -*- coding: utf-8 -*-
"""
penguin.components.solar.system
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""

from __future__ import annotations

import os
from typing import Dict, Mapping, Optional, Sequence

import pvlib as pv

import pandas as pd
from lori import ConfigurationException, Configurations, Constant, Context
from lori.components import Component, ComponentException, register_component_type
from penguin.components.solar.array import SolarArray
from penguin.components.solar.db import InverterDatabase
from penguin.components.solar.model import SolarModel


@register_component_type("pv", "solar")
# noinspection SpellCheckingInspection
class SolarSystem(Component, pv.pvsystem.PVSystem):
    INCLUDES = ["model", "inverter", "arrays", *SolarArray.INCLUDES]

    POWER = Constant(float, "pv_power", "PV Power", "W")
    POWER_EST = Constant(float, "pv_est_power", "Estimate PV Power", "W")
    POWER_EXP = Constant(float, "pv_exp_power", "Export PV Power", "W")
    POWER_DC = Constant(float, "pv_dc_power", "PV (DC) Power", "W")

    ENERGY = Constant(float, "pv_energy", "PV Energy", "kWh")
    ENERGY_DC = Constant(float, "pv_dc_energy", "PV (DC) Energy", "kWh")
    ENERGY_EXP = Constant(float, "pv_exp_energy", "Export PV Energy", "kWh")

    CURRENT_SC = Constant(float, "pv_current_sc", "Short Circuit Current", "A")
    CURRENT_MP = Constant(float, "pv_current_mp", "Maximum Power Point Current", "A")

    VOLTAGE_OC = Constant(float, "pv_voltage_oc", "Open Circuit Voltage", "V")
    VOLTAGE_MP = Constant(float, "pv_voltage_mp", "Maximum Power Point Voltage", "V")

    YIELD_SPECIFIC = Constant(float, "yield_specific", "Specific Yield", "kWh/kWp")
    YIELD_ENERGY_DC = Constant(float, "yield_energy_dc", "Energy Yield (DC)", "kWh")
    YIELD_ENERGY = Constant(float, "yield_energy", "Energy Yield", "kWh")

    arrays: Sequence[SolarArray]

    inverter: str = None
    inverter_parameters: dict = {}
    inverters_per_system: int = 1

    modules_per_inverter: int = SolarArray.modules_per_string * SolarArray.strings

    power_max: float = 0

    losses_parameters: dict = {}

    def __init__(self, context: Context, configs: Configurations) -> None:
        super().__init__(context=context, configs=configs)

    def __repr__(self) -> str:
        return Component.__repr__(self)

    def __str__(self) -> str:
        return Component.__str__(self)

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: Optional[str]) -> None:
        if name is None:
            return
        self._name = name

    def _load_arrays(self, configs: Configurations) -> None:
        self.arrays = tuple(
            self.components.load_from_type(
                SolarArray,
                configs,
                "arrays",
                key="array",
                name=f"{self.name} Array",
                includes=SolarArray.INCLUDES,
            )
        )

    # noinspection PyProtectedMembers
    def configure(self, configs: Configurations) -> None:
        super().configure(configs)
        self._load_arrays(configs)

        def add_channel(constant: Constant, **custom) -> None:
            channel = constant.to_dict()
            channel["name"] = constant.name.replace("PV", self.name, 1)
            channel["column"] = constant.key.replace("pv", self.key, 1)
            channel["aggregate"] = "mean"
            channel["connector"] = None
            channel.update(custom)
            self.data.add(**channel)

        add_channel(SolarSystem.POWER)
        add_channel(SolarSystem.POWER_EST)

    def _on_configure(self, configs: Configurations) -> None:
        super()._on_configure(configs)
        self.losses_parameters = configs.get("losses", default=SolarSystem.losses_parameters)
        try:
            # The converter needs to be configured, after all solar arrays were configured
            inverter = configs.get_section("inverter", defaults={})
            self.inverter = inverter.get("model", default=SolarSystem.inverter)
            self.inverter_parameters = self._infer_inverter_params()
            self.inverter_parameters = self._fit_inverter_params()
            self.inverters_per_system = inverter.get_int("count", default=SolarSystem.modules_per_inverter)

            self.modules_per_inverter = sum([array.modules_per_string * array.strings for array in self.arrays])

            if all(["pdc0" in a.module_parameters for a in self.arrays]):
                self.power_max = (
                    round(
                        sum(
                            [
                                array.modules_per_string * array.strings * array.module_parameters["pdc0"]
                                for array in self.arrays
                            ]
                        )
                    )
                    * self.inverters_per_system
                )
        except ConfigurationException as e:
            self._logger.warning(f"Unable to configure inverter for system '{self.key}': ", e)

    def _infer_inverter_params(self) -> dict:
        params = {}
        self._inverter_parameters_override = False
        if not self._read_inverter_params(params):
            self._read_inverter_database(params)

        inverter_params_exist = len(params) > 0
        if self._read_inverter_configs(params) and inverter_params_exist:
            self._inverter_parameters_override = True

        return params

    def _fit_inverter_params(self) -> dict:
        params = self.inverter_parameters

        if "pdc0" not in params and all(["pdc0" in a.module_parameters for a in self.arrays]):
            params["pdc0"] = round(
                sum(
                    [
                        array.modules_per_string * array.strings * array.module_parameters["pdc0"]
                        for array in self.arrays
                    ]
                )
            )

        if "eta_inv_nom" not in params and "Efficiency" in params:
            if params["Efficiency"] > 1:
                params["Efficiency"] /= 100.0
                self._logger.debug(
                    "Inverter efficiency configured in percent and will be adjusted: ", params["Efficiency"] * 100.0
                )
            params["eta_inv_nom"] = params["Efficiency"]

        return params

    def _read_inverter_params(self, params: dict) -> bool:
        if self.configs.has_section("Inverter"):
            module_params = dict({k: v for k, v in self.configs["Inverter"].items() if k not in ["count", "model"]})
            if len(module_params) > 0:
                _update_parameters(params, module_params)
                self._logger.debug("Extract inverter from config file")
                return True
        return False

    def _read_inverter_database(self, params: dict) -> bool:
        if self.inverter is not None:
            try:
                inverters = InverterDatabase(self.configs)
                inverter_params = inverters.read(self.inverter)
                _update_parameters(params, inverter_params)
            except IOError as e:
                self._logger.warning(f"Error reading inverter '{self.inverter}' from database: ", str(e))
                return False
            self._logger.debug(f"Read inverter '{self.inverter}' from database")
            return True
        return False

    def _read_inverter_configs(self, params: dict) -> bool:
        inverter_file = os.path.join(self.configs.dirs.conf, f"{self.key}.d", "inverter.conf")
        if os.path.exists(inverter_file):
            with open(inverter_file) as f:
                inverter_str = "[Inverter]\n" + f.read()

            from configparser import ConfigParser

            inverter_configs = ConfigParser()
            inverter_configs.optionxform = str
            inverter_configs.read_string(inverter_str)
            inverter_params = dict(inverter_configs["Inverter"])
            _update_parameters(params, inverter_params)
            self._logger.debug("Read inverter file: %s", inverter_file)
            return True
        return False

    # noinspection SpellCheckingInspection, PyMethodOverriding
    def pvwatts_losses(self, solar_position: pd.DataFrame):
        # noinspection SpellCheckingInspection
        def _pvwatts_losses(array: SolarArray):
            return pv.pvsystem.pvwatts_losses(**array.pvwatts_losses(solar_position))

        if self.num_arrays > 1:
            return tuple(_pvwatts_losses(array) for array in self.arrays)
        else:
            return _pvwatts_losses(self.arrays[0])

    def predict(self, weather: pd.DataFrame) -> pd.DataFrame:
        if len(self.arrays) < 1:
            raise ComponentException(self, "PV system must have at least one Array.")
        if not all(a.is_parametrized() for a in self.arrays):
            raise ComponentException(
                self,
                "PV array configurations of this system are not valid: ",
                ", ".join(a.name for a in self.arrays if not a.is_configured()),
            )

        model = SolarModel.load(self)
        return model(weather).rename(
            columns={
                SolarArray.POWER_AC: SolarSystem.POWER,
                SolarArray.POWER_DC: SolarSystem.POWER_DC,
                SolarArray.CURRENT_SC: SolarSystem.CURRENT_SC,
                SolarArray.VOLTAGE_OC: SolarSystem.VOLTAGE_OC,
                SolarArray.CURRENT_MP: SolarSystem.CURRENT_MP,
                SolarArray.VOLTAGE_MP: SolarSystem.VOLTAGE_MP,
            }
        )


def _update_parameters(parameters: Dict, update: Mapping):
    for key, value in update.items():
        try:
            parameters[key] = float(value)
        except ValueError:
            parameters[key] = value

    return parameters
