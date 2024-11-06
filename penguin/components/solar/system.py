# -*- coding: utf-8 -*-
"""
penguin.components.solar.system
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""

from __future__ import annotations

import glob
import os
from typing import Dict, List, Mapping

import pvlib as pv

import pandas as pd
from lori import ChannelState, ConfigurationException, Configurations, Context
from lori.components import Component, ComponentException, register_component_type
from lori.util import parse_key
from penguin.components.current import DirectCurrent
from penguin.components.solar.array import SolarArray
from penguin.components.solar.db import InverterDatabase


@register_component_type("solar", "pv")
# noinspection SpellCheckingInspection
class SolarSystem(pv.pvsystem.PVSystem, DirectCurrent):
    TYPE: str = "pv"

    SECTIONS = ["model", "inverter", "arrays", *SolarArray.SECTIONS]

    POWER: str = f"{TYPE}_power"
    POWER_EST: str = f"{TYPE}_est_power"
    POWER_EXP: str = f"{TYPE}_exp_power"

    ENERGY: str = f"{TYPE}_energy"
    ENERGY_EST: str = f"{TYPE}_est_energy"
    ENERGY_EXP: str = f"{TYPE}_exp_energy"

    CURRENT_SC: str = f"{TYPE}_current_sc"
    CURRENT_MP: str = f"{TYPE}_current_mp"

    VOLTAGE_OC: str = f"{TYPE}_voltage_oc"
    VOLTAGE_MP: str = f"{TYPE}_voltage_mp"

    YIELD_SPECIFIC: str = "specific_yield"

    arrays: List[SolarArray]

    inverter: str = None
    inverter_parameters: dict = {}
    inverters_per_system: int = 1

    modules_per_inverter: int = SolarArray.modules_per_string * SolarArray.strings

    power_max: float = 0

    losses_parameters: dict = {}

    def __init__(self, context: Context, configs: Configurations) -> None:  # noqa
        super(pv.pvsystem.PVSystem, self).__init__(context, configs)
        self.arrays = []

    def __repr__(self) -> str:
        return Component.__repr__(self)

    def __str__(self) -> str:
        return Component.__str__(self)

    # noinspection PyProtectedMembers
    def configure(self, configs: Configurations) -> None:
        super().configure(configs)
        self._load_arrays(configs)
        try:
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

            self.losses_parameters = configs.get("losses", default=SolarSystem.losses_parameters)

        except ConfigurationException as e:
            self._logger.warning(f"Unable to configure inverter for system '{self.key}': ", e)

        def _add_channel(key: str):
            from penguin.constants import COLUMNS

            channel = {}
            if key in COLUMNS:
                channel["name"] = COLUMNS[key]
            channel["column"] = key.replace(f"{SolarSystem.TYPE}_", self.key)
            channel["value_type"] = float
            channel["connector"] = None

            self.data.add(key=key, **channel)

        _add_channel(SolarSystem.POWER)
        _add_channel(SolarSystem.POWER_DC)
        _add_channel(SolarSystem.CURRENT_MP)
        _add_channel(SolarSystem.VOLTAGE_MP)
        _add_channel(SolarSystem.CURRENT_SC)
        _add_channel(SolarSystem.VOLTAGE_OC)

    def _load_arrays(self, configs: Configurations):
        array_dir = configs.path.replace(".conf", ".d")
        array_dirs = configs.dirs.to_dict()
        array_dirs["conf_dir"] = array_dir
        if "mounting" in configs and len(configs.get_section("mounting")) > 0:
            # TODO: verify parameter availability in 'General' by keys

            array_file = "array.conf"
            array_configs = Configurations.load(
                array_file,
                **array_dirs,
                **configs,
                require=False,
            )
            array = SolarArray(self, key="array", name=f"{self.name} Array")
            array.configure(array_configs)
            self.arrays.append(array)

        array_defaults = {}
        if "arrays" in configs:
            arrays_section = configs.get_section("arrays")
            array_keys = [k for k in arrays_section.sections if k not in [*Component.SECTIONS, *self.SECTIONS]]
            arrays_configs = {k: arrays_section.pop(k) for k in array_keys}
            array_defaults.update(arrays_section)

            for array_key, array_section in arrays_configs.items():
                array_key = parse_key(array_key)
                array_file = f"{array_key}.conf"
                array_configs = Configurations.load(
                    array_file,
                    **array_dirs,
                    **array_defaults,
                    require=False,
                )
                array_configs.update(arrays_section, replace=False)

                array = SolarArray(self, key=array_key, name=f"{self.name} {array_key.title()}")
                array.configure(array_configs)
                self.arrays.append(array)

        for array_path in glob.glob(os.path.join(array_dir, "array*.conf")):
            array_file = os.path.basename(array_path)
            array_key = parse_key(array_file.rsplit(".", maxsplit=1)[0])
            if any([array_key == a.key for a in self.arrays]):
                continue

            array_configs = Configurations.load(
                array_file,
                **array_dirs,
                **array_defaults,
            )
            array = SolarArray(self, key=array_key, name=f"{self.name} {array_key.title()}")
            array.configure(array_configs)
            self.arrays.append(array)

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

    def activate(self) -> None:
        for array in self.arrays:
            array.activate()

    def deactivate(self) -> None:
        for array in self.arrays:
            array.deactivate()

    # noinspection PyMethodOverriding
    def pvwatts_losses(self, solar_position: pd.DataFrame):
        def _pvwatts_losses(array: SolarArray):
            return pv.pvsystem.pvwatts_losses(**array.pvwatts_losses(solar_position))

        if self.num_arrays > 1:
            return tuple(_pvwatts_losses(array) for array in self.arrays)
        else:
            return _pvwatts_losses(self.arrays[0])

    def _run(self, weather: pd.DataFrame) -> pd.DataFrame:
        from penguin.model import Model

        if len(self.arrays) < 1:
            raise ComponentException("PV system must have at least one Array.")
        if not all(a.is_configured() for a in self.arrays):
            raise ComponentException(
                "PV array configurations of this system are not valid: ",
                ", ".join(a.name for a in self.arrays if not a.is_configured()),
            )

        model = Model.load(self)
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

    def run(self, weather: pd.DataFrame) -> pd.DataFrame:
        data = self._run(weather)

        for channel in self.data.channels:
            if channel.key not in data.columns or data[channel.key].empty:
                channel.state = ChannelState.NOT_AVAILABLE
                continue
            channel_data = data[channel.key]
            channel.set(channel_data.index[0], channel_data)

        return data


def _update_parameters(parameters: Dict, update: Mapping):
    for key, value in update.items():
        try:
            parameters[key] = float(value)
        except ValueError:
            parameters[key] = value

    return parameters
