# -*- coding: utf-8 -*-
"""
    penguin.components.pv.system
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""
from __future__ import annotations

import glob
import os
from typing import Dict, List, Mapping, Tuple

import pvlib as pv

import pandas as pd
from loris import Configurations, ConfigurationException
from loris.components import ComponentContext
from loris.util import parse_id
from penguin.components.dc import DirectCurrent
from penguin.components.pv.array import PVArray
from penguin.components.pv.db import InverterDatabase

# noinspection PyProtectedMember


# noinspection SpellCheckingInspection
class PVSystem(pv.pvsystem.PVSystem, DirectCurrent):
    TYPE: str = "pv"
    ALIAS: List[str] = ["solar"]  # , 'array']

    POWER: str = "power"
    POWER_EXP: str = "exp_power"

    ENERGY: str = "energy"
    ENERGY_EXP: str = "exp_energy"

    CURRENT_SC: str = "i_sc"
    CURRENT_MP: str = "i_mp"

    VOLTAGE_OC: str = "v_oc"
    VOLTAGE_MP: str = "v_mp"

    YIELD_SPECIFIC: str = "specific_yield"

    def __init__(self, context: ComponentContext, configs: Configurations) -> None:
        super(pv.pvsystem.PVSystem, self).__init__(context, configs)

    def __configure__(self, configs: Configurations) -> None:
        super().__configure__(configs)
        self.arrays = self._load_arrays(self._context, configs)
        for array in self.arrays:
            array.configure()
        try:
            inverter = configs.get_section("inverter", default={})
            self.inverter = inverter.get("model", default=None)
            self.inverter_parameters = self._infer_inverter_params()
            self.inverter_parameters = self._fit_inverter_params()
            self.inverters_per_system = inverter.get_int("count", default=1)

            self.modules_per_inverter = sum([array.modules_per_string * array.strings for array in self.arrays])

            if all(["pdc0" in a.module_parameters for a in self.arrays]):
                self.power_max = round(sum(
                    [
                        array.modules_per_string * array.strings * array.module_parameters["pdc0"] for array in self.arrays
                    ]
                )) * self.inverters_per_system

            self.losses_parameters = configs.get("losses", default={})

        except ConfigurationException as e:
            self._logger.warning(f"Unable to configure inverter for system '{self.id}': ", e)

        def _add_channel(channel_id: str):
            from penguin import COLUMNS

            channel = {}
            if channel_id in COLUMNS:
                channel["name"] = COLUMNS[channel_id]
            channel["column"] = f"{self.id}_{channel_id}"
            channel["value_type"] = float

            self.data.add(id=channel_id, **channel)

        _add_channel(self.POWER)
        _add_channel(self.POWER_DC)
        _add_channel(self.CURRENT_MP)
        _add_channel(self.VOLTAGE_MP)
        _add_channel(self.CURRENT_SC)
        _add_channel(self.VOLTAGE_OC)

    def _load_arrays(self, context: ComponentContext, configs: Configurations) -> Tuple[PVArray]:
        arrays = []
        array_dir = os.path.join(configs.dirs.conf, f"{parse_id(configs['id'])}.d")
        array_dirs = configs.dirs.encode()
        array_dirs["conf_dir"] = array_dir
        if "mounting" in configs and len(configs.get_section("mounting")) > 0:
            # TODO: verify parameter availability in 'General' by keys

            array_file = "array.conf"
            array_configs = Configurations.load(
                array_file,
                **array_dirs,
                **configs,
                require=False
            )
            array_configs.set("id", "array")
            arrays.append(self._new_array(context, array_configs))

        array_defaults = {}
        if "arrays" in configs:
            arrays_section = configs.get_section("arrays")
            array_ids = [
                i
                for i in arrays_section.keys()
                if (isinstance(arrays_section[i], Mapping) and i not in ["data", "mounting"])
            ]
            arrays_configs = {i: arrays_section.pop(i) for i in array_ids}
            array_defaults.update(arrays_section)

            for array_id, array_section in arrays_configs.items():
                array_file = f"{array_id}.conf"
                array_configs = Configurations.load(
                    array_file,
                    **array_dirs,
                    **array_defaults,
                    require=False
                )
                array_configs.update(arrays_section)
                array_configs.set("id", array_id)
                arrays.append(self._new_array(context, array_configs))

        for array_path in glob.glob(os.path.join(array_dir, "array*.conf")):
            array_file = os.path.basename(array_path)
            array_id = array_file.rsplit(".", maxsplit=1)[0]
            if any([array_id == a.name for a in arrays]):
                continue

            array_configs = Configurations.load(
                array_file,
                **array_dirs,
                **array_defaults
            )
            array_configs.set("id", array_id)
            arrays.append(self._new_array(context, array_configs))

        return tuple(arrays)

    # noinspection PyMethodMayBeStatic
    def _new_array(self, context: ComponentContext, configs: Configurations) -> PVArray:
        return PVArray(context, configs)

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
            params["pdc0"] = round(sum(
                [
                    array.modules_per_string * array.strings * array.module_parameters["pdc0"] for array in self.arrays
                ]
            ))

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
        inverter_file = os.path.join(self.configs.dirs.conf, f"{self.id}.d", "inverter.conf")
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

    def pvwatts_losses(self, solar_position: pd.DataFrame):
        def _pvwatts_losses(array: PVArray):
            return pv.pvsystem.pvwatts_losses(**array.pvwatts_losses(solar_position))

        if self.num_arrays > 1:
            return tuple(_pvwatts_losses(array) for array in self.arrays)
        else:
            return _pvwatts_losses(self.arrays[0])

    def _run(self, weather: pd.DataFrame) -> pd.DataFrame:
        from penguin.model import Model

        model = Model.load(self)
        return model(weather).rename(
            columns={
                "p_ac": PVSystem.POWER,
                "p_dc": PVSystem.POWER_DC,
                "i_sc": PVSystem.CURRENT_SC,
                "v_oc": PVSystem.VOLTAGE_OC,
                "i_mp": PVSystem.CURRENT_MP,
                "v_mp": PVSystem.VOLTAGE_MP,
            }
        )

    def run(self, weather: pd.DataFrame) -> pd.DataFrame:
        data = self._run(weather)

        for channel in self.data.values():
            if channel.id not in data.columns:
                # channel.state = ChannelState.NOT_AVAILABLE
                continue

            channel_data = data[channel.id]
            if not channel_data.empty:
                channel.set(channel_data.index[0], channel_data)

        return data

    def get_type(self) -> str:
        return self.TYPE


def _update_parameters(parameters: Dict, update: Mapping):
    for key, value in update.items():
        try:
            parameters[key] = float(value)
        except ValueError:
            parameters[key] = value

    return parameters
