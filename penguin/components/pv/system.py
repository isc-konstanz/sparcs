# -*- coding: utf-8 -*-
"""
    penguin.components.pv.system
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""
from __future__ import annotations

import glob
import os
from typing import Collection, Dict, List, Mapping

import pvlib as pv

import pandas as pd
from loris import ChannelState, ConfigurationException, Configurations, Configurator
from loris.components import Activator, ComponentException
from loris.util import parse_id
from penguin.components.dc import DirectCurrent
from penguin.components.pv.array import PVArray
from penguin.components.pv.db import InverterDatabase


# noinspection SpellCheckingInspection
class PVSystem(pv.pvsystem.PVSystem, DirectCurrent):
    TYPE: str = "pv"
    ALIAS: List[str] = ["solar"]  # , 'array']

    POWER: str = f"{TYPE}_power"
    POWER_CALC: str = f"{TYPE}_calc_power"
    POWER_EXP: str = f"{TYPE}_exp_power"

    ENERGY: str = f"{TYPE}_energy"
    ENERGY_CALC: str = f"{TYPE}_calc_energy"
    ENERGY_EXP: str = f"{TYPE}_exp_energy"

    CURRENT_SC: str = f"{TYPE}_current_sc"
    CURRENT_MP: str = f"{TYPE}_current_mp"

    VOLTAGE_OC: str = f"{TYPE}_voltage_oc"
    VOLTAGE_MP: str = f"{TYPE}_voltage_mp"

    YIELD_SPECIFIC: str = "specific_yield"

    arrays: List[PVArray]

    inverter: str = None
    inverter_parameters: dict = {}
    inverters_per_system: int = 1

    modules_per_inverter: int = PVArray.modules_per_string * PVArray.strings

    power_max: float = 0

    losses_parameters: dict = {}

    def __init__(self, context, configs: Configurations) -> None:
        super(pv.pvsystem.PVSystem, self).__init__(context, configs)
        self.arrays = []

    def __repr__(self) -> str:
        return Configurator.__repr__(self)

    @property
    def type(self) -> str:
        return self.TYPE

    # noinspection PyProtectedMembers
    def configure(self, configs: Configurations) -> None:
        super().configure(configs)
        self._do_load_arrays(configs)
        for array in self.arrays:
            array._do_configure()
        try:
            inverter = configs.get_section("inverter", defaults={})
            self.inverter = inverter.get("model", default=PVSystem.inverter)
            self.inverter_parameters = self._infer_inverter_params()
            self.inverter_parameters = self._fit_inverter_params()
            self.inverters_per_system = inverter.get_int("count", default=PVSystem.modules_per_inverter)

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

            self.losses_parameters = configs.get("losses", default=PVSystem.losses_parameters)

        except ConfigurationException as e:
            self._logger.warning(f"Unable to configure inverter for system '{self.id}': ", e)

        def _add_channel(channel_id: str):
            from penguin import COLUMNS
            channel = {}
            if channel_id in COLUMNS:
                channel["name"] = COLUMNS[channel_id]
            channel["column"] = channel_id.replace(f"{PVSystem.TYPE}_", self.id)
            channel["value_type"] = float
            channel["connector"] = None

            self.data.add(id=channel_id, **channel)

        _add_channel(PVSystem.POWER)
        _add_channel(PVSystem.POWER_DC)
        _add_channel(PVSystem.CURRENT_MP)
        _add_channel(PVSystem.VOLTAGE_MP)
        _add_channel(PVSystem.CURRENT_SC)
        _add_channel(PVSystem.VOLTAGE_OC)

    def _do_configure_members(self, configurators: Collection[Configurator]) -> None:
        configurators = [c for c in configurators if c not in self.arrays]
        super()._do_configure_members(configurators)

    def _do_load_arrays(self, configs: Configurations):
        array_dir = configs.path.replace(".conf", ".d")
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
            if "name" not in array_configs:
                array_configs.set("name", self.id)

            array = self._new_array(array_configs)
            self._add_array(array)

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
                array_id = parse_id(array_id)
                array_file = f"{array_id}.conf"
                array_configs = Configurations.load(
                    array_file,
                    **array_dirs,
                    **array_defaults,
                    require=False
                )
                array_configs.update(arrays_section)
                array_configs.set("id", array_id)
                if "name" not in array_configs:
                    array_configs.set("name", f"{self.id}_{array_id}")

                array = self._new_array(array_configs)
                self._add_array(array)

        for array_path in glob.glob(os.path.join(array_dir, "array*.conf")):
            array_file = os.path.basename(array_path)
            array_id = parse_id(array_file.rsplit(".", maxsplit=1)[0])
            if any([array_id == a.id for a in self.arrays]):
                continue

            array_configs = Configurations.load(
                array_file,
                **array_dirs,
                **array_defaults
            )
            array_configs.set("id", array_id)
            if "name" not in array_configs:
                array_configs.set("name", f"{self.id}_{array_id}")

            array = self._new_array(array_configs)
            self._add_array(array)

    # noinspection PyMethodMayBeStatic
    def _new_array(self, configs: Configurations) -> PVArray:
        return PVArray(self, configs)

    def _add_array(self, array: PVArray) -> None:
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

    def activate(self) -> None:
        pass

    def _do_activate_members(self, activators: Collection[Activator]) -> None:
        activators = list(activators)
        activators.extend([a for a in self.arrays if a not in activators and a.is_configured()])
        super()._do_activate_members(activators)

    def deactivate(self) -> None:
        pass

    def _do_deactivate_members(self, activators: Collection[Activator]) -> None:
        activators = list(activators)
        activators.extend([a for a in self.arrays if a not in activators and a.is_configured()])
        super()._do_deactivate_members(activators)

    # noinspection PyMethodOverriding
    def pvwatts_losses(self, solar_position: pd.DataFrame):
        def _pvwatts_losses(array: PVArray):
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
                PVArray.POWER_AC: PVSystem.POWER,
                PVArray.POWER_DC: PVSystem.POWER_DC,
                PVArray.CURRENT_SC: PVSystem.CURRENT_SC,
                PVArray.VOLTAGE_OC: PVSystem.VOLTAGE_OC,
                PVArray.CURRENT_MP: PVSystem.CURRENT_MP,
                PVArray.VOLTAGE_MP: PVSystem.VOLTAGE_MP,
            }
        )

    def run(self, weather: pd.DataFrame) -> pd.DataFrame:
        data = self._run(weather)

        for channel in self.data.values():
            if channel.id not in data.columns or data[channel.id].empty:
                channel.state = ChannelState.NOT_AVAILABLE
                continue
            channel_data = data[channel.id]
            channel.set(channel_data.index[0], channel_data)

        return data


def _update_parameters(parameters: Dict, update: Mapping):
    for key, value in update.items():
        try:
            parameters[key] = float(value)
        except ValueError:
            parameters[key] = value

    return parameters
