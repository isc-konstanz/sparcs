# -*- coding: utf-8 -*-
"""
    penguin.components.irrigation.system
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""
from __future__ import annotations

import glob
import os
from typing import List

import numpy as np
import pandas as pd
from loris import ChannelState, Configurations, Context
from loris.components import Component, register_component_type
from loris.util import parse_id
from penguin.components.irrigation import IrrigationArray


# noinspection SpellCheckingInspection
@register_component_type("irrigation", "watering")
class IrrigationSystem(Component):
    TYPE: str = "irrig"

    arrays: List[IrrigationArray]

    def __init__(self, context: Context, configs: Configurations) -> None:
        super().__init__(context, configs)
        self.arrays = []

    def configure(self, configs: Configurations) -> None:
        super().configure(configs)
        self._load_arrays(configs)
        for array in self.arrays:
            array._do_configure()

        self.data.add("humidity_mean", name="Humidity [%]", connector="random", type=float, min=50, max=100)
        self.data.add("storage_level", name="Storage Level [%]", connector="random", type=float, min=0, max=1000)

    def _load_arrays(self, configs: Configurations):
        array_dir = configs.path.replace(".conf", ".d")
        array_dirs = configs.dirs.encode()
        array_dirs["conf_dir"] = array_dir
        array_defaults = {}
        if "arrays" in configs:
            arrays_section = configs.get_section("arrays")
            array_keys = [k for k in arrays_section.sections if k not in ["data"]]
            arrays_configs = {k: arrays_section.pop(k) for k in array_keys}
            array_defaults.update(arrays_section)

            for array_key, array_section in arrays_configs.items():
                array_key = parse_id(array_key)
                array_file = f"{array_key}.conf"
                array_configs = Configurations.load(
                    array_file,
                    **array_dirs,
                    **array_defaults,
                    require=False
                )
                array_configs.update(arrays_section)
                array_configs.set("key", array_key)
                array = self._new_array(array_configs)
                self._add_array(array)

        for array_path in glob.glob(os.path.join(array_dir, "array*.conf")):
            array_file = os.path.basename(array_path)
            array_key = parse_id(array_file.rsplit(".", maxsplit=1)[0])
            if any([array_key == a.key for a in self.arrays]):
                continue

            array_configs = Configurations.load(
                array_file,
                **array_dirs,
                **array_defaults
            )
            array_configs.set("key", array_key)
            array = self._new_array(array_configs)
            self._add_array(array)

    # noinspection PyMethodMayBeStatic
    def _new_array(self, configs: Configurations) -> IrrigationArray:
        return IrrigationArray(self, configs)

    def _add_array(self, array: IrrigationArray) -> None:
        self.arrays.append(array)

    def run(self, weather: pd.DataFrame) -> pd.DataFrame:
        array_humidity = [i.data.humidity for i in self.arrays if i.data.humidity.is_valid()]
        if len(array_humidity) > 0:
            self.data.humidity_mean.set(
                np.array([t.timestamp for t in array_humidity]).max(),
                np.array([t.value for t in array_humidity]).mean(),
            )
        else:
            self.data.humidity_mean.state = ChannelState.NOT_AVAILABLE

        return pd.DataFrame()
