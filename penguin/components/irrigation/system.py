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
from lori import ChannelState, Configurations, Context
from lori.components import Component, register_component_type
from lori.util import parse_key
from penguin.components.irrigation import IrrigationSeries


@register_component_type("irrigation", "watering")
# noinspection SpellCheckingInspection
class IrrigationSystem(Component):
    TYPE: str = "irr"

    series: List[IrrigationSeries]

    def __init__(self, context: Context, configs: Configurations) -> None:
        super().__init__(context, configs)
        self.series = []

    def configure(self, configs: Configurations) -> None:
        super().configure(configs)
        self._load_series(configs)

        self.data.add("humidity_mean", name="Humidity [%]", connector="random", type=float, min=50, max=100)
        self.data.add("storage_level", name="Storage Level [%]", connector="random", type=float, min=0, max=1000)

    def _load_series(self, configs: Configurations):
        series_dir = configs.path.replace(".conf", ".d")
        series_dirs = configs.dirs.to_dict()
        series_dirs["conf_dir"] = series_dir
        series_defaults = {}
        if "series" in configs:
            series_section = configs.get_section("series")
            series_keys = [k for k in series_section.sections if k not in [*Component.SECTIONS, *self.SECTIONS]]
            series_configs = {k: series_section.pop(k) for k in series_keys}
            series_defaults.update(series_section)

            for series_key, series_section in series_configs.items():
                series_key = parse_key(series_key)
                series_file = f"{series_key}.conf"
                series_configs = Configurations.load(
                    series_file,
                    **series_dirs,
                    **series_defaults,
                    require=False,
                )
                series_configs.update(series_section)
                series_configs.set("key", series_key)

                series = IrrigationSeries(self, series_configs)
                series.configure(series_configs)
                self.series.append(series)

        for series_path in glob.glob(os.path.join(series_dir, "series*.conf")):
            series_file = os.path.basename(series_path)
            series_key = parse_key(series_file.rsplit(".", maxsplit=1)[0])
            if any([series_key == a.key for a in self.series]):
                continue

            series_configs = Configurations.load(
                series_file,
                **series_dirs,
                **series_defaults,
            )
            series_configs.set("key", series_key)

            series = IrrigationSeries(self)
            series.configure(series_configs)
            self.series.append(series)

    def run(self, weather: pd.DataFrame) -> pd.DataFrame:
        series_humidity = [i.data.humidity for i in self.series if i.data.humidity.is_valid()]
        if len(series_humidity) > 0:
            self.data.humidity_mean.set(
                np.array([t.timestamp for t in series_humidity]).max(),
                np.array([t.value for t in series_humidity]).mean(),
            )
        else:
            self.data.humidity_mean.state = ChannelState.NOT_AVAILABLE

        return pd.DataFrame()
