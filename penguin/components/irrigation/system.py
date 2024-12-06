# -*- coding: utf-8 -*-
"""
penguin.components.irrigation.system
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""

from __future__ import annotations

import glob
import os
from typing import List

import pandas as pd
from lori import ChannelState, Configurations, Context
from lori.components import Component, register_component_type
from lori.util import validate_key
from penguin.components.irrigation import IrrigationSeries


@register_component_type("irr", "irrigation", "watering")
class IrrigationSystem(Component):
    SECTIONS = ["storage", *IrrigationSeries.SECTIONS]

    series: List[IrrigationSeries]

    def __init__(self, context: Context, configs: Configurations) -> None:
        super().__init__(context, configs)
        self.series = []

    def configure(self, configs: Configurations) -> None:
        super().configure(configs)
        self._load_series(configs)

        # As % of plant available water capacity (PAWC)
        # self.data.add("water_supply_min", name="Water supply coverage min [%]", type=float)
        # self.data.add("water_supply_max", name="Water supply coverage max [%]", type=float)
        self.data.add("water_supply_mean", name="Water supply coverage mean [%]", type=float)

        self.data.add("storage_level", name="Storage Level [%]", type=float)

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
                series_key = validate_key(series_key)
                series_file = f"{series_key}.conf"
                series_configs = Configurations.load(
                    series_file,
                    **series_dirs,
                    **series_defaults,
                    require=False,
                )
                series_configs.update(series_section, replace=False)

                series = IrrigationSeries(self, key="series", name=f"{self.name} Series")
                series.configure(series_configs)
                self.series.append(series)

        for series_path in glob.glob(os.path.join(series_dir, "series*.conf")):
            series_file = os.path.basename(series_path)
            series_key = validate_key(series_file.rsplit(".", maxsplit=1)[0])
            if any([series_key == a.key for a in self.series]):
                continue

            series_configs = Configurations.load(
                series_file,
                **series_dirs,
                **series_defaults,
            )
            series = IrrigationSeries(self, key=series_key, name=f"{self.name} {series_key.title()}")
            series.configure(series_configs)
            self.series.append(series)

    # noinspection SpellCheckingInspection
    def activate(self) -> None:
        super().activate()
        water_supplies = []
        for series in self.series:
            series.activate()
            water_supplies.append(series.soil.data.water_supply)

        self.data.register(self._water_supply_callback, *water_supplies, how="all", unique=True)

    def deactivate(self) -> None:
        super().deactivate()
        for series in self.series:
            series.deactivate()

    def _water_supply_callback(self, data: pd.DataFrame) -> None:
        water_supply = data[[c for c in data.columns if "water_supply" in c]]
        if not water_supply.empty:
            water_supply_mean = water_supply.ffill().bfill().mean(axis="columns")
            if len(water_supply_mean) == 1:
                water_supply_mean = water_supply_mean.iloc[0]
            self.data.water_supply_mean.value = water_supply_mean
        else:
            self.data.water_supply_mean.state = ChannelState.NOT_AVAILABLE
