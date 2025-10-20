# -*- coding: utf-8 -*-
"""
sparcs.components.weather.static.tmy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""

from __future__ import annotations

import os
from typing import Any, Dict, Tuple

from pvlib.iotools import read_tmy2, read_tmy3

import pandas as pd
from lories import Configurations
from lories.components.weather import register_weather_type
from sparcs import Location
from sparcs.components.weather.static import WeatherFile


@register_weather_type("tmy")
class TMYWeather(WeatherFile):
    version: int

    year: int
    file: str
    path: str

    def configure(self, configs: Configurations) -> None:
        super().configure(configs)
        tmy = configs.get_member("epw", defaults={})
        self.version = tmy.get_int("version", default=3)

        self.year = tmy.get_int("year", default=None)
        self.file = tmy.get("file", default="weather.csv")
        self.path = self.file if os.path.isabs(self.file) else os.path.join(configs.dirs.data, self.file)

    def _read_from_file(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        if self.version == 3:
            return read_tmy3(filename=self.file, coerce_year=self.year, map_variables=True)

        elif self.version == 2:
            return read_tmy2(self.file)
        else:
            raise ValueError("Invalid TMY version: {}".format(self.version))

    # noinspection PyMethodMayBeStatic
    def _localize_from_meta(self, meta: Dict[str, Any]) -> Location:
        return Location.from_tmy(meta)
