# -*- coding: utf-8 -*-
"""
penguin.components.weather.static.tmy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""

from __future__ import annotations

import os
from typing import Any, Dict, Tuple

from pvlib.iotools import read_tmy2, read_tmy3

import pandas as pd
from lori import Configurations
from lori.components.weather import register_weather_type
from penguin import Location
from penguin.components.weather.static import WeatherFile


@register_weather_type("tmy")
class TMYWeather(WeatherFile):
    version: int

    year: int
    file: str
    path: str

    def configure(self, configs: Configurations) -> None:
        super().configure(configs)
        section = configs.get_section("epw", defaults={})
        self.version = section.get_int("version", default=3)

        self.year = section.get_int("year", default=None)
        self.file = section.get("file", default="weather.csv")
        self.path = self.file if os.path.isabs(self.file) else os.path.join(self.configs.dirs.data, self.file)

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
