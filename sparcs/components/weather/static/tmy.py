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

    _year: int
    _file: str
    _path: str

    def configure(self, configs: Configurations) -> None:
        super().configure(configs)
        tmy = configs.get_member("tmy", defaults={})
        self.version = tmy.get_int("version", default=3)

        self._year = tmy.get_int("year", default=None)
        self._file = tmy.get("file", default="weather.csv")
        self._path = self._file if os.path.isabs(self._file) else os.path.join(configs.dirs.data, self._file)

    def _read_from_file(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        if self.version == 3:
            data, meta = read_tmy3(filename=self._path, coerce_year=self._year, map_variables=True)

        elif self.version == 2:
            data, meta = read_tmy2(self._path)
        else:
            raise ValueError("Invalid TMY version: {}".format(self.version))

        return data.drop(columns=["Date (MM/DD/YYYY)", "Time (HH:MM)"]), meta

    # noinspection PyMethodMayBeStatic
    def _localize_from_meta(self, meta: Dict[str, Any]) -> Location:
        return Location.from_tmy(meta)
