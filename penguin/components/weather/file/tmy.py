# -*- coding: utf-8 -*-
"""
penguin.components.weather.tmy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""

from __future__ import annotations

import datetime as dt
import os
from typing import Dict

from pvlib.iotools import read_tmy2, read_tmy3

import pandas as pd
from loris import Configurations
from penguin import Location
from penguin.components.weather import Weather


class TMYWeather(Weather):
    _data: pd.DataFrame
    _meta: Dict

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

    def activate(self) -> None:
        super().activate()

        if self.version == 3:
            self._data, self._meta = read_tmy3(filename=self.file, coerce_year=self.year, map_variables=True)

        elif self.version == 2:
            self._data, self._meta = read_tmy2(self.file)
        else:
            raise ValueError("Invalid TMY version: {}".format(self.version))

        self._location = Location.from_epw(self._meta)

    def get(
        self,
        start: pd.Timestamp | dt.datetime = None,
        end: pd.Timestamp | dt.datetime = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Retrieves the weather data for a specified time interval

        :param start:
            the start timestamp for which weather data will be looked up for.
            For many applications, passing datetime.datetime.now() will suffice.
        :type start:
            :class:`pandas.Timestamp` or datetime

        :param end:
            the end timestamp for which weather data will be looked up for.
        :type end:
            :class:`pandas.Timestamp` or datetime

        :returns:
            the weather data, indexed in a specific time interval.

        :rtype:
            :class:`pandas.DataFrame`
        """
        return self._get_range(self._data, start, end)
