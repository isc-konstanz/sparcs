# -*- coding: utf-8 -*-
"""
penguin.components.weather.static.file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""

from __future__ import annotations

from abc import abstractmethod

import datetime as dt
from typing import Any, Dict, Tuple

import pandas as pd
from lori.components.weather import Weather
from penguin.location import Location


# noinspection SpellCheckingInspection
class WeatherFile(Weather):
    _data: pd.DataFrame
    _meta: Dict[str, Any]

    def activate(self) -> None:
        super().activate()
        self._data, self._meta = self._read_from_file()
        self.location = self._localize_from_meta(self._meta)

    @abstractmethod
    def _read_from_file(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        pass

    @abstractmethod
    def _localize_from_meta(self, meta: Dict[str, Any]) -> Location:
        pass

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
