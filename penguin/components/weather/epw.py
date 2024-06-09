# -*- coding: utf-8 -*-
"""
    penguin.components.weather.epw
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""
from __future__ import annotations

import datetime as dt
import os
import re
import numpy as np
import pandas as pd

from typing import Dict
from pvlib.iotools import read_epw
from loris import Configurations
from penguin import Location
from penguin.components.weather import Weather


class EPWWeather(Weather):
    _data: pd.DataFrame
    _meta: Dict

    def __configure__(self, configs: Configurations) -> None:
        super().__configure__(configs)
        section = configs.get_section("epw", default={})
        self.year = section.get_int("year", default=None)
        self.file = section.get("file", default="weather.epw")
        self.path = self.file if os.path.isabs(self.file) else os.path.join(self.configs.dirs.data, self.file)

    def __activate__(self) -> None:
        super().__activate__()
        if not os.path.isfile(self.path):
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            self._download(self.location)

        self._data, self._meta = read_epw(filename=self.path, coerce_year=self.year)
        columns = self._data.columns
        for column in columns:
            if self._data[column].sum() == 0:
                self._data.drop(column, axis=1, inplace=True)

        self.location = Location.from_epw(self._meta)

    # noinspection PyPackageRequirements
    def _download(self, location: Location) -> None:
        import requests
        import urllib3
        from urllib3.exceptions import InsecureRequestWarning
        urllib3.disable_warnings(InsecureRequestWarning)

        headers = {
            'User-Agent': "Magic Browser",
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
        }

        response = requests.get('https://github.com/NREL/EnergyPlus/raw/develop/weather/master.geojson', verify=False)
        data = response.json()  # metadata for available files
        # download lat/lon and url details for each .epw file into a dataframe

        locations = [{'url': [], 'lat': [], 'lon': [], 'name': []}]
        for features in data['features']:
            match = re.search(r'href=[\'"]?([^\'" >]+)', features['properties']['epw'])
            if match:
                url = match.group(1)
                name = url[url.rfind('/') + 1:]
                longitude = features['geometry']['coordinates'][0]
                latitude = features['geometry']['coordinates'][1]
                locations.append({'name': name, 'url': url, 'latitude': float(latitude), 'longitude': float(longitude)})

        locations = pd.DataFrame(locations)
        errorvec = np.sqrt(np.square(locations.latitude - location.latitude) +
                           np.square(locations.longitude - location.longitude))
        index = errorvec.idxmin()
        url = locations['url'][index]
        # name = locations['name'][index]

        response = requests.get(url, verify=False, headers=headers)
        if response.ok:
            with open(self.path, 'wb') as file:
                file.write(response.text.encode('ascii', 'ignore'))
        else:
            self._logger.warning("Connection error status code: %s" % response.status_code)
            response.raise_for_status()

    def get(
        self,
        start: pd.Timestamp | dt.datetime = None,
        end: pd.Timestamp | dt.datetime = None,
        **kwargs
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
