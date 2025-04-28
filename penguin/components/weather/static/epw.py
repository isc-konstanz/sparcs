# -*- coding: utf-8 -*-
"""
penguin.components.weather.static.epw
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, Tuple

from pvlib.iotools import read_epw

import numpy as np
import pandas as pd
from lori import Configurations
from lori.components.weather import register_weather_type
from penguin import Location
from penguin.components.weather.static import WeatherFile


@register_weather_type("epw")
class EPWWeather(WeatherFile):
    year: int
    file: str
    path: str

    def configure(self, configs: Configurations) -> None:
        super().configure(configs)
        section = configs.get_section("epw", defaults={})
        self.year = section.get_int("year", default=None)
        self.file = section.get("file", default="weather.epw")
        self.path = self.file if os.path.isabs(self.file) else os.path.join(self.configs.dirs.data, self.file)

    def activate(self) -> None:
        super().activate()
        if not os.path.isfile(self.path):
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            self._download(self.location)

    # noinspection PyPackageRequirements
    def _download(self, location: Location) -> None:
        import requests
        import urllib3
        from urllib3.exceptions import InsecureRequestWarning

        urllib3.disable_warnings(InsecureRequestWarning)

        headers = {
            "User-Agent": "Magic Browser",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }

        response = requests.get("https://github.com/NREL/EnergyPlus/raw/develop/weather/master.geojson", verify=False)
        data = response.json()  # metadata for available files
        # download lat/lon and url details for each .epw file into a dataframe

        locations = [{"url": [], "lat": [], "lon": [], "name": []}]
        for features in data["features"]:
            match = re.search(r'href=[\'"]?([^\'" >]+)', features["properties"]["epw"])
            if match:
                url = match.group(1)
                name = url[url.rfind("/") + 1 :]
                longitude = features["geometry"]["coordinates"][0]
                latitude = features["geometry"]["coordinates"][1]
                locations.append({"name": name, "url": url, "latitude": float(latitude), "longitude": float(longitude)})

        locations = pd.DataFrame(locations)
        errorvec = np.sqrt(
            np.square(locations.latitude - location.latitude) + np.square(locations.longitude - location.longitude)
        )
        index = errorvec.idxmin()
        url = locations["url"][index]
        # name = locations['name'][index]

        response = requests.get(url, verify=False, headers=headers)
        if response.ok:
            with open(self.path, "wb") as file:
                file.write(response.text.encode("ascii", "ignore"))
        else:
            self._logger.warning("Connection error status code: %s" % response.status_code)
            response.raise_for_status()

    def _read_from_file(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        return read_epw(filename=self.path, coerce_year=self.year)

    # noinspection PyMethodMayBeStatic
    def _localize_from_meta(self, meta: Dict[str, Any]) -> Location:
        return Location.from_epw(meta)
