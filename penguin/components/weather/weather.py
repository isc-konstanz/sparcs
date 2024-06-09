# -*- coding: utf-8 -*-
"""
    penguin.components.weather.weather
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
from __future__ import annotations

import loris
from loris import Configurations
from loris.components import ComponentContext
from loris.components.weather import WeatherException
from penguin import Location


class Weather(loris.Weather):
    # noinspection PyShadowingBuiltins
    @classmethod
    def load(cls, context: ComponentContext, configs: Configurations) -> Weather:
        type = configs.get("type", default="default").lower()
        if type == "epw":
            from penguin.components.weather.epw import EPWWeather

            return EPWWeather(context, configs)
        elif type == "tmy":
            return None
        else:
            return cls(context, configs)

    # noinspection PyMethodMayBeStatic
    def __localize__(self, configs: Configurations) -> None:
        if hasattr(self._context, "location"):
            location = getattr(self._context, "location")
            if not isinstance(location, Location):
                raise WeatherException(f"Invalid location type for weather '{self._uuid}': {type(location)}")
            self.location = location

        elif configs.has_section(Location.SECTION):
            self.location = Location(
                configs.get_float("latitude"),
                configs.get_float("longitude"),
                timezone=configs.get("timezone", default="UTC"),
                altitude=configs.get_float("altitude", default=None),
                country=configs.get("country", default=None),
                state=configs.get("state", default=None),
            )
        else:
            raise WeatherException(f"Unable to find valid location for weather configuration: {self.configs.name}")
