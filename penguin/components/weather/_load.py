# -*- coding: utf-8 -*-
"""
    penguin.components.weather.weather
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
from __future__ import annotations

from functools import wraps

from loris import Configurations, Context, LocationUnavailableException
from loris.components import Weather
from loris.components.weather import WeatherException
from penguin import Location


# noinspection PyShadowingBuiltins
def load(context: Context, configs: Configurations) -> Weather:
    type = configs.get("type", default="default").lower()
    if type == "epw":
        from penguin.components.weather.epw import EPWWeather
        return EPWWeather(context, configs)
    elif type == "tmy":
        from penguin.components.weather.tmy import TMYWeather
        return TMYWeather(context, configs)

    # noinspection PyShadowingNames, PyProtectedMember
    @wraps(Weather.localize)
    def localize(configs: Configurations) -> None:
        if configs.enabled and all(k in configs for k in ["latitude", "longitude"]):
            weather._location = Location(
                configs.get_float("latitude"),
                configs.get_float("longitude"),
                timezone=configs.get("timezone", default="UTC"),
                altitude=configs.get_float("altitude", default=None),
                country=configs.get("country", default=None),
                state=configs.get("state", default=None),
            )
        else:
            try:
                weather._location = weather.context.location
                if not isinstance(weather._location, Location):
                    raise WeatherException(
                        f"Invalid location type for weather '{weather.uuid}': {type(weather._location)}"
                    )
            except (LocationUnavailableException, AttributeError):
                raise WeatherException(f"Missing location for weather '{weather.uuid}'")

    weather = Weather.load(context, configs)
    weather.localize = localize

    return weather
