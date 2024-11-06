# -*- coding: utf-8 -*-
"""
penguin.components.weather.weather
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""

from __future__ import annotations

import datetime as dt
import types
from typing import Optional, Type

from pvlib import atmosphere

import pandas as pd
from lori.components import register_component_type
from lori.components.weather import Weather, WeatherException, WeatherMeta
from lori.core import Configurations
from penguin.components.weather.input import (
    direct_diffuse_from_global_irradiance,
    direct_normal_from_global_diffuse_irradiance,
    global_irradiance_from_cloud_cover,
    precipitable_water_from_relative_humidity,
    relative_humidity_from_dewpoint,
)
from penguin.location import Location, LocationUnavailableException


class ValidatedWeatherMeta(WeatherMeta):
    # noinspection PyTypeChecker
    def __call__(cls, *args, **kwargs) -> Weather:
        weather = super().__call__(*args, **kwargs)
        weather._get_weather = weather.get
        weather.get = types.MethodType(ValidatedWeather.get, weather)
        weather.validate = types.MethodType(ValidatedWeather.validate, weather)
        weather.localize = types.MethodType(ValidatedWeather.localize, weather)
        return weather

    # noinspection PyShadowingBuiltins
    def _get_class(cls: Type[Weather], type: str) -> Type[Weather]:
        from penguin.components.weather.file import EPWWeather, TMYWeather

        if type == "epw":
            return EPWWeather
        elif type == "tmy":
            return TMYWeather
        else:
            return super()._get_class(type)


# noinspection SpellCheckingInspection
@register_component_type(replace=True)
class ValidatedWeather(Weather, metaclass=ValidatedWeatherMeta):
    # noinspection PyShadowingNames, PyProtectedMember
    def localize(self, configs: Configurations) -> None:
        if configs.enabled and all(k in configs for k in ["latitude", "longitude"]):
            self._location = Location(
                configs.get_float("latitude"),
                configs.get_float("longitude"),
                timezone=configs.get("timezone", default="UTC"),
                altitude=configs.get_float("altitude", default=None),
                country=configs.get("country", default=None),
                state=configs.get("state", default=None),
            )
        else:
            try:
                self._location = self.context.location
                if not isinstance(self._location, Location):
                    raise WeatherException(f"Invalid location type for weather '{self.key}': {type(self._location)}")
            except (LocationUnavailableException, AttributeError):
                raise WeatherException(f"Missing location for weather '{self.key}'")

    # noinspection PyUnresolvedReferences, PyTypeChecker
    def validate(self, weather: pd.DataFrame) -> pd.DataFrame:
        # noinspection PyShadowingBuiltins
        def assert_columns(*columns):
            if any(column for column in columns if column not in weather.columns or weather[column].isna().any()):
                return False
            return True

        # noinspection PyShadowingBuiltins
        def insert_column(column, data):
            if column not in weather.columns:
                weather[column] = data
            else:
                weather[column] = weather[column].combine_first(data)

        if not assert_columns(Weather.PRESSURE_SEA):
            pressure = pd.Series(index=weather.index, data=atmosphere.alt2pres(self.location.altitude))
            insert_column(Weather.PRESSURE_SEA, pressure)

        solar_position = self.location.get_solarposition(weather.index, pressure=weather[Weather.PRESSURE_SEA])

        if not assert_columns(Weather.GHI, Weather.DHI, Weather.DNI):
            if not assert_columns(Weather.GHI):
                if not assert_columns(Weather.CLOUD_COVER):
                    raise WeatherException(
                        f"Unable to estimate missing '{Weather.GHI}' data with "
                        f"missing or invalid column: {Weather.CLOUD_COVER}"
                    )
                ghi = global_irradiance_from_cloud_cover(self.location, weather[Weather.CLOUD_COVER], solar_position)
                insert_column(Weather.GHI, ghi)

            if not assert_columns(Weather.DHI, Weather.DNI):
                if not assert_columns(Weather.DHI):
                    if not assert_columns(Weather.GHI):
                        raise WeatherException(
                            f"Unable to estimate missing '{Weather.DHI}' and '{Weather.DNI}' data with "
                            f"missing or invalid columns: {', '.join([Weather.GHI])}"
                        )
                    kwargs = {}
                    if assert_columns(Weather.TEMP_DEW_POINT):
                        kwargs["temp_dew"] = weather[Weather.TEMP_DEW_POINT]
                    if assert_columns(Weather.PRESSURE_SEA):
                        kwargs["pressure"] = weather[Weather.PRESSURE_SEA]

                    dni, dhi = direct_diffuse_from_global_irradiance(solar_position, weather[Weather.GHI], **kwargs)
                    insert_column(Weather.DHI, dhi)
                    insert_column(Weather.DNI, dni)
                else:
                    if not assert_columns(Weather.GHI, Weather.DHI):
                        raise WeatherException(
                            f"Unable to estimate missing '{Weather.DNI}' data with "
                            f"missing or invalid columns: {', '.join([Weather.GHI, Weather.DHI])}"
                        )
                    dni = direct_normal_from_global_diffuse_irradiance(
                        solar_position, weather[Weather.GHI], weather[Weather.DHI]
                    )
                    insert_column(Weather.DNI, dni)

        if not assert_columns(Weather.HUMIDITY_REL):
            if not assert_columns(Weather.TEMP_AIR, Weather.TEMP_DEW_POINT):
                raise WeatherException(
                    f"Unable to estimate missing '{Weather.HUMIDITY_REL}' data with "
                    f"missing or invalid columns: {', '.join([Weather.TEMP_AIR, Weather.TEMP_DEW_POINT])}"
                )
            else:
                insert_column(
                    Weather.HUMIDITY_REL,
                    relative_humidity_from_dewpoint(weather[Weather.TEMP_AIR], weather[Weather.TEMP_DEW_POINT]),
                )
        if not assert_columns(Weather.PRECIPITABLE_WATER):
            if not assert_columns(Weather.TEMP_AIR, Weather.HUMIDITY_REL):
                raise WeatherException(
                    f"Unable to estimate missing '{Weather.PRECIPITABLE_WATER}' data with "
                    f"missing or invalid columns: {', '.join([Weather.TEMP_AIR, Weather.HUMIDITY_REL])}"
                )
            else:
                insert_column(
                    Weather.PRECIPITABLE_WATER,
                    precipitable_water_from_relative_humidity(weather[Weather.TEMP_AIR], weather[Weather.HUMIDITY_REL]),
                )

        insert_column("solar_azimuth", solar_position["azimuth"])
        insert_column("solar_zenith", solar_position["apparent_zenith"])
        insert_column("solar_elevation", solar_position["apparent_elevation"])

        return weather

    # noinspection PyUnresolvedReferences
    def get(
        self,
        start: Optional[pd.Timestamp, dt.datetime, str] = None,
        end: Optional[pd.Timestamp, dt.datetime, str] = None,
        validate: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        weather = self._get_weather(start, end, **kwargs)
        if validate:
            self.validate(weather)

        return weather
