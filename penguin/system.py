# -*- coding: utf-8 -*-
"""
    penguin.system
    ~~~~~~~~~~~~~~


"""
from __future__ import annotations

import datetime as dt

from pvlib import solarposition

import loris
import pandas as pd
from loris import ComponentException, Configurations, Weather, WeatherException
from penguin import Location, PVSystem
from penguin.input import (
    direct_diffuse_from_global_irradiance,
    direct_normal_from_global_diffuse_irradiance,
    global_irradiance_from_cloud_cover,
    precipitable_water_from_relative_humidity,
    relative_humidity_from_dewpoint,
)


class System(loris.System):
    POWER_EL:      str = "el_power"
    POWER_EL_IMP:  str = "el_import_power"
    POWER_EL_EXP:  str = "el_export_power"
    POWER_TH:      str = "th_power"
    POWER_TH_HT:   str = "th_ht_power"
    POWER_TH_DOM:  str = "th_dom_power"

    ENERGY_EL:      str = "el_energy"
    ENERGY_EL_IMP:  str = "el_import_energy"
    ENERGY_EL_EXP:  str = "el_export_energy"
    ENERGY_TH:      str = "th_energy"
    ENERGY_TH_HT:   str = "th_ht_energy"
    ENERGY_TH_DOM:  str = "th_dom_energy"

    def localize(self, configs: Configurations) -> None:
        if configs.enabled:
            self._location = Location(
                configs.get_float("latitude"),
                configs.get_float("longitude"),
                timezone=configs.get("timezone", default="UTC"),
                altitude=configs.get_float("altitude", default=None),
                country=configs.get("country", default=None),
                state=configs.get("state", default=None),
            )
        else:
            self._location = None

    def _on_configure(self, configs: Configurations) -> None:
        super()._on_configure(configs)
        if self.has_type(PVSystem.TYPE):
            from penguin import COLUMNS

            self.data.add(id=PVSystem.POWER_CALC, name=COLUMNS[PVSystem.POWER_CALC], connector=None, value_type=float)

    # # noinspection PyShadowingBuiltins
    # def evaluate(self, **kwargs):
    #     from .evaluation import Evaluation
    #     eval = Evaluation(self)
    #     return eval(**kwargs)

    # noinspection PyShadowingBuiltins, PyUnresolvedReferences
    def run(
        self, start: pd.Timestamp | dt.datetime = None, end: pd.Timestamp | dt.datetime = None, **kwargs
    ) -> pd.DataFrame:
        try:
            input = self._get_input(start, end, **kwargs)
            result = pd.DataFrame(columns=[], index=input.index)
            result.index.name = "timestamp"
            if self.has_type(PVSystem.TYPE):
                result.loc[:, [PVSystem.POWER, PVSystem.POWER_DC]] = 0.0
                for pv in self.get_all(PVSystem.TYPE):
                    pv_result = pv.run(input)
                    result[[PVSystem.POWER, PVSystem.POWER_DC]] += pv_result[[PVSystem.POWER, PVSystem.POWER_DC]].abs()

                pv_power_channel = self.data[PVSystem.POWER_CALC]
                pv_power = result[PVSystem.POWER]
                if not pv_power.empty:
                    pv_power_channel.set(pv_power.index[0], pv_power)
                else:
                    pv_power_channel.state = ChannelState.NOT_AVAILABLE

            return pd.concat([result, input], axis="columns")

        except ComponentException as e:
            self._logger.warning(f"Unable to run system '{self.name}': {str(e)}")

    # noinspection PyShadowingBuiltins, PyUnresolvedReferences
    def _get_input(self, *args, **kwargs) -> pd.DataFrame:
        weather = self.weather.get(*args, **kwargs)
        weather = self._validate_weather(weather)
        solar_position = self._get_solar_position(weather.index)
        return pd.concat([weather, solar_position], axis="columns")

    # noinspection PyUnresolvedReferences, PyTypeChecker
    def _validate_weather(self, weather: pd.DataFrame) -> pd.DataFrame:
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

        if not assert_columns(Weather.GHI, Weather.DHI, Weather.DNI):
            solar_position = self.location.get_solarposition(weather.index)
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
                        kwargs["pressure"] = weather[Weather.TEMP_DEW_POINT]
                    if assert_columns(Weather.PRESSURE_SEA):
                        kwargs["temp_dew"] = weather[Weather.PRESSURE_SEA]

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
                logger.warning(
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
                logger.warning(
                    f"Unable to estimate missing '{Weather.PRECIPITABLE_WATER}' data with "
                    f"missing or invalid columns: {', '.join([Weather.TEMP_AIR, Weather.HUMIDITY_REL])}"
                )
            else:
                insert_column(
                    Weather.PRECIPITABLE_WATER,
                    precipitable_water_from_relative_humidity(weather[Weather.TEMP_AIR], weather[Weather.HUMIDITY_REL]),
                )
        return weather

    def _get_solar_position(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        data = pd.DataFrame(index=index)
        try:
            data = solarposition.get_solarposition(
                index, self.location.latitude, self.location.longitude, altitude=self.location.altitude
            )
            data = data.loc[:, ["azimuth", "apparent_zenith", "apparent_elevation"]]
            data.columns = ["solar_azimuth", "solar_zenith", "solar_elevation"]

        except ImportError as e:
            self._logger.warning("Unable to generate solar position: {}".format(str(e)))

        return data
