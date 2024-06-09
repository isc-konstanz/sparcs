# -*- coding: utf-8 -*-
"""
    penguin.system
    ~~~~~~~~~~~~~~
    
    
"""
from __future__ import annotations

import datetime as dt
import loris
import pandas as pd
from pvlib import solarposition
from loris import components, Configurations
from penguin import Location, Weather, PVSystem, PVArray, ElectricVehicle, ElectricalEnergyStorage, ThermalEnergyStorage
from penguin.input import (
    relative_humidity_from_dewpoint,
    precipitable_water_from_relative_humidity,
    direct_normal_from_global_diffuse_irradiance,
    direct_diffuse_from_global_irradiance,
    global_irradiance_from_cloud_cover
)

components.register(Weather, Weather.TYPE, factory=Weather.load, replace=True)
components.register(PVArray, PVArray.TYPE)
components.register(PVSystem, PVSystem.TYPE, *PVSystem.ALIAS)
components.register(ElectricVehicle, ElectricVehicle.TYPE)
components.register(ElectricalEnergyStorage, ElectricalEnergyStorage.TYPE)
components.register(ThermalEnergyStorage, ThermalEnergyStorage.TYPE)


class System(loris.System):

    POWER_EL:     str = 'el_power'
    POWER_EL_IMP: str = 'el_import_power'
    POWER_EL_EXP: str = 'el_export_power'
    POWER_TH:     str = 'th_power'
    POWER_TH_HT:  str = 'th_ht_power'
    POWER_TH_DOM: str = 'th_dom_power'

    ENERGY_EL:     str = 'el_energy'
    ENERGY_EL_IMP: str = 'el_import_energy'
    ENERGY_EL_EXP: str = 'el_export_energy'
    ENERGY_TH:     str = 'th_energy'
    ENERGY_TH_HT:  str = 'th_ht_energy'
    ENERGY_TH_DOM: str = 'th_dom_energy'

    # noinspection PyMethodMayBeStatic
    def __localize__(self, configs: Configurations) -> None:
        if configs.has_section(Location.SECTION):
            location_configs = configs.get_section(Location.SECTION)
            self._location = Location(
                location_configs.get_float("latitude"),
                location_configs.get_float("longitude"),
                timezone=location_configs.get("timezone", default="UTC"),
                altitude=location_configs.get_float("altitude", default=None),
                country=location_configs.get("country", default=None),
                state=location_configs.get("state", default=None),
            )

    # noinspection PyShadowingBuiltins
    def run(
        self,
        start: pd.Timestamp | dt.datetime = None,
        end: pd.Timestamp | dt.datetime = None,
        **kwargs
    ) -> pd.DataFrame:
        # if start is None:
        #     start = pd.Timestamp.now(tz=self.location.timezone)
        input = self._get_input(start, end, **kwargs)
        result = pd.DataFrame(columns=[PVSystem.POWER, PVSystem.POWER_DC], index=input.index).fillna(0.0)
        result.index.name = 'time'
        for pv in self.get_all(PVSystem.TYPE):
            result_pv = pv.run(input)
            result[[PVSystem.POWER, PVSystem.POWER_DC]] += result_pv[[PVSystem.POWER, PVSystem.POWER_DC]].abs()

        return pd.concat([result, input], axis='columns')

    # noinspection PyUnresolvedReferences, PyTypeChecker
    def _validate_input(self, weather: pd.DataFrame) -> pd.DataFrame:

        # noinspection PyShadowingBuiltins
        def assert_inputs(*inputs):
            if any(input for input in inputs if input not in weather.columns or weather[input].isna().any()):
                return False
            return True

        # noinspection PyShadowingBuiltins
        def insert_input(input, data):
            if input not in weather.columns:
                weather[input] = data
            else:
                weather[input] = weather[input].combine_first(data)

        if not assert_inputs(Weather.GHI, Weather.DHI, Weather.DNI):
            solar_position = self.location.get_solarposition(weather.index)
            if not assert_inputs(Weather.GHI):
                if not assert_inputs(Weather.CLOUD_COVER):
                    raise ValueError(f'Unable to estimate missing "{Weather.GHI}" data with '
                                     f'missing or invalid column: {Weather.CLOUD_COVER}')
                ghi = global_irradiance_from_cloud_cover(self.location, weather[Weather.CLOUD_COVER], solar_position)
                insert_input(Weather.GHI, ghi)

            if not assert_inputs(Weather.DHI, Weather.DNI):
                if not assert_inputs(Weather.DHI):
                    if not assert_inputs(Weather.GHI):
                        raise ValueError(f'Unable to estimate missing "{Weather.DHI}" and "{Weather.DNI}" data with '
                                         f'missing or invalid columns: {", ".join([Weather.GHI])}')
                    kwargs = {}
                    if assert_inputs(Weather.TEMP_DEW_POINT):
                        kwargs['pressure'] = weather[Weather.TEMP_DEW_POINT]
                    if assert_inputs(Weather.PRESSURE_SEA):
                        kwargs['temp_dew'] = weather[Weather.PRESSURE_SEA]

                    dni, dhi = direct_diffuse_from_global_irradiance(
                        solar_position,
                        weather[Weather.GHI],
                        **kwargs
                    )
                    insert_input(Weather.DHI, dhi)
                    insert_input(Weather.DNI, dni)
                else:
                    if not assert_inputs(Weather.GHI, Weather.DHI):
                        raise ValueError(f'Unable to estimate missing "{Weather.DNI}" data with '
                                         f'missing or invalid columns: {", ".join([Weather.GHI, Weather.DHI])}')
                    dni = direct_normal_from_global_diffuse_irradiance(
                        solar_position,
                        weather[Weather.GHI],
                        weather[Weather.DHI]
                    )
                    insert_input(Weather.DNI, dni)

        if not assert_inputs(Weather.HUMIDITY_REL):
            if not assert_inputs(Weather.TEMP_AIR, Weather.TEMP_DEW_POINT):
                logger.warning(f'Unable to estimate missing "{Weather.HUMIDITY_REL}" data with '
                               f'missing or invalid columns: {", ".join([Weather.TEMP_AIR, Weather.TEMP_DEW_POINT])}')
            else:
                insert_input(Weather.HUMIDITY_REL, relative_humidity_from_dewpoint(
                    weather[Weather.TEMP_AIR],
                    weather[Weather.TEMP_DEW_POINT])
                )
        if not assert_inputs(Weather.PRECIPITABLE_WATER):
            if not assert_inputs(Weather.TEMP_AIR, Weather.HUMIDITY_REL):
                logger.warning(f'Unable to estimate missing "{Weather.PRECIPITABLE_WATER}" data with '
                               f'missing or invalid columns: {", ".join([Weather.TEMP_AIR, Weather.HUMIDITY_REL])}')
            else:
                insert_input(Weather.PRECIPITABLE_WATER, precipitable_water_from_relative_humidity(
                    weather[Weather.TEMP_AIR],
                    weather[Weather.HUMIDITY_REL])
                )
        solar_position = self._get_solar_position(weather.index)
        return pd.concat([weather, solar_position], axis=1)

    # noinspection PyShadowingBuiltins
    def _get_input(self, *args, **kwargs) -> pd.DataFrame:
        weather = self.weather.get(*args, **kwargs)
        input = self._validate_input(weather)
        return input

    def _get_solar_position(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        data = pd.DataFrame(index=index)
        try:
            data = solarposition.get_solarposition(index,
                                                   self.location.latitude,
                                                   self.location.longitude,
                                                   altitude=self.location.altitude)
            data = data.loc[:, ['azimuth', 'apparent_zenith', 'apparent_elevation']]
            data.columns = ['solar_azimuth', 'solar_zenith', 'solar_elevation']

        except ImportError as e:
            self._logger.warning("Unable to generate solar position: {}".format(str(e)))

        return data

    # # noinspection PyShadowingBuiltins
    # def evaluate(self, **kwargs):
    #     from .evaluation import Evaluation
    #     eval = Evaluation(self)
    #     return eval(**kwargs)
