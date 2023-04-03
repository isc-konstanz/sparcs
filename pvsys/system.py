# -*- coding: utf-8 -*-
"""
    pvsys.system
    ~~~~~~~~~~~~
    
    
"""
from __future__ import annotations
from typing import List

import logging
import pandas as pd
import corsys as core
from corsys import Component
from corsys.weather import Weather
from corsys.configs import Configurations
from pvlib import solarposition
from .pv import PVSystem
from .model import Model
from .location import Location
from .input import (
    relative_humidity_from_dewpoint,
    precipitable_water_from_relative_humidity,
    global_diffuse_to_direct_normal_irradiance,
    cloud_cover_to_irradiance
)

logger = logging.getLogger(__name__)


class System(core.System):

    def __location__(self, configs: Configurations) -> Location:
        # FIXME: location necessary for for weather instantiation, but called afterwards here
        # if isinstance(self.weather, TMYWeather):
        #     return Location.from_tmy(self.weather.meta)
        # elif isinstance(self.weather, EPWWeather):
        #     return Location.from_epw(self.weather.meta)

        return Location(configs.getfloat('Location', 'latitude'),
                        configs.getfloat('Location', 'longitude'),
                        timezone=configs.get('Location', 'timezone', fallback='UTC'),
                        altitude=configs.getfloat('Location', 'altitude', fallback=None),
                        country=configs.get('Location', 'country', fallback=None),
                        state=configs.get('Location', 'state', fallback=None))

    def __weather__(self, configs: Configurations) -> Weather:
        conf_file = 'weather.cfg'
        configs = Configurations.from_configs(self.configs, conf_file)
        type = configs.get('General', 'type', fallback='default').lower()
        if type == 'tmy':
            from .weather.tmy import TMYWeather
            return TMYWeather(self, configs)
        elif type == 'epw':
            from .weather.epw import EPWWeather
            return EPWWeather(self, configs)

        return Weather.read(self, conf_file)

    def __cmpt_types__(self, *args: str) -> List[str]:
        return super().__cmpt_types__('solar', 'array', *args)

    # noinspection PyShadowingBuiltins
    def __cmpt__(self, configs: Configurations, type: str) -> Component:
        if type in ['pv', 'solar', 'array']:
            return PVSystem(self, configs)

        return super().__cmpt__(configs, type)

    # noinspection PyShadowingBuiltins
    def __call__(self) -> pd.DataFrame:
        input = self._get_input()
        result = pd.DataFrame(columns=['pv_power', 'dc_power'], index=input.index).fillna(0)
        result.index.name = 'time'
        for cmpt in self.values():
            if cmpt.type == 'pv':
                result_pv = self._get_solar_yield(cmpt, input)
                result[['pv_power', 'dc_power']] += result_pv[['pv_power', 'dc_power']].abs()

        return pd.concat([result, input], axis=1)

    # noinspection PyUnresolvedReferences, PyTypeChecker
    def _validate_input(self, weather: pd.DataFrame) -> pd.DataFrame:

        # noinspection PyShadowingBuiltins
        def assert_inputs(*inputs, error=False):
            if any(input for input in inputs if input not in weather.columns or weather[input].isna().any()):
                if error:
                    raise ValueError(f"Unable to complete input data with missing or invalid features: "
                                     ', '.join(inputs))
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
                assert_inputs(Weather.CLOUD_COVER, error=True)
                ghi, dhi, dni = cloud_cover_to_irradiance(self.location, weather[Weather.CLOUD_COVER], solar_position)
                insert_input(Weather.GHI, ghi)
                insert_input(Weather.DHI, dhi)
                insert_input(Weather.DNI, dni)
            if not assert_inputs(Weather.DNI):
                assert_inputs(Weather.GHI, Weather.DHI, error=True)
                insert_input(Weather.DNI, global_diffuse_to_direct_normal_irradiance(
                    weather[Weather.GHI],
                    weather[Weather.DHI],
                    solar_position)
                )
        if not assert_inputs(Weather.HUMIDITY_REL):
            assert_inputs(Weather.TEMP_AIR, Weather.TEMP_DEW_POINT, error=True)
            insert_input(Weather.HUMIDITY_REL, relative_humidity_from_dewpoint(
                weather[Weather.TEMP_AIR],
                weather[Weather.TEMP_DEW_POINT])
            )
        if not assert_inputs(Weather.PRECIPITABLE_WATER):
            assert_inputs(Weather.TEMP_AIR, Weather.HUMIDITY_REL, error=True)
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
            logger.warning("Unable to generate solar position: {}".format(str(e)))

        return data

    # noinspection PyMethodMayBeStatic
    def _get_solar_yield(self, pv: PVSystem, weather: pd.DataFrame) -> pd.DataFrame:
        model = Model.read(pv)
        return model(weather).rename(columns={'p_ac': 'pv_power',
                                              'p_dc': 'dc_power'})

    # noinspection PyShadowingBuiltins
    def evaluate(self, **kwargs):
        from .evaluation import Evaluation
        eval = Evaluation(self)
        return eval(**kwargs)
