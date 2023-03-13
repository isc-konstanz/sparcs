# -*- coding: utf-8 -*-
"""
    pvsys.system
    ~~~~~~~~~~~~
    
    
"""
from __future__ import annotations

import logging
import pandas as pd
import corsys
from corsys import Component
from corsys.configs import Configurations
from pvlib import solarposition
from .pv import PVSystem
from .model import Model
from .location import Location

logger = logging.getLogger(__name__)

CMPTS = {
    'tes': 'Buffer Storage',
    'ees': 'Battery Storage',
    'ev': 'Electric Vehicle',
    'pv': 'Photovoltaics'
}

AC_E = 'Energy yield [kWh]'
AC_Y = 'Specific yield [kWh/kWp]'


class System(corsys.System):

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

    def __cmpt_types__(self):
        return super().__cmpt_types__('solar', 'array')

    # noinspection PyShadowingBuiltins
    def __cmpt__(self, configs: Configurations, type: str) -> Component:
        if type in ['pv', 'solar', 'array']:
            return PVSystem(self, configs)

        return super().__cmpt__(configs, type)

    def __call__(self) -> pd.DataFrame:
        weather = self._get_weather()

        result = pd.DataFrame(columns=['pv_power', 'dc_power'], index=weather.index).fillna(0)
        result.index.name = 'time'
        for cmpt in self.values():
            if cmpt.type == 'pv':
                result_pv = self._get_solar_yield(cmpt, weather)
                result[['pv_power', 'dc_power']] += result_pv[['pv_power', 'dc_power']].abs()

        return pd.concat([result, weather], axis=1)

    def _get_weather(self) -> pd.DataFrame:
        weather = self.weather.get()
        if 'precipitable_water' not in weather.columns or weather['precipitable_water'].sum() == 0:
            from pvlib.atmosphere import gueymard94_pw
            weather['precipitable_water'] = gueymard94_pw(weather['temp_air'], weather['relative_humidity'])
        if 'albedo' in weather.columns and weather['albedo'].sum() == 0:
            weather.drop('albedo', axis=1, inplace=True)

        solar_position = self._get_solar_position(weather.index)
        return pd.concat([weather, solar_position], axis=1)

    def _get_solar_position(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        data = pd.DataFrame(index=index)
        try:
            # TODO: use weather pressure for solar position
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
