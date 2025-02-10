# -*- coding: utf-8 -*-
"""
penguin.system
~~~~~~~~~~~~~~


"""

from __future__ import annotations

import datetime as dt
from typing import Optional

import lori
import pandas as pd
from lori import ChannelState, ComponentException, Configurations, WeatherUnavailableException
from penguin import Location, SolarSystem, Weather


class System(lori.System):
    # fmt: off
    POWER_EL:       str = "el_power"
    POWER_EL_IMP:   str = "el_import_power"
    POWER_EL_EXP:   str = "el_export_power"
    POWER_TH:       str = "th_power"
    POWER_TH_HT:    str = "th_ht_power"
    POWER_TH_DOM:   str = "th_dom_power"

    ENERGY_EL:      str = "el_energy"
    ENERGY_EL_IMP:  str = "el_import_energy"
    ENERGY_EL_EXP:  str = "el_export_energy"
    ENERGY_TH:      str = "th_energy"
    ENERGY_TH_HT:   str = "th_ht_energy"
    ENERGY_TH_DOM:  str = "th_dom_energy"
    # fmt: on

    _location: Optional[Location] = None

    def configure(self, configs: Configurations) -> None:
        super().configure(configs)
        if self.has_type(SolarSystem):
            from penguin.constants import COLUMNS

            self.data.add(key=SolarSystem.POWER_EST, name=COLUMNS[SolarSystem.POWER_EST], connector=None, type=float)

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

    def activate(self) -> None:
        super().activate()
        if self.has_type(SolarSystem):
            try:
                weather_channels = self.weather.data.channels.filter(lambda c: c.key in Weather.VALIDATED)
                if self.weather.forecast.is_enabled():
                    weather_channels += self.weather.forecast.data.channels.filter(lambda c: c.key in Weather.VALIDATED)
                self.data.register(self.predict_solar, *weather_channels, how="all", unique=False)

            except WeatherUnavailableException:
                pass

    # # noinspection PyShadowingBuiltins
    # def evaluate(self, **kwargs):
    #     from .evaluation import Evaluation
    #     eval = Evaluation(self)
    #     return eval(**kwargs)

    # noinspection PyShadowingBuiltins, PyUnresolvedReferences, PyUnusedLocal
    def predict_solar(self, weather: pd.DataFrame) -> pd.DataFrame:
        try:
            weather = self.weather.validate(weather)
            result = pd.DataFrame(data=0.0, index=weather.index, columns=[SolarSystem.POWER, SolarSystem.POWER_DC])
            result.index.name = "timestamp"
            for solar in self.get_all(SolarSystem):
                solar_result = solar.predict(weather)
                solar_columns = [SolarSystem.POWER, SolarSystem.POWER_DC]
                result[solar_columns] += solar_result[solar_columns].abs()

            solar_power_channel = self.data[SolarSystem.POWER_EST]
            solar_power = result[SolarSystem.POWER]
            if not solar_power.empty:
                solar_power_channel.set(solar_power.index[0], solar_power)
            else:
                solar_power_channel.state = ChannelState.NOT_AVAILABLE

            return result

        except ComponentException as e:
            self._logger.warning(f"Unable to predict system '{self.name}' PV: {str(e)}")

    # noinspection PyShadowingBuiltins, PyUnresolvedReferences
    def predict(
        self,
        start: pd.Timestamp | dt.datetime = None,
        end: pd.Timestamp | dt.datetime = None,
        **kwargs,
    ) -> pd.DataFrame:
        try:
            weather = self.weather.get(start, end, **kwargs)  # , validate=True, **kwargs)
            solar = self.predict_solar(weather)

            return pd.concat([solar, weather], axis="columns")

        except ComponentException as e:
            self._logger.warning(f"Unable to run system '{self.name}': {str(e)}")
