# -*- coding: utf-8 -*-
"""
penguin.system
~~~~~~~~~~~~~~


"""

from __future__ import annotations

from typing import Optional

import lori
import pandas as pd
from lori import Channel, ChannelState, Configurations, Constant, Weather, WeatherUnavailableException
from lori.typing import TimestampType
from penguin import Location
from penguin.components import SolarSystem
from penguin.components.weather import validated_meteo_inputs, validate_meteo_inputs


class System(lori.System):
    POWER_EL = Constant(float, "el_power", "Electrical Power", "W")
    POWER_EL_EST = Constant(float, "el_est_power", "Estimate Electrical Power", "W")
    POWER_EL_IMP = Constant(float, "el_import_power", "Import Electrical Power", "W")
    POWER_EL_EXP = Constant(float, "el_export_power", "Export Electrical Power", "W")

    POWER_TH = Constant(float, "th_power", "Thermal Power", "W")
    POWER_TH_EST = Constant(float, "th_est_power", "Estimate Thermal Power", "W")
    POWER_TH_DOM = Constant(float, "th_dom_power", "Domestic Water Thermal Power", "W")
    POWER_TH_HT = Constant(float, "th_ht_power", "Heating Water Thermal Power", "W")

    ENERGY_EL = Constant(float, "el_energy", "Electrical Energy", "kWh")
    ENERGY_EL_IMP = Constant(float, "el_import_energy", "Import Electrical Energy", "kWh")
    ENERGY_EL_EXP = Constant(float, "el_export_energy", "Export Electrical Energy", "kWh")

    ENERGY_TH = Constant(float, "th_energy", "Thermal Energy", "kWh")
    ENERGY_TH_HT = Constant(float, "th_ht_energy", "Heating Water Thermal Energy", "kWh")
    ENERGY_TH_DOM = Constant(float, "th_dom_energy", "Domestic Water Thermal Energy", "kWh")

    _location: Optional[Location] = None

    def configure(self, configs: Configurations) -> None:
        super().configure(configs)

        def add_channel(constant: Constant, **custom) -> None:
            self.data.add(
                key = constant,
                aggregate = "mean",
                connector = None,
                **custom,
            )

        if self.components.has_type(SolarSystem):
            add_channel(SolarSystem.POWER_DC)
            add_channel(SolarSystem.POWER)
            add_channel(SolarSystem.POWER_EST)

        # TODO: Improve channel setup based on available components
        add_channel(System.POWER_EL)
        add_channel(System.POWER_EL_EST)
        add_channel(System.POWER_TH)
        add_channel(System.POWER_TH_EST)


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

    # noinspection PyUnresolvedReferences
    def activate(self) -> None:
        super().activate()
        try:
            self._register_weather(self.weather)
            self._register_weather(self.weather.forecast)

        except WeatherUnavailableException:
            pass

    def _register_weather(self, weather: Weather) -> None:
        if not weather.is_enabled():
            return

        weather_channels = []
        for input in validated_meteo_inputs:
            if input not in weather.data:
                weather.data.add(key=input, aggregate="mean", connector=None)
                continue
            weather_channels.append(weather.data[input])
        weather.data.register(self._on_weather_received, weather_channels, how="all", unique=False)

    def _on_weather_received(self, weather: pd.DataFrame) -> None:
        predictions = self._predict(weather)
        timestamp = predictions.index[0]

        def update_channel(channel: Channel, column: str) -> None:
            if column in predictions.columns:
                channel.set(timestamp, predictions[column])
            else:
                channel.state = ChannelState.NOT_AVAILABLE

        if self.components.has_type(SolarSystem):
            for solar in self.components.get_all(SolarSystem):
                solar_column = solar.data[SolarSystem.POWER].column
                update_channel(solar.data[SolarSystem.POWER_EST], solar_column)
            update_channel(self.data[SolarSystem.POWER_EST], SolarSystem.POWER)
        update_channel(self.data[System.POWER_EL_EST], System.POWER_EL)
        update_channel(self.data[System.POWER_TH_EST], System.POWER_TH)

    def _predict(self, weather: pd.DataFrame) -> pd.DataFrame:
        weather = validate_meteo_inputs(weather, self.location)
        predictions = pd.DataFrame(index=weather.index)
        predictions.index.name = Channel.TIMESTAMP

        if self.components.has_type(SolarSystem):
            solar_columns = [SolarSystem.POWER, SolarSystem.POWER_DC]
            predictions[solar_columns] = 0.0
            for solar in self.components.get_all(SolarSystem):
                solar_column = solar.data[SolarSystem.POWER].column
                solar_prediction = solar.predict(weather)
                predictions[solar_column] = solar_prediction[SolarSystem.POWER]
                predictions[solar_columns] += solar_prediction[solar_columns].fillna(0)

        return predictions

    def predict(
        self,
        start: TimestampType = None,
        end: TimestampType = None,
        **kwargs,
    ) -> pd.DataFrame:
        weather = self.weather.get(start, end, **kwargs)
        predictions = self._predict(weather)

        return pd.concat([predictions, weather], axis="columns")
