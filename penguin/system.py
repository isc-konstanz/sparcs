# -*- coding: utf-8 -*-
"""
penguin.system
~~~~~~~~~~~~~~


"""

from __future__ import annotations

from typing import Any, Optional

import lori
import pandas as pd
from lori import Channel, ChannelState, Configurations, Constant, Weather, WeatherUnavailableException
from lori.simulation import Result, Results
from lori.typing import TimestampType
from penguin import Location
from penguin.components import ElectricalEnergyStorage, SolarSystem
from penguin.components.weather import validate_meteo_inputs, validated_meteo_inputs


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
                key=constant,
                aggregate="mean",
                connector=None,
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

    def simulate(
        self,
        start: TimestampType,
        end: TimestampType,
        prior: Optional[pd.DataFrame] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        data = self.predict(start, end, **kwargs)

        if System.POWER_EL not in data.columns:
            if self.data.has_logged(System.POWER_EL, start=start, end=end):
                self._logger.debug(f"Reference {System.POWER_EL.name} will be as missing prediction.")
                data.insert(0, System.POWER_EL, self.data.from_logger([System.POWER_EL], start=start, end=end))
            else:
                self._logger.debug(f"Reference {System.POWER_EL.name} cannot be found.")

        data = self._simulate_solar(data, start, end, prior)
        data = self._simulate_storage(data, start, end, prior)

        return data.dropna(axis="columns", how="all")

    # noinspection PyUnusedLocal
    def _simulate_solar(
        self,
        data: pd.DataFrame,
        start: TimestampType,
        end: TimestampType,
        prior: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        if System.POWER_EL not in data.columns or not self.components.has_type(SolarSystem):
            return data

        for solar in self.components.get_all(SolarSystem):
            solar_column = solar.data[SolarSystem.POWER].column
            if not solar.data.has_logged(SolarSystem.POWER, start, end):
                # Solar System does not have a measured reference and will be simulated to effect residual power
                data[System.POWER_EL] -= data[solar_column]
        return data

    # noinspection PyUnusedLocal, PyUnresolvedReferences
    def _simulate_storage(
        self,
        data: pd.DataFrame,
        start: TimestampType,
        end: TimestampType,
        prior: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        if System.POWER_EL not in data.columns or not self.components.has_type(ElectricalEnergyStorage):
            return data

        columns = [
            ElectricalEnergyStorage.STATE_OF_CHARGE,
            ElectricalEnergyStorage.POWER_CHARGE,
        ]
        for ees in self.components.get_all(ElectricalEnergyStorage):
            if not ees.data.has_logged(columns, start, end):
                ees_columns = [ees.data[c].column for c in columns]
                ees_soc_column = ees.data[ElectricalEnergyStorage.STATE_OF_CHARGE].column
                ees_soc = prior.iloc[-1][ees_soc_column] if prior is not None else 50.0
                ees_result = ees.infer_soc(data, ees_soc)

                # EES does not have a measured reference and will be simulated to effect residual power
                data[System.POWER_EL] += ees_result[ElectricalEnergyStorage.POWER_CHARGE]
                data[ees_columns] = ees_result[columns]

        # TODO: Aggregate SoC and Power if more than one EES exist
        # def aggregate(column: str, how: str) -> None:
        #     for ees in self.get_all(ElectricalEnergyStorage):
        #         ees_columns = [ees.data[c].column for c in columns]
        #     data = pd.concat(ees_results, axis='columns').loc[:, [column]]
        #     if len(data.columns) > 1:
        #         if how == 'sum':
        #             data = data.sum(axis='columns')
        #         elif how == 'mean':
        #             data = data.mean(axis='columns')
        #     results.loc[:, column] = data
        #
        # aggregate(ElectricalEnergyStorage.STATE_OF_CHARGE, how='mean')
        # aggregate(ElectricalEnergyStorage.POWER_CHARGE, how='sum')

        return data

    def evaluate(self, results: Results) -> pd.DataFrame:
        predictions = results.data
        references = self.data.from_logger(start=results.start, end=results.end)

        columns = list(dict.fromkeys([*predictions.columns, *references.columns]))
        data = pd.DataFrame(columns=pd.MultiIndex.from_product([["predictions", "references"], columns]))
        data["predictions"] = predictions.reindex(columns=columns)
        data["references"] = references.reindex(columns=columns)
        data.dropna(axis="columns", how="all", inplace=True)

        self._evaluate_yield(results, data)
        self._evaluate_storage(results, data)
        self._evaluate_energy(results, data)
        self._evaluate_weather(results, data)

        if "references" in data.columns.get_level_values(0):
            errors = (data["predictions"] - data["references"]).dropna(axis="columns", how="all")
            errors.columns = pd.MultiIndex.from_product([["errors"], errors.columns])
            data = pd.concat([data, errors], axis="columns")
        return data

    # noinspection PyUnresolvedReferences
    def _evaluate_yield(self, results: Results, data: pd.DataFrame) -> None:
        if not self.components.has_type(SolarSystem) or SolarSystem.POWER not in data["predictions"].columns:
            return
        solar_simulated = False
        for solar in self.components.get_all(SolarSystem):
            solar_column = solar.data[SolarSystem.POWER].column
            solar_reference = solar.data.from_logger(start=results.start, end=results.end)
            if solar_reference.empty:
                solar_simulated = True
            else:
                data[("references", solar_column)] = solar_reference[SolarSystem.POWER]

        if solar_simulated:
            # One or more solar system does not have reference measurements.
            # The reference value does not correspond to the total prediction and should be dropped
            for column in [System.POWER_EL, SolarSystem.POWER]:
                if column in data["references"].columns:
                    data.drop(columns=[("references", column)], inplace=True)

        hours = pd.Series(data.index, index=data.index)
        hours = (hours - hours.shift(1)).bfill().dt.total_seconds() / 3600.0

        solar_kwp = sum(solar.power_max / 1000.0 for solar in self.components.get_all(SolarSystem))
        solar_power = data[("predictions", SolarSystem.POWER)]
        solar_energy = solar_power / 1000.0 * hours
        solar_energy.name = SolarSystem.ENERGY

        yield_months_file = results.dirs.tmp.joinpath("yield_months.png")
        try:
            from lori.io import plot

            plot_data = solar_energy.to_frame().groupby(data.index.month).sum()
            plot.bar(
                x=plot_data.index,
                y=SolarSystem.ENERGY,
                data=plot_data,
                xlabel="Month",
                ylabel="Energy [kWh]",
                title="Monthly Yield",
                colors=list(reversed(plot.COLORS)),
                file=str(yield_months_file),
            )

            # plot_data = pd.concat(
            #     [
            #         pd.Series(
            #             data=solar_power[solar_power.index.month == m]/1000.,
            #             name=calendar.month_name[m],
            #         ) for m in range(1, 13)
            #     ],
            #     axis='columns',
            # )
            # plot_data['hour'] = plot_data.index.hour + plot_data.index.minute/60.
            # plot_melt = plot_data.melt(id_vars='hour', var_name='Months')
            # plot.line(
            #     x='hour',
            #     y='value',
            #     data=plot_melt,
            #     xlabel='Hour of the Day',
            #     ylabel='Power [kW]',
            #     title='Yield Profile',
            #     hue='Months',
            #     colors=list(reversed(plot.COLORS)),
            #     file=str(yield_profiles_file),
            # )
        except ImportError:
            pass

        yield_specific = round((solar_energy / solar_kwp).sum(), 2)
        yield_energy = solar_energy.sum()
        yield_images = {
            "yield_months": yield_months_file,
        }
        results.append(Result.from_const(SolarSystem.YIELD_SPECIFIC, yield_specific, header="Yield"))
        results.append(Result.from_const(SolarSystem.YIELD_ENERGY, yield_energy, header="Yield", images=yield_images))

        if SolarSystem.POWER_DC in data["predictions"].columns:
            dc_energy = (data[("predictions", SolarSystem.POWER_DC)] / 1000.0 * hours).sum()
            results.append(Result.from_const(SolarSystem.YIELD_ENERGY_DC, dc_energy, header="Yield"))

    # noinspection PyUnresolvedReferences
    def _evaluate_storage(self, results: Results, data: pd.DataFrame) -> None:
        columns = [
            ElectricalEnergyStorage.STATE_OF_CHARGE,
            ElectricalEnergyStorage.POWER_CHARGE,
        ]
        if not self.components.has_type(ElectricalEnergyStorage):
            return

        hours = pd.Series(data.index, index=data.index)
        hours = (hours - hours.shift(1)).bfill().dt.total_seconds() / 3600.0

        ees_simulated = False
        for ees in self.components.get_all(ElectricalEnergyStorage):
            ees_columns = [ees.data[c].column for c in columns]
            ees_power_column = ees.data[ElectricalEnergyStorage.POWER_CHARGE].column
            if ees_power_column not in data["predictions"].columns:
                continue

            ees_power = data[("predictions", ees_power_column)]
            ees_reference = ees.data.from_logger(start=data.index[0], end=data.index[-1])
            if ees_reference.empty:
                ees_simulated = True
            else:
                data[("references", ees_columns)] = ees_reference[columns]

            ees_capacity = sum([ees.capacity for ees in self.components.get_all(ElectricalEnergyStorage)])
            ees_cycles = (ees_power.where(ees_power >= 0, other=0) / 1000 * hours).sum() / ees_capacity
            ees_cycles_name = ElectricalEnergyStorage.CYCLES.name.replace("EES", ees.name)

            results.add(ElectricalEnergyStorage.CYCLES, ees_cycles_name, ees_cycles, header="Battery Storage")

        if ees_simulated:
            # One or more solar system does not have reference measurements.
            # The reference value does not correspond to the total prediction and should be dropped
            for column in [System.POWER_EL, *columns]:
                if column in data["references"].columns:
                    data.drop(columns=[("references", column)], inplace=True)

    # noinspection PyMethodMayBeStatic
    def _evaluate_energy(self, results: Results, data: pd.DataFrame) -> None:
        if System.POWER_EL not in data["predictions"].columns:
            return
        hours = pd.Series(data.index, index=data.index)
        hours = (hours - hours.shift(1)).bfill().dt.total_seconds() / 3600.0

        active_power = data[("predictions", System.POWER_EL)]
        import_power = active_power.where(active_power >= 0, other=0)
        import_energy = (import_power / 1000 * hours).sum()
        export_power = active_power.where(active_power <= 0, other=0).abs()
        export_energy = (export_power / 1000 * hours).sum()

        results.add("grid_import", "Import [kWh]", import_energy, header="Grid")
        results.add("grid_export", "Export [kWh]", export_energy, header="Grid")
        results.add("grid_import_max", "Import Peak [W]", import_power.max(), header="Grid")
        results.add("grid_export_max", "Export Peak [W]", export_power.max(), header="Grid")

    # noinspection PyMethodMayBeStatic
    def _evaluate_weather(self, results: Results, data: pd.DataFrame) -> None:
        if not all(c in data["predictions"].columns for c in [Weather.GHI, Weather.DHI]):
            return
        hours = pd.Series(data.index, index=data.index)
        hours = (hours - hours.shift(1)).bfill().dt.total_seconds() / 3600.0

        ghi = (data[("predictions", Weather.GHI)] / 1000.0 * hours).sum()
        dhi = (data[("predictions", Weather.DHI)] / 1000.0 * hours).sum()

        results.add(Weather.GHI, f"{Weather.GHI.name} [kWh/m^2]", ghi, header="Weather")
        results.add(Weather.DHI, f"{Weather.DHI.name} [kWh/m^2]", dhi, header="Weather")
