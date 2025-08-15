# -*- coding: utf-8 -*-
"""
penguin.system
~~~~~~~~~~~~~~


"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Optional

import numpy as np

from filterpy.kalman import KalmanFilter # MIT licence

import lori
import pandas as pd
from lori import Channel, ChannelState, Configurations, Constant, ResourceUnavailableException
from lori.components.weather import Weather, WeatherUnavailableException
from lori.components.tariff import Tariff, TariffUnavailableException
from lori.simulation import Result, Results
from lori.typing import TimestampType
from lori.components import Tariff, TariffUnavailableException
from lori.components.tariff.entsoe import EntsoeProvider
from penguin import Location
from penguin.components import ElectricalEnergyStorage, SolarSystem

from penguin.components.weather import validate_meteo_inputs, validated_meteo_inputs

from penguin.components.control.predictive import Optimization
from penguin.components.control.predictive.problems.grid_cost import GridCostProblem

# from lori.data.forecast import get_forecast


class System(lori.System):
    POWER_EL = Constant(float, "el_power", "Electrical Power", "W")
    POWER_EL_EST = Constant(float, "el_est_power", "Estimate Electrical Power", "W")
    POWER_EL_CON = Constant(float, "el_cons_power", "Consumption Electrical Power", "W")
    POWER_EL_IMP = Constant(float, "el_import_power", "Import Electrical Power", "W")
    POWER_EL_EXP = Constant(float, "el_export_power", "Export Electrical Power", "W")

    POWER_TH = Constant(float, "th_power", "Thermal Power", "W")
    POWER_TH_EST = Constant(float, "th_est_power", "Estimate Thermal Power", "W")
    POWER_TH_DOM = Constant(float, "th_dom_power", "Domestic Water Thermal Power", "W")
    POWER_TH_HT = Constant(float, "th_ht_power", "Heating Water Thermal Power", "W")

    ENERGY_EL = Constant(float, "el_energy", "Electrical Energy", "kWh")
    ENERGY_EL_CON = Constant(float, "el_cons_energy", "Consumed Electrical Energy", "W")
    ENERGY_EL_IMP = Constant(float, "el_import_energy", "Import Electrical Energy", "kWh")
    ENERGY_EL_EXP = Constant(float, "el_export_energy", "Export Electrical Energy", "kWh")

    ENERGY_TH = Constant(float, "th_energy", "Thermal Energy", "kWh")
    ENERGY_TH_HT = Constant(float, "th_ht_energy", "Heating Water Thermal Energy", "kWh")
    ENERGY_TH_DOM = Constant(float, "th_dom_energy", "Domestic Water Thermal Energy", "kWh")

    _location: Optional[Location] = None

    def has_tariff(self) -> bool:
        return self.components.has_type(Tariff)

    # noinspection PyTypeChecker
    @property
    def tariff(self) -> Tariff:
        tariff = self.components.get_first(Tariff)
        if tariff is None:
            raise TariffUnavailableException(f"System '{self.name}' has no tariff configured")
        return tariff

    def configure(self, configs: Configurations) -> None:
        super().configure(configs)

        def add_channel(constant: Constant, aggregate: str = "mean", **custom) -> None:
            self.data.add(
                key=constant,
                aggregate=aggregate,
                connector=None,
                **custom,
            )

        # TODO: Improve channel setup based on available components
        add_channel(System.POWER_EL)
        add_channel(System.POWER_EL_EST)
        add_channel(System.POWER_TH)
        add_channel(System.POWER_TH_EST)

        if self.components.has_type(SolarSystem):
            add_channel(SolarSystem.POWER_DC)
            add_channel(SolarSystem.POWER)
            add_channel(SolarSystem.POWER_EST)
            add_channel(System.POWER_EL_CON)

        if self.components.has_type(ElectricalEnergyStorage):
            add_channel(ElectricalEnergyStorage.POWER_CHARGE)
            add_channel(ElectricalEnergyStorage.STATE_OF_CHARGE, aggregate="last")

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

        if self.components.has_type(SolarSystem):
            power_channels = [
                self.data[SolarSystem.POWER],
                self.data[System.POWER_EL_CON],
                self.data[System.POWER_EL],
            ]
            self.data.register(self._on_power_received, power_channels, how="any", unique=False)

        try:
            pass
            #self._register_tariff(self.components.get_first(EntsoeProvider))

        except TariffUnavailableException:
            pass

    # noinspection PyShadowingBuiltins
    def _register_weather(self, weather: Weather) -> None:
        if not weather.is_enabled():
            return

        weather_channels = []
        for input in validated_meteo_inputs:
            if input not in weather.data:
                weather.data.add(key=input, aggregate="mean", connector=None)
                continue
            weather_channel = weather.data[input]
            if weather_channel.has_connector():
                weather_channels.append(weather_channel)
        weather.data.register(self._on_weather_received, weather_channels, how="all", unique=False)

    def _on_weather_received(self, weather: pd.DataFrame) -> None:
        predictions = self._predict_solar(weather.dropna(axis="columns"))
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

    def _register_tariff(self, tariff: Tariff) -> None:
        if not tariff.is_enabled():
            return

        #TODO: Validate tariff inputs, not weather inputs
        #      also rename input (buildin function)
        tariff_channels = []
        for input in validated_meteo_inputs:
            if input not in tariff.data:
                tariff.data.add(key=input, aggregate="mean", connector=None)
                continue
            tariff_channels.append(tariff.data[input])
        tariff.data.register(self._on_weather_received, tariff_channels, how="all", unique=False)

    def _on_power_received(self, data: pd.DataFrame) -> None:
        if data[System.POWER_EL_CON].dropna().empty:
            power = data.loc[:, System.POWER_EL].dropna()
            power += data.loc[power.index, SolarSystem.POWER].fillna(0)
            power.name = System.POWER_EL_CON
            self.data[System.POWER_EL_CON].set(power.index[0], power)
        elif data[System.POWER_EL].dropna().empty:
            power = data.loc[:, System.POWER_EL_CON].dropna()
            power -= data.loc[power.index, SolarSystem.POWER].fillna(0)
            power.name = System.POWER_EL
            self.data[System.POWER_EL].set(power.index[0], power)

    def _on_tariff_received(self, tariff: pd.DataFrame) -> None:
        predictions = self._predict(tariff)
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





    def predict(
        self,
        start: TimestampType = None,
        end: TimestampType = None,
        prior: Optional[pd.DataFrame] = None,
        **kwargs,
    ) -> pd.DataFrame:
        real_end = end

        has_opti = self.components.has_type(Optimization)
        if has_opti:
            optimization_problem = self.components.get_first(Optimization)
            optimization_end = start + pd.Timedelta(seconds=optimization_problem.total_duration)
            if end > optimization_end:
                self._logger.warning(f"Simulation duration is greater than optimization duration")
            end = optimization_end



        has_solar = self.components.has_type(SolarSystem)
        if has_solar:
            weather = self.weather.get(start, end, **kwargs)
            solar = self._predict_solar(weather)
        else:
            weather = pd.DataFrame(index=pd.date_range(start, end, freq="1min"))
            solar = weather.copy()



        # hacky tariff implementation
        tariff_component: Tariff = self.components.get_first(Tariff)
        if tariff_component is None:
            # df with index start:end, columns: [Tariff.PRICE_IMPORT, Tariff.PRICE_EXPORT]
            tariff = pd.DataFrame(index=pd.date_range(start - pd.Timedelta(hours=1), end, freq="1min"))
            tariff[Tariff.PRICE_IMPORT] = 25.0
            tariff[Tariff.PRICE_EXPORT] = -5.0

        tariff = tariff_component.get(start - pd.Timedelta(hours=1), end, **kwargs)
        tariff = tariff.resample("1min").ffill()
        tariff[Tariff.PRICE_EXPORT] = -5
        tariff_mode = tariff_component.configs.get("mode", default=None)
        if tariff_mode == "static":
            tariff[Tariff.PRICE_IMPORT] = 25.0
        elif tariff_mode == "sinus":
            tariff[Tariff.PRICE_IMPORT] = 25.0 + 10 * np.sin(
                (tariff.index - tariff.index[0]).total_seconds() / (20 * 3600) * 2 * np.pi
            )
        elif tariff_mode == "2025":
            tariff[Tariff.PRICE_IMPORT] = (tariff[Tariff.PRICE_IMPORT] - 20) * 4 + 20


        # solve mpc if available
        if has_opti:
            required_components = self.components.get_all(*self.components.get_first(Optimization).required_components)
            if not required_components:
                opti = pd.DataFrame(index=pd.date_range(start, end, freq="1min"))
            else:

                # hacky forecasting #TODO: remove this when predictors are ready
                #from lori.data.forecast import get_forecast
                #el_power_forecast = get_forecast(
                #    self,
                #    System.POWER_EL,
                #    start=start,
                #    end=end,
                #    method="real",
                #    #method="persistence_3weeks",
                #)
                # upsample load predictions
                #real = self.data.from_logger(["el_power"], start=start, end=end)["el_power"].copy()


                el_power_forecast = self.residual_forecast(start, end)
                el_power_forecast = el_power_forecast.resample("1min").ffill()
                # is not has logged solar power, subtract that power from the forecast
                for solar_system in self.components.get_all(SolarSystem):
                    if not solar_system.data.has_logged(SolarSystem.POWER, start, end):
                        solar_column = solar_system.data[SolarSystem.POWER].column
                        el_power_forecast["forecast"] -= solar[solar_column]
                        #real -= solar[solar_column]


                # plot real vs forecast
                # import matplotlib.pyplot as plt
                # plt.figure(figsize=(10, 5))
                # plt.plot(real.index, real, label="Real Power", color="blue")
                # plt.plot(el_power_forecast.index, el_power_forecast["forecast"], label="Forecast Power", color="orange")
                # plt.xlabel("Time")
                # plt.ylabel("Power (W)")
                # plt.legend()
                # plt.show()


                opti_input = pd.concat([solar, tariff, el_power_forecast], axis="columns")
                opti_input.dropna(axis="index", how="any", inplace=True)

                opti_component: GridCostProblem = self.components.get_first(GridCostProblem)
                opti = opti_component.solve(opti_input, start, prior=prior)
                opti = opti.resample("1min").ffill()

        else:
            opti = pd.DataFrame(index=pd.date_range(start, end, freq="1min"))

        pass



        data_df = (
            pd.concat([weather, solar, opti, tariff], axis="columns")
            .dropna(axis="index", how="any")
            .sort_index()

            [:real_end]
        )


        return data_df


    def _predict_solar(self,
        weather: pd.DataFrame,
    ) -> pd.DataFrame:

        weather = validate_meteo_inputs(weather, self.location)
        predictions = pd.DataFrame(index=weather.index)
        predictions.index.name = Channel.TIMESTAMP

        if self.components.has_type(SolarSystem):
            columns = [SolarSystem.POWER, SolarSystem.POWER_DC]
            predictions[columns] = 0.0
            for solar_system in self.components.get_all(SolarSystem):
                system_column = solar_system.data[SolarSystem.POWER].column
                system_prediction = solar_system.predict(weather)
                predictions[system_column] = system_prediction[SolarSystem.POWER]
                predictions[columns] += system_prediction[columns].fillna(0)

        return predictions


    def simulate(
        self,
        start: TimestampType,
        end: TimestampType,
        prior: Optional[pd.DataFrame] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        data = self.predict(start, end, prior, **kwargs)

        if System.POWER_EL not in data.columns:
            if self.data.has_logged(System.POWER_EL, start=start, end=end):
                self._logger.debug(f"Reference {System.POWER_EL.name} will be as missing prediction.")
                data.insert(0, System.POWER_EL, self.data.from_logger([System.POWER_EL], start=start, end=end))
            else:
                self._logger.debug(f"Reference {System.POWER_EL.name} cannot be found.")

        data = self._simulate_solar(data, start, end, prior)
        data = self._simulate_storage(data, start, end, prior)

        data = data.dropna(axis="columns", how="all")
        data = data.dropna(axis="index", how="any")
        
        if Tariff.PRICE_EXPORT in data.columns:
            data.drop(columns=[Tariff.PRICE_EXPORT], inplace=True)
        return data
    
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
                # Solar System does not have a measured reference and will be subtracted from residual power
                data[System.POWER_EL] -= data[solar_column]
                pass
        return data

    # noinspection PyUnusedLocal
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
        if ElectricalEnergyStorage.POWER_CHARGE not in data.columns:
            data[ElectricalEnergyStorage.POWER_CHARGE] = 0

        total_capacity = 0
        total_energy = pd.Series(index=data.index, data=0)

        for ees in self.components.get_all(ElectricalEnergyStorage):
            ees_data = ees.data.from_logger(columns, start, end)
            if ees_data.empty:
                ees_columns = [ees.data[c].column for c in columns]
                ees_soc_column = ees.data[ElectricalEnergyStorage.STATE_OF_CHARGE].column
                ees_soc = prior.iloc[-1][ees_soc_column] if prior is not None else 50.0
                ees_data = ees.predict(data=data, soc=ees_soc)
                ees_power = ees_data[ElectricalEnergyStorage.POWER_CHARGE]

                data[ees_columns] = ees_data[columns]
                data[ElectricalEnergyStorage.POWER_CHARGE] += ees_power

                # EES does not have a measured reference and will be added to residual power
                data[System.POWER_EL] += ees_power

            total_capacity += ees.capacity
            total_energy += ees_data[ElectricalEnergyStorage.STATE_OF_CHARGE] / 100 * ees.capacity

        data[ElectricalEnergyStorage.STATE_OF_CHARGE] = total_energy / total_capacity * 100

        return data

    def evaluate(self, results: Results) -> pd.DataFrame:
        predictions = deepcopy(results.data)
        predictions.columns = pd.MultiIndex.from_product([["predictions"], predictions.columns])
        references = self.data.from_logger(start=results.start, end=results.end).dropna(axis="columns", how="all")
        references.columns = pd.MultiIndex.from_product([["references"], references.columns])
        data = pd.concat([predictions, references], axis="columns")

        self._evaluate_yield(results, data)
        self._evaluate_storage(results, data)
        self._evaluate_system(results, data)
        self._evaluate_weather(results, data)
        self._evaluate_costs(results, data)


        if "references" in data.columns.get_level_values(0):
            errors = (data["predictions"] - data["references"]).dropna(axis="columns", how="all")
            errors.columns = pd.MultiIndex.from_product([["errors"], errors.columns])
            data = pd.concat([data, errors], axis="columns")
        return data

    def _plot(self, data: pd.DataFrame) -> None:
        from penguin.simulation.report import plots
        pass


        del plots

    def _evaluate_yield(self, results: Results, data: pd.DataFrame) -> None:
        if not self.components.has_type(SolarSystem) or SolarSystem.POWER not in data["predictions"].columns:
            return

        solar_simulated = False
        solar_columns = [SolarSystem.POWER]
        for solar in self.components.get_all(SolarSystem):
            solar_column = solar.data[SolarSystem.POWER].column
            solar_columns.append(solar_column)
            solar_reference = solar.data.from_logger(start=results.start, end=results.end)
            if solar_reference.empty:
                solar_simulated = True
            else:
                data[("references", solar_column)] = solar_reference[SolarSystem.POWER]

        if solar_simulated and "references" in data.columns.get_level_values(0):
            # One or more solar system does not have reference measurements.
            # The reference value does not correspond to the total prediction and should be dropped
            for column in [System.POWER_EL, SolarSystem.POWER]:
                if column in data["references"].columns:
                    data.drop(columns=[("references", column)], inplace=True)

        data = data.copy()
        data = data.drop(columns=[("predictions", col) for col in solar_columns if col in data["references"].columns])
        data = data.droplevel(0, axis=1)

        solar_kwps = pd.DataFrame([{solar.id: solar.power_max for solar in self.components.get_all(SolarSystem)}])
        solar_kwps["total"] = solar_kwps.sum(axis=1)
        print("")
        print("kWp of solar systems:")
        print(solar_kwps.iloc[0])

        solar_kwp = solar_kwps["total"].iloc[0]

        data["time"] = data.index
        data["hours"] =  (data["time"] - data["time"].shift(1)).bfill().dt.total_seconds() / 3600.0

        solar_powers = data[solar_columns]
        solar_energies = (solar_powers / 1000.0).multiply(data["hours"], axis=0)
        solar_energies.columns = [col.replace("power", "energy") for col in solar_energies.columns]
        solar_yield = solar_energies.sum(axis=0)
        print("")
        print("yeald per year:")
        print(solar_yield)

        # percentage against pv_energy column
        share = solar_yield / solar_yield["pv_energy"] * 100
        print("")
        print("share of total yield:")
        print(share)

        solar_energy = solar_energies[SolarSystem.ENERGY]

        plot_data = pd.concat([solar_energies, solar_powers], axis="columns")

        from penguin.simulation.report.plots import plot_yield
        plot_yield(plot_data)

        yield_months_file = results.dirs.tmp.joinpath("yield_months.png")
        yield_hours_file = results.dirs.tmp.joinpath("yield_hours.png")
        yield_hours_stats_file = results.dirs.tmp.joinpath("yield_hours_stats.png")
        # power_avg_week_file = results.dirs.tmp.joinpath("power_avg_week.png")
        # try:
        #     from lori.io import plot
        #
        #
        #     # Monthly Yield
        #     plot_data = solar_energys.groupby(data.index.month).sum()
        #     plot.bar(
        #         x=plot_data.index,
        #         y=SolarSystem.ENERGY,
        #         data=plot_data,
        #         xlabel="Month",
        #         ylabel="Energy [kWh]",
        #         title="Monthly Yield",
        #         colors=list(reversed(plot.COLORS)),
        #         file=str(yield_months_file),
        #     )
        #
        #     # Hourly Yield
        #     plot_data = solar_energy.to_frame().groupby(data.index.hour).mean()
        #     plot.bar(
        #         x=plot_data.index,
        #         y=SolarSystem.ENERGY,
        #         data=plot_data,
        #         xlabel="Hour of day",
        #         ylabel="Power [kW]",
        #         title="Hourly Yield Mean",
        #         colors=list(reversed(plot.COLORS)),
        #         file=str(yield_hours_file),
        #     )
        #
        #     # Hourly Yield Boxplot
        #     plot_data = solar_energy.to_frame()
        #     plot_data["Hour"] = plot_data.index.hour
        #     plot_data["Month"] = plot_data.index.month_name()
        #     plot.quartiles(
        #         x=plot_data["Hour"],
        #         y=SolarSystem.ENERGY,
        #         data=plot_data,
        #         xlabel="Hour of day",
        #         ylabel="Power [kW]",
        #         title="Hourly Yield Statistics",
        #         colors=list(reversed(plot.COLORS)),
        #         file=str(yield_hours_stats_file),
        #         hue="Month",
        #         width=48,
        #     )
        #
        #     plot_data["time_of_week"] = plot_data.index.dayofweek + plot_data.index.hour / 24.0 + plot_data.index.minute / 1440.0
        #     plot.line(
        #         x="time_of_week",
        #         y=SolarSystem.ENERGY,
        #         data=plot_data,
        #         xlabel="Week",
        #         ylabel="Power [kW]",
        #         title="Average Power per Week",
        #         file=str(power_avg_week_file),
        #         #percentile=100,
        #         #hue="Month"
        #     )
        #
        #     # TODO: safe for report, show only total?
        #     summary_stats = plot_data.groupby(['Month', 'Hour'])[SolarSystem.ENERGY].agg(['mean', 'std'])
        #     #print(summary_stats)
        #
        #     # plot_data = pd.concat(
        #     #     [
        #     #         pd.Series(
        #     #             data=solar_power[solar_power.index.month == m]/1000.,
        #     #             name=calendar.month_name[m],
        #     #         ) for m in range(1, 13)
        #     #     ],
        #     #     axis='columns',
        #     # )
        #     # plot_data['hour'] = plot_data.index.hour + plot_data.index.minute/60.
        #     # plot_melt = plot_data.melt(id_vars='hour', var_name='Months')
        #     # plot.line(
        #     #     x='hour',
        #     #     y='value',
        #     #     data=plot_melt,
        #     #     xlabel='Hour of the Day',
        #     #     ylabel='Power [kW]',
        #     #     title='Yield Profile',
        #     #     hue='Months',
        #     #     colors=list(reversed(plot.COLORS)),
        #     #     file=str(yield_profiles_file),
        #     # )
        # except ImportError:
        #     pass

        yield_specific = round((solar_energy / solar_kwp).sum(), 2)
        yield_energy = solar_energy.sum()
        yield_images = {
            "yield_months": yield_months_file,
            "yield_hours": yield_hours_file,
            "yield_hours_stats": yield_hours_stats_file,
        }
        results.append(Result.from_const(SolarSystem.YIELD_SPECIFIC, yield_specific, header="Yield"))
        results.append(Result.from_const(SolarSystem.YIELD_ENERGY, yield_energy, header="Yield", images=yield_images))

        if SolarSystem.POWER_DC in data.columns:
            dc_energy = (data[SolarSystem.POWER_DC] / 1000.0 * data["hours"]).sum()
            results.append(Result.from_const(SolarSystem.YIELD_ENERGY_DC, dc_energy, header="Yield"))

    def _evaluate_storage(self, results: Results, data: pd.DataFrame) -> None:
        if not self.components.has_type(ElectricalEnergyStorage):
            return

        columns = [
            ElectricalEnergyStorage.STATE_OF_CHARGE,
            ElectricalEnergyStorage.POWER_CHARGE,
        ]
        ees_simulated = False
        for ees in self.components.get_all(ElectricalEnergyStorage):
            ees_columns = [ees.data[c].column for c in columns]
            ees_reference = ees.data.from_logger(start=results.start, end=results.end)
            if ees_reference.empty:
                ees_simulated = True
            else:
                data[("references", ees_columns)] = ees_reference[SolarSystem.POWER]

        if ees_simulated and "references" in data.columns.get_level_values(0):
            # One or more solar system does not have reference measurements.
            # The reference value does not correspond to the total prediction and should be dropped
            for column in [System.POWER_EL, *columns]:
                try:
                    if column in data["references"].columns:
                        data.drop(columns=[("references", column)], inplace=True)
                except KeyError:
                    # "references" is not in the columns...
                    pass

        hours = pd.Series(data.index, index=data.index)
        hours = (hours - hours.shift(1)).bfill().dt.total_seconds() / 3600.0

        ees_soc = data[("predictions", ElectricalEnergyStorage.STATE_OF_CHARGE)]
        ees_power = data[("predictions", ElectricalEnergyStorage.POWER_CHARGE)]
        ees_capacity = sum(ees.capacity for ees in self.components.get_all(ElectricalEnergyStorage))
        ees_cycles = (ees_power.where(ees_power >= 0, other=0) / 1000 * hours).sum() / ees_capacity

        results.add("ees_cycles", "EES Cycles", ees_cycles, header="Storage")
        results.add("ees_soc_min", "EES SoC Minimum [%]", ees_soc.min(), header="Storage")

        ees_soc.name = ElectricalEnergyStorage.STATE_OF_CHARGE
        ees_power.name = ElectricalEnergyStorage.POWER_CHARGE
        ees_power_file = results.dirs.tmp.joinpath("ees_power_hours.png")
        ees_soc_file = results.dirs.tmp.joinpath("ees_soc_hours.png")
        try:
            from lori.io import plot

            # Hourly Power
            plot_data = ees_power.to_frame().groupby(data.index.hour).mean()
            plot.bar(
                x=plot_data.index,
                y=ElectricalEnergyStorage.POWER_CHARGE,
                data=plot_data,
                xlabel="Hour of day",
                ylabel="Power [kW]",
                title="Hourly Electrical Energy Storage Power Mean",
                file=str(ees_power_file),
            )

            # Hourly Soc
            plot_data = ees_soc.to_frame()
            plot_data["time_of_day"] = plot_data.index.hour + plot_data.index.minute / 60.0
            plot_data["Hour"] = plot_data.index.hour
            plot_data["Month"] = plot_data.index.month_name()
            plot.line(
                x="time_of_day",
                y=ElectricalEnergyStorage.STATE_OF_CHARGE,
                data=plot_data,
                xlabel="Hour of day",
                ylabel="SoC [%]",
                title="Hourly Electrical Energy Storage SoC Mean",
                file=str(ees_soc_file),
                #percentile=20,
                ylim=(0, 100),
                hue="Month"
            )

        except ImportError:
            pass



    # noinspection PyMethodMayBeStatic
    def _evaluate_system(self, results: Results, data: pd.DataFrame) -> None:
        if System.POWER_EL not in data["predictions"].columns:
            return
        
        data = data.copy()
        data = data.resample("15min").mean()
        
        hours = pd.Series(data.index, index=data.index)
        hours = (hours - hours.shift(1)).bfill().dt.total_seconds() / 3600.0

        active_power = data[("predictions", System.POWER_EL)]
        import_power = active_power.where(active_power >= 0, other=0)
        import_energy = import_power / 1000 * hours
        export_power = active_power.where(active_power <= 0, other=0).abs()
        export_energy = export_power / 1000 * hours

        results.add("grid_export_max", "Export Peak [W]", export_power.max(), header="Grid", order=10)
        results.add("grid_import_max", "Import Peak [W]", import_power.max(), header="Grid", order=10)
        results.add("grid_export", "Export [kWh]", export_energy.sum(), header="Grid", order=10)
        results.add("grid_import", "Import [kWh]", import_energy.sum(), header="Grid", order=10)

        if SolarSystem.POWER in data["predictions"].columns:
            solar_power = data[("predictions", SolarSystem.POWER)]
            solar_energy = solar_power / 1000 * hours

            cons_energy = import_energy + solar_energy - export_energy
            cons_self = (solar_energy - export_energy).sum() / solar_energy.sum() * 100
            suff_self = (1 - (import_energy.sum() / cons_energy.sum())) * 100

            results.add("consumption", "Energy [kWh]", cons_energy.sum(), header="Load", order=10)
            results.add("self_consumption", "Self-Consumption [%]", cons_self, header="Consumption", order=10)
            results.add("self_sufficiency", "Self-Sufficiency [%]", suff_self, header="Consumption", order=10)

        try:
            # import_peak_energy = import_energy[import_power >= import_power.max()]
            # import_peak_time = import_peak_energy.index.time.min()
            # import_peak_date = import_peak_energy[import_peak_energy.index.time == import_peak_time].index.date[0]
            # self._plot_system(
            #     data["predictions"][data.index.date == import_peak_date],
            #     title="Day with earliest Peak",
            #     file=str(results.dirs.tmp.joinpath("power_peak.png")),
            #     width=16,
            # )
            #from penguin.simulation.report.plots import plot_system

            import_week_energy = import_energy.groupby(import_energy.index.isocalendar().week).sum()
            import_week_energy_max = import_week_energy[import_week_energy == import_week_energy.max()].index[0]
            #plot_system(
            #    data["predictions"][data.index.isocalendar().week == import_week_energy_max],
            #    title="Week with highest Grid Import",
            #    file=str(results.dirs.tmp.joinpath("week_max_import.png")),
            #)

            self._plot_system(
                data["predictions"][data.index.isocalendar().week == import_week_energy_max],
                title="Week with highest Grid Import",
                file=str(results.dirs.tmp.joinpath("week_max_import.png")),
            )

            if self.components.has_type(SolarSystem):
                solar_power = data[("predictions", SolarSystem.POWER)]
                solar_energy = solar_power / 1000 * hours

                cons_self = solar_energy - export_energy
                cons_self_week_energy = cons_self.groupby(cons_self.index.isocalendar().week).sum()
                cons_self_week_energy_max = cons_self_week_energy[
                    cons_self_week_energy == cons_self_week_energy.max()
                ].index[0]
                self._plot_system(
                    data["predictions"][data.index.isocalendar().week == cons_self_week_energy_max],
                    title="Week with highest Self-Consumption",
                    file=str(results.dirs.tmp.joinpath("week_max_self-cons.png")),
                )

        except ImportError:
            pass

    # noinspection PyMethodMayBeStatic
    def _evaluate_weather(self, results: Results, data: pd.DataFrame) -> None:
        if not all(c in data["predictions"].columns for c in [Weather.GHI, Weather.DHI]):
            return
        hours = pd.Series(data.index, index=data.index)
        hours = (hours - hours.shift(1)).bfill().dt.total_seconds() / 3600.0

        ghi = (data[("predictions", Weather.GHI)] / 1000.0 * hours).sum()
        dhi = (data[("predictions", Weather.DHI)] / 1000.0 * hours).sum()

        results.add(Weather.GHI, f"{Weather.GHI.name} [kWh/m²]", ghi, header="Weather")
        results.add(Weather.DHI, f"{Weather.DHI.name} [kWh/m²]", dhi, header="Weather")

    def _evaluate_costs(self, results: Results, data: pd.DataFrame) -> None:
        dynamic = True
        if not self.components.has_type(Tariff):
            pass
            #return

            dynamic = False

        data = data.copy()

        hours = pd.Series(data.index, index=data.index)
        hours = (hours - hours.shift(1)).bfill().dt.total_seconds() / 3600.0

        power = data[("predictions", System.POWER_EL)]
        energy = power * hours / 1000.0  # kWh
        neg_energy = energy.where(energy < 0, other=0)
        pos_energy = energy.where(energy > 0, other=0)
        price_import = data[("predictions", Tariff.PRICE_IMPORT)]
        price_export = -5.0
        pos_cost = (pos_energy * price_import / 100.0).sum()
        neg_cost = (neg_energy * price_export / 100.0).sum()

        peak_power = power.resample("15min").mean().max()
        peak_price = 137 # €/kW
        peak_cost = peak_power * peak_price / 1000.0 # €


        results.add(Tariff.PRICE_IMPORT, "Import Price [ct/kWh]", price_import.mean(), header="Tariff")
        results.add(Tariff.PRICE_EXPORT, "Export Price [ct/kWh]", price_export, header="Tariff")
        results.add("tariff_dynamic", "Dynamic Tariff", "y" if dynamic else "n", header="Tariff")
        results.add("tariff_costs_import", "Tariff Costs [€]", pos_cost, header="Tariff")
        results.add("tariff_costs_export", "Tariff Revenue [€]", neg_cost, header="Tariff")
        results.add("tariff_costs_peak", "Tariff Peak Costs [€]", peak_cost, header="Tariff")
        results.add("tariff_costs_total", "Tariff Total Costs [€]", pos_cost - neg_cost + peak_cost, header="Tariff")




    # noinspection PyTypeChecker
    def _plot_system(
        self,
        data: pd.DataFrame,
        title: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        show: bool = False,
        file: str = None,
    ) -> None:
        # Ignore this error, as pandas implements its own matplotlib converters for handling datetime or period values.
        # When seaborn and pandas plots are mixed, converters may conflict and this warning is shown.
        import warnings

        import matplotlib.dates as dates
        import matplotlib.pyplot as plt
        import seaborn as sns

        from lori.io import plot

        data = data.copy().resample("15min").mean()

        warnings.filterwarnings(
            "ignore",
            message="This axis already has a converter set and is updating to a potentially incompatible converter",
        )

        columns_power = [System.POWER_EL]

        # TODO: Replace with tariff component constants
        has_tariff = Tariff.PRICE_IMPORT in data.columns
        has_solar = self.components.has_type(SolarSystem)
        if has_solar:
            columns_power.append(SolarSystem.POWER)
        has_ees = self.components.has_type(ElectricalEnergyStorage)
        if has_ees:
            columns_power.append(ElectricalEnergyStorage.POWER_CHARGE)
        if "grid_expected" in data.columns:
            columns_power.append("grid_expected")
        if "grid_solution" in data.columns:
            columns_power.append("grid_solution")

        data_power = deepcopy(data[columns_power])
        data_power /= 1000

        if width is None:
            width = plot.WIDTH
        if height is None:
            height = plot.HEIGHT
        figure, ax_power = plt.subplots(figsize=(width / plot.INCH, height / plot.INCH), dpi=120)
        axes = [ax_power]

        sns.lineplot(
            data_power[System.POWER_EL],
            linewidth=0.25,
            color="#004f9e",
            label="_hidden",
            ax=axes[0],
        )
        if has_ees:
            ax_soc = axes[0].twinx()
            axes.append(ax_soc)

            sns.lineplot(
                data_power[ElectricalEnergyStorage.POWER_CHARGE],
                linewidth=0.25,
                color="#30a030",
                label="_hidden",
                ax=ax_power,
            )
            # sns.lineplot(
            #     data_power["grid_expected"],
            #     linewidth=0.5,
            #     color="#00a0F0",
            #     label="_hidden",
            #     ax=ax_power,
            # )
            # sns.lineplot(
            #     data_power["grid_solution"],
            #     linewidth=0.5,
            #     color="#0060B0",
            #     label="_hidden",
            #     ax=ax_power,
            # )
            sns.lineplot(
                data[ElectricalEnergyStorage.STATE_OF_CHARGE],
                linewidth=1,
                color="#037003",
                label="Battery State",
                ax=ax_soc,
            )

            data_ref = data_power[System.POWER_EL] - data_power[ElectricalEnergyStorage.POWER_CHARGE]
            data_ref.plot.area(
                stacked=False,
                label="_hidden",
                color={"_hidden": "#dddddd"},
                linewidth=0,
                alpha=0.75,
                ax=ax_power,
            )

            ax_soc.set_ylim(-1, 119)
            ax_soc.yaxis.set_label_text("State of Charge [%]")
            if has_tariff:
                ax_soc.legend(ncol=1, loc="upper right", bbox_to_anchor=(0.84, 1), frameon=False)
            else:
                ax_soc.legend(ncol=1, loc="upper right", frameon=False)

        if has_solar:
            data_power[SolarSystem.POWER].plot.area(
                stacked=False,
                label="PV Generation",
                color={"PV Generation": "#ffeb9b"},
                linewidth=0,
                alpha=0.75,
                ax=ax_power,
            )

        data_power[System.POWER_EL].plot.area(
            stacked=False,
            label="Residual Load",
            alpha=0.25,
            ax=ax_power,
        )

        if has_ees:
            data_power[ElectricalEnergyStorage.POWER_CHARGE].plot.area(
                stacked=False,
                label="Battery Charging",
                color={"Battery Charging": "#80df95"},
                alpha=0.25,
                ax=ax_power,
            )

        if has_tariff:
            # TODO: Replace with tariff component constants
            tariff = data[Tariff.PRICE_IMPORT]

            ax_price = axes[0].twinx()
            axes.append(ax_price)

            sns.lineplot(tariff, linewidth=1, color="#999999", label="Dynamic Tariff", ax=ax_price)

            ax_price.spines.right.set_position(("axes", 1.07))
            ax_price.set_ylim(tariff.min() - 1, tariff.max() + 1)
            ax_price.yaxis.set_label_text("Price [ct/kWh]")
            ax_price.legend(ncol=1, loc="upper right", frameon=False)

        ax_power.set_xlim(data_power.index[0], data_power.index[-1])
        ax_power.set_ylim(min(data_power.min()), max(data_power.max()) + 50)
        ax_power.xaxis.set_minor_locator(dates.HourLocator(interval=12))
        ax_power.xaxis.set_minor_formatter(dates.DateFormatter("%H:%M", tz="Europe/Berlin"))
        ax_power.xaxis.set_major_locator(dates.DayLocator(interval=1))
        ax_power.xaxis.set_major_formatter(dates.DateFormatter("\n%A", tz="Europe/Berlin"))
        ax_power.xaxis.set_label_text(f"{data.index[0].strftime('%d. %B')} to " f"{data.index[-1].strftime('%d. %B')}")
        # ax_power.xaxis.label.set_visible(False)
        ax_power.yaxis.set_label_text("Power [kW]")
        ax_power.legend(ncol=3, loc="upper left", frameon=False)

        for pos in ["right", "top", "bottom", "left"]:
            for ax in axes:
                ax.spines[pos].set_visible(False)

        axes[0].grid(color="grey", linestyle="--", linewidth=0.25, alpha=0.5, axis="both")
        axes[0].set_title(title)
        figure.tight_layout()

        if file is not None:
            figure.savefig(file)
        if show:
            figure.show()
            # figure.waitforbuttonpress()

        plt.close(figure)
        plt.clf()

    def residual_forecast(self, start, end):

        def apply_kalman_filter(series: pd.Series, process_var=1e-4, meas_var=0.1) -> pd.Series:
            """
            Apply a basic 1D Kalman Filter to a pandas Series.

            Parameters:
                series (pd.Series): The input time series.
                process_var (float): Process variance (Q).
                meas_var (float): Measurement variance (R).

            Returns:
                pd.Series: Smoothed series.
            """
            kf = KalmanFilter(dim_x=2, dim_z=1)

            # State: [position, velocity]
            kf.x = np.array([[series.iloc[0]], [0.]])  # initial state
            kf.F = np.array([[1., 1.], [0., 1.]])  # linear state transition matrix
            kf.H = np.array([[1., 0.]])  # measurement function
            kf.P *= 1000.  # covariance matrix
            kf.R = meas_var  # measurement noise
            kf.Q = np.array([[0.25, 0.5],  # process noise
                             [0.5, 1.0]]) * process_var

            results = []
            for z in series:
                kf.predict()
                kf.update([z])
                results.append(kf.x[0, 0])  # position estimate

            return pd.Series(results, index=series.index, name="kalman")

        def persistence_3_week(
                start: TimestampType,
                last_3w_ts: pd.DataFrame,
        ) -> (pd.Series, pd.Series):
            past_ts = last_3w_ts.copy()

            def timestamp_to_week(timestamp: pd.Timestamp) -> int:
                return timestamp.weekday() * 24 * 60 + timestamp.hour * 60 + timestamp.minute

            week_minutes = 7 * 24 * 60  # Total minutes in a week
            start = past_ts.index[0].floor("h")
            week_offset = timestamp_to_week(start)
            past_ts.index = pd.to_datetime(past_ts.index).tz_convert(None)
            past_ts.index = (past_ts.index.map(timestamp_to_week) - week_offset) % week_minutes

            agg_df = past_ts.groupby(past_ts.index).agg(['mean', 'std'])

            # Align aggregated df to start time  till start + 1 week
            aligned_index = ((pd.Series(range(week_minutes)) + week_offset) % week_minutes)
            aligned_index.index = pd.date_range(
                start=start,  # + pd.Timedelta(weeks=20),
                periods=week_minutes,
                freq="min")

            persistence_mean = agg_df['mean']
            persistence_mean.index = aligned_index.index

            persistence_std = agg_df['std']
            persistence_std.index = aligned_index.index

            return persistence_mean, persistence_std

        def deterministic_forecast_model(
                current,
                forecast_values: pd.Series,
                t_half,
                dt=1):
            """
            F = external forecast values
            R = model results
            R_k+1 = F_k+1 + kappa * (R_k - F_k)
            kappa = 2^dt/t_half
            """
            kappa = 2 ** -(dt / t_half)
            results = [current]
            for index in range(1, len(forecast_values)):
                results.append(
                    forecast_values.iloc[index] + kappa * (results[index - 1] - forecast_values.iloc[index - 1]))
            return pd.Series(results, index=forecast_values.index, name=forecast_values.name)

        column = System.POWER_EL
        try:
            if start > self.buffered_forecast.index[0] and end < self.buffered_forecast.index[-1]:
                return self.buffered_forecast
        except Exception:
            pass

        _start = start - pd.Timedelta(weeks=3)
        _end = start

        # Todo: Remove this, it is only for testing
        # Shift forward if data is not available before this date
        while _start < pd.Timestamp("2016-06-01T00:00:00+02:00"):
            _start += pd.Timedelta(weeks=1)
            _end += pd.Timedelta(weeks=1)

        if _start > pd.Timestamp("2020-01-01T00:00:00+01:00"):
            while _start < pd.Timestamp("2024-01-01T00:00:00+01:00"):
                _start += pd.Timedelta(weeks=1)
                _end += pd.Timedelta(weeks=1)
        # Load the last 3 weeks of data
        # last_3w_ts = last_3w_ts.resample("15min").mean()



        last_3w_ts = self.data.from_logger([column], start=_start, end=_end)[column]

        persistence_mean, persistence_std = persistence_3_week(start, last_3w_ts)

        persistence_mean.index = persistence_mean.index + pd.Timedelta(weeks=3)
        persistence_std.index = persistence_std.index + pd.Timedelta(weeks=3)

        last_3w_filtered_ts = apply_kalman_filter(last_3w_ts[_end - pd.Timedelta(hours=1):_end])

        current_filtered = last_3w_filtered_ts.iloc[-1]

        deterministic = deterministic_forecast_model(
            current=current_filtered,
            forecast_values=persistence_mean,
            t_half=12,  # Default half-life of 12 hours
            dt=60  # Default time step of 1 hour
        )

        # resample last_3w_ts to 10 frequency
        if self.configs.get_bool("no_forecast", default=False):
            deterministic = self.data.from_logger([column], start=_start + pd.Timedelta(weeks=3), end=_start + pd.Timedelta(weeks=4))[column]


        plot_model = False
        if plot_model == True:
            real = self.data.from_logger([column], start=_start + pd.Timedelta(weeks=3), end=_start + pd.Timedelta(weeks=4))[column]
            real = real.resample("15min").mean()
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 6))
            plt.plot(last_3w_ts.index, last_3w_ts, label='Last 3 Weeks', color='C0', alpha=0.5)
            plt.plot(last_3w_filtered_ts.index, last_3w_filtered_ts, label='Kalman Filtered', color='C0')
            plt.plot(persistence_mean.index, persistence_mean, label='Persistence Mean', color='C1', alpha=0.5)
            plt.plot(deterministic.index, deterministic, label='Deterministic Forecast', color='C1')
            plt.plot(real.index, real, label='Real Data', color='C2', alpha=0.5)
            plt.fill_between(
                deterministic.index,
                deterministic - persistence_std,
                deterministic + persistence_std,
                color='C1',
                alpha=0.2,
                label='Persistence Std'
            )
            plt.legend()

            plt.show()
            pass





        return_df = pd.DataFrame({
            'forecast': deterministic,
            'forecast_std': persistence_std
        })

        self.buffered_forecast = return_df.copy()

        # shift index to index[0] = start time
        t_diff = start - return_df.index[0]
        return_df.index = return_df.index + t_diff

        if self.configs.get_bool("no_forecast", default=False):
            real =  self.data.from_logger([column], start=start, end=start + pd.Timedelta(weeks=1))[column]
            return_df = pd.DataFrame({
                "forecast" : real,
                "forecast_std": real * 0.0 + 5000
            })

        return return_df
