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
from lori import Channel, ChannelState, Configurations, Constant, Weather, WeatherUnavailableException, \
    ResourceUnavailableException
from lori.simulation import Result, Results
from lori.typing import TimestampType
from lori.components import Tariff, TariffUnavailableException
from lori.components.tariff.entsoe import EntsoeProvider
from penguin import Location
from penguin.components import ElectricalEnergyStorage, SolarSystem

from penguin.components.weather import validate_meteo_inputs, validated_meteo_inputs

from penguin.components.control.predictive import Optimization
from penguin.components.control.predictive.problems.grid_cost import GridCostProblem


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

        try:
            pass
            #self._register_tariff(self.components.get_first(EntsoeProvider))

        except TariffUnavailableException:
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
        predictions = self._predict_solar(weather)
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

    def _predict(self,
        weather: pd.DataFrame,
        start: TimestampType = None,
        end: TimestampType = None,
    ) -> pd.DataFrame:
        weather = validate_meteo_inputs(weather, self.location)
        predictions = pd.DataFrame(index=weather.index)
        predictions.index.name = Channel.TIMESTAMP

        if self.components.has_type(SolarSystem):
            solar_columns = [SolarSystem.POWER, SolarSystem.POWER_DC]
            predictions[solar_columns] = 0.0
            for solar_system in self.components.get_all(SolarSystem):
                solar_column = solar_system.data[SolarSystem.POWER].column
                solar_prediction = solar_system.predict(weather)
                predictions[solar_column] = solar_prediction[SolarSystem.POWER]
                predictions[solar_columns] += solar_prediction[solar_columns].fillna(0)

        #TODO: implement load consumption prediction

        if self.components.has_type(ElectricalEnergyStorage) and self.components.has_type(GridCostProblem):
            # get el power from logger and simulate ontop
            opti_df = pd.DataFrame(index=weather.index)
            opti_df.insert(0, System.POWER_EL, self.data.from_logger([System.POWER_EL], start=start, end=end))
            for solar in self.components.get_all(SolarSystem):
                solar_column = solar.data[SolarSystem.POWER].column
                if not solar.data.has_logged(SolarSystem.POWER, start, end):
                    # Solar System does not have a measured reference and will be subtracted from residual power
                    opti_df[System.POWER_EL] -= predictions[solar_column]

            # get tariff data
            # TODO: Callback of channels
            # opti_df.insert(0, Tariff.EXPORT, self.data.from_logger([Tariff.EXPORT], start=start, end=end))
            # opti_df.insert(0, Tariff.IMPORT, self.data.from_logger([Tariff.IMPORT], start=start, end=end))
            tariff_component:Tariff = self.components.get_first(Tariff)
            self.connectors.get_first(EntsoeProvider)
            #tariff_component.

            problem:GridCostProblem = self.components.get_first(GridCostProblem)


            #problem.solve()

        return predictions

    def predict(
        self,
        start: TimestampType = None,
        end: TimestampType = None,
        **kwargs,
    ) -> pd.DataFrame:
        if self.components.has_type(Optimization):
            optimization_problem = self.components.get_first(Optimization)
            optimization_end = start + pd.Timedelta(seconds=optimization_problem.total_duration)
            if end > optimization_end:
                self._logger.warning(f"Simulation duration is greater than optimization duration")
            end = optimization_end

        weather = self.weather.get(start, end, **kwargs)
        solar_predictions = self._predict_solar(weather)

        load_predictions = self._predict_load(start, end, index=weather.index)
        predictions = pd.concat([solar_predictions, load_predictions], axis="columns")
        predictions = self._predict_residual(start, end, predictions)

        #if self.components.has_type(Tariff):
        #    tariff: Tariff = self.components.get_first(Tariff)
        #    tariff.get(start, end, **kwargs)
        #tariff = self.components..get(start, end, **kwargs)

        forecaster = self.forecast.get()
        tariff = pd.DataFrame(index=weather.index)
        #tariff["import"] = 0.25
        tariff["import"] = 0.3 + 0.1 * (np.sin(weather.index.hour / 24 * 2 * np.pi) + 1)
        tariff["export"] = -0.05
        predictions = pd.concat([predictions, tariff], axis="columns")

        self._predict_optimization(start, end, predictions)

        pass






        return pd.concat([predictions, weather], axis="columns")

    def _predict_solar(self, weather: pd.DataFrame) -> pd.DataFrame:
        weather = validate_meteo_inputs(weather, self.location)
        solar_predictions = pd.DataFrame(index=weather.index)
        solar_predictions.index.name = Channel.TIMESTAMP

        if self.components.has_type(SolarSystem):
            solar_columns = [SolarSystem.POWER, SolarSystem.POWER_DC]
            solar_predictions[solar_columns] = 0.0
            for solar_system in self.components.get_all(SolarSystem):
                solar_column = solar_system.data[SolarSystem.POWER].column
                solar_prediction = solar_system.predict(weather)
                solar_predictions[solar_column] = solar_prediction[SolarSystem.POWER]
                solar_predictions[solar_columns] += solar_prediction[solar_columns].fillna(0)

        return solar_predictions

    def logged_data(
        self,
        start: TimestampType,
        end: TimestampType,
        column: str,
    ) -> pd.DataFrame:
        return self.data.from_logger([column], start=start, end=end)



    def apply_kalman_filter(self, series: pd.Series, process_var=1e-5, meas_var=0.1) -> pd.Series:
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
        kf.F = np.array([[1., 1.], [0., 1.]])  # state transition matrix
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

    def persistence_avg_3_week(self,
        start: TimestampType,
        column: str,
        prediction_index: pd.Index
    ) -> pd.DataFrame:
        # Define window of past 3 weeks
        _start = start - pd.Timedelta(weeks=3)
        _end = start

        # Todo: Remove this, it is only for testing
        # Shift forward if data is not available before this date
        while _start < pd.Timestamp("2016-06-01T00:00:00+02:00"):
            _start += pd.Timedelta(weeks=1)
            _end += pd.Timedelta(weeks=1)

        def timestamp_to_week(timestamp: pd.Timestamp) -> int:
            return timestamp.weekday() * 24 * 60 + timestamp.hour * 60 + timestamp.minute

        week_minutes = 7 * 24 * 60
        week_offset = timestamp_to_week(start)

        # Load historical data
        past_df = self.data.from_logger([column], start=_start, end=_end)
        past_df = past_df.copy()
        past_df.index = pd.to_datetime(past_df.index).tz_convert(None)

        # Normalize time of week
        past_df["min_of_week"] = (past_df.index.map(timestamp_to_week) - week_offset) % week_minutes

        # Average power per minute offset in week
        avg_week = past_df.groupby("min_of_week")[column].mean()

        # Align average week data to the prediction time
        aligned_index = ((prediction_index.map(timestamp_to_week) - week_offset) % week_minutes)
        aligned_index.index = prediction_index

        # Map predictions
        predictions = pd.DataFrame(index=prediction_index)
        predictions.index.name = Channel.TIMESTAMP
        predictions[column] = aligned_index.map(avg_week.get)

        return predictions

    def deterministic_forecast_model(self, current, forecast_values:pd.Series, t_half, dt=1):
        """
        F = external forecast values
        R = model results
        R_k+1 = F_k+1 + kappa * (R_k - F_k)
        kappa = 2^dt/t_half
        """
        kappa = 2 ** -(dt / t_half)
        results = [current]
        for index in range(1, len(forecast_values)):
            results.append(forecast_values.iloc[index] + kappa * (results[index - 1] - forecast_values.iloc[index - 1]))
        print(results)
        return pd.Series(results, index=forecast_values.index, name=forecast_values.name)



    def _predict_load(self,
        start: TimestampType,
        end: TimestampType,
        index: pd.Index = None,
    ) -> pd.DataFrame:
        real = self.logged_data(start, end, System.POWER_EL)
        persistence = self.persistence_avg_3_week(start, System.POWER_EL, index)
        forcast_model = self.deterministic_forecast_model(
            current=real.iloc[0][System.POWER_EL],
            forecast_values=persistence[System.POWER_EL],
            t_half= 6 * 60,  # 6h half-life
            dt=1,  # 1 minute step
        )


        combined = pd.concat([real, persistence, forcast_model], axis=1, keys=["real", "persistence", "forecast_model"])
        combined["real_kalman"] = self.apply_kalman_filter(real[System.POWER_EL], process_var=1, meas_var=100_000)

        plot_persistence_against_real = True
        if plot_persistence_against_real:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(12, 6))
            for col in combined.columns:
                plt.plot(combined.index, combined[col], linestyle='-', label=col)
            plt.xlabel("Time")
            plt.ylabel("Power [W]")
            plt.title("Real vs. Persistence Load Prediction")
            plt.legend()
            plt.tight_layout()
            #plt.show()








        """
        Predicts electric load using 3-week persistence-based average.
        """
        # Define window of past 3 weeks
        _start = start - pd.Timedelta(weeks=3)
        _end = start

        def timestamp_to_week(timestamp: pd.Timestamp) -> int:
            return timestamp.weekday() * 24 * 60 + timestamp.hour * 60 + timestamp.minute

        # Todo: Remove this, it is only for testing
        # Shift forward if data is not available before this date
        while _start < pd.Timestamp("2016-06-01T00:00:00+02:00"):
            _start += pd.Timedelta(weeks=1)
            _end += pd.Timedelta(weeks=1)

        week_minutes = 7 * 24 * 60
        week_offset = timestamp_to_week(start)

        # Load historical data
        historical_load = self.data.from_logger([System.POWER_EL], start=_start, end=_end)
        historical_load = historical_load.copy()
        historical_load.index = pd.to_datetime(historical_load.index).tz_convert(None)

        # Normalize time of week
        historical_load["tow_time"] = (historical_load.index.map(timestamp_to_week) - week_offset) % week_minutes

        # Average power per minute offset in week
        avg_week = historical_load.groupby("tow_time")[System.POWER_EL].mean()

        # Align average week data to the prediction time
        aligned_index = ((index.map(timestamp_to_week) - week_offset) % week_minutes)
        aligned_index.index = index

        # Map predictions
        load_predictions = pd.DataFrame(index=index)
        load_predictions.index.name = Channel.TIMESTAMP
        load_predictions[System.POWER_EL] = aligned_index.map(avg_week.get)

        return load_predictions


    def _predict_residual(
        self,
        start: TimestampType,
        end: TimestampType,
        predictions: pd.DataFrame,
    ) -> pd.DataFrame:
        for solar in self.components.get_all(SolarSystem):
            solar_column = solar.data[SolarSystem.POWER].column
            if not solar.data.has_logged(SolarSystem.POWER, start, end):
                # Solar System does not have a measured reference and will be subtracted from residual power
                predictions[System.POWER_EL] -= predictions[solar_column]
        return predictions

    def _predict_optimization(
        self,
        start: TimestampType,
        end: TimestampType,
        predictions: pd.DataFrame,
    ) -> pd.DataFrame:
        if self.components.has_type(GridCostProblem):
            problem: GridCostProblem = self.components.get_first(GridCostProblem)
            results = problem.solve(predictions, start)
            predictions = pd.concat([predictions, results], axis="columns")
        return predictions



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
                # Solar System does not have a measured reference and will be subtracted from residual power
                data[System.POWER_EL] -= data[solar_column]
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
                ees_data = ees.predict(data, ees_soc)
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

        if "references" in data.columns.get_level_values(0):
            errors = (data["predictions"] - data["references"]).dropna(axis="columns", how="all")
            errors.columns = pd.MultiIndex.from_product([["errors"], errors.columns])
            data = pd.concat([data, errors], axis="columns")
        return data

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

        if solar_simulated and "references" in data.columns.get_level_values(0):
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
        yield_hours_file = results.dirs.tmp.joinpath("yield_hours.png")
        yield_hours_stats_file = results.dirs.tmp.joinpath("yield_hours_stats.png")
        power_avg_week_file = results.dirs.tmp.joinpath("power_avg_week.png")
        try:
            from lori.io import plot

            # Monthly Yeald
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

            # Hourly Yield
            plot_data = solar_energy.to_frame().groupby(data.index.hour).mean()
            plot.bar(
                x=plot_data.index,
                y=SolarSystem.ENERGY,
                data=plot_data,
                xlabel="Hour of day",
                ylabel="Power [kW]",
                title="Hourly Yield Mean",
                colors=list(reversed(plot.COLORS)),
                file=str(yield_hours_file),
            )

            # Hourly Yield Boxplot
            plot_data = solar_energy.to_frame()
            plot_data["Hour"] = plot_data.index.hour
            plot_data["Month"] = plot_data.index.month_name()
            plot.quartiles(
                x=plot_data["Hour"],
                y=SolarSystem.ENERGY,
                data=plot_data,
                xlabel="Hour of day",
                ylabel="Power [kW]",
                title="Hourly Yield Statistics",
                colors=list(reversed(plot.COLORS)),
                file=str(yield_hours_stats_file),
                hue="Month",
                width=48,
            )

            plot_data["time_of_week"] = plot_data.index.dayofweek + plot_data.index.hour / 24.0 + plot_data.index.minute / 1440.0
            plot.line(
                x="time_of_week",
                y=SolarSystem.ENERGY,
                data=plot_data,
                xlabel="Week",
                ylabel="Power [kW]",
                title="Average Power per Week",
                file=str(power_avg_week_file),
                percentile=100,
                #hue="Month"
            )

            # TODO: safe for report, show only total?
            summary_stats = plot_data.groupby(['Month', 'Hour'])[SolarSystem.ENERGY].agg(['mean', 'std'])
            #print(summary_stats)

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
            "yield_hours": yield_hours_file,
            "yield_hours_stats": yield_hours_stats_file,
        }
        results.append(Result.from_const(SolarSystem.YIELD_SPECIFIC, yield_specific, header="Yield"))
        results.append(Result.from_const(SolarSystem.YIELD_ENERGY, yield_energy, header="Yield", images=yield_images))

        if SolarSystem.POWER_DC in data["predictions"].columns:
            dc_energy = (data[("predictions", SolarSystem.POWER_DC)] / 1000.0 * hours).sum()
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
                if column in data["references"].columns:
                    data.drop(columns=[("references", column)], inplace=True)

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
                percentile=20,
                ylim=(0, 100),
                hue="Month"
            )

        except ImportError:
            pass



    # noinspection PyMethodMayBeStatic
    def _evaluate_system(self, results: Results, data: pd.DataFrame) -> None:
        if System.POWER_EL not in data["predictions"].columns:
            return
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

            import_week_energy = import_energy.groupby(import_energy.index.isocalendar().week).sum()
            import_week_energy_max = import_week_energy[import_week_energy == import_week_energy.max()].index[0]
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

        warnings.filterwarnings(
            "ignore",
            message="This axis already has a converter set and is updating to a potentially incompatible converter",
        )

        columns_power = [System.POWER_EL]

        # TODO: Replace with tariff component constants
        has_tariff = "tariff" in data.columns
        has_solar = self.components.has_type(SolarSystem)
        if has_solar:
            columns_power.append(SolarSystem.POWER)
        has_ees = self.components.has_type(ElectricalEnergyStorage)
        if has_ees:
            columns_power.append(ElectricalEnergyStorage.POWER_CHARGE)

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
            tariff = data["tariff"]

            ax_price = axes[0].twinx()
            axes.append(ax_price)

            sns.lineplot(tariff, linewidth=1, color="#999999", label="Dynamic Tariff", ax=ax_price)

            ax_price.spines.right.set_position(("axes", 1.07))
            ax_price.set_ylim(min(tariff.min() - 0.05), max(tariff.max()) + 0.1)
            ax_price.yaxis.set_label_text("Price [€/kWh]")
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
