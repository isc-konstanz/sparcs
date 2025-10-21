# -*- coding: utf-8 -*-
"""
sparcs.system
~~~~~~~~~~~~~


"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Optional

import lories
import pandas as pd
from lories import Channel, ChannelState, Configurations, Constant, Weather
from lories.components import ComponentUnavailableError
from lories.components.tariff import Tariff
from lories.simulation import Result, Results
from lories.typing import Timestamp
from sparcs import Location
from sparcs.components import ElectricalEnergyStorage, SolarSystem
from sparcs.components.weather import validate_meteo_inputs, validated_meteo_inputs


class System(lories.System):
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
            raise ComponentUnavailableError(f"System '{self.name}' has no tariff configured")
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
            add_channel(SolarSystem.POWER)
            add_channel(SolarSystem.POWER_EST)
            add_channel(SolarSystem.POWER_DC)
            add_channel(System.POWER_EL_CON)
            add_channel(System.POWER_EL_IMP)
            add_channel(System.POWER_EL_EXP)

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

        except ComponentUnavailableError:
            pass

        if self.components.has_type(SolarSystem):
            power_channels = [
                self.data[SolarSystem.POWER],
                self.data[System.POWER_EL_CON],
                self.data[System.POWER_EL],
            ]
            self.data.register(self._on_power_received, power_channels, how="any", unique=False)

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
        predictions = self._predict(weather.dropna(axis="columns"))
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
        start: Timestamp = None,
        end: Timestamp = None,
        **kwargs,
    ) -> pd.DataFrame:
        # predictions = super().predict(start, end, **kwargs)
        weather = self.weather.get(start, end, **kwargs)
        predictions = self._predict(weather)

        return pd.concat([predictions, weather], axis="columns")

    def simulate(
        self,
        start: Timestamp,
        end: Timestamp,
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
        start: Timestamp,
        end: Timestamp,
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
        start: Timestamp,
        end: Timestamp,
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
        try:
            from lories.io import plot

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

        from lories.io import plot

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
                color="#ff9995",
                label="_hidden",
                ax=ax_power,
            )
            sns.lineplot(
                data[ElectricalEnergyStorage.STATE_OF_CHARGE],
                linewidth=1,
                color="#333333",
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
                color={"Battery Charging": "#ff9995"},
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
