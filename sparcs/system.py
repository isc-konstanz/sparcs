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
from lories import Channel, Configurations, Constant
from lories.components import ComponentUnavailableError
from lories.components.tariff import Tariff
from lories.components.weather import Weather, WeatherProvider
from lories.simulation import Result, Results
from lories.typing import Timestamp
from sparcs import Location
from sparcs.components import ElectricalEnergyStorage, SolarSystem, ThermalEnergyStorage
from sparcs.components.weather import validate_meteo_inputs, validated_meteo_inputs


class System(lories.System):
    POWER_AC = Constant(float, "ac_power", "Power", "W")
    POWER_AC_GEN = Constant(float, "ac_gen_power", "Generation Power", "W")
    POWER_AC_CON = Constant(float, "ac_cons_power", "Consumption Power", "W")
    POWER_AC_STOR = Constant(float, "ac_stor_power", "Storage Power", "W")
    POWER_AC_IMP = Constant(float, "ac_import_power", "Import Power", "W")
    POWER_AC_EXP = Constant(float, "ac_export_power", "Export Power", "W")

    POWER_DC = Constant(float, "dc_power", "Power (DC)", "W")
    POWER_DC_GEN = Constant(float, "dc_gen_power", "Generation Power (DC)", "W")
    POWER_DC_STOR = Constant(float, "dc_stor_power", "Storage Power (DC)", "W")

    POWER_TH = Constant(float, "th_power", "Power (Thermal)", "W")
    POWER_TH_DOM = Constant(float, "th_dom_power", "Domestic Power (Thermal)", "W")
    POWER_TH_HT = Constant(float, "th_ht_power", "Heating Power (Thermal)", "W")

    ENERGY_AC_GEN = Constant(float, "ac_gen_energy", "Generated Energy", "kWh")
    ENERGY_AC_CON = Constant(float, "ac_cons_energy", "Consumed Energy", "kWh")
    ENERGY_AC_STOR = Constant(float, "ac_stor_energy", "Stored Energy", "kWh")
    ENERGY_AC_IMP = Constant(float, "ac_import_energy", "Import Energy", "kWh")
    ENERGY_AC_EXP = Constant(float, "ac_export_energy", "Export Energy", "kWh")

    ENERGY_DC_GEN = Constant(float, "dc_gen_energy", "Generated Energy (DC)", "kWh")

    ENERGY_TH = Constant(float, "th_energy", "Thermal Energy", "kWh")
    ENERGY_TH_DOM = Constant(float, "th_dom_energy", "Domestic Energy (Thermal)", "kWh")
    ENERGY_TH_HT = Constant(float, "th_ht_energy", "Heating Energy (Thermal)", "kWh")

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

        has_pv = self.components.has_type(SolarSystem)
        has_ess = self.components.has_type(ElectricalEnergyStorage)
        has_ht = self.components.has_type(ThermalEnergyStorage)  # or self.components.has_type(Heating)

        add_channel(System.POWER_AC)
        if has_pv:
            add_channel(System.POWER_AC_GEN)
        if has_ess:
            add_channel(System.POWER_AC_STOR)
        if has_ess or has_pv:
            add_channel(System.POWER_AC_CON)
        if has_ess or has_pv:
            add_channel(System.POWER_AC_IMP)
            add_channel(System.POWER_AC_EXP)
            add_channel(System.POWER_DC)
        if has_ess:
            add_channel(System.POWER_DC_STOR)
        if has_pv:
            add_channel(System.POWER_DC_GEN)
        if has_ht:
            add_channel(System.POWER_TH)
        if has_ess:
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

            if isinstance(self.weather, WeatherProvider):
                self._register_weather(self.weather.forecast)

        except ComponentUnavailableError:
            pass

        if self.components.has_type(SolarSystem):
            power_channels = [
                self.data[SolarSystem.POWER_AC],
                self.data[System.POWER_AC_CON],
                self.data[System.POWER_AC],
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

        # def update_channel(channel: Channel, column: str) -> None:
        #     if column in predictions.columns:
        #         channel.set(timestamp, predictions[column])
        #     else:
        #         channel.state = ChannelState.NOT_AVAILABLE
        #
        # if self.components.has_type(SolarSystem):
        #     for solar in self.components.get_all(SolarSystem):
        #         solar_column = solar.data[SolarSystem.POWER_AC].column
        #         update_channel(solar.data[SolarSystem.POWER_EST], solar_column)
        #     update_channel(self.data[SolarSystem.POWER_EST], SolarSystem.POWER_AC)
        # update_channel(self.data[System.POWER_AC_EST], System.POWER_AC)
        # update_channel(self.data[System.POWER_TH_EST], System.POWER_TH)

    def _on_power_received(self, data: pd.DataFrame) -> None:
        if data[System.POWER_AC_CON].dropna().empty:
            power = data.loc[:, System.POWER_AC].dropna()
            power += data.loc[power.index, System.POWER_AC_GEN].fillna(0)
            power.name = System.POWER_AC_CON
            self.data[System.POWER_AC_CON].set(power.index[0], power)
        elif data[System.POWER_AC].dropna().empty:
            power = data.loc[:, System.POWER_AC_CON].dropna()
            power -= data.loc[power.index, System.POWER_AC_GEN].fillna(0)
            power.name = System.POWER_AC
            self.data[System.POWER_AC].set(power.index[0], power)

    def run(self, weather: pd.DataFrame) -> pd.DataFrame:
        weather = validate_meteo_inputs(weather, self.location)
        predictions = pd.DataFrame(index=weather.index)

        if self.components.has_type(SolarSystem):
            system_ac_gen = self.data[System.POWER_AC_GEN]
            system_dc_gen = self.data[System.POWER_DC_GEN]
            predictions[[system_ac_gen.id, system_dc_gen.id]] = 0.0
            for solar in self.components.get_all(SolarSystem):
                solar_prediction = solar.run(weather)
                predictions = pd.concat([predictions, solar_prediction], axis="columns")
                predictions[system_ac_gen.id] += solar_prediction[solar.data[SolarSystem.POWER_AC].id].fillna(0)
                predictions[system_dc_gen.id] += solar_prediction[solar.data[SolarSystem.POWER_DC].id].fillna(0)

        predictions.index.name = Channel.TIMESTAMP
        return predictions

    def predict(
        self,
        start: Timestamp = None,
        end: Timestamp = None,
        **kwargs,
    ) -> pd.DataFrame:
        # predictions = super().predict(start, end, **kwargs)
        weather = self.weather.get(start, end, **kwargs)
        predictions = self.run(weather)

        return pd.concat(
            [
                predictions,
                weather.rename(columns={c.key: c.id for c in self.weather.data.channels}),
            ],
            axis="columns",
        )

    def simulate(
        self,
        start: Timestamp,
        end: Timestamp,
        prior: Optional[pd.DataFrame] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        data = self.predict(start, end, **kwargs)

        system_ac = self.data[System.POWER_AC]
        if system_ac.id not in data.columns:
            if self.data.has_logged(system_ac, start=start, end=end):
                self._logger.debug(f"Reference {System.POWER_AC.name} will be as missing prediction.")
                data.insert(0, system_ac.id, self.data.from_logger(system_ac, start=start, end=end))
            else:
                self._logger.debug(f"Reference {System.POWER_AC.name} cannot be found.")

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
        system_ac = self.data[System.POWER_AC]
        system_ac_gen = self.data[System.POWER_AC_GEN]

        # TODO: Verify if total generation power will be always only ever be calculated
        if system_ac_gen.id not in data.columns:
            data[system_ac_gen.id] = 0.0

        for solar in self.components.get_all(SolarSystem):
            solar_ac = solar.data[SolarSystem.POWER_AC]
            solar_data = solar.data.from_logger(solar_ac, start, end)
            if solar_data.empty:
                if system_ac.id in data.columns:
                    data[system_ac.id] -= data[solar_ac.id]

                # Solar System does not have a measured reference and will be subtracted from residual power
                data[system_ac_gen.id] += data[solar_ac.id]
            else:
                data[system_ac_gen.id] += solar_data[solar_ac.id]
        return data

    # noinspection PyUnusedLocal
    def _simulate_storage(
        self,
        data: pd.DataFrame,
        start: Timestamp,
        end: Timestamp,
        prior: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        if not self.components.has_type(ElectricalEnergyStorage):
            return data

        system_ac = self.data[System.POWER_AC]
        system_ac_stor = self.data[System.POWER_AC_STOR]
        system_soc = self.data[ElectricalEnergyStorage.STATE_OF_CHARGE]

        # TODO: Verify if total charging power will be always only ever be calculated
        if system_ac_stor.id not in data.columns:
            data[system_ac_stor.id] = 0.0

        total_capacity = 0
        total_energy = pd.Series(index=data.index, data=0)

        for ees in self.components.get_all(ElectricalEnergyStorage):
            ees_ac = ees.data[ElectricalEnergyStorage.STATE_OF_CHARGE]
            ees_soc = ees.data[ElectricalEnergyStorage.POWER_AC]
            ees_data = ees.data.from_logger(system_soc, start, end)
            if ees_data.empty:
                ees_soc_prior = prior.iloc[-1][ees_soc.id] if prior is not None else 50.0
                ees_data = ees.predict(data, ees_soc_prior)
                ess_power = ees_data[ees_ac.id]

                # EES does not have a measured reference and will be added to residual power
                data = pd.concat([data, ees_data], axis="columns")
                data[system_ac_stor.id] += ess_power

                if system_ac.id in data.columns:
                    data[system_ac.id] += ess_power
            else:
                data[system_ac_stor.id] += ees_data[ees_ac.id]

            total_capacity += ees.capacity
            total_energy += ees_data[ees_soc.id] / 100 * ees.capacity

        data[system_soc.id] = total_energy / total_capacity * 100

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
        if not self.components.has_type(SolarSystem):
            return
        simulation = data["predictions"]

        solar_simulated = False
        solar_systems = self.components.get_all(SolarSystem)
        solar_data = pd.DataFrame(
            data=0.0,
            index=results.data.index,
            columns=[SolarSystem.POWER_AC, SolarSystem.POWER_DC],
        )
        has_bifaciality = any(a.is_bifacial() for solar_system in solar_systems for a in solar_system.arrays)
        if has_bifaciality:
            solar_data[SolarSystem.POWER_DC_FRONT] = 0.0
        for solar in solar_systems:
            solar_dc = solar.data[SolarSystem.POWER_DC]
            solar_ac = solar.data[SolarSystem.POWER_AC]
            solar_data[SolarSystem.POWER_AC] += simulation[solar_ac.id]
            solar_data[SolarSystem.POWER_DC] += simulation[solar_dc.id]
            if has_bifaciality:
                solar_data[SolarSystem.POWER_DC_FRONT] += simulation[solar.data[SolarSystem.POWER_DC_FRONT].id]

            solar_reference = solar.data.from_logger(start=results.start, end=results.end)
            if solar_reference.empty:
                solar_simulated = True
            else:
                data[("references", solar_ac.id)] = solar_reference[solar_ac.id]

        if solar_simulated and "references" in data.columns.get_level_values(0):
            # One or more solar system does not have reference measurements.
            # The reference value does not correspond to the total prediction and should be dropped
            for column in [self.data[System.POWER_AC].id, self.data[System.POWER_DC].id]:
                if column in data["references"].columns:
                    data.drop(columns=[("references", column)], inplace=True)

        hours = pd.Series(data.index, index=data.index)
        hours = (hours - hours.shift(1)).bfill().dt.total_seconds() / 3600.0

        solar_kwp = sum(solar.power_max / 1000.0 for solar in solar_systems)
        solar_power = solar_data[SolarSystem.POWER_AC]
        solar_energy = solar_power / 1000.0 * hours
        solar_energy.name = SolarSystem.ENERGY_AC

        yield_months_file = results.dirs.tmp.joinpath("yield_months.png")
        try:
            from lories.io import plot

            plot_data = solar_energy.to_frame().groupby(data.index.month).sum()
            plot.bar(
                x=plot_data.index,
                y=SolarSystem.ENERGY_AC,
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

        dc_energy = (solar_data[SolarSystem.POWER_DC] / 1000.0 * hours).sum()
        results.append(Result.from_const(SolarSystem.YIELD_ENERGY_DC, dc_energy, header="Yield"))

        if has_bifaciality:
            dc_front_energy = (solar_data[SolarSystem.POWER_DC_FRONT] / 1000.0 * hours).sum()
            bifacial_gain = max(round((dc_energy / dc_front_energy - 1) * 100, 2), 0)
            results.append(Result.from_const(SolarSystem.BIFACIAL_GAIN, bifacial_gain, header="Yield"))

    def _evaluate_storage(self, results: Results, data: pd.DataFrame) -> None:
        if not self.components.has_type(ElectricalEnergyStorage):
            return
        simulation = data["predictions"]

        ees_simulated = False
        ees_capacity = 0
        ees_systems = self.components.get_all(ElectricalEnergyStorage)
        ees_data = pd.DataFrame(
            data=0.0,
            index=results.data.index,
            columns=[ElectricalEnergyStorage.POWER_AC],
        )
        for ees in ees_systems:
            ees_ac = ees.data[ElectricalEnergyStorage.POWER_AC]
            ees_data[ElectricalEnergyStorage.POWER_AC] += simulation[ees_ac.id]
            ees_capacity += ees.capacity
            ees_reference = ees.data.from_logger(ees_ac, start=results.start, end=results.end)
            if ees_reference.empty:
                ees_simulated = True
            else:
                data[("references", ees_ac.id)] = ees_reference[ees_ac.id]

        if ees_simulated and "references" in data.columns.get_level_values(0):
            # One or more solar system does not have reference measurements.
            # The reference value does not correspond to the total prediction and should be dropped
            for column in [
                self.data[System.POWER_AC].id,
                self.data[System.POWER_DC].id,
                self.data[ElectricalEnergyStorage.STATE_OF_CHARGE].id,
            ]:
                if column in data["references"].columns:
                    data.drop(columns=[("references", column)], inplace=True)

        hours = pd.Series(data.index, index=data.index)
        hours = (hours - hours.shift(1)).bfill().dt.total_seconds() / 3600.0

        ees_soc = simulation[ElectricalEnergyStorage.STATE_OF_CHARGE]
        ees_power = ees_data[ElectricalEnergyStorage.POWER_AC]
        ees_cycles = (ees_power.where(ees_power >= 0, other=0) / 1000 * hours).sum() / ees_capacity

        results.append(Result.from_const(ElectricalEnergyStorage.CYCLES, ees_cycles, header="Storage"))
        results.append(Result.from_const(ElectricalEnergyStorage.SOC_MIN, ees_soc.min(), header="Storage"))

    # noinspection PyMethodMayBeStatic
    def _evaluate_system(self, results: Results, data: pd.DataFrame) -> None:
        simulation = data["predictions"]
        system_ac = self.data[System.POWER_AC]
        if system_ac.id not in simulation.columns:
            return
        hours = pd.Series(data.index, index=data.index)
        hours = (hours - hours.shift(1)).bfill().dt.total_seconds() / 3600.0

        active_power = simulation[system_ac.id]
        import_power = active_power.where(active_power >= 0, other=0)
        import_energy = import_power / 1000 * hours
        export_power = active_power.where(active_power <= 0, other=0).abs()
        export_energy = export_power / 1000 * hours

        results.add("grid_export_max", "Export Peak [W]", export_power.max(), header="Grid", order=10)
        results.add("grid_import_max", "Import Peak [W]", import_power.max(), header="Grid", order=10)
        results.add("grid_export", "Export [kWh]", export_energy.sum(), header="Grid", order=10)
        results.add("grid_import", "Import [kWh]", import_energy.sum(), header="Grid", order=10)

        try:
            # import_peak_energy = import_energy[import_power >= import_power.max()]
            # import_peak_time = import_peak_energy.index.time.min()
            # import_peak_date = import_peak_energy[import_peak_energy.index.time == import_peak_time].index.date[0]
            # self._plot_system(
            #     simulation[data.index.date == import_peak_date],
            #     title="Day with earliest Peak",
            #     file=str(results.dirs.tmp.joinpath("power_peak.png")),
            #     width=16,
            # )

            import_week_energy = import_energy.groupby(import_energy.index.isocalendar().week).sum()
            import_week_energy_max = import_week_energy[import_week_energy == import_week_energy.max()].index[0]
            self._plot_system(
                simulation[data.index.isocalendar().week == import_week_energy_max],
                title="Week with highest Grid Import",
                file=str(results.dirs.tmp.joinpath("week_max_import.png")),
            )
        except ImportError:
            pass

        system_ac_gen = self.data[System.POWER_AC_GEN]
        if system_ac_gen.id in simulation.columns:
            gen_power = simulation[system_ac_gen.id]
            gen_energy = gen_power / 1000 * hours
            cons_energy = import_energy + gen_energy - export_energy
            cons_self = (gen_energy - export_energy).sum() / gen_energy.sum() * 100
            suff_self = (1 - (import_energy.sum() / cons_energy.sum())) * 100

            results.add("consumption", "Energy [kWh]", cons_energy.sum(), header="Load", order=10)
            results.add("self_consumption", "Self-Consumption [%]", cons_self, header="Consumption", order=10)
            results.add("self_sufficiency", "Self-Sufficiency [%]", suff_self, header="Consumption", order=10)

            try:
                cons_self = gen_energy - export_energy
                cons_self_week_energy = cons_self.groupby(cons_self.index.isocalendar().week).sum()
                cons_self_week_energy_max = cons_self_week_energy[
                    cons_self_week_energy == cons_self_week_energy.max()
                ].index[0]
                self._plot_system(
                    simulation[data.index.isocalendar().week == cons_self_week_energy_max],
                    title="Week with highest Self-Consumption",
                    file=str(results.dirs.tmp.joinpath("week_max_self-cons.png")),
                )
            except ImportError:
                pass

    # noinspection PyMethodMayBeStatic
    def _evaluate_weather(self, results: Results, data: pd.DataFrame) -> None:
        weather = self.components.get_first(Weather)
        if weather is None:
            return
        predictions = data["predictions"]
        ghi = weather.data[Weather.GHI]
        dhi = weather.data[Weather.DHI]
        if not all(c in predictions.columns for c in [ghi.id, dhi.id]):
            return
        hours = pd.Series(data.index, index=data.index)
        hours = (hours - hours.shift(1)).bfill().dt.total_seconds() / 3600.0

        total_ghi = (predictions[ghi.id] / 1000.0 * hours).sum()
        total_dhi = (predictions[dhi.id] / 1000.0 * hours).sum()

        results.add(Weather.GHI, f"{Weather.GHI.name} [kWh/m²]", total_ghi, header="Weather")
        results.add(Weather.DHI, f"{Weather.DHI.name} [kWh/m²]", total_dhi, header="Weather")

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
        column_ac = self.data[System.POWER_AC].id
        columns = [column_ac]

        # TODO: Replace with tariff component constants
        has_tariff = "tariff" in data.columns

        column_gen = self.data[System.POWER_AC_GEN].id
        has_gen = self.components.has_type(SolarSystem)
        if has_gen:
            columns.append(column_gen)

        column_stor = self.data[System.POWER_AC_STOR].id
        has_stor = self.components.has_type(ElectricalEnergyStorage)
        if has_stor:
            columns.append(column_stor)

        data_power = deepcopy(data[columns])
        data_power /= 1000

        if width is None:
            width = plot.WIDTH
        if height is None:
            height = plot.HEIGHT
        figure, ax_power = plt.subplots(figsize=(width / plot.INCH, height / plot.INCH), dpi=120)
        axes = [ax_power]

        sns.lineplot(
            data_power[column_ac],
            linewidth=0.25,
            color="#004f9e",
            label="_hidden",
            ax=axes[0],
        )
        if has_stor:
            ax_soc = axes[0].twinx()
            axes.append(ax_soc)

            sns.lineplot(
                data_power[column_stor],
                linewidth=0.25,
                color="#ff9995",
                label="_hidden",
                ax=ax_power,
            )
            sns.lineplot(
                data[self.data[ElectricalEnergyStorage.STATE_OF_CHARGE].id],
                linewidth=1,
                color="#333333",
                label="Battery State",
                ax=ax_soc,
            )

            data_ref = data_power[column_ac] - data_power[column_stor]
            data_ref.plot.area(
                stacked=False,
                label="_hidden",
                color={"_hidden": "#dddddd"},
                linewidth=0,
                alpha=0.75,
                ax=ax_power,
            )

            ax_soc.set_ylim(-1, 119)
            ax_soc.yaxis.set_labac_text("State of Charge [%]")
            if has_tariff:
                ax_soc.legend(ncol=1, loc="upper right", bbox_to_anchor=(0.84, 1), frameon=False)
            else:
                ax_soc.legend(ncol=1, loc="upper right", frameon=False)

        if has_gen:
            data_power[column_gen].plot.area(
                stacked=False,
                label="PV Generation",
                color={"PV Generation": "#ffeb9b"},
                linewidth=0,
                alpha=0.75,
                ax=ax_power,
            )

        data_power[column_ac].plot.area(
            stacked=False,
            label="Residual Load",
            alpha=0.25,
            ax=ax_power,
        )

        if has_stor:
            data_power[column_stor].plot.area(
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
            ax_price.yaxis.set_labac_text("Price [€/kWh]")
            ax_price.legend(ncol=1, loc="upper right", frameon=False)

        ax_power.set_xlim(data_power.index[0], data_power.index[-1])
        ax_power.set_ylim(min(data_power.min()), max(data_power.max()) + 50)
        ax_power.xaxis.set_minor_locator(dates.HourLocator(interval=12))
        ax_power.xaxis.set_minor_formatter(dates.DateFormatter("%H:%M", tz="Europe/Berlin"))
        ax_power.xaxis.set_major_locator(dates.DayLocator(interval=1))
        ax_power.xaxis.set_major_formatter(dates.DateFormatter("\n%A", tz="Europe/Berlin"))
        ax_power.xaxis.set_labac_text(f"{data.index[0].strftime('%d. %B')} to " f"{data.index[-1].strftime('%d. %B')}")
        # ax_power.xaxis.label.set_visible(False)
        ax_power.yaxis.set_labac_text("Power [kW]")
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
