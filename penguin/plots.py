# -*- coding: utf-8 -*-
"""
penguin.system
~~~~~~~~~~~~~~


"""
from __future__ import annotations

from lori.io import plot


from copy import deepcopy
from typing import Any, Optional

import numpy as np

from filterpy.kalman import KalmanFilter # MIT licence

import lori
import pandas as pd
import colorsys

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



class Color:
    hex_color: str

    def __init__(self, hex_color: str) -> None:
        self.hex_color = hex_color

    def __str__(self):
        return f"{self.hex_color}"

    def __repr__(self):
        return f"{self.hex_color}\trgb:{self.rgb}\thsv:{self.hsv}"

    @property
    def hex(self) -> str:
        return self.hex_color

    @property
    def rgb(self) -> tuple[float, float, float]:
        r = int(self.hex_color[1:3], 16) / 255.0
        g = int(self.hex_color[3:5], 16) / 255.0
        b = int(self.hex_color[5:7], 16) / 255.0
        return r, g, b

    @property
    def hsv(self) -> tuple[float, float, float]:
        return colorsys.rgb_to_hsv(*self.rgb)

    def set_hex(self, hex_color: str) -> None:
        if not isinstance(hex_color, str) or not hex_color.startswith("#") or len(hex_color) != 7:
            raise ValueError("Hex color must be a string in the format '#RRGGBB'.")
        self.hex_color = hex_color

    def set_rgb(self, r: float, g: float, b: float) -> None:
        self.hex_color = "#{:02X}{:02X}{:02X}".format(int(r * 255), int(g * 255), int(b * 255))

    def set_hsv(self, h: float, s: float, v: float) -> None:
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        self.set_rgb(r, g, b)

    def copy(self) -> Color:
        return Color(self.hex_color)

    def range(self, n: int) -> list[str]:
        if n <= 0:
            return []
        colors = []
        for i in range(n):
            color = self.copy()
            h, s, v = color.hsv

            h = h - 0.03 + 0.06 * i / (n-1)
            h = h % 1.0  # Ensure hue wraps around
            v = v + 0.1 - 0.2 * i / (n-1)
            v = max(0, min(1, v))  # Ensure value is between 0 and 1

            color.set_hsv(h, s, v)

            colors.append(color.hex)
        return colors



class SystemPlots(lori.System):
    COLORS = {
        "solar_yellow": Color("#FFB800"),
        "consumption_red": Color("#EF1932"),
        "residual_blue": None,
        "storage_green": None,
    }
    MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


    @classmethod
    def plot_yield(cls, data: pd.DataFrame, solar_columns: list[str]) -> None:
        if SolarSystem.POWER not in data.columns:
            return

        data = data.copy()

        power = data[SolarSystem.POWER]
        power.name = SolarSystem.POWER

        hours = pd.Series(data.index, index=data.index)
        hours = (hours - hours.shift(1)).bfill().dt.total_seconds() / 3600.0
        energy = power * hours  / 1000.0
        energy.name = SolarSystem.ENERGY

        color = cls.COLORS["solar_yellow"]


        cls._plot_monthly(
            energy,
            base_color=color,
            xlabel= "Month",
            ylabel= "Energy Yield [kWh]",
            title= "Monthly Energy Yield",
            # file_name=results.dirs.tmp.joinpath("yield_months.png"),
        )

        cls._plot_interval_quartiles(
            power,
            base_color=color,
            xlabel="Time of day",
            ylabel="Power [kW]",
            title="Daily power",
            interval="day",
            freq="1min",
            # file_name=results.dirs.tmp.joinpath("yield_months.png"),
        )

        cls._plot_interval_monthly_hue(
            power,
            base_color=color,
            xlabel="Time of day",
            ylabel="Power [kW]",
            title="Daily power",
            interval="day",
            freq="1min",
            # file_name=results.dirs.tmp.joinpath("yield_months.png"),
        )

        if solar_columns and len(solar_columns) > 1:
            solar_df = data[[solar_column for solar_column in solar_columns]]

            cls._plot_normalized(
                solar_df,
                base_color=color,
                xlabel="Time of day",
                ylabel="Power normalized",
                title="Solar systems normalized",
                interval="day",
                freq="1min",
                # file_name=results.dirs.tmp.joinpath("yield_months.png"),
            )
            pass



    @classmethod
    def _plot_monthly(
            cls,
            data: pd.Series,
            base_color: Color,
            xlabel: str = "Month",
            ylabel: str = "Energy [kWh]",
            title: str = "Monthly Energy",
            file_name: Optional[str] = None,
    ) -> None:
        base_color = base_color.hex

        # Monthly Yield
        # TODO: How to handle x-axis labels?
        x_ticks = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        plot_data = data.to_frame().groupby(data.index.month).sum()
        plot.bar(
            x=plot_data.index,
            y=SolarSystem.ENERGY,
            data=plot_data,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            colors=[base_color],
            file=str(file_name) if file_name else None,
            show=False,#False if file_name else True,
        )



        pass

    #noinstect PyMethodMayBeStatic,
    @classmethod
    def _plot_interval_quartiles(
            cls,
            data: pd.Series,
            base_color: Color,
            xlabel: str = "Time",
            ylabel: str = "",
            title: str = "Monthly Energy",
            interval: str = "week",
            freq: str = "15min",
            file_name: Optional[str] = None,
    ) -> None:
        base_color = base_color.hex

        plot_data = data.copy()
        plot_data.index = plot_data.index.floor(freq)

        plot_data.index, xlim = cls._conv_interval(interval, plot_data.index, freq=freq)
        plot_data = plot_data.groupby(plot_data.index)

        mean = plot_data.mean().to_frame()
        std = plot_data.std()
        minimum = plot_data.min().to_frame()
        maximum = plot_data.max().to_frame()


        # noinspection PyTypeChecker
        fig, safe_func = plot.line(
            x=mean.index,
            y=data.name,
            data=mean,
            xlabel=xlabel,
            ylabel=ylabel,
            xlim=xlim,
            title=title,
            colors=[base_color],
            file=str(file_name) if file_name else None,
            show=False,#False if file_name else True,
            await_safe=True
        )

        fig.axes[0].fill_between(
            mean.index,
            mean[data.name] - std,
            mean[data.name] + std,
            color=base_color,
            alpha=0.3,
            label='±1σ interval'
        )

        # TODO: imporve df handlig
        fig.axes[0].fill_between(
            minimum.index,
            minimum[data.name],
            maximum[data.name],
            color=base_color,
            alpha=0.2,
            label='Min–Max'
        )

        safe_func()

        pass


    @classmethod
    def _plot_interval_monthly_hue(
            cls,
            data: pd.Series,
            base_color: Color,
            xlabel: str = "Time",
            ylabel: str = "",
            title: str = "Monthly Energy",
            interval: str = "week",
            freq: str = "15min",
            file_name: Optional[str] = None,
    ) -> None:
        plot_data = data.to_frame()
        plot_data["month"] = plot_data.index.month
        plot_data["month"] = plot_data["month"].apply(lambda x: SystemPlots.MONTHS[x - 1])

        plot_data["time"], xlim = cls._conv_interval(interval, plot_data.index)
        plot_data = plot_data.groupby(["time", "month"]).mean().reset_index()

        months_count = plot_data["month"].nunique()

        plot.line(
            x="time",
            y=data.name,
            data=plot_data,
            xlabel=xlabel,
            ylabel=ylabel,
            xlim=xlim,
            title=title,
            colors=base_color.range(months_count),
            file=str(file_name) if file_name else None,
            show=False,  # False if file_name else True,
            #await_safe=True,
            hue="month",
            estimator=None,
            errorbar=None,
        )

        plot.ridge(
            x="time",
            y=data.name,
            data=plot_data,
            xlabel=xlabel,
            ylabel=ylabel,
            xlim=xlim,
            title=title,
            colors=base_color.range(months_count),
            file=str(file_name) if file_name else None,
            show=False,  # False if file_name else True,
            #await_safe=True,
            hue="month",
            estimator=None,
            errorbar=None,
        )

        pass

    @classmethod
    def _plot_normalized(
            cls,
            data: pd.DataFrame,
            base_color: Color,
            xlabel: str = "Time",
            ylabel: str = "",
            title: str = "Monthly Energy",
            interval: str = "week",
            freq: str = "15min",
            file_name: Optional[str] = None,
    ) -> None:
        plot_data = data.copy()
        plot_data = plot_data.reset_index().melt(
            id_vars="timestamp",
            var_name="solar_system",
            value_name="value"
        )
        plot_data.set_index("timestamp", inplace=True)

        plot_data["time"], xlim = cls._conv_interval(interval, plot_data.index)
        plot_data = plot_data.groupby(["time", "solar_system"]).mean().reset_index()
        plot_data["value"] = plot_data["value"] / plot_data.groupby("solar_system")["value"].transform("max")



        plot.line(
            x="time",
            y="value",
            data=plot_data,
            xlabel=xlabel,
            ylabel=ylabel,
            xlim=xlim,
            title=title,
            colors=base_color.range(len(plot_data.columns)),
            file=str(file_name) if file_name else None,
            show=False,  # False if file_name else True,
            # await_safe=True,
            estimator=None,
            errorbar=None,
            hue="solar_system",
        )

        plot.ridge(
            x="time",
            y="value",
            data=plot_data,
            xlabel=xlabel,
            ylabel=ylabel,
            xlim=xlim,
            title=title,
            colors=base_color.range(len(plot_data.columns)),
            file=str(file_name) if file_name else None,
            show=True,  # False if file_name else True,
            # await_safe=True,
            estimator=None,
            errorbar=None,
            hue="solar_system",
        )
        pass

    def _plot_daily(cls):
        pass

    @classmethod
    def _conv_interval(
            cls,
            interval: str,
            index: pd.DatetimeIndex,
            freq: str = None
    ) -> tuple[pd.Series, list[float]]:
        if freq is not None:
            index = index.floor(freq)

        if interval == "year":
            index = index.day_of_year
            xlim = [0, 365]
        elif interval == "month":
            index = index.day_of_month
            xlim = [1, 31]
        elif interval == "week":
            index = index.day_of_week + index.hour / 24.0
            xlim = [0, 7]
        elif interval == "day":
            index = index.hour + index.minute / 60.0
            xlim = [0, 24]
        else:
            raise ValueError(f"Invalid interval '{interval}'.")

        return index, xlim

    # # noinspection PyTypeChecker
    # def _plot_system(
    #     self,
    #     data: pd.DataFrame,
    #     title: Optional[str] = None,
    #     width: Optional[int] = None,
    #     height: Optional[int] = None,
    #     show: bool = False,
    #     file: str = None,
    # ) -> None:
    #     # Ignore this error, as pandas implements its own matplotlib converters for handling datetime or period values.
    #     # When seaborn and pandas plots are mixed, converters may conflict and this warning is shown.
    #     import warnings
    #
    #     import matplotlib.dates as dates
    #     import matplotlib.pyplot as plt
    #     import seaborn as sns
    #
    #     from lori.io import plot
    #
    #     warnings.filterwarnings(
    #         "ignore",
    #         message="This axis already has a converter set and is updating to a potentially incompatible converter",
    #     )
    #
    #     columns_power = [System.POWER_EL]
    #
    #     # TODO: Replace with tariff component constants
    #     has_tariff = "tariff" in data.columns
    #     has_solar = self.components.has_type(SolarSystem)
    #     if has_solar:
    #         columns_power.append(SolarSystem.POWER)
    #     has_ees = self.components.has_type(ElectricalEnergyStorage)
    #     if has_ees:
    #         columns_power.append(ElectricalEnergyStorage.POWER_CHARGE)
    #
    #     data_power = deepcopy(data[columns_power])
    #     data_power /= 1000
    #
    #     if width is None:
    #         width = plot.WIDTH
    #     if height is None:
    #         height = plot.HEIGHT
    #     figure, ax_power = plt.subplots(figsize=(width / plot.INCH, height / plot.INCH), dpi=120)
    #     axes = [ax_power]
    #
    #     sns.lineplot(
    #         data_power[System.POWER_EL],
    #         linewidth=0.25,
    #         color="#004f9e",
    #         label="_hidden",
    #         ax=axes[0],
    #     )
    #     if has_ees:
    #         ax_soc = axes[0].twinx()
    #         axes.append(ax_soc)
    #
    #         sns.lineplot(
    #             data_power[ElectricalEnergyStorage.POWER_CHARGE],
    #             linewidth=0.25,
    #             color="#30a030",
    #             label="_hidden",
    #             ax=ax_power,
    #         )
    #         sns.lineplot(
    #             data[ElectricalEnergyStorage.STATE_OF_CHARGE],
    #             linewidth=1,
    #             color="#037003",
    #             label="Battery State",
    #             ax=ax_soc,
    #         )
    #
    #         data_ref = data_power[System.POWER_EL] - data_power[ElectricalEnergyStorage.POWER_CHARGE]
    #         data_ref.plot.area(
    #             stacked=False,
    #             label="_hidden",
    #             color={"_hidden": "#dddddd"},
    #             linewidth=0,
    #             alpha=0.75,
    #             ax=ax_power,
    #         )
    #
    #         ax_soc.set_ylim(-1, 119)
    #         ax_soc.yaxis.set_label_text("State of Charge [%]")
    #         if has_tariff:
    #             ax_soc.legend(ncol=1, loc="upper right", bbox_to_anchor=(0.84, 1), frameon=False)
    #         else:
    #             ax_soc.legend(ncol=1, loc="upper right", frameon=False)
    #
    #     if has_solar:
    #         data_power[SolarSystem.POWER].plot.area(
    #             stacked=False,
    #             label="PV Generation",
    #             color={"PV Generation": "#ffeb9b"},
    #             linewidth=0,
    #             alpha=0.75,
    #             ax=ax_power,
    #         )
    #
    #     data_power[System.POWER_EL].plot.area(
    #         stacked=False,
    #         label="Residual Load",
    #         alpha=0.25,
    #         ax=ax_power,
    #     )
    #
    #     if has_ees:
    #         data_power[ElectricalEnergyStorage.POWER_CHARGE].plot.area(
    #             stacked=False,
    #             label="Battery Charging",
    #             color={"Battery Charging": "#80df95"},
    #             alpha=0.25,
    #             ax=ax_power,
    #         )
    #
    #     if has_tariff:
    #         # TODO: Replace with tariff component constants
    #         tariff = data["tariff"]
    #
    #         ax_price = axes[0].twinx()
    #         axes.append(ax_price)
    #
    #         sns.lineplot(tariff, linewidth=1, color="#999999", label="Dynamic Tariff", ax=ax_price)
    #
    #         ax_price.spines.right.set_position(("axes", 1.07))
    #         ax_price.set_ylim(min(tariff.min() - 0.05), max(tariff.max()) + 0.1)
    #         ax_price.yaxis.set_label_text("Price [€/kWh]")
    #         ax_price.legend(ncol=1, loc="upper right", frameon=False)
    #
    #     ax_power.set_xlim(data_power.index[0], data_power.index[-1])
    #     ax_power.set_ylim(min(data_power.min()), max(data_power.max()) + 50)
    #     ax_power.xaxis.set_minor_locator(dates.HourLocator(interval=12))
    #     ax_power.xaxis.set_minor_formatter(dates.DateFormatter("%H:%M", tz="Europe/Berlin"))
    #     ax_power.xaxis.set_major_locator(dates.DayLocator(interval=1))
    #     ax_power.xaxis.set_major_formatter(dates.DateFormatter("\n%A", tz="Europe/Berlin"))
    #     ax_power.xaxis.set_label_text(f"{data.index[0].strftime('%d. %B')} to " f"{data.index[-1].strftime('%d. %B')}")
    #     # ax_power.xaxis.label.set_visible(False)
    #     ax_power.yaxis.set_label_text("Power [kW]")
    #     ax_power.legend(ncol=3, loc="upper left", frameon=False)
    #
    #     for pos in ["right", "top", "bottom", "left"]:
    #         for ax in axes:
    #             ax.spines[pos].set_visible(False)
    #
    #     axes[0].grid(color="grey", linestyle="--", linewidth=0.25, alpha=0.5, axis="both")
    #     axes[0].set_title(title)
    #     figure.tight_layout()
    #
    #     if file is not None:
    #         figure.savefig(file)
    #     if show:
    #         figure.show()
    #         # figure.waitforbuttonpress()
    #
    #     plt.close(figure)
    #     plt.clf()
