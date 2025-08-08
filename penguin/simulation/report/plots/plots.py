# -*- coding: utf-8 -*-
"""
penguin.simulation.report.plots.plots
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""

from __future__ import annotations

from copy import deepcopy
import warnings
from typing import Optional
import pandas as pd
import matplotlib.dates as dates
import matplotlib.pyplot as plt
import seaborn as sns

    
from lori.io import plot
from penguin import System
from penguin.components import ElectricalEnergyStorage, SolarSystem
from penguin.simulation.report.plots.base import (
    Color,
    plot_monthly_bars, 
    plot_interval_quartiles, 
    plot_interval_monthly_hue, 
    plot_normalized
)

COLORS = {
    "solar_yellow": Color("#FFB800"),
    "consumption_red": Color("#EF1932"),
    "residual_blue": Color("#004F9E"),
    "storage_green": Color("#037003"),
}

def plot_yield(
        data: pd.DataFrame,
) -> None:
    if SolarSystem.POWER not in data.columns:
        return

    data = data.copy()

    power = data[SolarSystem.POWER]
    power.name = SolarSystem.POWER

    hours = pd.Series(data.index, index=data.index)
    hours = (hours - hours.shift(1)).bfill().dt.total_seconds() / 3600.0
    energy = power * hours  / 1000.0
    energy.name = SolarSystem.ENERGY

    solar_color = COLORS["solar_yellow"]


    plot_monthly_bars(
        energy,
        base_color=solar_color,
        xlabel= "Month",
        ylabel= "Energy Yield [kWh]",
        title= "Monthly Energy Yield",
        # file_name=results.dirs.tmp.joinpath("yield_months.png"),
    )

    plot_interval_quartiles(
        power,
        base_color=solar_color,
        xlabel="Time of day",
        ylabel="Power [kW]",
        title="Daily power",
        interval="day",
        freq="1min",
        # file_name=results.dirs.tmp.joinpath("yield_months.png"),
    )

    plot_interval_monthly_hue(
        power,
        base_color=solar_color,
        xlabel="Time of day",
        ylabel="Power [kW]",
        title="Daily power",
        interval="day",
        freq="1min",
        # file_name=results.dirs.tmp.joinpath("yield_months.png"),
    )

    #if solar_columns and len(solar_columns) > 1:
    #    solar_df = data[[solar_column for solar_column in solar_columns]]
#
    #    plot_normalized(
    #        solar_df,
    #        base_color=solar_color,
    #        xlabel="Time of day",
    #        ylabel="Power normalized",
    #        title="Solar systems normalized",
    #        interval="day",
    #        freq="1min",
    #        # file_name=results.dirs.tmp.joinpath("yield_months.png"),
    #    )
    #    pass


# noinspection PyTypeChecker
def plot_system(
    self,
    data: pd.DataFrame,
    title: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    show: bool = False,
    file: str = None,
) -> None:
    

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
        color=COLORS["residual_blue"].hex,
        label="_hidden",
        ax=axes[0],
    )
    if has_ees:
        ax_soc = axes[0].twinx()
        axes.append(ax_soc)

        sns.lineplot(
            data_power[ElectricalEnergyStorage.POWER_CHARGE],
            linewidth=0.25,
            color=COLORS["storage_green"].light.hex,
            label="_hidden",
            ax=ax_power,
        )
        sns.lineplot(
            data[ElectricalEnergyStorage.STATE_OF_CHARGE],
            linewidth=1,
            color=COLORS["storage_green"].hex,
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
