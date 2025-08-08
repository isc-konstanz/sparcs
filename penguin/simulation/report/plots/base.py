# -*- coding: utf-8 -*-
"""
penguin.system
~~~~~~~~~~~~~~


"""
from __future__ import annotations

from typing import Any, Optional
import pandas as pd
import colorsys

from lori.io import plot
from penguin.components import SolarSystem


class Color:
    hex_color: str

    def __init__(self, hex_color: str) -> None:
        self.hex_color = hex_color

    def __str__(self):
        return f"{self.hex_color}"

    def __repr__(self):
        return f"{self.hex_color}\trgb:{self.rgb}\thsv:{self.hsv}"
    
    @classmethod
    def from_rgb(cls, r: float, g: float, b: float) -> Color:
        hex_color = "#{:02X}{:02X}{:02X}".format(int(r * 255), int(g * 255), int(b * 255))
        return cls(hex_color)
    
    @classmethod
    def from_hsv(cls, h: float, s: float, v: float) -> Color:
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return cls.from_rgb(r, g, b) 

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
    
    @property
    def light(self) -> Color:
        h, s, v = self.hsv
        v = min(1.0, v + 0.2)  # Increase value by 20%
        return Color.from_hsv(h, s, v)
    
    @property
    def dark(self) -> Color:
        h, s, v = self.hsv
        v = max(0.0, v - 0.2)  # Decrease value by 20%
        return Color.from_hsv(h, s, v)

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



def plot_monthly_bars(
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
        show=False,  # False if file_name else True,
    )


def plot_interval_quartiles(
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
    #plot_data.index = plot_data.index.floor(freq)

    plot_data.index, xlim = _conv_interval(interval, plot_data.index, freq=freq)
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
        show=False,  # False if file_name else True,
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

    # TODO: imporve df handling
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


def plot_interval_monthly_hue(
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
    #plot_data["month"] = plot_data["month"].apply(lambda x: SystemPlots.MONTHS[x - 1])

    plot_data["time"], xlim = _conv_interval(interval, plot_data.index)
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
        # await_safe=True,
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
        # await_safe=True,
        hue="month",
        estimator=None,
        errorbar=None,
    )

    pass


def plot_normalized(
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
        id_vars="index",
        var_name="solar_system",
        value_name="value"
    )
    plot_data.set_index("index", inplace=True)

    plot_data["time"], xlim = _conv_interval(interval, plot_data.index)
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


def _plot_daily():
    pass


def _conv_interval(
        interval: str,
        index: pd.DatetimeIndex,
        freq: str = None
) -> tuple[pd.Series, list[float]]:
    if freq is not None:
        pass
        #index = index.floor(freq)

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