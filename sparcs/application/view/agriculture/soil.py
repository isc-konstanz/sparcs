# -*- coding: utf-8 -*-
"""
sparcs.application.view.agriculture.soil
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Dict, Union

import dash_bootstrap_components as dbc
from dash import Input, Output, callback, html

from lories import Channel, Constant
from lories.application.view.pages import ComponentPage, PageLayout, register_component_page
from sparcs.components.agriculture import SoilMoisture


@register_component_page(SoilMoisture)
class SoilModelPage(ComponentPage[SoilMoisture]):
    def create_layout(self, layout: PageLayout) -> None:
        super().create_layout(layout)

        moisture = [
            dbc.Row(dbc.Col(html.H5("Soil moisture"), width="auto")),
            dbc.Row(
                [
                    dbc.Col(html.H6("Water supply coverage", style={"min-width": "14rem"}), width="auto"),
                    dbc.Col(html.H6("Water content", style={"min-width": "10rem"}), width="auto"),
                ],
            ),
            dbc.Row(
                [
                    dbc.Col(
                        self._build_value(
                            SoilMoisture.WATER_SUPPLY,
                            color="#68adff",
                            style={"min-width": "14rem"},
                        ),
                        width="auto",
                    ),
                    dbc.Col(
                        self._build_value(
                            SoilMoisture.WATER_CONTENT,
                            color="#8fd0ff",
                            style={"min-width": "10rem"},
                        ),
                        width="auto",
                    ),
                ],
            ),
        ]
        layout.card.append(html.Div(moisture), focus=True)
        layout.append(html.Div(moisture))

        if self.data[SoilMoisture.TEMPERATURE].has_connector():
            temperature = [
                dbc.Row(dbc.Col(html.H5("Soil"))),
                dbc.Row(dbc.Col(html.H6("Temperature"))),
                dbc.Row(dbc.Col(self._build_value(SoilMoisture.TEMPERATURE, color="#ff746c"))),
            ]
            layout.card.append(html.Div(temperature))
            layout.append(html.Div(temperature))

    # noinspection PyShadowingBuiltins
    def _build_value(self, constant: Constant, *args, **kwargs) -> html.Div:
        id = f"{self.id}-{constant.key.replace('_', '-')}"
        channel = self.data[constant.key]
        channel_callback = callback(
            Output(id, "children"),
            Input("view-update", "n_intervals"),
        )(ChannelCallback(channel, unit=constant.unit, *args, **kwargs))
        return html.Div(channel_callback(), id=id)


class ChannelCallback(Callable[[int], Union[html.P, dbc.Spinner]]):
    channel: Channel

    unit: str
    decimal_digits: int

    style: Dict[str, Any]

    # noinspection PyShadowingBuiltins
    def __init__(
        self,
        channel: Channel,
        unit: str = "",
        color: str = "#373f43",
        font_size="4rem",
        decimal_digits: int = 1,
        style: Dict[str, Any] = None,
    ) -> None:
        self.channel = channel
        if style is None:
            style = {}
        if "color" not in style:
            style["color"] = color
        if "fontSize" not in style:
            style["fontSize"] = font_size
        self.style = style
        self.unit = unit
        self.decimal_digits = decimal_digits

    def __call__(self, *_) -> html.P | dbc.Spinner:
        if self.channel.is_valid():
            return html.P(
                f"{round(self.channel.value, self.decimal_digits)}{self.unit}",
                style=self.style,
            )
        return dbc.Spinner(
            color=self.style["color"],
            spinner_style={
                "width": self.style["fontSize"],
                "height": self.style["fontSize"],
            },
        )
