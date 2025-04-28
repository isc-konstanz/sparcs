# -*- coding: utf-8 -*-
"""
penguin.application.view.irrigation.series
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Dict, Union

import dash_bootstrap_components as dbc
from dash import Input, Output, callback, dcc, html

from lori import Channel, Channels, Constant
from lori.application.view.pages import ComponentPage, PageLayout, register_component_page
from penguin.components.irrigation import IrrigationSeries, SoilMoisture


@register_component_page(IrrigationSeries)
class IrrigationSeriesPage(ComponentPage[IrrigationSeries]):
    @property
    def soil(self) -> SoilMoisture:
        return self._component.soil

    def create_layout(self, layout: PageLayout) -> None:
        super().create_layout(layout)

        switch = self._build_switch()
        layout.card.append(switch, focus=True)
        layout.append(html.Div(switch))
        layout.append(html.Hr())

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
                        self._build_soil_value(
                            SoilMoisture.WATER_SUPPLY,
                            color="#68adff",
                            style={"min-width": "14rem"},
                        ),
                        width="auto",
                    ),
                    dbc.Col(
                        self._build_soil_value(
                            SoilMoisture.WATER_CONTENT,
                            color="#8fd0ff",
                            style={"min-width": "10rem"},
                        ),
                        width="auto",
                    ),
                ],
            ),
        ]
        temperature = [
            dbc.Row(dbc.Col(html.H5("Soil"))),
            dbc.Row(dbc.Col(html.H6("Temperature"))),
            dbc.Row(dbc.Col(self._build_soil_value(SoilMoisture.TEMPERATURE, color="#ff746c"))),
        ]

        layout.card.append(html.Div(moisture), focus=True)
        layout.card.append(html.Div(temperature))

        layout.append(html.Div(moisture))
        layout.append(html.Div(temperature))

    def _create_data_layout(self, layout: PageLayout, channels: Channels, **kwargs) -> None:
        super()._create_data_layout(layout, channels + self.soil.data.channels, **kwargs)

    # noinspection PyShadowingBuiltins
    def _build_switch(self) -> html.Div:
        id = f"{self.id}-state"

        @callback(
            Input(id, "value"),
            force_no_output=True,
        )
        def _update_state(state: bool) -> None:
            _state = self.data.irrigation_state
            if _state.is_valid() and _state.value != state:
                _state.write(state)

        @callback(
            Output(id, "value"),
            Input(f"{id}-update", "n_intervals"),
        )
        def _update_switch(*_) -> bool:
            _state = self.data.irrigation_state
            if _state.is_valid():
                return _state.value
            return False

        return html.Div(
            [
                html.H5("State"),
                dbc.Switch(
                    id=id,
                    # label="State",
                    style={"fontSize": "1.5rem"},
                    value=_update_switch(),
                ),
                dcc.Interval(
                    id=f"{id}-update",
                    interval=60000,
                    n_intervals=0,
                ),
            ]
        )

    # noinspection PyShadowingBuiltins
    def _build_soil_value(self, constant: Constant, *args, **kwargs) -> html.Div:
        id = f"{self.id}-{constant.key.replace('_', '-')}"
        channel = self.soil.data[constant.key]
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
