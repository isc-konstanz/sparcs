# -*- coding: utf-8 -*-
"""
penguin.application.view.storage.water
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Dict, Union

import dash_bootstrap_components as dbc
from dash import Input, Output, callback, html

from lori import Channel, Constant
from lori.application.view.pages import ComponentPage, PageLayout, register_component_page
from penguin.components.storage import WaterStorage


@register_component_page(WaterStorage)
class WaterStoragePage(ComponentPage[WaterStorage]):
    def create_layout(self, layout: PageLayout) -> None:
        super().create_layout(layout)

        storage = self._build_storage_layout()
        layout.card.append(storage, focus=True)
        layout.append(storage)

    def _build_storage_layout(self) -> html.Div:
        return html.Div(
            [
                dbc.Row([dbc.Col(html.H5("State", style={"min-width": "14rem"}), width="auto")]),
                dbc.Row([dbc.Col(html.H6("Level", style={"min-width": "14rem"}), width="auto")]),
                dbc.Row(
                    [
                        dbc.Col(
                            self._build_channel_layout(
                                WaterStorage.LEVEL,
                                color="#0a4f8c",
                                style={"min-width": "14rem"},
                            ),
                            width="auto",
                        ),
                    ],
                ),
            ]
        )

    # noinspection PyShadowingBuiltins
    def _build_channel_layout(self, constant: Constant, *args, **kwargs) -> html.Div:
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
