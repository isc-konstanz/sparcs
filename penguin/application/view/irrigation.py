# -*- coding: utf-8 -*-
"""
penguin.components.irrigation.view
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""

from __future__ import annotations

from typing import Optional

import dash_bootstrap_components as dbc
from dash import Input, Output, callback, dcc, html

from lori.application.view.pages import (
    ComponentGroup,
    ComponentPage,
    PageLayout,
    register_component_group,
    register_component_page,
)
from penguin.components.irrigation import IrrigationSeries, IrrigationSystem

KEY = "irrigation"
NAME = "Irrigation"


@register_component_page(IrrigationSystem)
class IrrigationSystemPage(ComponentGroup, ComponentPage[IrrigationSystem]):
    def __init__(self, irrigation: IrrigationSystem, *args, **kwargs) -> None:
        super().__init__(component=irrigation, *args, **kwargs)
        for series in irrigation.series:
            self.append(IrrigationSeriesPage(self, series))

    @property
    def path(self) -> str:
        return f"/{KEY}/{self._encode_id(self.key)}"

    def create_layout(self, layout: PageLayout) -> None:
        super().create_layout(layout)

        humidity_mean = self._build_humidity_mean()
        layout.card.append(humidity_mean, focus=True)

    def _create_data_layout(self, layout: PageLayout, title: Optional[str] = "Data") -> None:
        if len(self.data.channels) > 0:
            data = []
            if title is not None:
                data.append(html.H5(f"{title}:"))
            data.append(self._build_data())
            layout.append(dbc.Row(dbc.Col(dbc.Card(dbc.CardBody(data)))))

    def _do_create_layout(self) -> PageLayout:
        for page in self:
            page._do_create_layout()
        return super()._do_create_layout()

    def _do_register(self) -> None:
        super()._do_register()
        for page in self:
            page._do_register()

    def _build_humidity_mean(self) -> html.Div:
        @callback(
            Output(f"{self.id}-humidity-mean", "children"),
            Input(f"{self.id}-humidity-mean-update", "n_intervals"),
        )
        def _update_humidity(*_) -> html.P | dbc.Spinner:
            humidity = self.data.humidity_mean
            if humidity.is_valid():
                return html.P(f"{round(humidity.value, 1)}%", style={"color": "#a7c7e7", "fontSize": "4rem"})
            return dbc.Spinner(html.Div(id=f"{self.id}-humidity-mean-loader"))

        return html.Div(
            [
                html.H5("Soil moisture"),
                html.Div(
                    _update_humidity(),
                    id=f"{self.id}-humidity-mean",
                ),
                dcc.Interval(
                    id=f"{self.id}-humidity-mean-update",
                    interval=1000,
                    n_intervals=0,
                ),
            ]
        )


class IrrigationSeriesPage(ComponentPage[IrrigationSeries]):
    def __init__(self, system: IrrigationSystemPage, *args, **kwargs) -> None:
        super().__init__(group=system, *args, **kwargs)

    def create_layout(self, layout: PageLayout) -> None:
        super().create_layout(layout)

        humidity = self._build_humidity()
        layout.card.append(humidity, focus=True)
        layout.append(dbc.Row(dbc.Col(humidity, width="auto")))

    def _build_humidity(self) -> html.Div:
        @callback(
            Output(f"{self.id}-humidity", "children"),
            Input(f"{self.id}-humidity-update", "n_intervals"),
        )
        def _update_humidity(*_) -> html.P | dbc.Spinner:
            humidity = self.data.humidity
            if humidity.is_valid():
                return html.P(f"{round(humidity.value, 1)}%", style={"color": "#a7c7e7", "fontSize": "4rem"})
            return dbc.Spinner(html.Div(id=f"{self.id}-humidity-loader"))

        return html.Div(
            [
                html.H5("Soil moisture"),
                html.Div(
                    _update_humidity(),
                    id=f"{self.id}-humidity",
                ),
                dcc.Interval(
                    id=f"{self.id}-humidity-update",
                    interval=1000,
                    n_intervals=0,
                ),
            ]
        )


@register_component_group(IrrigationSystem, key=KEY, name=NAME)
class IrrigationGroup(ComponentGroup[IrrigationSystem]):
    pass
