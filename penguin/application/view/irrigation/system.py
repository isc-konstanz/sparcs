# -*- coding: utf-8 -*-
"""
penguin.application.view.irrigation.system
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""

from __future__ import annotations

from functools import wraps
from typing import Optional

import dash_bootstrap_components as dbc
from dash import Input, Output, callback, html

from lori.application.view.pages import (
    ComponentGroup,
    ComponentPage,
    PageLayout,
    register_component_group,
    register_component_page,
)
from penguin.application.view.irrigation import IrrigationSeriesPage
from penguin.components.irrigation import IrrigationSystem

KEY = "irrigation"
NAME = "Irrigation"


@register_component_page(IrrigationSystem)
class IrrigationSystemPage(ComponentGroup, ComponentPage[IrrigationSystem]):
    def __init__(self, irrigation: IrrigationSystem, *args, **kwargs) -> None:
        super().__init__(component=irrigation, *args, **kwargs)
        for series in irrigation.series:
            self.append(IrrigationSeriesPage(self, series))

    @property
    def key(self) -> str:
        return self._component.key

    @property
    def path(self) -> str:
        return f"/{KEY}/{self._encode_id(self.key)}"

    def create_layout(self, layout: PageLayout) -> None:
        super().create_layout(layout)

        water_supply = self._build_water_supply()
        layout.card.append(water_supply, focus=True)

    def _create_data_layout(self, layout: PageLayout, title: Optional[str] = "Data") -> None:
        if len(self.data.channels) > 0:
            data = []
            if title is not None:
                data.append(html.H5(f"{title}:"))
            data.append(self._build_data())
            layout.append(dbc.Row(dbc.Col(dbc.Card(dbc.CardBody(data)))))

    @wraps(create_layout, updated=())
    def _do_create_layout(self, *args, **kwargs) -> None:
        for page in self:
            page._do_create_layout(*args, **kwargs)
        super()._do_create_layout(*args, **kwargs)

    def _do_register(self) -> None:
        super()._do_register()
        for page in self:
            page._do_register()

    def _build_water_supply(self) -> html.Div:
        @callback(
            Output(f"{self.id}-water-supply-mean", "children"),
            Input("view-update", "n_intervals"),
        )
        def _update_water_supply(*_) -> html.P | dbc.Spinner:
            water_supply = self.data.water_supply_mean
            if water_supply.is_valid():
                return html.P(f"{round(water_supply.value, 1)}%", style={"color": "#68adff", "fontSize": "4rem"})
            return dbc.Spinner(html.Div(id=f"{self.id}-water-supply-mean-loader"))

        return html.Div(
            [
                html.H5("Soil moisture"),
                html.H6("Water supply coverage"),
                html.Div(
                    _update_water_supply(),
                    id=f"{self.id}-water-supply-mean",
                ),
            ]
        )


@register_component_group(IrrigationSystem, key=KEY, name=NAME)
class IrrigationGroup(ComponentGroup[IrrigationSystem]):
    pass
