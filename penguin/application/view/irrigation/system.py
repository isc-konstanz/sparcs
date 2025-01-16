# -*- coding: utf-8 -*-
"""
penguin.application.view.irrigation.system
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import Input, Output, callback, html

from lori.application.view.pages import ComponentGroup, PageLayout, register_component_group, register_component_page
from penguin.application.view.irrigation import IrrigationSeriesPage
from penguin.components.irrigation import IrrigationSystem


@register_component_page(IrrigationSystem, children={"series": IrrigationSeriesPage})
@register_component_group(IrrigationSystem, name="Irrigation")
class IrrigationSystemPage(ComponentGroup[IrrigationSystem]):
    def create_layout(self, layout: PageLayout) -> None:
        super().create_layout(layout)

        water_supply = self._build_water_supply()
        layout.card.append(water_supply, focus=True)

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
