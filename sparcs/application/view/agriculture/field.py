# -*- coding: utf-8 -*-
"""
sparcs.application.view.agriculture.field
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""

from __future__ import annotations

from typing import Sequence

import dash_bootstrap_components as dbc
from dash import Input, Output, callback, html

from lories.application.view.pages import ComponentGroup, PageLayout, register_component_page
from sparcs.components.agriculture import AgriculturalField, Irrigation, SoilMoisture


@register_component_page(AgriculturalField)
class AgriculturalFieldPage(ComponentGroup[AgriculturalField]):
    @property
    def soil(self) -> Sequence[SoilMoisture]:
        return self._component.soil

    @property
    def irrigation(self) -> Irrigation:
        return self._component.irrigation

    def has_irrigation(self) -> bool:
        return self._component.has_irrigation()

    def create_layout(self, layout: PageLayout) -> None:
        super().create_layout(layout)
        if self.has_irrigation():
            irrigation = self.get_page(self.irrigation)
            layout.card.extend(irrigation.layout.card)

        soil = self._build_soil_layout()
        layout.card.append(soil, focus=True)

    def _build_soil_layout(self) -> html.Div:
        @callback(
            Output(f"{self.id}-water-supply-mean", "children"),
            Input("view-update", "n_intervals"),
        )
        def _update_water_supply(*_) -> html.P | dbc.Spinner:
            water_supply = self.data.water_supply_mean
            if water_supply.is_valid():
                return html.P(
                    f"{round(water_supply.value, 1)}%",
                    style={"min-width": "14rem", "color": "#68adff", "fontSize": "4rem"},
                )
            return dbc.Spinner(html.Div(id=f"{self.id}-water-supply-mean-loader"))

        return html.Div(
            [
                dbc.Row([dbc.Col(html.H5("Soil moisture", style={"min-width": "14rem"}), width="auto")]),
                dbc.Row([dbc.Col(html.H6("Water supply coverage", style={"min-width": "14rem"}), width="auto")]),
                dbc.Row([dbc.Col(html.Div(_update_water_supply(), id=f"{self.id}-water-supply-mean"), width="auto")]),
            ]
        )
