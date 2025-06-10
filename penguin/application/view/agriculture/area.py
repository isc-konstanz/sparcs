# -*- coding: utf-8 -*-
"""
penguin.application.view.agriculture.area
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""

from __future__ import annotations

from typing import Optional

import dash_bootstrap_components as dbc
from dash import Input, Output, callback, html

from lori.application.view.pages import ComponentGroup, PageLayout, register_component_group, register_component_page
from penguin.components.agriculture import AgriculturalArea
from penguin.components.storage import WaterStorage


@register_component_page(AgriculturalArea)
@register_component_group(AgriculturalArea, name="Agriculture")
class AgriculturalAreaPage(ComponentGroup[AgriculturalArea]):
    @property
    def water_storage(self) -> Optional[WaterStorage]:
        return self._component.water_storage

    def has_water_storage(self) -> bool:
        return self._component.has_water_storage()

    def create_layout(self, layout: PageLayout) -> None:
        super().create_layout(layout)
        if self.has_water_storage():
            water_storage = self.get_page(self.water_storage)
            layout.card.extend(water_storage.layout.card)

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
