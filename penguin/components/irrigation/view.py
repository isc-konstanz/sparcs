# -*- coding: utf-8 -*-
"""
penguin.components.irrigation.view
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import Input, Output, callback, dcc, html

from loris.app.view.pages import (
    ComponentGroup,
    ComponentPage,
    PageLayout,
    register_component_group,
    register_component_page,
)
from penguin.components.irrigation import IrrigationSystem


@register_component_page(IrrigationSystem)
class IrrigationPage(ComponentPage[IrrigationSystem]):
    def create_layout(self, layout: PageLayout) -> None:
        super().create_layout(layout)

        humidity = self._build_humidity()
        layout.card.append(humidity, focus=True)
        layout.append(
            dbc.Row(
                dbc.Col(humidity, width="auto")
            )
        )

    def _build_humidity(self) -> html.Div:

        @callback(Output(f"{self.id}-humidity", "children"),
                  Input(f"{self.id}-humidity-update", "n_intervals"))
        def _update_humidity(*_) -> html.P | dbc.Spinner:
            humidity = self.data.humidity_mean
            if humidity.is_valid():
                return html.P(
                    f"{round(humidity.value, 1)}%",
                    style={
                        "color": "#a7c7e7",
                        "fontSize": "4rem"
                    }
                )
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


@register_component_group(IrrigationSystem, name="Irrigation")
class IrrigationGroup(ComponentGroup[IrrigationSystem]):
    pass
