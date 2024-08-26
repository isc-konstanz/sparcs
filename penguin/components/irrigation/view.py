# -*- coding: utf-8 -*-
"""
loris.components.irrigation.view
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""

from __future__ import annotations

from typing import Collection

import dash_bootstrap_components as dbc
from dash import Input, Output, html, callback, dcc

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

    # noinspection PyTypeChecker
    def create_layout(self, layout: PageLayout) -> None:
        super().create_layout(layout)

        def _get_humidity() -> Collection[html.Col]:
            style = {
                "color": "#a7c7e7",
                "fontSize": "5rem"
            }
            return [
                dbc.Col(html.Span(f"{round(self.data.humidity_mean.value, 2)}%", style=style), width=3),
            ]

        @callback(Output(f"{self.id}-humidity-data", "children"),
                  Input(f"{self.id}-humidity-data-update-interval", "n_intervals"))
        def _update_data(n_intervals: int):
            return _get_humidity()

        layout.append(
            dbc.Row(
                [
                    dbc.Col(html.H5("Humidity"), width=3),
                ],
            )
        )
        layout.append(
            dbc.Row(id=f"{self.id}-humidity-data")
        )
        layout.append(
            dcc.Interval(
                id=f"{self.id}-humidity-data-update-interval",
                interval=1000,
                n_intervals=0,
            )
        )


@register_component_group(IrrigationSystem, name="Irrigation")
class IrrigationGroup(ComponentGroup[IrrigationSystem]):
    pass
