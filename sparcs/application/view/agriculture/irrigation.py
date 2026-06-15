# -*- coding: utf-8 -*-
"""
sparcs.application.view.agriculture.irrigation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""

from __future__ import annotations

from typing import Optional

import dash_bootstrap_components as dbc
from dash import Input, Output, callback, dcc, html

from lories.application.view.pages import ComponentPage, PageLayout, register_component_page
from sparcs.components.agriculture import Irrigation


@register_component_page(Irrigation)
class IrrigationPage(ComponentPage[Irrigation]):
    def create_layout(self, layout: PageLayout) -> None:
        super().create_layout(layout)

        irrigation = self._build_irrigation_layout()
        layout.card.append(irrigation, focus=True)
        layout.append(html.Div(irrigation))
        layout.append(html.Hr())

    # noinspection PyShadowingBuiltins
    def _build_irrigation_layout(self) -> html.Div:
        _switch = self._build_switch_layout()

        def _build_flow_unit(unit: Optional[str]) -> html.Span:
            if unit is None:
                return html.Span()
            fraction = unit.split("/")
            if len(fraction) != 2:
                return html.Span(unit, style={"fontSize": "1rem"})
            return html.Span(
                [
                    html.Span(fraction[0], style={"display": "block"}),
                    html.Span(fraction[1], style={"display": "block", "border-top": "2px solid"}),
                ],
                style={"display": "inline-block", "text-align": "center", "fontSize": "1.25rem", "margin": "0 0.5rem"},
            )

        @callback(
            Output(f"{self.id}-flow", "children"),
            Input("view-update", "n_intervals"),
        )
        def _update_flow(*_) -> html.P | dbc.Spinner:
            _flow = self.data.flow
            if _flow.is_valid():
                return html.P(
                    [round(_flow.value, 1), _build_flow_unit(_flow.get("unit", None))],
                    style={"min-width": "7rem", "color": "#68adff", "fontSize": "4rem"},
                )
            return dbc.Spinner(html.Div(id=f"{self.id}-flow-loader"))

        return html.Div(
            [
                dbc.Row([dbc.Col(html.H5("Irrigation", style={"min-width": "14rem"}), width="auto")]),
                dbc.Row(
                    [
                        dbc.Col(html.H6("Watering State", style={"min-width": "7rem"}), width="auto"),
                        dbc.Col(html.H6("Water flow", style={"min-width": "7rem"}), width="auto"),
                    ],
                ),
                dbc.Row(
                    [
                        dbc.Col(html.Div(_switch), width="auto"),
                        dbc.Col(html.Div(_update_flow(), id=f"{self.id}-flow"), width="auto"),
                    ],
                ),
            ]
        )

    # noinspection PyShadowingBuiltins
    def _build_switch_layout(self) -> html.Div:
        _state_id = f"{self.id}-state"

        @callback(
            Input(_state_id, "value"),
            force_no_output=True,
        )
        def _update_state(state: bool) -> None:
            _state = self.data.state
            if _state.is_valid() and _state.value != state:
                _state.write(state)

        @callback(
            Output(_state_id, "value"),
            Input(f"{_state_id}-update", "n_intervals"),
        )
        def _update_switch(*_) -> bool:
            _state = self.data.state
            if _state.is_valid():
                return _state.value
            return False

        return html.Div(
            [
                dbc.Switch(
                    id=_state_id,
                    # label="State",
                    style={"min-width": "7rem", "padding-top": "1rem", "fontSize": "1.5rem"},
                    value=_update_switch(),
                ),
                dcc.Interval(
                    id=f"{_state_id}-update",
                    interval=60000,
                    n_intervals=0,
                ),
            ]
        )
