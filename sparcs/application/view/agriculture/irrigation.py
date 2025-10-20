# -*- coding: utf-8 -*-
"""
sparcs.application.view.agriculture.irrigation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import Input, Output, callback, dcc, html

from lories.application.view.pages import ComponentPage, PageLayout, register_component_page
from sparcs.components.agriculture import Irrigation


@register_component_page(Irrigation)
class IrrigationPage(ComponentPage[Irrigation]):
    def create_layout(self, layout: PageLayout) -> None:
        super().create_layout(layout)

        switch = self._build_switch()
        layout.card.append(switch, focus=True)
        layout.append(html.Div(switch))
        layout.append(html.Hr())

    # noinspection PyShadowingBuiltins
    def _build_switch(self) -> html.Div:
        id = f"{self.id}-state"

        @callback(
            Input(id, "value"),
            force_no_output=True,
        )
        def _update_state(state: bool) -> None:
            _state = self.data.state
            if _state.is_valid() and _state.value != state:
                _state.write(state)

        @callback(
            Output(id, "value"),
            Input(f"{id}-update", "n_intervals"),
        )
        def _update_switch(*_) -> bool:
            _state = self.data.state
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
