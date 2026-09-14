# -*- coding: utf-8 -*-
"""
sparcs.application.view.agriculture.field_simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Detailed dashboard for ``FieldSimulation``: renders the latest soil-
saturation and ground-shading PNGs.

The dashboard is **channel-driven**: every panel reads its data straight
from the chain's ``self.data[...]`` channels and refreshes only when the
underlying channel timestamp moves.
"""

from __future__ import annotations

import base64
from typing import Optional

import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, dcc, html, no_update

import pandas as pd
from lories import Channel, Constant
from lories.application.view.pages import ComponentPage, PageLayout, register_component_page
from sparcs.components.agriculture.simulation import (
    FieldSimulation,
    GroundShading,
    SoilSimulation,
)


def _png_data_uri(data: bytes) -> str:
    return f"data:image/png;base64,{base64.b64encode(data).decode('ascii')}"


def progress_image_channel(component, constant: Constant) -> Optional[Channel]:
    """The component's progress-image channel, or None when it was never
    registered (the channel only exists when ``[plot] enabled`` is true)."""
    if component is None:
        return None
    try:
        return component.data[constant.key]
    except KeyError:
        return None


def _channel_ts(channel: Channel) -> Optional[str]:
    """Return the channel timestamp as ISO string, or None if invalid."""
    if not channel.is_valid() or pd.isna(channel.timestamp):
        return None
    return channel.timestamp.isoformat()


@register_component_page(FieldSimulation)
class FieldSimulationPage(ComponentPage[FieldSimulation]):
    def create_layout(self, layout: PageLayout) -> None:
        super().create_layout(layout)

        # Cards are pure image panels, so gate on the progress-image channel:
        # with [plot] enabled = false the component never registers it.
        shading_image = progress_image_channel(self._component.ground_shading, GroundShading.SHADING_PROGRESS_IMAGE)
        soil_image = progress_image_channel(self._component.soil_simulation, SoilSimulation.SOIL_PROGRESS_IMAGE)

        cards = []
        if shading_image is not None:
            card = self._build_shading_card(shading_image)
            layout.card.append(card)
            cards.append(card)

        if soil_image is not None:
            card = self._build_soil_card(soil_image)
            layout.card.append(card, focus=True)
            cards.append(card)

        if len(cards) == 2:
            # Side-by-side at md+ breakpoints; stacks on small screens.
            layout.append(
                dbc.Row(
                    [dbc.Col(card, md=6, xs=12) for card in cards],
                    className="g-3",
                )
            )
        else:
            for card in cards:
                layout.append(card)

    # ------------------------------------------------------------------ soil

    def _build_soil_card(self, image_channel: Channel) -> html.Div:
        image_id = f"{self.id}-soil-image"
        image_store_id = f"{image_id}-store"

        @callback(
            Output(image_id, "src"),
            Output(image_store_id, "data"),
            Input("view-update", "n_intervals"),
            State(image_store_id, "data"),
        )
        def _update_image(_, last_ts):
            ts = _channel_ts(image_channel)
            if ts is None or ts == last_ts:
                return no_update, no_update
            return _png_data_uri(image_channel.value), ts

        return html.Div(
            [
                dcc.Store(id=image_store_id),
                dbc.Row(dbc.Col(html.H5("Soil simulation"), width="auto")),
                dbc.Row(
                    dbc.Col(
                        html.Img(
                            id=image_id,
                            alt="Soil saturation",
                            style={
                                "width": "100%",
                                "height": "auto",
                                "display": "block",
                                "border": "1px solid #ddd",
                                "borderRadius": "4px",
                            },
                        ),
                        width=12,
                        style={"minWidth": 0},
                    ),
                ),
            ],
            className="h-100",
            style={"marginBottom": "1.5rem"},
        )

    # --------------------------------------------------------------- shading

    def _build_shading_card(self, image_channel: Channel) -> html.Div:
        image_id = f"{self.id}-shading-image"
        image_store_id = f"{image_id}-store"

        @callback(
            Output(image_id, "src"),
            Output(image_store_id, "data"),
            Input("view-update", "n_intervals"),
            State(image_store_id, "data"),
        )
        def _update_image(_, last_ts):
            ts = _channel_ts(image_channel)
            if ts is None or ts == last_ts:
                return no_update, no_update
            return _png_data_uri(image_channel.value), ts

        return html.Div(
            [
                dcc.Store(id=image_store_id),
                dbc.Row(dbc.Col(html.H5("Ground shading"), width="auto")),
                dbc.Row(
                    dbc.Col(
                        html.Img(
                            id=image_id,
                            alt="Ground shading",
                            style={
                                "width": "100%",
                                "height": "auto",
                                "display": "block",
                                "border": "1px solid #ddd",
                                "borderRadius": "4px",
                            },
                        ),
                        width=12,
                        style={"minWidth": 0},
                    ),
                ),
            ],
            className="h-100",
            style={"marginBottom": "1.5rem"},
        )
