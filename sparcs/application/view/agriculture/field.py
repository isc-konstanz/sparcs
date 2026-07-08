# -*- coding: utf-8 -*-
"""
sparcs.application.view.agriculture.field
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""

from __future__ import annotations

import base64
import logging
from typing import Optional

import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, dcc, html, no_update

import pandas as pd
from lories.application.view.pages import ComponentGroup, PageLayout, register_component_page
from sparcs.components.agriculture import AgriculturalField, Irrigation
from sparcs.components.agriculture.simulation import (
    Evapotranspiration,
    FieldSimulation,
    GroundShading,
    SoilSimulation,
)

from .field_simulation import progress_image_channel

logger = logging.getLogger(__name__)


@register_component_page(AgriculturalField)
class AgriculturalFieldPage(ComponentGroup[AgriculturalField]):
    @property
    def irrigation(self) -> Irrigation:
        return self._component.irrigation

    @property
    def simulation(self) -> Optional[FieldSimulation]:
        return self._component.simulation

    def has_irrigation(self) -> bool:
        return self._component.has_irrigation()

    def has_simulation(self) -> bool:
        return self._component.has_simulation()

    def create_layout(self, layout: PageLayout) -> None:
        super().create_layout(layout)
        if self.has_irrigation():
            irrigation = self.get_page(self.irrigation)
            layout.card.extend(irrigation.layout.card)

        if self.has_simulation():
            sim_thumbnail = self._build_simulation_thumbnail()
            layout.card.append(sim_thumbnail)

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

    def _build_simulation_thumbnail(self) -> html.Div:
        sim = self.simulation
        sim_page = self.get_page(sim)
        sim_href = sim_page.path if sim_page is not None else None

        image_id = f"{self.id}-sim-thumbnail-image"
        image_store_id = f"{image_id}-store"
        scalars_id = f"{self.id}-sim-thumbnail-scalars"

        soil = sim.soil_simulation
        shading = sim.ground_shading
        et = sim.evapotranspiration
        # None when plot_progress = false: the component never registers the channel.
        image_channel = progress_image_channel(soil, SoilSimulation.SOIL_PROGRESS_IMAGE)
        et_channel = et.data[Evapotranspiration.EVAPOTRANSPIRATION.key] if et is not None else None
        drainage_channel = soil.data[SoilSimulation.WATER_BOTTOM.key] if soil is not None else None
        shading_channel = shading.data[GroundShading.SHADING_FACTOR.key] if shading is not None else None

        if image_channel is not None:

            @callback(
                Output(image_id, "src"),
                Output(image_store_id, "data"),
                Input("view-update", "n_intervals"),
                State(image_store_id, "data"),
            )
            def _update_image(_, last_ts):
                if not image_channel.is_valid() or pd.isna(image_channel.timestamp):
                    return no_update, no_update
                ts = image_channel.timestamp.isoformat()
                if ts == last_ts:
                    return no_update, no_update
                b64 = base64.b64encode(image_channel.value).decode("ascii")
                return f"data:image/png;base64,{b64}", ts

        @callback(
            Output(scalars_id, "children"),
            Input("view-update", "n_intervals"),
        )
        def _update_scalars(*_):
            return [
                _scalar_block("ET", et_channel, "#2ca02c"),
                _scalar_block("Drainage", drainage_channel, "#d62728"),
                _scalar_block("Shading", shading_channel, "#946a00"),
            ]

        thumbnail = html.Div(
            [
                dcc.Store(id=image_store_id),
                html.Img(
                    id=image_id,
                    alt="Soil saturation",
                    style={
                        "maxWidth": "100%",
                        "maxHeight": "10rem",
                        "borderRadius": "4px",
                    },
                ),
            ]
        )
        if sim_href is not None:
            thumbnail = dcc.Link(thumbnail, href=sim_href)

        return html.Div(
            [
                dbc.Row(dbc.Col(html.H5("Field simulation"), width="auto")),
                dbc.Row(dbc.Col(thumbnail, width="auto")),
                dbc.Row(
                    id=scalars_id,
                    children=_update_scalars(),
                    className="g-3",
                    style={"marginTop": "0.5rem"},
                ),
            ],
            style={"marginBottom": "1rem"},
        )


def _scalar_block(label: str, channel, color: str) -> dbc.Col:
    if channel is None:
        return dbc.Col()
    if channel.is_valid():
        try:
            text = f"{round(float(channel.value), 2)}"
        except (TypeError, ValueError):
            logger.debug("Channel %s value %r is not numeric; displaying raw.", channel.id, channel.value)
            text = str(channel.value)
        unit = channel.unit or ""
        body = html.Div(
            [
                html.Div(text, style={"fontSize": "1.25rem", "color": color, "fontWeight": "bold"}),
                html.Small(unit, className="text-muted"),
            ]
        )
    else:
        body = html.Div(html.Small("—", className="text-muted"))
    return dbc.Col(
        [html.Small(label, className="text-muted"), body],
        width="auto",
    )
