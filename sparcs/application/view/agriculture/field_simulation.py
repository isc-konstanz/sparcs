# -*- coding: utf-8 -*-
"""
sparcs.application.view.agriculture.field_simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Detailed dashboard for ``FieldSimulation`` — renders the latest soil-
saturation and ground-shading PNGs plus a 24h time-series of the per-tick
diagnostic flux channels and bulk evapotranspiration.

The dashboard is **channel-driven**: every panel reads its data straight
from the chain's ``self.data[...]`` channels and refreshes only when the
underlying channel timestamp moves. The matplotlib live-plot pipeline in
``soil.py`` / ``ground_shading.py`` keeps writing PNG bytes to its
channel; we just consume.
"""

from __future__ import annotations

import base64
from typing import List, Optional, Sequence

import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from dash import Input, Output, State, callback, dcc, html, no_update

import pandas as pd
from lories import Channel
from lories.application.view.pages import ComponentPage, PageLayout, register_component_page
from sparcs.components.agriculture.simulation import (
    Evapotranspiration,
    FieldSimulation,
    GroundShading,
    SoilSimulation,
)


HISTORY_WINDOW = pd.Timedelta(hours=24)


def _png_data_uri(data: bytes) -> str:
    return f"data:image/png;base64,{base64.b64encode(data).decode('ascii')}"


def _channel_ts(channel: Channel) -> Optional[str]:
    """Return the channel timestamp as ISO string, or None if invalid."""
    if not channel.is_valid() or pd.isna(channel.timestamp):
        return None
    return channel.timestamp.isoformat()


def _empty_figure(message: str) -> dict:
    return {
        "data": [],
        "layout": go.Layout(
            xaxis={"visible": False},
            yaxis={"visible": False},
            annotations=[
                {
                    "text": message,
                    "xref": "paper",
                    "yref": "paper",
                    "showarrow": False,
                    "font": {"size": 14, "color": "#888"},
                }
            ],
            margin={"l": 32, "r": 16, "t": 16, "b": 32},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        ),
    }


class _ChannelHistory:
    """Per-page rolling buffer of the latest channel samples.

    Filled tick-by-tick from the live channel value rather than from the
    logger backend — keeps the page self-contained for projects that have
    not enabled per-channel logging on the diagnostic flux / shading-factor
    channels (model-side default is ``logger={"enabled": False}``).
    """

    def __init__(self, channels: Sequence[Channel], window: pd.Timedelta = HISTORY_WINDOW) -> None:
        self.channels = list(channels)
        self.window = window
        self._records: dict[pd.Timestamp, dict[str, float]] = {}
        self._last_ts: Optional[pd.Timestamp] = None

    def update(self) -> bool:
        ts = None
        sample: dict[str, float] = {}
        for c in self.channels:
            if not c.is_valid() or pd.isna(c.timestamp):
                continue
            try:
                sample[c.id] = float(c.value)
            except (TypeError, ValueError):
                continue
            if ts is None or c.timestamp > ts:
                ts = c.timestamp
        if ts is None or ts == self._last_ts:
            return False
        self._last_ts = ts
        self._records[ts] = sample
        cutoff = ts - self.window
        self._records = {t: v for t, v in self._records.items() if t >= cutoff}
        return True

    @property
    def last_ts(self) -> Optional[pd.Timestamp]:
        return self._last_ts

    def to_frame(self) -> pd.DataFrame:
        if not self._records:
            return pd.DataFrame()
        return pd.DataFrame.from_dict(self._records, orient="index").sort_index()


@register_component_page(FieldSimulation)
class FieldSimulationPage(ComponentPage[FieldSimulation]):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._flux_history: Optional[_ChannelHistory] = None
        self._shading_history: Optional[_ChannelHistory] = None
        self._et_history: Optional[_ChannelHistory] = None

    def create_layout(self, layout: PageLayout) -> None:
        super().create_layout(layout)

        if self._has_ground_shading():
            card = self._build_shading_card()
            layout.card.append(card)
            layout.append(card)

        if self._has_evapotranspiration():
            card = self._build_et_card()
            layout.card.append(card)
            layout.append(card)

        if self._has_soil_simulation():
            card = self._build_soil_card()
            layout.card.append(card, focus=True)
            layout.append(card)

    # ------------------------------------------------------------------ chain

    def _has_soil_simulation(self) -> bool:
        return self._component.soil_simulation is not None

    def _has_ground_shading(self) -> bool:
        return self._component.ground_shading is not None

    def _has_evapotranspiration(self) -> bool:
        return self._component.evapotranspiration is not None

    # ------------------------------------------------------------------ soil

    def _build_soil_card(self) -> html.Div:
        image_id = f"{self.id}-soil-image"
        image_store_id = f"{image_id}-store"
        graph_id = f"{self.id}-soil-flux-graph"
        graph_store_id = f"{graph_id}-store"

        soil = self._component.soil_simulation
        image_channel = soil.data[SoilSimulation.SOIL_PROGRESS_IMAGE.key]
        flux_channels = [
            soil.data[SoilSimulation.WATER_TOP_IN.key],
            soil.data[SoilSimulation.WATER_TOP_OUT.key],
            soil.data[SoilSimulation.WATER_BOTTOM.key],
            soil.data[SoilSimulation.WATER_TRANSP.key],
        ]
        self._flux_history = _ChannelHistory(flux_channels)

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

        @callback(
            Output(graph_id, "figure"),
            Output(graph_store_id, "data"),
            Input("view-update", "n_intervals"),
            State(graph_store_id, "data"),
        )
        def _update_flux(_, last_ts):
            if not self._flux_history.update():
                return no_update, no_update
            new_ts = self._flux_history.last_ts.isoformat() if self._flux_history.last_ts else None
            return self._build_flux_figure(flux_channels, self._flux_history.to_frame()), new_ts

        return html.Div(
            [
                dcc.Store(id=image_store_id),
                dcc.Store(id=graph_store_id),
                dbc.Row(dbc.Col(html.H5("Soil simulation"), width="auto")),
                dbc.Row(
                    dbc.Col(
                        html.Img(
                            id=image_id,
                            alt="Soil saturation",
                            style={
                                "maxWidth": "100%",
                                "maxHeight": "32rem",
                                "border": "1px solid #ddd",
                                "borderRadius": "4px",
                            },
                        ),
                        width="auto",
                    )
                ),
                dbc.Row(
                    dbc.Col(
                        dcc.Graph(
                            id=graph_id,
                            figure=_empty_figure("Waiting for first tick…"),
                            config={"displayModeBar": False},
                            style={"height": "20rem"},
                        )
                    )
                ),
            ],
            style={"marginBottom": "1.5rem"},
        )

    @staticmethod
    def _build_flux_figure(channels: List[Channel], frame: pd.DataFrame) -> dict:
        if frame.empty:
            return _empty_figure("No samples yet")

        traces = []
        colors = {
            SoilSimulation.WATER_TOP_IN.key: "#1f77b4",
            SoilSimulation.WATER_TOP_OUT.key: "#ff7f0e",
            SoilSimulation.WATER_BOTTOM.key: "#d62728",
            SoilSimulation.WATER_TRANSP.key: "#2ca02c",
        }
        for c in channels:
            if c.id not in frame.columns:
                continue
            traces.append(
                go.Scatter(
                    x=frame.index,
                    y=frame[c.id].values,
                    mode="lines",
                    name=c.name,
                    line={"color": colors.get(c.key, "#777"), "width": 2},
                )
            )

        unit = next((c.unit for c in channels if c.unit), "")
        layout = go.Layout(
            xaxis={"title": ""},
            yaxis={"title": unit, "rangemode": "tozero"},
            margin={"l": 56, "r": 16, "t": 16, "b": 40},
            legend={"orientation": "h", "y": -0.2},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            hovermode="x unified",
        )
        return {"data": traces, "layout": layout}

    # --------------------------------------------------------------- shading

    def _build_shading_card(self) -> html.Div:
        image_id = f"{self.id}-shading-image"
        image_store_id = f"{image_id}-store"
        graph_id = f"{self.id}-shading-graph"
        graph_store_id = f"{graph_id}-store"

        shading = self._component.ground_shading
        image_channel = shading.data[GroundShading.SHADING_PROGRESS_IMAGE.key]
        factor_channel = shading.data[GroundShading.SHADING_FACTOR.key]
        self._shading_history = _ChannelHistory([factor_channel])

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

        @callback(
            Output(graph_id, "figure"),
            Output(graph_store_id, "data"),
            Input("view-update", "n_intervals"),
            State(graph_store_id, "data"),
        )
        def _update_factor(_, last_ts):
            if not self._shading_history.update():
                return no_update, no_update
            new_ts = self._shading_history.last_ts.isoformat() if self._shading_history.last_ts else None
            frame = self._shading_history.to_frame()
            if frame.empty:
                return _empty_figure("No samples yet"), new_ts
            trace = go.Scatter(
                x=frame.index,
                y=frame[factor_channel.id].values,
                mode="lines",
                name=factor_channel.name,
                line={"color": "#946a00", "width": 2},
            )
            layout = go.Layout(
                xaxis={"title": ""},
                yaxis={"title": "shading factor", "range": [0, 1]},
                margin={"l": 56, "r": 16, "t": 16, "b": 40},
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                hovermode="x unified",
            )
            return {"data": [trace], "layout": layout}, new_ts

        return html.Div(
            [
                dcc.Store(id=image_store_id),
                dcc.Store(id=graph_store_id),
                dbc.Row(dbc.Col(html.H5("Ground shading"), width="auto")),
                dbc.Row(
                    dbc.Col(
                        html.Img(
                            id=image_id,
                            alt="Ground shading",
                            style={
                                "maxWidth": "100%",
                                "maxHeight": "24rem",
                                "border": "1px solid #ddd",
                                "borderRadius": "4px",
                            },
                        ),
                        width="auto",
                    )
                ),
                dbc.Row(
                    dbc.Col(
                        dcc.Graph(
                            id=graph_id,
                            figure=_empty_figure("Waiting for first tick…"),
                            config={"displayModeBar": False},
                            style={"height": "16rem"},
                        )
                    )
                ),
            ],
            style={"marginBottom": "1.5rem"},
        )

    # -------------------------------------------------------------------- ET

    def _build_et_card(self) -> html.Div:
        graph_id = f"{self.id}-et-graph"
        store_id = f"{graph_id}-store"

        et = self._component.evapotranspiration
        et_channel = et.data[Evapotranspiration.EVAPOTRANSPIRATION.key]
        rad_channel = et.data[Evapotranspiration.RAD_TERM.key]
        aer_channel = et.data[Evapotranspiration.AER_TERM.key]
        self._et_history = _ChannelHistory([et_channel, rad_channel, aer_channel])

        @callback(
            Output(graph_id, "figure"),
            Output(store_id, "data"),
            Input("view-update", "n_intervals"),
            State(store_id, "data"),
        )
        def _update_et(_, last_ts):
            if not self._et_history.update():
                return no_update, no_update
            new_ts = self._et_history.last_ts.isoformat() if self._et_history.last_ts else None
            frame = self._et_history.to_frame()
            if frame.empty:
                return _empty_figure("No samples yet"), new_ts

            traces = []
            for channel, color in (
                (et_channel, "#2ca02c"),
                (rad_channel, "#1f77b4"),
                (aer_channel, "#ff7f0e"),
            ):
                if channel.id not in frame.columns:
                    continue
                traces.append(
                    go.Scatter(
                        x=frame.index,
                        y=frame[channel.id].values,
                        mode="lines",
                        name=channel.name,
                        line={"color": color, "width": 2},
                        yaxis="y" if channel is et_channel else "y2",
                    )
                )
            layout = go.Layout(
                xaxis={"title": ""},
                yaxis={"title": et_channel.unit or "ET", "rangemode": "tozero"},
                yaxis2={
                    "title": "PM terms",
                    "overlaying": "y",
                    "side": "right",
                    "showgrid": False,
                },
                margin={"l": 56, "r": 56, "t": 16, "b": 40},
                legend={"orientation": "h", "y": -0.2},
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                hovermode="x unified",
            )
            return {"data": traces, "layout": layout}, new_ts

        return html.Div(
            [
                dcc.Store(id=store_id),
                dbc.Row(dbc.Col(html.H5("Evapotranspiration"), width="auto")),
                dbc.Row(
                    dbc.Col(
                        dcc.Graph(
                            id=graph_id,
                            figure=_empty_figure("Waiting for first tick…"),
                            config={"displayModeBar": False},
                            style={"height": "20rem"},
                        )
                    )
                ),
            ],
            style={"marginBottom": "1.5rem"},
        )
