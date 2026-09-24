# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.fieldsim.core.plots
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Progress images as pure functions from data to PNG bytes. Today's
``plot_style.py`` and ``plot_render.py`` move here unchanged; the strike
counter and the ``plot_strikes`` channel write leave for the IO layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import pandas as pd


@dataclass(frozen=True)
class PlotConfig:
    enabled: bool = False
    interval: pd.Timedelta = pd.Timedelta("1h")
    disable_after_failures: int = 5


def render_due(last: Optional[pd.Timestamp], now: pd.Timestamp, config: PlotConfig) -> bool:
    return config.enabled and (last is None or now - last >= config.interval)


def render_shading_png(ts: pd.Timestamp, pv_rows: list[tuple], factors: Mapping[str, float], envelope: Any) -> bytes:
    """Shading pattern frame (today ``GroundShading._render_progress``)."""
    raise NotImplementedError


def render_rel_sat_png(ts: pd.Timestamp, mesh: Any, se: Any, probes: Any) -> bytes:
    """Relative-saturation cross-section (today ``plot_render.RenderSession.render``)."""
    raise NotImplementedError
