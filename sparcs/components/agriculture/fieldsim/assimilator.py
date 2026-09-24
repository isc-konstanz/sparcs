# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.fieldsim.assimilator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tensiometer assimilation ("anchoring"), as an owned instance with its own
state. Replaces ``AnchorRuntime(self)`` being constructed at five call sites
with its state living on ``SoilSimulation``.

The math stays in ``simulation._anchor`` (fipy-free, AST-guarded by a test);
this class only holds the per-sensor history and the last-anchored stamp and
decides when to blend.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Mapping

import pandas as pd

from .engine import SoilEngine
from .state import SoilState


class Assimilator:
    def __init__(self, config: Any, engine: SoilEngine) -> None:
        self.config = config  # simulation._anchor.AnchorConfig
        self.engine = engine
        self.history: dict[str, pd.Series] = {}
        self.last_anchored: dt.datetime | None = None

    @property
    def enabled(self) -> bool:
        return bool(getattr(self.config, "enabled", False))

    def ingest(self, history: Mapping[str, pd.Series]) -> None:
        """Store the tick's ranged sensor reads (today ``AnchorRuntime.load_history``)."""
        raise NotImplementedError

    def update(self, state: SoilState, now: dt.datetime) -> SoilState:
        """Blend fresh observations into ``state`` and return the anchored
        state (today ``AnchorRuntime.apply`` -> ``anchor_update`` ->
        ``anchor_field``). Returns ``state`` unchanged when nothing is fresh."""
        raise NotImplementedError
