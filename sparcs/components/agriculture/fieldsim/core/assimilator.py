# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.fieldsim.core.assimilator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tensiometer assimilation ("anchoring"), as an owned instance with its own
state. Replaces ``AnchorRuntime(self)`` being constructed at five call sites
with its state living on ``SoilSimulation``.

The math stays in ``.anchor`` (fipy-free, a verbatim copy of
``simulation._anchor``); this class only holds the per-sensor history and
the last-anchored stamp and decides when to blend.
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Any, Mapping, Optional, Sequence

import pandas as pd
from lories.util import to_timedelta

from .anchor import AnchorConfig, AnchorResult, AnchorSensor, SensorOverrides, anchor_update, latest_reading_at
from .state import SoilState

logger = logging.getLogger(__name__)


def _opt_float(spec: Any, key: str) -> Optional[float]:
    """Read an optional float override from a per-sensor sub-block (``None`` if absent)."""
    value = spec.get(key, default=None)
    return None if value is None else float(value)


def parse_anchor_config(configs: Any) -> AnchorConfig:
    """Parse an ``[anchor]`` block into the FiPy-free ``AnchorConfig``.

    Off by default, so existing configs and calibration runs are unaffected.
    ``configs`` is a lories ``Configurations``-like object (``has_member`` /
    ``get`` / ``get_bool`` / ``get_member``), or a falsy value (``None`` or
    ``{}``) for "no ``[anchor]`` block at all", which parses to a disabled
    config with no sensors. Two mutually exclusive ways to name the
    allowlist: a bare ``sensors`` value (list of keys, or a comma string) --
    every sensor inherits the ``[anchor]`` globals; or per-sensor
    ``[anchor.sensors.<key>]`` sub-blocks (mirroring the
    ``[probes.points.<name>]`` precedent), each optionally overriding
    ``sigma_meas_pf``/``staleness``/``r_horizontal``/``r_vertical``; any
    omitted key inherits the global. ``sigma_sys`` stays global (see
    ``SensorOverrides``).
    """
    if not configs:
        return AnchorConfig(
            enabled=False,
            sigma_sys=0.05,
            sigma_meas_pf=0.15,
            r_horizontal=0.5,
            r_vertical=0.2,
            staleness=to_timedelta("6h"),
            sensors={},
        )
    if configs.has_member("sensors"):
        sensors: dict[str, Optional[SensorOverrides]] = {
            str(key): SensorOverrides(
                sigma_meas_pf=_opt_float(spec, "sigma_meas_pf"),
                staleness=(
                    None if spec.get("staleness", default=None) is None else to_timedelta(spec.get("staleness"))
                ),
                r_horizontal=_opt_float(spec, "r_horizontal"),
                r_vertical=_opt_float(spec, "r_vertical"),
            )
            for key, spec in configs.get_member("sensors").items()
        }
    else:
        raw = configs.get("sensors", default=[]) or []
        if isinstance(raw, str):
            raw = [s.strip() for s in raw.split(",") if s.strip()]
        sensors = {str(key): None for key in raw}
    return AnchorConfig(
        enabled=configs.get_bool("enabled", default=False),
        sigma_sys=float(configs.get("sigma_sys", default=0.05)),
        sigma_meas_pf=float(configs.get("sigma_meas_pf", default=0.15)),
        r_horizontal=float(configs.get("r_horizontal", default=0.5)),
        r_vertical=float(configs.get("r_vertical", default=0.2)),
        staleness=to_timedelta(configs.get("staleness", default="6h")),
        sensors=sensors,
    )


class Assimilator:
    """Owns the sensor list, the per-sensor tension history and the
    last-anchored stamp; ``update`` blends fresh tensiometer readings into a
    ``SoilState`` exactly as the live ``AnchorRuntime.apply`` would."""

    def __init__(self, config: Optional[Any], engine: Any) -> None:
        self.config: AnchorConfig = config if isinstance(config, AnchorConfig) else parse_anchor_config(config)
        self.engine = engine
        self.sensors: list[AnchorSensor] = []
        self.history: dict[str, pd.Series] = {}
        # Per-sensor key -> the timestamp of the reading last assimilated for
        # that sensor (mirrors the live SoilSimulation._last_anchored dict;
        # anchor_update gates each sensor on its OWN previous timestamp).
        self.last_anchored: dict[str, Any] = {}
        self.last_result: Optional[AnchorResult] = None
        self._stale_warned: set[str] = set()

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled) and bool(self.sensors)

    def set_sensors(self, sensors: Sequence[AnchorSensor]) -> None:
        """Register the discovered sensors (the adapter calls this after discovery)."""
        self.sensors = list(sensors)

    def ingest(self, history: Mapping[str, pd.Series]) -> None:
        """Store the tick's ranged sensor reads (today ``AnchorRuntime.load_history``).

        An empty series for a key means the read produced nothing this tick;
        the previous series (if any) is kept and a warning is logged once,
        latched until a non-empty read for that key arrives."""
        for key, series in history.items():
            if series is None or series.empty:
                if key not in self._stale_warned:
                    logger.warning(
                        "anchor sensor %s produced no readings this tick; it stays predict-only until data returns.",
                        key,
                    )
                    self._stale_warned.add(key)
                continue
            self.history[key] = series
            self._stale_warned.discard(key)

    def update(self, state: SoilState, now: dt.datetime) -> SoilState:
        """Blend fresh observations into ``state`` and return the anchored
        state (today ``AnchorRuntime.apply`` -> ``anchor_update`` ->
        ``anchor_field``). Returns ``state`` unchanged when nothing is fresh,
        including when the assimilator is disabled or has no sensors."""
        if not self.enabled:
            return state
        sensors = [sensor for sensor in self.sensors if sensor.key in self.config.sensors]
        if not sensors:
            return state

        def read_tension(sensor: AnchorSensor):
            return latest_reading_at(self.history.get(sensor.key), now)

        result = anchor_update(
            state.se,
            self.engine.cell_centers,
            sensors,
            read_tension,
            now,
            self.config,
            self.engine.model,
            self.engine.mesh_config.width,
            self.last_anchored,
            self.engine.se_bounds[0],
            self.engine.se_bounds[1],
        )
        if result is None:
            return state
        self.last_anchored.update(result.anchored_at)
        self.last_result = result
        return SoilState(se=result.se_new, se_old=result.se_new.copy(), surface_h=state.surface_h, at=now)
