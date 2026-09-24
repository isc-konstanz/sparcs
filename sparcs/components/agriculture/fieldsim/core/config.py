# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.fieldsim.core.config
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Frozen configuration tree, parsed exactly once by the lories adapter from the
unchanged ``field_simulation.conf`` + ``field_simulation.d/`` layout.

Every ``from_mapping`` takes a plain ``Mapping`` (a materialized nested dict),
never a lories ``Configurations``. Validation lives in ``__post_init__``:
ranges, positivity, cross-field rules, and unknown keys raise ``ValueError``.
Frozen means picklable, which is what process-pool planner workers need.

The nested mesh / pde / ponding / feddes / drip / anchor / probe configs
already exist as dataclasses in ``simulation._soil`` and ``simulation._anchor``
and are reused as-is; they are typed here as ``Any`` only to keep this skeleton
import-free of FiPy.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .plots import PlotConfig
from .shading import ShadingConfig


@dataclass(frozen=True)
class SoilConfig:
    """``[soil_simulation]`` block: everything the engine needs, nothing more."""

    mesh: Any  # simulation._soil.MeshConfig
    pde: Any  # simulation._soil.PDEConfig, with .ponding / .feddes attached
    drip: Any  # simulation._soil.DripConfig
    anchor: Any  # simulation._anchor.AnchorConfig
    probes: Sequence[Any] = ()  # simulation._soil.ProbeSpec, from [probes.points.*]
    total_drip_line_length_m: float = 1.0

    @classmethod
    def from_mapping(cls, block: Mapping[str, Any], *, bay_width: float, model: Mapping[str, Any]) -> SoilConfig:
        """Parse ``[soil_simulation]``.

        ``model`` is the already-cascaded ``[soil_simulation.model]`` over
        field-level ``[model]`` mapping; the cascade is computed by the adapter
        with lories' own defaults mechanism, once, and never re-derived here.
        """
        raise NotImplementedError

    def __post_init__(self) -> None:
        if self.total_drip_line_length_m <= 0:
            raise ValueError("total_drip_line_length_m must be positive")


@dataclass(frozen=True)
class WateringWindow:
    start: dt.time


@dataclass(frozen=True)
class PlannerConfig:
    """``[soil_predictor]`` block. The planner never reads ``[soil_simulation]``."""

    windows: Sequence[WateringWindow] = ()
    durations_min: Sequence[int] = ()
    grid_mode: str = "ladder"
    threshold_hpa: float = 300.0
    decision_probes: Sequence[str] = ()
    horizon: dt.timedelta = dt.timedelta(days=3)
    parallel: bool = False
    max_workers: int | None = None
    drip: Any = None  # DripConfig override, falls back to SoilConfig.drip in the adapter

    @classmethod
    def from_mapping(cls, block: Mapping[str, Any], *, drip_fallback: Any) -> PlannerConfig:
        raise NotImplementedError

    def __post_init__(self) -> None:
        if self.grid_mode not in ("ladder", "full"):
            raise ValueError(f"grid_mode must be 'ladder' or 'full', got {self.grid_mode!r}")


@dataclass(frozen=True)
class FieldConfig:
    """``[field_simulation]`` block plus its two configured children."""

    soil: SoilConfig
    planner: PlannerConfig | None
    shading: ShadingConfig = field(default_factory=ShadingConfig)
    plots: PlotConfig | None = None
    lai_type: str = "grass"
    roughness: float = 0.002
    plant_height: float = 0.1
    ndvi: float = 0.25
    bare_lai: float = 1.0
    bare_roughness: float = 0.002
    bare_plant_height: float = 0.1
    bare_ndvi: float = 0.25
    bay_width: float = 3.5
    interval: dt.timedelta = dt.timedelta(minutes=30)
    offset: dt.timedelta = dt.timedelta(0)
    intake_delay: dt.timedelta = dt.timedelta(minutes=30)
    extra: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, block: Mapping[str, Any]) -> FieldConfig:
        """Parse the whole tree from one materialized mapping.

        Expected shape (member blocks are nested dicts, exactly as the
        ``.d/`` files lay them out)::

            {
              "lai_type": ..., "interval": ..., "intake_delay": ...,
              "model": {...},                       # field-level, optional
              "ground_shading": {...}, "evapotranspiration": {...},
              "soil_simulation": {"mesh": {...}, "pde": {...}, "model": {...}, ...},
              "soil_predictor":  {"windows": {...}, ...},   # optional
            }
        """
        raise NotImplementedError

    def __post_init__(self) -> None:
        if self.interval <= dt.timedelta(0):
            raise ValueError("interval must be positive")
        if not dt.timedelta(0) <= self.offset < self.interval:
            raise ValueError("offset must satisfy 0 <= offset < interval")
