# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.fieldsim.core.shading
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ground shading under the PV rows, as a pure model: geometry in, per-segment
shade factors out. This is the pvfactors part of today's ``ground_shading.py``
(the numpy-2 compat patch, ``_PVSetup``, the ground report and combine
helpers, the horizontal / trackable / as-is setup builders) with every
``self.data`` call and the progress figure removed. Publishing goes through
``Outputs.chain``; rendering lives in ``plots``.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import pandas as pd
from lories.core import ConfigurationError
from lories.core.configs.configurations import Configurations
from lories.core.configs.parameters import Parameter, ParameterGroup, SelectParameter

from .config import Config

MODE_AS_IS = "as_is"  # fixed-tilt rows as configured; supports mirrored
MODE_HORIZONTAL = "horizontal"  # row geometry forced flat (surface_tilt = 0)
MODE_TRACKABLE = "trackable"  # single-axis tracker via pvlib.tracking.singleaxis
MODE_FREE_FIELD = "free_field"  # no PV array; open-sky reference baseline
MODES = (MODE_AS_IS, MODE_HORIZONTAL, MODE_TRACKABLE, MODE_FREE_FIELD)


class ShadingConfig(Config):
    """``[ground_shading]`` plus the PV geometry it is evaluated against.

    ``bay_width``, ``pv_rows`` and ``segment_ranges`` are derived by the
    adapter from the field config, the PV system and the soil mesh; they are
    attributes, not config keys.
    """

    _CONFIGS_ALLOWED_KEYS = Config._CONFIGS_ALLOWED_KEYS | {"plot"}

    mode = SelectParameter(choices=list(MODES), default=MODE_AS_IS, desc="as_is, horizontal, trackable or free_field")
    albedo = Parameter(type=float, default=0.2, min=0.0, max=1.0, desc="Ground albedo")
    surface_azimuth = Parameter(type=float, default=180.0, min=0.0, max=360.0, desc="PV surface azimuth (deg)")
    surface_tilt = Parameter(type=float, default=0.0, min=0.0, max=90.0, desc="PV surface tilt (deg)")
    mirrored = Parameter(type=bool, default=False, desc="Mirror the row arrangement about the bay centre")
    axis_azimuth = Parameter(type=float, default=100.0, min=0.0, max=360.0, desc="Row-axis bearing (deg)")
    tracker = ParameterGroup(
        key="tracker",
        desc="[tracker] single-axis tracking geometry, required for mode = trackable",
        children=[
            Parameter(key="axis_azimuth", type=float, default=180.0, min=0.0, max=360.0, desc="Axis azimuth (deg)"),
            Parameter(key="max_angle", type=float, default=60.0, min=0.0, max=90.0, desc="Maximum rotation (deg)"),
            Parameter(key="backtrack", type=bool, default=True, desc="Backtrack to avoid row-to-row shading"),
        ],
    )

    # derived, set by the adapter
    bay_width: float = 3.5
    pv_rows: Sequence[Mapping[str, float]] = ()
    segment_ranges: Optional[Mapping[str, tuple[float, float]]] = None

    def _on_configure(self, configs: Configurations) -> None:
        if self.mode == MODE_TRACKABLE and not self.tracker:
            raise ConfigurationError("mode = trackable requires a [tracker] section")

    def derive(
        self, *, bay_width: float, pv_rows: Sequence[Mapping[str, float]] = (), segment_ranges=None
    ) -> ShadingConfig:
        """Today ``GroundShading._configure_geometry`` + ``_resolve_segment_ranges``."""
        self.bay_width = bay_width
        self.pv_rows = tuple(pv_rows)
        self.segment_ranges = segment_ranges
        return self


class ShadingModel:
    """Per-row pvfactors evaluation, aggregated to soil segments."""

    def __init__(self, config: ShadingConfig) -> None:
        self.config = config
        self._setups: list[Any] = []  # one _PVSetup per PV row arrangement

    def evaluate(self, weather: pd.DataFrame) -> pd.DataFrame:
        """Return one row per weather row with a ``<segment>`` column per
        soil top segment (shade factor in [0, 1]) plus ``open_sky_ghi``.

        Free-field mode returns all ones without touching pvfactors
        (today ``GroundShading.evaluate`` + ``_evaluate_free_field``).
        """
        raise NotImplementedError

    def pv_rows_at(self, ts: pd.Timestamp) -> list[tuple]:
        """PV-row geometry for the plot at ``ts``; night frames reuse the
        last sun-up geometry (today ``_synthesize_pv_rows``)."""
        raise NotImplementedError

    def _build_setups(self) -> None:
        """Today ``_build_horizontal_setups`` / ``_build_trackable_setups`` /
        ``_build_as_is_setups``, chosen by ``config.mode``."""
        raise NotImplementedError

    def _aggregate_per_segment(self, ground: list[tuple]) -> dict[str, float]:
        """Today ``_aggregate_per_segment`` over ``_qinc_in_range``."""
        raise NotImplementedError
