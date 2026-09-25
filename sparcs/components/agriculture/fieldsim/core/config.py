# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.fieldsim.core.config
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Configuration classes for the unchanged ``field_simulation.conf`` +
``field_simulation.d/`` layout, declared with lories' parameter structure.

Each class is a lories ``Configurator`` and mirrors one config section. Every
key is one class-level ``Parameter`` (type, default, bounds, choices,
description), so what is needed and allowed is declared once and known at
runtime: lories resolves and coerces the TOML, applies defaults, and
``schema()`` returns the declaration. Unknown keys are a hard error
(``Config._assert_configs``), recursively through declared groups, the same
rule lories' converters and processors follow.

This is the only lories import in ``core``: the config machinery. Channels,
components and threads stay out. ``from_dict`` builds a configured instance
from a plain mapping for scenarios, notebooks and tests.

Sections that still live as dataclasses next to FiPy code in
``simulation._soil`` / ``simulation._anchor`` (mesh, pde, ponding, feddes,
drip, anchor, probes) are declared as passthrough groups: accepted verbatim,
not validated. Moving them here is the first real implementation step.
"""

from __future__ import annotations

from typing import Any, ClassVar, Iterable, Mapping, Optional, Sequence

from lories.core import ConfigurationError
from lories.core.configs.configurations import Configurations
from lories.core.configs.configurator import Configurator
from lories.core.configs.directories import Directories
from lories.core.configs.parameters import (
    DurationParameter,
    ListParameter,
    Parameter,
    ParameterGroup,
    SelectParameter,
    _Parameter,
    _TypedParameter,
)

LAI_TYPES = ("fao", "grass", "apple")
GRID_MODES = ("ladder", "full")


class Config(Configurator):
    """Base for all fieldsim config sections: strict keys, schema, from_dict."""

    # Keys every section may carry without declaring them: the lories
    # reserved set plus the component-level keys that share the same file.
    _CONFIGS_ALLOWED_KEYS: ClassVar[frozenset] = frozenset({"type", "data"})
    # Child sections handled by other Config classes (set per class).
    _CONFIGS_CHILD_KEYS: ClassVar[Iterable[str]] = ()

    @classmethod
    def _assert_configs(cls, configs: Optional[Configurations]) -> Optional[Configurations]:
        configs = super()._assert_configs(configs)
        if configs is None:
            return None
        params = {p._resolve_key(): p for p in cls.__config_parameters__.values()}
        allowed = set(cls._CONFIGS_RESERVED_KEYS) | set(cls._CONFIGS_ALLOWED_KEYS) | set(cls._CONFIGS_CHILD_KEYS)
        unknown = _unknown_keys(configs, params, allowed, path=cls.__name__)
        if unknown:
            raise ConfigurationError(f"{cls.__name__}: unknown configuration keys {unknown}")
        return configs

    @classmethod
    def schema(cls) -> dict[str, Any]:
        """Needed and allowed keys with type, default, bounds and description."""
        return {p._resolve_key(): p.to_schema() for p in cls.__config_parameters__.values()}

    @classmethod
    def from_dict(cls, values: Optional[Mapping[str, Any]] = None, name: str = "fieldsim"):
        """A configured instance from a plain mapping; for scenarios and tests."""
        instance = cls()
        instance.configure(Configurations(name, Directories(), defaults=dict(values or {})))
        return instance

    def values(self) -> dict[str, Any]:
        return {attr: getattr(self, attr) for attr in type(self).__config_parameters__}


def _unknown_keys(configs: Any, params: Mapping[str, _Parameter], allowed: set, *, path: str) -> list[str]:
    """Keys of ``configs`` that no parameter declares, recursing into groups
    with declared children. A sub-table is a Mapping value; ``has_member``
    is not used because lories counts bool values as members."""
    unknown: list[str] = []
    for key in configs:
        if key in allowed:
            continue
        param = params.get(key)
        is_section = isinstance(configs.get(key), Mapping)
        if param is None:
            unknown.append(f"{path}.{key}")
        elif isinstance(param, ParameterGroup) and param.children and is_section:
            unknown += _unknown_keys(configs.get_member(key), param.children, set(), path=f"{path}.{key}")
        elif isinstance(param, _TypedParameter) and is_section:
            unknown.append(f"{path}.{key} (section given, scalar expected)")
    return unknown


# --------------------------------------------------------------------------- sections


class PlotConfig(Config):
    """``[plot]``: field-level progress-image defaults, cascaded to every child."""

    enabled = Parameter(type=bool, default=False, desc="Render progress images")
    interval = DurationParameter(default="1h", desc="Minimum time between two rendered frames")
    disable_after_failures = Parameter(
        type=int, default=5, min=1, desc="Consecutive render failures before plotting stops"
    )


class SoilConfig(Config):
    """``[soil_simulation]``: everything the engine needs, nothing more."""

    _CONFIGS_ALLOWED_KEYS = Config._CONFIGS_ALLOWED_KEYS | {"plot"}

    mesh = ParameterGroup(key="mesh", required=True, desc="[mesh] geometry (simulation._soil.MeshConfig, passthrough)")
    pde = ParameterGroup(key="pde", desc="[pde] solver settings (simulation._soil.PDEConfig, passthrough)")
    ponding = ParameterGroup(key="ponding", desc="[ponding] surface storage, sibling of [pde] (passthrough)")
    feddes = ParameterGroup(key="feddes", desc="[feddes] root water uptake, sibling of [pde] (passthrough)")
    drip = ParameterGroup(key="drip", desc="[drip] design flow and line geometry (passthrough)")
    anchor = ParameterGroup(
        key="anchor", desc="[anchor] tensiometer assimilation incl. [anchor.sensors.*] (passthrough)"
    )
    probes = ParameterGroup(key="probes", desc="[probes.points.<key>] virtual tensiometers (passthrough)")
    model = ParameterGroup(key="model", desc="[model] van Genuchten override of the field-level [model]")
    total_drip_line_length_m = Parameter(
        type=float, default=1.0, min=0.0, desc="Total drip line length under the mesh (m)"
    )
    discover_sensor_probes = Parameter(
        type=bool, default=False, desc="Register a probe per discovered SoilMoisture sensor"
    )
    plot_structure = Parameter(type=bool, default=False, desc="Render the mesh structure plot once at configure")
    testing = ParameterGroup(
        key="testing", desc="[testing] rig-only options: history_window, poll_interval (passthrough)"
    )

    # derived by the adapter from [probes.points.*] and sensor discovery (simulation._soil.ProbeSpec)
    probe_specs: Sequence[Any] = ()

    def _on_configure(self, configs: Configurations) -> None:
        if self.total_drip_line_length_m <= 0:
            raise ConfigurationError("total_drip_line_length_m must be positive")


class PlannerConfig(Config):
    """``[soil_predictor]``. The planner never reads ``[soil_simulation]``."""

    _CONFIGS_ALLOWED_KEYS = Config._CONFIGS_ALLOWED_KEYS | {"plot"}

    windows = ParameterGroup(key="windows", desc="[windows] watering windows, start time per key (passthrough)")
    interval = Parameter(type=int, default=30, min=1, desc="Planner schedule cadence (minutes)")
    offset = Parameter(type=int, default=0, min=0, desc="Planner schedule offset within the interval (minutes)")
    state = ParameterGroup(
        key="state",
        desc="[state] predictor state-blob debug sink",
        children=[
            Parameter(key="save", type=bool, default=False, desc="Persist predict_state blobs"),
            DurationParameter(key="interval", default="1h", desc="Blob cadence"),
        ],
    )
    durations_min = ListParameter(item_type=int, default=[], desc="Candidate watering durations (min)")
    grid_mode = SelectParameter(choices=list(GRID_MODES), default="ladder", desc="Shared-prefix ladder or full rolls")
    combo_cap = Parameter(type=int, default=64, min=1, desc="Upper bound on candidates per run")
    threshold_hpa = Parameter(
        type=float, default=300.0, min=0.0, desc="Tension threshold the score measures against (hPa)"
    )
    decision_probes = ListParameter(item_type=str, default=[], desc="Probe keys the score is evaluated at")
    horizon = DurationParameter(default="3D", desc="Forecast horizon the candidates are rolled over")
    max_windows = Parameter(type=int, default=4, min=1, desc="Windows considered per day")
    parallel = Parameter(type=bool, default=False, desc="Roll candidates in a process pool")
    # no min bound: lories numeric validation compares a None default against min (TypeError)
    max_workers = Parameter(type=int, default=None, required=False, desc="Pool size; default cpu_count - 1")
    logger = Parameter(type=str, default=None, required=False, desc="Logger connector id for the forecast tables")
    drip = ParameterGroup(key="drip", desc="[drip] override; falls back to [soil_simulation.drip] (passthrough)")


class FieldConfig(Config):
    """``[field_simulation]``. Child sections are separate Config objects,
    attached by the adapter after it configured the child components."""

    _CONFIGS_ALLOWED_KEYS = Config._CONFIGS_ALLOWED_KEYS | {"plot"}
    _CONFIGS_CHILD_KEYS = ("ground_shading", "evapotranspiration", "soil_simulation", "soil_predictor")

    model = ParameterGroup(key="model", desc="[model] field-level van Genuchten parameters (passthrough)")
    lai_type = SelectParameter(choices=list(LAI_TYPES), default="grass", desc="Leaf area index profile")
    roughness = Parameter(type=float, default=0.002, min=0.0, desc="Canopy surface roughness length (m)")
    plant_height = Parameter(type=float, default=0.1, min=0.0, desc="Canopy plant height (m)")
    ndvi = Parameter(type=float, default=0.25, min=0.0, max=1.0, desc="Canopy NDVI")
    bare_lai = Parameter(type=float, default=1.0, min=0.0, desc="LAI of the bare segments")
    bare_roughness = Parameter(type=float, default=0.002, min=0.0, desc="Roughness length of the bare segments (m)")
    bare_plant_height = Parameter(type=float, default=0.1, min=0.0, desc="Plant height of the bare segments (m)")
    bare_ndvi = Parameter(type=float, default=0.25, min=0.0, max=1.0, desc="NDVI of the bare segments")
    bay_width = Parameter(type=float, default=3.5, min=0.0, desc="Distance between PV rows (m)")
    interval = Parameter(type=int, default=30, min=1, desc="Tick cadence, wall-clock aligned (minutes)")
    offset = Parameter(type=int, default=0, min=0, desc="Tick offset within the interval (minutes)")
    intake_delay = DurationParameter(default="30min", desc="How far behind now the frontier may advance")

    # attached by the adapter (not config keys of this section)
    soil: SoilConfig
    planner: Optional[PlannerConfig] = None
    shading: Any = None  # shading.ShadingConfig
    plots: Optional[PlotConfig] = None

    def _on_configure(self, configs: Configurations) -> None:
        if not 0 <= self.offset < self.interval:
            raise ConfigurationError("offset must satisfy 0 <= offset < interval")

    def attach(
        self,
        *,
        soil: SoilConfig,
        planner: Optional[PlannerConfig] = None,
        shading: Any = None,
        plots: Optional[PlotConfig] = None,
    ) -> FieldConfig:
        self.soil = soil
        self.planner = planner
        self.shading = shading
        self.plots = plots
        return self

    @property
    def interval_td(self):
        import datetime as dt

        return dt.timedelta(minutes=self.interval)
