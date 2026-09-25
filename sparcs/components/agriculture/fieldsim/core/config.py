# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.fieldsim.core.config
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Configuration sections for the unchanged ``field_simulation.conf`` +
``field_simulation.d/`` layout, declared with lories' parameter structure.

Each section is a lories ``Configurator`` mirroring one TOML table. Every
key is one class-level ``Parameter`` (type, default, bounds, choices,
description), so what is needed and allowed is declared once and known at
runtime: lories resolves and coerces the TOML, applies defaults, and
``schema()`` returns the declaration. Unknown keys are a hard error
(``Config._assert_configs``), recursively through declared groups, the same
rule lories' converters and processors follow.

A nested table that has its own section class (``[mesh]``, ``[drip]``) is
declared with ``section()``: the strict check and the schema see its keys,
and after resolution the attribute holds a configured instance of that
class, not a dict. Tables still owned by dataclasses next to FiPy code
(pde, ponding, feddes, anchor, probes, windows) remain passthrough groups:
accepted verbatim, flagged ``passthrough`` in the schema, not validated.

``FieldSetup`` is the frozen bundle the runtime receives: nothing on a
section changes after ``configure``. ``from_dict`` builds a configured
section from a plain mapping for scenarios, notebooks and tests.

This is the only lories import in ``core``: the config machinery. Channels,
components and threads stay out.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
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


# --------------------------------------------------------------------------- base


class Config(Configurator):
    """Base for all fieldsim sections: strict keys, typed sub-sections, schema, from_dict."""

    # Keys every section may carry without declaring them: the lories
    # reserved set plus the component-level keys that share the same file.
    _CONFIGS_ALLOWED_KEYS: ClassVar[frozenset] = frozenset({"type", "data"})
    # Child sections handled by other Config classes (set per class).
    _CONFIGS_CHILD_KEYS: ClassVar[Iterable[str]] = ()
    # attribute -> section class, for tables declared with section()
    _SECTIONS: ClassVar[Mapping[str, type]] = {}

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

    def _at_configure(self, configs: Configurations) -> None:
        super()._at_configure(configs)
        for attr, cls in self._SECTIONS.items():
            key = type(self).__config_parameters__[attr]._resolve_key()
            if isinstance(configs.get(key), Mapping):
                obj = cls()
                obj.configure(configs.get_member(key))
                setattr(self, attr, obj)
            else:
                setattr(self, attr, None)

    @classmethod
    def schema(cls) -> dict[str, Any]:
        """Needed and allowed keys with type, default, bounds and description.
        Groups without declared children are flagged ``passthrough``."""
        out: dict[str, Any] = {}
        for p in cls.__config_parameters__.values():
            sch = p.to_schema()
            if isinstance(p, ParameterGroup) and not p.children:
                sch["passthrough"] = True
            out[p._resolve_key()] = sch
        return out

    @classmethod
    def from_dict(cls, values: Optional[Mapping[str, Any]] = None, name: str = "fieldsim"):
        """A configured instance from a plain mapping; for scenarios and tests."""
        instance = cls()
        instance.configure(Configurations(name, Directories(), defaults=dict(values or {})))
        return instance

    def values(self) -> dict[str, Any]:
        return {attr: getattr(self, attr) for attr in type(self).__config_parameters__}


def section(cls: type, key: str, *, required: bool = False, desc: Optional[str] = None) -> ParameterGroup:
    """Declare a nested table validated by section class ``cls``. Pair it
    with an entry in ``_SECTIONS`` so the attribute becomes a ``cls`` instance."""
    return ParameterGroup(key=key, desc=desc, required=required, children=list(cls.__config_parameters__.values()))


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


# --------------------------------------------------------------------------- leaf sections


class PlotConfig(Config):
    """``[plot]``: progress-image settings, field-level default cascaded to every child."""

    enabled = Parameter(type=bool, default=False, desc="Render progress images")
    interval = DurationParameter(default="1h", desc="Minimum time between two rendered frames")
    disable_after_failures = Parameter(
        type=int, default=5, min=1, desc="Consecutive render failures before plotting stops"
    )


class MeshConfig(Config):
    """``[soil_simulation.mesh]``: the 2D bay cross-section handed to Gmsh.

    ``width`` defaults to the field-level ``bay_width`` so mesh and PV bay
    agree; the adapter fills it through ``derive``. The top boundary is cut
    into ``d_x``-wide segments, so ``(width - plant_width) / (2 * d_x)`` must
    be a non-negative integer.
    """

    filename = Parameter(type=str, default="soil.msh", desc="Gmsh mesh file, built when missing")
    dl = Parameter(type=float, default=0.1, min=0.0, desc="Element characteristic length (m)")
    width = Parameter(
        type=float, default=None, required=False, desc="Cross-section width (m); defaults to the field bay_width"
    )
    height = Parameter(type=float, default=5.0, min=0.0, desc="Cross-section depth (m)")
    plant_width = Parameter(type=float, default=2.0, min=0.0, desc="Root zone width in the bay centre (m)")
    plant_height = Parameter(type=float, default=2.0, min=0.0, desc="Root zone depth in the bay centre (m)")
    watering_width = Parameter(type=float, default=1.0, min=0.0, desc="Drip strip width (m); keep >= dl")
    dx = Parameter(key="d_x", type=float, default=0.5, min=0.0, desc="Top-boundary segment width (m); keep > dl")

    def _on_configure(self, configs: Configurations) -> None:
        if self.dl <= 0:
            raise ConfigurationError("mesh.dl must be positive")
        if self.watering_width < self.dl:
            raise ConfigurationError("mesh.watering_width must be >= dl so the drip strip is resolved")
        if self.dx <= self.dl:
            raise ConfigurationError("mesh.d_x must be larger than dl")
        if self.width is not None:
            self._check_segments(self.width)

    def derive(self, *, bay_width: float) -> MeshConfig:
        """Fill ``width`` from the field-level bay width when not set."""
        if self.width is None:
            self.width = float(bay_width)
            self._check_segments(self.width)
        return self

    def _check_segments(self, width: float) -> None:
        n = (width - self.plant_width) / (2.0 * self.dx)
        if n < 0 or abs(n - round(n)) > 1e-9:
            raise ConfigurationError(
                f"mesh: (width - plant_width) / (2 * d_x) = {n:.4f} must be a non-negative integer"
            )

    @property
    def top_segments(self) -> int:
        """Bare top segments per side of the root zone."""
        return int(round((self.width - self.plant_width) / (2.0 * self.dx)))


class DripConfig(Config):
    """``[drip]``: whole-field drip layout; ``design_flow_lpm`` is derived.

    ``explicit`` records whether the table was present. The live sim only
    trusts the valve-state fallback feed when it was: a bare state channel
    with no explicit block would otherwise roll at the 1 nozzle x 1 l/h
    placeholder.
    """

    nozzle_count = Parameter(type=int, default=1, min=1, desc="Number of drip nozzles on the field")
    nozzle_flow_lph = Parameter(type=float, default=1.0, min=0.0, desc="Flow per nozzle (l/h)")

    explicit: bool = True

    @property
    def design_flow_lpm(self) -> float:
        return self.nozzle_count * self.nozzle_flow_lph / 60.0

    def override(self, values: Mapping[str, Any]) -> DripConfig:
        """A copy with the given keys replaced (the predictor's per-key
        ``[soil_predictor.drip]`` merge over the sim's block)."""
        merged = {**self.values(), **dict(values)}
        out = DripConfig.from_dict(merged)
        out.explicit = self.explicit
        return out


# --------------------------------------------------------------------------- component sections


class SoilConfig(Config):
    """``[soil_simulation]``: everything the engine needs, nothing more."""

    _CONFIGS_ALLOWED_KEYS = Config._CONFIGS_ALLOWED_KEYS | {"plot"}
    _SECTIONS = {"mesh": MeshConfig, "drip": DripConfig}

    mesh = section(MeshConfig, "mesh", required=True, desc="[mesh] bay cross-section geometry")
    drip = section(DripConfig, "drip", desc="[drip] whole-field drip layout; defaults to 1 nozzle x 1 l/h")
    pde = ParameterGroup(key="pde", desc="[pde] solver settings (simulation._soil.PDEConfig, passthrough)")
    ponding = ParameterGroup(key="ponding", desc="[ponding] surface storage, sibling of [pde] (passthrough)")
    feddes = ParameterGroup(key="feddes", desc="[feddes] root water uptake, sibling of [pde] (passthrough)")
    anchor = ParameterGroup(
        key="anchor", desc="[anchor] tensiometer assimilation incl. [anchor.sensors.*] (passthrough)"
    )
    probes = ParameterGroup(key="probes", desc="[probes.points.<key>] virtual tensiometers (passthrough)")
    model = ParameterGroup(key="model", desc="[model] van Genuchten override of the field-level [model]")
    testing = ParameterGroup(
        key="testing", desc="[testing] rig-only options: history_window, poll_interval (passthrough)"
    )
    total_drip_line_length_m = Parameter(
        type=float, default=1.0, min=0.0, desc="Total drip line length under the mesh (m)"
    )
    discover_sensor_probes = Parameter(
        type=bool, default=False, desc="Register a probe per discovered SoilMoisture sensor"
    )
    plot_structure = Parameter(type=bool, default=False, desc="Render the mesh structure plot once at configure")

    # derived by the adapter from [probes.points.*] and sensor discovery (simulation._soil.ProbeSpec)
    probe_specs: Sequence[Any] = ()

    def _on_configure(self, configs: Configurations) -> None:
        if self.drip is None:
            self.drip = DripConfig.from_dict()
            self.drip.explicit = False
        if self.total_drip_line_length_m <= 0:
            raise ConfigurationError("total_drip_line_length_m must be positive")


class PlannerConfig(Config):
    """``[soil_predictor]``. The planner never reads ``[soil_simulation]``;
    its ``[drip]`` is a per-key override the adapter merges over the sim's."""

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
    drip = ParameterGroup(
        key="drip",
        desc="[drip] per-key override of [soil_simulation.drip]",
        children=[
            Parameter(key="nozzle_count", type=int, required=False, min=1, desc="Number of drip nozzles"),
            Parameter(key="nozzle_flow_lph", type=float, required=False, min=0.0, desc="Flow per nozzle (l/h)"),
        ],
    )

    def drip_override(self) -> dict[str, Any]:
        """Only the keys the table restates; absent table -> empty."""
        return {k: v for k, v in (self.drip or {}).items() if v is not None}


class FieldConfig(Config):
    """``[field_simulation]``: the field-level keys. Child tables are their
    own sections, bundled by the adapter into a ``FieldSetup``."""

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

    def _on_configure(self, configs: Configurations) -> None:
        if not 0 <= self.offset < self.interval:
            raise ConfigurationError("offset must satisfy 0 <= offset < interval")

    @property
    def interval_td(self) -> dt.timedelta:
        return dt.timedelta(minutes=self.interval)

    @property
    def offset_td(self) -> dt.timedelta:
        return dt.timedelta(minutes=self.offset)


# --------------------------------------------------------------------------- bundle


@dataclass(frozen=True)
class FieldSetup:
    """Everything the runtime needs, assembled once by the adapter after
    every section configured. Nothing here changes afterwards."""

    field: FieldConfig
    soil: SoilConfig
    shading: Any  # shading.ShadingConfig; typed Any to avoid an import cycle
    planner: Optional[PlannerConfig] = None
    plots: Optional[PlotConfig] = None

    @property
    def planner_drip(self) -> DripConfig:
        """The predictor's drip layout: its per-key override over the sim's."""
        if self.planner is None:
            return self.soil.drip
        return self.soil.drip.override(self.planner.drip_override())
