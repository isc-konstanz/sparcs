# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.fieldsim.core.config
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Configuration sections for the ``field_simulation.conf`` +
``field_simulation.d/`` layout, declared as lories ``Configurator`` classes.
Unknown keys are a hard error; ``FieldSetup`` bundles the configured sections.
"""

from __future__ import annotations

import datetime as dt
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Iterable, Mapping, Optional

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
GRID_MODES = ("fill_order", "full")

# Never created, so ``from_dict`` cannot pick up a "<name>.d/<key>.conf" from disk.
_NO_CONF_DIR = Path(tempfile.gettempdir(), "fieldsim-no-conf-dir")


# --------------------------------------------------------------------------- base


class Config(Configurator):
    """Base for all fieldsim sections: strict keys, typed sub-sections, schema, from_dict."""

    _CONFIGS_RESERVED_KEYS: ClassVar[frozenset] = Configurator._CONFIGS_RESERVED_KEYS | {
        "type",
        "data",
        "components",
        "connectors",
        "converters",
    }
    # Child sections handled by other Config classes (set per class).
    _CONFIGS_CHILD_KEYS: ClassVar[Iterable[str]] = ()
    # attribute -> section class, for tables declared with section()
    _SECTIONS: ClassVar[Mapping[str, type]] = {}

    @classmethod
    def _assert_configs(cls, configs: Optional[Configurations]) -> Optional[Configurations]:
        configs = super()._assert_configs(configs)
        if configs is None:
            return None
        unknown = _unknown_keys(
            configs,
            cls._params_by_key(),
            cls._allowed_keys(),
            sections=cls._sections_by_key(),
            path=cls.__name__,
        )
        if unknown:
            raise ConfigurationError(f"{cls.__name__}: unknown configuration keys {unknown}")
        return configs

    @classmethod
    def _params_by_key(cls) -> dict[str, _Parameter]:
        return {p._resolve_key(): p for p in cls.__config_parameters__.values()}

    @classmethod
    def _allowed_keys(cls) -> set:
        return set(cls._CONFIGS_RESERVED_KEYS) | set(cls._CONFIGS_CHILD_KEYS)

    @classmethod
    def _sections_by_key(cls) -> dict[str, type]:
        """``_SECTIONS`` keyed by config key instead of attribute name."""
        params = cls.__config_parameters__
        return {params[attr]._resolve_key(): section_cls for attr, section_cls in cls._SECTIONS.items()}

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
        """Declared keys with type, default, bounds and description."""
        sections = cls._sections_by_key()
        out: dict[str, Any] = {}
        for p in cls.__config_parameters__.values():
            key = p._resolve_key()
            sch = p.to_schema()
            if key in sections:
                sch["children"] = sections[key].schema()
            elif isinstance(p, ParameterGroup) and not p.children:
                sch["passthrough"] = True
            out[key] = sch
        return out

    @classmethod
    def from_dict(cls, values: Optional[Mapping[str, Any]] = None, name: str = "fieldsim"):
        """A configured instance from a plain mapping; reads nothing from disk."""
        instance = cls()
        instance.configure(Configurations(name, Directories(conf_dir=str(_NO_CONF_DIR)), defaults=dict(values or {})))
        return instance

    def values(self) -> dict[str, Any]:
        return {param._resolve_key(): getattr(self, attr) for attr, param in type(self).__config_parameters__.items()}


def section(key: str, *, required: bool = False, desc: Optional[str] = None) -> ParameterGroup:
    """Declare a nested table whose keys belong to the ``_SECTIONS`` class for it.

    Childless on purpose: the group only enforces presence, while ``_SECTIONS``
    carries the class to the strict key check, the schema and ``_at_configure``.
    """
    return ParameterGroup(key=key, desc=desc, required=required)


def _unknown_keys(
    configs: Any,
    params: Mapping[str, _Parameter],
    allowed: set,
    *,
    sections: Optional[Mapping[str, type]] = None,
    path: str,
) -> list[str]:
    """Keys of ``configs`` that no parameter declares, recursing into sections and groups."""
    unknown: list[str] = []
    sections = sections or {}
    for key in configs:
        if key in allowed:
            continue
        param = params.get(key)
        is_section = isinstance(configs.get(key), Mapping)
        section_cls = sections.get(key)
        if param is None:
            unknown.append(f"{path}.{key}")
        elif section_cls is not None and is_section:
            unknown += _unknown_keys(
                configs.get_member(key),
                section_cls._params_by_key(),
                section_cls._allowed_keys(),
                sections=section_cls._sections_by_key(),
                path=f"{path}.{key}",
            )
        elif isinstance(param, ParameterGroup) and param.children and is_section:
            unknown += _unknown_keys(configs.get_member(key), param.children, set(), path=f"{path}.{key}")
        elif isinstance(param, _TypedParameter) and is_section:
            unknown.append(f"{path}.{key} (section given, scalar expected)")
    return unknown


def _positive(value: float) -> None:
    if value <= 0:
        raise ValueError("must be greater than zero")


# --------------------------------------------------------------------------- leaf sections


class PlotConfig(Config):
    """``[plot]``: progress-image settings. ``enabled`` is lories' own table switch."""

    interval = DurationParameter(default="1h", desc="Minimum time between two rendered frames")
    dir = Parameter(
        type=str, default=None, required=False, desc="Directory the images are written to; defaults to the data dir"
    )
    disable_after_failures = Parameter(
        type=int, default=3, min=1, desc="Consecutive render failures before plotting stops"
    )


class MeshConfig(Config):
    """``[soil_simulation.mesh]``: the 2D bay cross-section handed to Gmsh."""

    filename = Parameter(type=str, default="soil.msh", desc="Gmsh mesh file, built when missing")
    dl = Parameter(type=float, default=0.1, validator=_positive, desc="Element characteristic length (m), > 0")
    width = Parameter(
        type=float, default=None, required=False, desc="Cross-section width (m); defaults to the field bay_width"
    )
    height = Parameter(type=float, default=5.0, validator=_positive, desc="Cross-section depth (m), > 0")
    plant_width = Parameter(type=float, default=2.0, min=0.0, desc="Root zone width in the bay centre (m)")
    plant_height = Parameter(type=float, default=2.0, min=0.0, desc="Root zone depth in the bay centre (m)")
    watering_width = Parameter(type=float, default=1.0, min=0.0, desc="Drip strip width (m); keep >= dl")
    dx = Parameter(key="d_x", type=float, default=0.5, min=0.0, desc="Top-boundary segment width (m); keep > dl")

    def _on_configure(self, configs: Configurations) -> None:
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
        if width < self.plant_width + 2 * self.dx:
            raise ConfigurationError("mesh.width must be at least plant_width + 2 * d_x")
        if self.height <= self.plant_height:
            raise ConfigurationError("mesh.height must be greater than plant_height")
        half_width = (width - self.plant_width) / 2
        surface_count = round(half_width / self.dx)
        if surface_count < 1 or abs(half_width - surface_count * self.dx) > 1e-9 * max(1.0, half_width):
            raise ConfigurationError("mesh: (width - plant_width) must be a multiple of 2 * d_x")

    @property
    def top_segments(self) -> int:
        """Bare top segments per side of the root zone."""
        return int(round((self.width - self.plant_width) / (2.0 * self.dx)))


class DripConfig(Config):
    """``[drip]``: whole-field drip layout; ``explicit`` records whether the table was present."""

    nozzle_count = Parameter(type=int, default=1, min=1, desc="Number of drip nozzles on the field")
    nozzle_flow_lph = Parameter(type=float, default=1.0, min=0.0, desc="Flow per nozzle (l/h)")

    explicit: bool = True

    @property
    def design_flow_lpm(self) -> float:
        return self.nozzle_count * self.nozzle_flow_lph / 60.0


# --------------------------------------------------------------------------- component sections


class SoilConfig(Config):
    """``[soil_simulation]``: everything the engine needs, nothing more."""

    _CONFIGS_RESERVED_KEYS = Config._CONFIGS_RESERVED_KEYS | {"plot"}
    _SECTIONS = {"mesh": MeshConfig, "drip": DripConfig}

    mesh = section("mesh", required=True, desc="[mesh] bay cross-section geometry")
    drip = section("drip", desc="[drip] whole-field drip layout; defaults to 1 nozzle x 1 l/h")
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
        type=float, default=1.0, validator=_positive, desc="Total drip line length under the mesh (m), > 0"
    )
    discover_sensor_probes = Parameter(
        type=bool, default=False, desc="Register a probe per discovered SoilMoisture sensor"
    )
    plot_structure = Parameter(type=bool, default=False, desc="Render the mesh structure plot once at configure")

    def _on_configure(self, configs: Configurations) -> None:
        if self.drip is None:
            self.drip = DripConfig.from_dict()
            self.drip.explicit = False


class PlannerConfig(Config):
    """``[soil_predictor]``: its ``[drip]`` defaults per key to the sim's."""

    _CONFIGS_RESERVED_KEYS = Config._CONFIGS_RESERVED_KEYS | {"plot"}
    _SECTIONS = {"drip": DripConfig}

    windows = ParameterGroup(key="windows", desc="[windows.<name>] start time and durations (passthrough)")
    interval = Parameter(type=int, default=1440, min=1, desc="Planner schedule cadence (minutes)")
    offset = Parameter(type=int, default=60, min=0, desc="Planner schedule offset within the interval (minutes)")
    state = ParameterGroup(
        key="state",
        desc="[state] predictor state-blob debug sink",
        children=[
            Parameter(key="save", type=bool, default=False, desc="Persist predict_state blobs"),
            DurationParameter(key="interval", default="1h", desc="Blob cadence"),
        ],
    )
    grid_mode = SelectParameter(
        choices=list(GRID_MODES), default="fill_order", desc="Shared-prefix ladder or full rolls"
    )
    combo_cap = Parameter(type=int, default=16, min=1, desc="Upper bound on candidates per run")
    threshold_hpa = Parameter(
        type=float, default=300.0, min=0.0, desc="Tension threshold the score measures against (hPa)"
    )
    decision_probes = ListParameter(item_type=str, default=[], desc="Probe keys the score is evaluated at")
    horizon = DurationParameter(default="24h", desc="Forecast horizon the candidates are rolled over")
    max_windows = Parameter(type=int, default=4, min=1, desc="Windows considered per day")
    parallel = Parameter(type=bool, default=False, desc="Roll candidates in a process pool")
    # no min bound: lories compares a None default against min (TypeError)
    max_workers = Parameter(type=int, default=None, required=False, desc="Pool size; default cpu_count - 1")
    logger = Parameter(type=str, default=None, required=False, desc="Logger connector id for the forecast tables")
    drip = section("drip", desc="[drip] the planner's drip layout; unset keys fall back to [soil_simulation.drip]")

    def _on_configure(self, configs: Configurations) -> None:
        if len(self.windows) > self.max_windows:
            raise ConfigurationError(f"{len(self.windows)} windows configured, max_windows is {self.max_windows}")
        if self.max_workers is not None and self.max_workers < 1:
            raise ConfigurationError("max_workers must be at least 1")
        if not 0 <= self.offset < self.interval:
            raise ConfigurationError("offset must satisfy 0 <= offset < interval")


class EvapotranspirationConfig(Config):
    """``[evapotranspiration]``: no keys of its own; its ``.d`` file carries only channels."""

    _CONFIGS_RESERVED_KEYS = Config._CONFIGS_RESERVED_KEYS | {"plot"}


class FieldConfig(Config):
    """``[field_simulation]``: the field-level keys."""

    _CONFIGS_RESERVED_KEYS = Config._CONFIGS_RESERVED_KEYS | {"plot"}
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
    interval = Parameter(type=int, default=60, min=1, desc="Tick cadence, wall-clock aligned (minutes)")
    offset = Parameter(type=int, default=0, min=0, desc="Tick offset within the interval (minutes)")
    intake_delay = DurationParameter(default="0min", desc="How far behind now the frontier may advance")

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
    """The configured sections the runtime receives; nothing here changes afterwards."""

    field: FieldConfig
    soil: SoilConfig
    shading: Any  # shading.ShadingConfig; typed Any to avoid an import cycle
    planner: Optional[PlannerConfig] = None
    plots: Optional[PlotConfig] = None
    location: Any = None  # lories Location of the field; None offline
