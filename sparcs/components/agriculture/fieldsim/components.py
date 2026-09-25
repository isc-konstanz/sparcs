# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.fieldsim.components
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The lories layer: the only module in this package that imports lories
components and channels.

``FieldSimulation`` configures the ``FieldConfig`` section from its own
``.conf``, lets each ``.d`` child configure its own section, attaches them,
builds the ``Simulation``, the runner and the scheduler, owns the thread
through activate/deactivate, and exposes the ``Snapshot`` for dash.

``GroundShading``, ``Evapotranspiration``, ``SoilSimulation`` and
``SoilPredictor`` remain Components for one reason: channel ids and logger
table names derive from the component id, and those are frozen. Each is a
``ChannelNamespace``: a TYPE, a ``CHANNELS`` spec list, and the ``Config``
class its file is validated against. ``ChannelInputs`` and ``ChannelOutputs``
implement the two runtime ports over those channels.

Every section's keys are declared once, as lories ``Parameter`` descriptors
on the ``Config`` classes in ``core.config``; unknown keys fail the load.
``FieldSimulation.schema()`` returns the whole tree.

Not registered as a component type; the live ``simulation.FieldSimulation``
keeps the ``field_simulation`` type. Class names are reused on purpose so the
skeleton reads against the current diagram.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any, ClassVar, Mapping, Optional, Sequence, Type, TypeVar

import pandas as pd
from lories.components import Component
from lories.core import Configurations

from .core.assimilator import Assimilator
from .core.chain import WeatherChain
from .core.config import Config, FieldConfig, PlannerConfig, PlotConfig, SoilConfig
from .core.engine import SoilEngine
from .core.evapotranspiration import ETModel
from .core.planner import IrrigationPlanner
from .core.shading import ShadingConfig, ShadingModel
from .core.simulation import Simulation
from .core.state import ChainResult, Plan, Snapshot, SoilState, StepResult
from .runtime.ports import InputKey
from .runtime.runner import FieldRunner
from .runtime.scheduler import TickScheduler

_C = TypeVar("_C", bound="ChannelNamespace")


@dataclass(frozen=True)
class ChannelSpec:
    key: str
    type: type
    name: str
    unit: str = ""
    aggregate: str = "mean"
    logged: bool = False  # default; the .d file's [data.channels.<key>.logger] overrides


class ChannelNamespace(Component):
    """A Component that exists to own channels under its id and to validate
    its ``.d`` file against one ``Config`` class. No logic."""

    CHANNELS: ClassVar[Sequence[ChannelSpec]] = ()
    CONFIG: ClassVar[Optional[Type[Config]]] = None

    config: Optional[Config] = None

    def configure(self, configs: Configurations) -> None:
        super().configure(configs)
        if self.CONFIG is not None:
            self.config = self.CONFIG()
            self.config.configure(configs)  # resolves, applies defaults, rejects unknown keys
        for spec in self.CHANNELS:
            self.data.add(
                spec.key,
                type=spec.type,
                name=spec.name,
                unit=spec.unit,
                aggregate=spec.aggregate,
                logger={"enabled": spec.logged},
            )

    @classmethod
    def schema(cls) -> dict[str, Any]:
        return cls.CONFIG.schema() if cls.CONFIG is not None else {}


class GroundShading(ChannelNamespace):
    TYPE: str = "ground_shading"
    CONFIG = ShadingConfig
    CHANNELS = (  # excerpt; the full list is today's GroundShading.CHANNELS + per-segment GHI
        ChannelSpec("shading_factor", float, "Ground Shading Factor", logged=True),
        ChannelSpec("shading_progress_image", bytes, "Ground Shading Progress Image", "png", "last", logged=True),
        ChannelSpec("plot_strikes", float, "Plot Strikes", aggregate="last"),
    )


class Evapotranspiration(ChannelNamespace):
    TYPE: str = "evapotranspiration"
    CONFIG = None  # no keys of its own today; its .d file carries only channels
    CHANNELS = (  # excerpt; today's Evapotranspiration.CHANNELS, Penman-Monteith intermediates
        ChannelSpec("net_irradiance", float, "Net Irradiance", "W/m^2"),
        ChannelSpec("aerodynamic_resistance", float, "Aerodynamic Resistance", "s/m"),
        ChannelSpec("evapotranspiration", float, "Evapotranspiration", "kg/(m^2*h)"),
    )


class SoilSimulation(ChannelNamespace):
    TYPE: str = "soil_simulation"
    CONFIG = SoilConfig
    CHANNELS = (  # excerpt; WATER_* / WALK_* / stall counters + per-probe channels from [probes.points.*]
        ChannelSpec("simulation_state", bytes, "Simulation State", aggregate="last", logged=True),
        ChannelSpec("water_total", float, "Total Soil Water", "m^3/m"),
        ChannelSpec("water_anchor", float, "Anchor Correction", "m^3/m"),
    )


class SoilPredictor(ChannelNamespace):
    TYPE: str = "soil_predictor"
    CONFIG = PlannerConfig
    CHANNELS = ()  # forecast header / detail / irrigation / image are logger tables, written directly


class ChannelInputs:
    """``Inputs`` over lories connector reads. Every key is one ranged read;
    no decisions live here."""

    def __init__(self, field: FieldSimulation, soil: SoilSimulation) -> None:
        self.field = field
        self.soil = soil

    def read(self, key: InputKey, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
        # WEATHER: self.field.weather.data.read(start, end), trimmed + validated
        # IRRIGATION: meter channel, else valve state x design flow, else empty
        # TENSION: discovered SoilMoisture sensors, one ranged read each
        # FORECAST: forecast channels for the planner horizon
        raise NotImplementedError

    def load_state(self) -> SoilState | None:
        raise NotImplementedError  # soil.data.simulation_state -> SoilState.from_blob


class ChannelOutputs:
    """``Outputs`` over channel sets and logger tables. Fans a ``ChainResult``
    out per row; writes plan tables best effort, one try/except each."""

    def __init__(
        self,
        shading: Optional[GroundShading],
        et: Optional[Evapotranspiration],
        soil: SoilSimulation,
        predictor: Optional[SoilPredictor],
    ) -> None:
        self.shading = shading
        self.et = et
        self.soil = soil
        self.predictor = predictor

    def chain(self, now: dt.datetime, result: ChainResult) -> None:
        raise NotImplementedError  # per row: shading.data.*.set(...), et.data.*.set(...); image when present

    def step(self, result: StepResult) -> None:
        raise NotImplementedError  # soil channels .set(...) incl. probe tensions

    def plan(self, plan: Plan) -> None:
        raise NotImplementedError  # predictor logger tables, four best-effort writes

    def save_state(self, state: SoilState) -> None:
        raise NotImplementedError  # soil.data.simulation_state.set(state.to_blob())


class FieldSimulation(Component):
    """Configure own section, let children configure theirs, attach, assemble, own the thread."""

    TYPE: str = "field_simulation"
    CHILDREN: ClassVar[Sequence[Type[ChannelNamespace]]] = (
        GroundShading,
        Evapotranspiration,
        SoilSimulation,
        SoilPredictor,
    )
    INCLUDES = [c.TYPE for c in CHILDREN]

    def configure(self, configs: Configurations) -> None:
        super().configure(configs)
        field = FieldConfig()
        field.configure(configs)  # own keys; child sections and [plot] are allowed, not resolved here

        self.ground_shading = self._child(GroundShading, configs)
        self.evapotranspiration = self._child(Evapotranspiration, configs)
        self.soil = self._child(SoilSimulation, configs)
        self.predictor = self._child(SoilPredictor, configs)
        if self.soil is None or self.soil.config is None:
            raise ValueError(f"{self.id}: [soil_simulation] block is required")

        plots = None
        if configs.has_member("plot"):
            plots = PlotConfig()
            plots.configure(configs.get_member("plot"))
        shading = self.ground_shading.config if self.ground_shading is not None else ShadingConfig.from_dict()
        shading.derive(bay_width=field.bay_width)  # pv_rows / segment_ranges follow from the PV system and mesh
        self.config = field.attach(
            soil=self.soil.config,
            planner=self.predictor.config if self.predictor is not None else None,
            shading=shading,
            plots=plots,
        )

        engine = SoilEngine.build(self.config.soil)
        self.simulation = Simulation(
            self.config,
            engine=engine,
            chain=WeatherChain(self.config, ShadingModel(self.config.shading), ETModel(), self.config.plots),
            assimilator=Assimilator(self.config.soil.anchor, engine),
            planner=IrrigationPlanner(self.config.planner, engine) if self.config.planner else None,
        )
        self.runner = FieldRunner(
            self.config,
            self.simulation,
            inputs=ChannelInputs(self, self.soil),
            outputs=ChannelOutputs(self.ground_shading, self.evapotranspiration, self.soil, self.predictor),
        )
        self.scheduler = TickScheduler(self.config, self.runner)

    def activate(self) -> None:
        super().activate()
        # resolve weather / irrigation siblings, validate inputs, discover sensors
        self.scheduler.start()

    def deactivate(self) -> None:
        self.scheduler.stop()
        super().deactivate()

    @property
    def snapshot(self) -> Snapshot:
        """What dash reads. Never a channel round-trip."""
        return self.simulation.snapshot()

    @classmethod
    def schema(cls) -> dict[str, Any]:
        """The whole tree of needed and allowed keys: own section, [plot],
        and one entry per child type. What a config editor reads."""
        tree: dict[str, Any] = dict(FieldConfig.schema())
        tree["plot"] = {"type": "group", "children": PlotConfig.schema()}
        for child in cls.CHILDREN:
            tree[child.TYPE] = {"type": "component", "children": child.schema()}
        return tree

    def _child(self, cls: Type[_C], configs: Configurations) -> Optional[_C]:
        """Build one channel namespace from its ``.d`` member, if present."""
        if not configs.has_member(cls.TYPE, includes=True):
            return None
        child = cls(self, configs.get_member(cls.TYPE))
        self.components.add(child)
        return child


Mapping  # imported for annotations in subclasses
