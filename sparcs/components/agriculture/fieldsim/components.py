# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.fieldsim.components
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The lories layer: the only module in this package that imports lories.

``FieldSimulation`` parses the unchanged ``.d`` tree into ``FieldConfig``
once, builds engine, chain, runner and scheduler, and owns the thread through
activate/deactivate. ``GroundShading``, ``Evapotranspiration``,
``SoilSimulation`` and ``SoilPredictor`` remain Components for one reason:
channel ids and logger table names derive from the component id, and those
are frozen. They hold channels and no logic. ``ChannelIO`` implements
``FieldIO`` over those channels.

Not registered as a component type; the live ``simulation.FieldSimulation``
keeps the ``field_simulation`` type. Class names are reused on purpose so the
skeleton reads against the current diagram.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Mapping, Optional, Type, TypeVar

import pandas as pd
from lories.components import Component
from lories.core import Configurations

from .base.runner import FieldRunner
from .base.scheduler import TickScheduler
from .core.assimilator import Assimilator
from .core.chain import WeatherChain
from .core.config import FieldConfig
from .core.engine import SoilEngine
from .core.evapotranspiration import ETModel
from .core.planner import IrrigationPlanner
from .core.shading import ShadingModel
from .core.state import ChainResult, Plan, SoilState, StepResult

_C = TypeVar("_C", bound=Component)


class GroundShading(Component):
    """Channel namespace: ``shading_factor`` and ``shading_progress_image``
    (both logged to named tables), per-segment GHI, ``plot_strikes``."""

    TYPE: str = "ground_shading"

    def configure(self, configs: Configurations) -> None:
        super().configure(configs)
        raise NotImplementedError


class Evapotranspiration(Component):
    """Channel namespace: the Penman-Monteith intermediates and ``evapotranspiration``."""

    TYPE: str = "evapotranspiration"

    def configure(self, configs: Configurations) -> None:
        super().configure(configs)
        raise NotImplementedError


class SoilSimulation(Component):
    """Channel namespace for probe, state, mass-balance and walk channels."""

    TYPE: str = "soil_simulation"

    def configure(self, configs: Configurations) -> None:
        super().configure(configs)
        raise NotImplementedError


class SoilPredictor(Component):
    """Channel namespace for the forecast header / detail / irrigation / image tables."""

    TYPE: str = "soil_predictor"

    def configure(self, configs: Configurations) -> None:
        super().configure(configs)
        raise NotImplementedError


class ChannelIO:
    """``FieldIO`` over lories channels. Every method is one ranged read or
    one channel/table write; no decisions live here."""

    def __init__(
        self,
        field: FieldSimulation,
        shading: Optional[GroundShading],
        et: Optional[Evapotranspiration],
        soil: SoilSimulation,
        predictor: Optional[SoilPredictor],
    ) -> None:
        self.field = field
        self.shading = shading
        self.et = et
        self.soil = soil
        self.predictor = predictor

    def read_weather(self, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
        raise NotImplementedError  # self.field.weather.data.read(start, end), trimmed + validated

    def read_irrigation_lpm(self, start: dt.datetime, end: dt.datetime) -> pd.Series:
        raise NotImplementedError  # meter -> state x design flow -> zeros

    def read_tension_history(self, start: dt.datetime, end: dt.datetime) -> Mapping[str, pd.Series]:
        raise NotImplementedError  # discovered SoilMoisture sensors, ranged read each

    def read_forecast(self, creation: dt.datetime | None) -> pd.DataFrame:
        raise NotImplementedError

    def load_state(self) -> SoilState | None:
        raise NotImplementedError  # soil.data.simulation_state -> SoilState.from_blob

    def save_state(self, state: SoilState) -> None:
        raise NotImplementedError

    def publish_chain(self, now: dt.datetime, result: ChainResult) -> None:
        raise NotImplementedError  # shading.data.*.set(...), et.data.*.set(...), image bytes when present

    def publish(self, now: dt.datetime, result: StepResult, probe_tension: Mapping[str, float]) -> None:
        raise NotImplementedError  # soil channels .set(...)

    def write_plan(self, plan: Plan) -> None:
        raise NotImplementedError  # predictor logger tables, four best-effort writes


class FieldSimulation(Component):
    """Parse once, assemble, own the thread."""

    TYPE: str = "field_simulation"
    INCLUDES = [GroundShading.TYPE, Evapotranspiration.TYPE, SoilSimulation.TYPE, SoilPredictor.TYPE]

    def configure(self, configs: Configurations) -> None:
        super().configure(configs)
        self.config = FieldConfig.from_mapping(self._materialize(configs))
        self.ground_shading = self._child(GroundShading, configs)
        self.evapotranspiration = self._child(Evapotranspiration, configs)
        self.soil = self._child(SoilSimulation, configs)
        self.predictor = self._child(SoilPredictor, configs)
        if self.soil is None:
            raise ValueError(f"{self.id}: [soil_simulation] block is required")

        engine = SoilEngine.build(self.config.soil)
        chain = WeatherChain(self.config, ShadingModel(self.config.shading), ETModel(), self.config.plots)
        planner = IrrigationPlanner(self.config.planner, engine) if self.config.planner else None
        self.runner = FieldRunner(
            self.config,
            io=ChannelIO(self, self.ground_shading, self.evapotranspiration, self.soil, self.predictor),
            engine=engine,
            chain=chain,
            assimilator=Assimilator(self.config.soil.anchor, engine),
            planner=planner,
        )
        self.scheduler = TickScheduler(self.config, self.runner)

    def activate(self) -> None:
        super().activate()
        # resolve weather / irrigation siblings, validate inputs, discover sensors
        self.scheduler.start()

    def deactivate(self) -> None:
        self.scheduler.stop()
        super().deactivate()

    def _child(self, cls: Type[_C], configs: Configurations) -> Optional[_C]:
        """Build one thin channel Component from its ``.d`` member, if present."""
        if not configs.has_member(cls.TYPE, includes=True):
            return None
        child = cls(self, configs.get_member(cls.TYPE))
        self.components.add(child)
        return child

    @staticmethod
    def _materialize(configs: Configurations) -> Mapping[str, Any]:
        """Walk the known member keys with ``get_member(..., ensure_exists=True)``
        and return a plain nested dict, so ``.d`` files declared only on disk
        are loaded and the core never sees a ``Configurations``. The
        ``[soil_simulation.model]`` over ``[model]`` cascade is applied here
        with ``Component._build_defaults`` and nowhere else."""
        raise NotImplementedError
