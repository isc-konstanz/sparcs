# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.fieldsim.components
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The lories layer: the only module in this package that imports lories.

``FieldSimulation`` parses the unchanged ``.d`` tree into ``FieldConfig``
once, builds engine, runner and scheduler, and owns the thread through
activate/deactivate. ``SoilSimulation`` and ``SoilPredictor`` remain
Components for one reason: channel ids and logger table names derive from
the component id, and those are frozen. They hold channels and no logic.
``ChannelIO`` implements ``FieldIO`` over those channels.

Not registered as a component type; the live ``simulation.FieldSimulation``
keeps the ``field_simulation`` type. Class names are reused on purpose so the
skeleton reads against the current diagram.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Mapping

import pandas as pd
from lories.components import Component
from lories.core import Configurations

from .assimilator import Assimilator
from .chain import WeatherChain
from .config import FieldConfig
from .engine import SoilEngine
from .planner import IrrigationPlanner
from .runner import FieldRunner
from .scheduler import TickScheduler
from .state import Plan, SoilState, StepResult


class SoilSimulation(Component):
    """Channel namespace for probe, state, mass-balance and walk channels."""

    TYPE: str = "soil_simulation"

    def configure(self, configs: Configurations) -> None:
        super().configure(configs)
        # register WATER_* / WALK_* / simulation_state / per-probe channels only
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

    def __init__(self, field: FieldSimulation, soil: SoilSimulation, predictor: SoilPredictor | None) -> None:
        self.field = field
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

    def publish(self, now: dt.datetime, result: StepResult, probe_tension: Mapping[str, float]) -> None:
        raise NotImplementedError  # soil channels .set(...)

    def write_plan(self, plan: Plan) -> None:
        raise NotImplementedError  # predictor logger tables, four best-effort writes


class FieldSimulation(Component):
    """Parse once, assemble, own the thread."""

    TYPE: str = "field_simulation"
    INCLUDES = [SoilSimulation.TYPE, SoilPredictor.TYPE]

    def configure(self, configs: Configurations) -> None:
        super().configure(configs)
        self.config = FieldConfig.from_mapping(self._materialize(configs))
        self.soil = SoilSimulation(self, configs.get_member(SoilSimulation.TYPE))
        self.components.add(self.soil)
        self.predictor: SoilPredictor | None = None
        if configs.has_member(SoilPredictor.TYPE, includes=True):
            self.predictor = SoilPredictor(self, configs.get_member(SoilPredictor.TYPE))
            self.components.add(self.predictor)

        engine = SoilEngine.build(self.config.soil)
        planner = IrrigationPlanner(self.config.planner, engine) if self.config.planner else None
        self.runner = FieldRunner(
            self.config,
            io=ChannelIO(self, self.soil, self.predictor),
            engine=engine,
            chain=WeatherChain(self.config),
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

    @staticmethod
    def _materialize(configs: Configurations) -> Mapping[str, Any]:
        """Walk the known member keys with ``get_member(..., ensure_exists=True)``
        and return a plain nested dict, so ``.d`` files declared only on disk
        are loaded and the core never sees a ``Configurations``. The
        ``[soil_simulation.model]`` over ``[model]`` cascade is applied here
        with ``Component._build_defaults`` and nowhere else."""
        raise NotImplementedError
