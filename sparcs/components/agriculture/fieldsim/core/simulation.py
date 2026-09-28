# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.fieldsim.core.simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The simulation session: owns the state, composes engine, chain, assimilator
and planner, and runs the per-chunk sequence. No I/O, no clock, no gating.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any, Mapping, Optional, Sequence

import pandas as pd

from .anchor import AnchorSensor
from .assimilator import Assimilator, parse_anchor_config
from .chain import WeatherChain
from .config import FieldSetup
from .engine import Cancel, SoilEngine
from .evapotranspiration import ETModel
from .pde import top_segment_count
from .planner import IrrigationPlanner
from .pv import _GROUND_SHADING_N_ROWS
from .shading import MODE_FREE_FIELD, ShadingModel
from .state import ChainResult, Forcing, Plan, Snapshot, SoilState, StepResult

logger = logging.getLogger(__name__)

FORECAST_CREATION_KEY = "timestamp_creation"


def _resolve_segment_ranges(mesh, shading_config, bay_width: float) -> Optional[dict[str, tuple[float, float]]]:
    """Soil-mesh top-segment x-ranges in pvfactors coordinates; ``None`` without a mesh."""
    if mesh is None:
        return None
    dx = mesh.dx
    plant_width = mesh.plant_width
    watering_width = mesh.watering_width
    n_pv_segments = top_segment_count(mesh)

    plant_left = n_pv_segments * dx
    plant_right = plant_left + plant_width
    watering_left = plant_left + (plant_width - watering_width) / 2
    watering_right = watering_left + watering_width
    plant_center = (plant_left + plant_right) / 2.0

    if shading_config.mode != MODE_FREE_FIELD:
        distance = shading_config.distance if shading_config.distance is not None else bay_width
        pv_center = (_GROUND_SHADING_N_ROWS - 1) * distance / 2.0
        shift = plant_center - pv_center
    else:
        shift = plant_center

    ranges: dict[str, tuple[float, float]] = {}
    for i in range(n_pv_segments):
        ranges[f"LeftTopSegment_{i}"] = (i * dx - shift, (i + 1) * dx - shift)
    ranges["PlantTopLeftSegment"] = (plant_left - shift, watering_left - shift)
    ranges["PlantTopRightSegment"] = (watering_right - shift, plant_right - shift)
    for i in range(n_pv_segments):
        x0 = plant_right + i * dx
        ranges[f"RightTopSegment_{i}"] = (x0 - shift, x0 + dx - shift)
    return ranges


class Simulation:
    @classmethod
    def build(
        cls,
        setup: FieldSetup,
        *,
        anchor_sensors: Sequence[AnchorSensor] = (),
        rel_sat_name: str = "relative saturation",
        name: str = "fieldsim",
    ) -> "Simulation":
        """Assemble engine, chain, assimilator and planner from a configured ``FieldSetup``."""
        engine = SoilEngine.build(setup.soil, rel_sat_name=rel_sat_name)

        probes = list(engine.probes(setup.soil.configs.get_member("probes", defaults={}, ensure_exists=True)))
        anchor_cfg = parse_anchor_config(setup.soil.configs.get_member("anchor", defaults={}))

        segment_ranges = _resolve_segment_ranges(engine.mesh_config, setup.shading, setup.field.bay_width)
        setup.shading.derive(bay_width=setup.field.bay_width, segment_ranges=segment_ranges)
        shading = ShadingModel(setup.shading)

        chain = WeatherChain(
            setup,
            shading,
            ETModel(),
            setup.plots,
            top_segment_names=engine.top_segment_names,
            segment_face_length=engine.segment_face_length,
        )

        assimilator = Assimilator(anchor_cfg, engine)

        planner = None
        if setup.planner is not None:
            planner = IrrigationPlanner(
                setup.planner,
                engine,
                probes=probes,
                drip=setup.planner.drip if setup.planner.drip is not None else setup.soil.drip,
                total_drip_line_length_m=setup.soil.total_drip_line_length_m,
                name=name,
            )

        simulation = cls(setup, engine, chain, assimilator, planner, probes=probes)
        simulation.add_sensors(anchor_sensors)
        return simulation

    def __init__(
        self,
        setup: FieldSetup,
        engine: SoilEngine,
        chain: WeatherChain,
        assimilator: Assimilator,
        planner: Optional[IrrigationPlanner] = None,
        *,
        probes: Sequence[Any] = (),
    ) -> None:
        self.setup = setup
        self.engine = engine
        self.chain = chain
        self.assimilator = assimilator
        self.planner = planner
        self.probes = list(probes)
        self.state: Optional[SoilState] = None
        self._last_chain: Optional[ChainResult] = None
        self._last_step: Optional[StepResult] = None
        self._last_plan: Optional[Plan] = None
        self._creation_warned = False

    def add_sensors(self, sensors: Sequence[AnchorSensor]) -> None:
        """Hand discovered tensiometers to the assimilator; sample each as a probe
        when sensor probes are on or anchoring is live."""
        self.assimilator.set_sensors(sensors)
        if self.setup.soil.discover_sensor_probes or self.assimilator.enabled:
            self.probes += [self.engine.probe_from_sensor(sensor) for sensor in sensors]

    def resume(self, state: SoilState) -> None:
        """Continue from ``state``; ``ValueError`` when the engine cannot load it, the
        session then keeps its current state."""
        self.engine.load(state)
        self.state = state

    def run(
        self,
        weather: pd.DataFrame,
        irrigation_lpm: pd.Series,
        tension_history: Optional[Mapping[str, pd.Series]] = None,
        cancel: Cancel = None,
        *,
        extra_diagnostics: Optional[Mapping[str, float]] = None,
    ) -> tuple[Sequence[StepResult], ChainResult]:
        """Advance over one weather chunk: chain once, then per row advance,
        assimilate, sample probes. Stops early on cancel; committed rows stay
        committed. Without a state yet, the first row is the cold start.
        ``extra_diagnostics`` is merged into every ``StepResult.diagnostics``.
        """
        frontier = self.state.at if self.state is not None else None
        first_dt_s = self.engine.cold_start_s if self.state is None else 0.0
        forcing, chain = self.chain.forcing_series(weather, irrigation_lpm, frontier=frontier, first_dt_s=first_dt_s)
        self._last_chain = chain
        if tension_history and self.assimilator.enabled:
            self.assimilator.ingest(tension_history)
        extra = dict(extra_diagnostics or {})
        results: list[StepResult] = []

        if self.state is None and forcing:
            cold = forcing[0]
            initial = self.engine.initial_state(cold.at)
            if cold.dt_s > 0:
                logger.info("cold start spin-up: %.0fs with weather at %s", cold.dt_s, cold.at)
                result = self._advance(initial, cold, extra, cancel)
                if result is None:
                    logger.info("cold start cancelled at %s", cold.at)
                    return results, chain
                results.append(result)
            else:
                self.state = initial

        for step in forcing:
            if step.at <= self.state.at:
                continue
            result = self._advance(self.state, step, extra, cancel)
            if result is None:
                logger.info("advance cancelled at %s, %d rows committed", step.at, len(results))
                break
            results.append(result)
        if results:
            self._last_step = results[-1]
        return results, chain

    def _advance(
        self, state: SoilState, step: Forcing, extra: Mapping[str, float], cancel: Cancel
    ) -> Optional[StepResult]:
        """One committed row from ``state``; ``None`` when the walk was cancelled. With
        anchoring live, ``anchor`` carries the summed innovations of this row's update."""
        result = self.engine.advance(state, step, cancel=cancel)
        if result.cancelled:
            return None
        state = result.state
        diagnostics = {**result.diagnostics, **extra}
        if self.assimilator.enabled:
            previous = self.assimilator.last_result
            state = self.assimilator.update(state, step.end)
            anchored = self.assimilator.last_result
            diagnostics["anchor"] = float(sum(anchored.innovations.values())) if anchored is not previous else 0.0
        tension = {p.channel_id: self.engine.tension_at(state, p) for p in self.probes}
        self.state = state
        return replace(result, state=state, probe_tension=tension, diagnostics=diagnostics)

    def plan(self, forecast: pd.DataFrame) -> Optional[Plan]:
        """Roll the planner over a forecast horizon from the current state."""
        if self.planner is None or self.state is None or forecast.empty:
            return None
        weather, seg_et = self.chain.horizon_inputs(forecast)
        if weather.empty:
            return None
        run_timestamp = pd.Timestamp(self.state.at)
        plan = self.planner.plan(
            self.state,
            weather,
            seg_et,
            weather.index[0],
            weather.index[-1],
            run_timestamp=run_timestamp,
            weather_creation=self._weather_creation(forecast, run_timestamp),
        )
        self._last_plan = plan
        return plan

    def _weather_creation(self, forecast: pd.DataFrame, run_timestamp: pd.Timestamp) -> pd.Timestamp:
        """The forecast's latest issue time; the run time when the frame carries none."""
        creation = pd.NaT
        if FORECAST_CREATION_KEY in forecast.columns:
            creation = pd.to_datetime(forecast[FORECAST_CREATION_KEY], utc=True).max()
        if pd.isna(creation):
            if not self._creation_warned:
                logger.warning(
                    "weather forecast issue time unavailable (no valid '%s' in the forecast yet); "
                    "using the run time %s as weather_creation",
                    FORECAST_CREATION_KEY,
                    run_timestamp,
                )
                self._creation_warned = True
            return run_timestamp
        return creation

    def snapshot(self) -> Snapshot:
        return Snapshot(
            state=self.state,
            last_chain=self._last_chain,
            last_step=self._last_step,
            last_plan=self._last_plan,
        )
