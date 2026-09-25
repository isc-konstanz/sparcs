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
from typing import Mapping, Optional, Sequence

import pandas as pd

from .anchor import AnchorSensor
from .assimilator import Assimilator, parse_anchor_config
from .chain import WeatherChain
from .config import FieldSetup
from .engine import Cancel, SoilEngine
from .evapotranspiration import ETModel
from .planner import IrrigationPlanner
from .shading import _N_ROWS, MODE_FREE_FIELD, ShadingModel
from .state import ChainResult, Forcing, Plan, Snapshot, SoilState, StepResult

logger = logging.getLogger(__name__)


def _resolve_segment_ranges(mesh, shading_config, bay_width: float) -> Optional[dict[str, tuple[float, float]]]:
    """Soil-mesh top-segment x-ranges in pvfactors coordinates; ``None`` without a mesh."""
    if mesh is None:
        return None
    dx = mesh.dx
    plant_width = mesh.plant_width
    watering_width = mesh.watering_width
    n_pv_segments = int((mesh.width - plant_width) / (2 * dx))

    plant_left = n_pv_segments * dx
    plant_right = plant_left + plant_width
    watering_left = plant_left + (plant_width - watering_width) / 2
    watering_right = watering_left + watering_width
    plant_center = (plant_left + plant_right) / 2.0

    if shading_config.mode != MODE_FREE_FIELD:
        distance = shading_config.distance if shading_config.distance is not None else bay_width
        pv_center = (_N_ROWS - 1) * distance / 2.0
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
        """Assemble engine, chain, assimilator and planner from a configured ``FieldSetup``.

        Sets ``setup.soil.probe_specs`` in place.
        """
        engine = SoilEngine.build(setup.soil, setup.model, rel_sat_name=rel_sat_name)

        probes = (
            list(engine.probes(setup.soil.configs.get_member("probes", defaults={}, ensure_exists=True)))
            if (setup.soil.configs is not None)
            else []
        )
        anchor_cfg = parse_anchor_config(setup.soil.anchor)
        if setup.soil.discover_sensor_probes or anchor_cfg.enabled:
            probes += [engine.probe_from_sensor(sensor) for sensor in anchor_sensors]
        setup.soil.probe_specs = probes

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
        assimilator.set_sensors(anchor_sensors)

        planner = None
        if setup.planner is not None:
            planner = IrrigationPlanner(
                setup.planner,
                engine,
                probes=probes,
                drip=setup.planner_drip,
                total_drip_line_length_m=setup.soil.total_drip_line_length_m,
                name=name,
            )

        return cls(setup, engine, chain, assimilator, planner)

    def __init__(
        self,
        setup: FieldSetup,
        engine: SoilEngine,
        chain: WeatherChain,
        assimilator: Assimilator,
        planner: Optional[IrrigationPlanner] = None,
    ) -> None:
        self.setup = setup
        self.engine = engine
        self.chain = chain
        self.assimilator = assimilator
        self.planner = planner
        self.state: Optional[SoilState] = None
        self._last_chain: Optional[ChainResult] = None
        self._last_step: Optional[StepResult] = None
        self._last_plan: Optional[Plan] = None

    def resume(self, state: SoilState) -> None:
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
            self.state = self.engine.initial_state(cold.at)
            if cold.dt_s > 0:
                logger.info("cold start spin-up: %.0fs with weather at %s", cold.dt_s, cold.at)
                result = self._advance(cold, extra, cancel)
                if result is None:
                    logger.info("cold start cancelled at %s", cold.at)
                    return results, chain
                results.append(result)

        for step in forcing:
            if step.at <= self.state.at:
                continue
            result = self._advance(step, extra, cancel)
            if result is None:
                logger.info("advance cancelled at %s, %d rows committed", step.at, len(results))
                break
            results.append(result)
        if results:
            self._last_step = results[-1]
        return results, chain

    def _advance(self, step: Forcing, extra: Mapping[str, float], cancel: Cancel) -> Optional[StepResult]:
        """One committed row; ``None`` when the walk was cancelled."""
        result = self.engine.advance(self.state, step, cancel=cancel)
        if result.cancelled:
            return None
        state = result.state
        if self.assimilator.enabled:
            state = self.assimilator.update(state, step.end)
        tension = {p.channel_id: self.engine.tension_at(state, p) for p in self.setup.soil.probe_specs}
        self.state = state
        return replace(
            result,
            state=state,
            probe_tension=tension,
            diagnostics={**result.diagnostics, **extra},
        )

    def plan(self, forecast: pd.DataFrame) -> Optional[Plan]:
        """Roll the planner over a forecast horizon from the current state."""
        if self.planner is None or self.state is None or forecast.empty:
            return None
        weather, seg_et = self.chain.horizon_inputs(forecast)
        if weather.empty:
            return None
        plan = self.planner.plan(
            self.state,
            weather,
            seg_et,
            weather.index[0],
            weather.index[-1],
            run_timestamp=pd.Timestamp(self.state.at),
        )
        self._last_plan = plan
        return plan

    def snapshot(self) -> Snapshot:
        return Snapshot(
            state=self.state,
            last_chain=self._last_chain,
            last_step=self._last_step,
            last_plan=self._last_plan,
        )
