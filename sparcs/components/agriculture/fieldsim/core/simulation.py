# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.fieldsim.core.simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The simulation session: the scenario and test API. Owns the current state
and the latest results, composes engine, chain, assimilator and planner, and
runs the per-chunk sequence. No I/O, no clock, no gating; those belong to
``runtime.runner``. Notebooks, tuning campaigns and dt benches call ``run``
directly with frames.
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
from .state import ChainResult, Plan, Snapshot, SoilState, StepResult

logger = logging.getLogger(__name__)


def _resolve_segment_ranges(mesh, shading_config, bay_width: float) -> Optional[dict[str, tuple[float, float]]]:
    """Soil-mesh top-segment x-ranges in pvfactors coordinates (port of the
    live ``GroundShading._resolve_segment_ranges``): aligns the PV array
    centre over the plant centre of the soil mesh, then maps each mesh
    segment. ``None`` when there is no mesh. In ``free_field`` mode (no PV
    array) the shift is the plant centre alone, matching the live no-setups
    branch."""
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
        """Assemble engine, chain, assimilator and planner from a configured
        ``FieldSetup`` plus the sensors discovered at activation. Lories-free:
        every argument is a plain config/value object, never a Component.

        Mutates ``setup.soil.probe_specs`` in place (``SoilConfig`` is not
        frozen) so every later reader of the setup sees the resolved probes.
        """
        engine = SoilEngine.build(setup.soil, setup.model, rel_sat_name=rel_sat_name)

        probes = (
            list(engine.probes(setup.soil.raw.get_member("probes", defaults={}, ensure_exists=True)))
            if (setup.soil.raw is not None)
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
    ) -> tuple[Sequence[StepResult], ChainResult]:
        """Advance over one weather chunk: chain once, then per row advance,
        assimilate, sample probes. Stops early on cancel; committed rows stay
        committed. Returns the completed steps and the chain outputs."""
        forcing, chain = self.chain.forcing_series(weather, irrigation_lpm)
        if tension_history and self.assimilator.enabled:
            self.assimilator.ingest(tension_history)
        probes = self.setup.soil.probe_specs
        results: list[StepResult] = []
        for step in forcing:
            if self.state is None:
                self.state = self.engine.initial_state(step.start)
            result = self.engine.advance(self.state, step, cancel=cancel)
            if result.cancelled:
                logger.info("advance cancelled at %s, %d rows committed", step.start, len(results))
                break
            state = result.state
            if self.assimilator.enabled:
                state = self.assimilator.update(state, step.end)
            # ProbeSpec's id attribute is `channel_id`; `key` stays as a fallback
            # for lightweight test doubles that predate this attribute name.
            tension = {
                getattr(p, "channel_id", None) or getattr(p, "key", None): self.engine.tension_at(state, p)
                for p in probes
            }
            result = replace(result, state=state, probe_tension=tension)
            self.state = state
            results.append(result)
        self._last_chain = chain
        if results:
            self._last_step = results[-1]
        return results, chain

    def plan(self, forecast: pd.DataFrame) -> Optional[Plan]:
        """Roll the planner over a forecast horizon from the current state.
        Returns None when there is no planner, no state yet, or the chain
        cannot yet produce the per-segment ET the planner needs
        (``WeatherChain.horizon_inputs``; not yet landed -- see the chain
        unit)."""
        if self.planner is None or self.state is None or forecast.empty:
            return None
        if not hasattr(self.chain, "horizon_inputs"):
            logger.warning("plan skipped: chain has no horizon_inputs yet")
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
