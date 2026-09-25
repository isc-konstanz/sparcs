# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.fieldsim.runtime.scenario
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Offline driver: the same ``FieldRunner`` as production, fed by
``FrameInputs`` and recorded by ``Recorder``, with a synthetic clock stepping
``run_tick`` from ``start`` to ``end``. Every scenario therefore exercises
frontier handling, catch-up chunking and the planner gate, not just the
core. This is the entry point for parameter sweeps, model-vs-tensiometer
benches and dt-sensitivity runs.
"""

from __future__ import annotations

import datetime as dt
from typing import Mapping, Optional

import pandas as pd

from ..core.config import FieldConfig
from ..core.simulation import Simulation
from ..core.state import SoilState
from .memory import FrameInputs, Recorder
from .ports import InputKey
from .runner import FieldRunner


class ScenarioRunner:
    def __init__(self, config: FieldConfig, simulation: Simulation) -> None:
        self.config = config
        self.simulation = simulation

    def run(
        self,
        frames: Mapping[InputKey, pd.DataFrame],
        start: dt.datetime,
        end: dt.datetime,
        step: Optional[dt.timedelta] = None,
        initial_state: Optional[SoilState] = None,
    ) -> Recorder:
        """Tick from ``start`` to ``end`` every ``step`` (default: the
        configured interval) and return everything the runner wrote."""
        inputs = FrameInputs(frames, state=initial_state)
        recorder = Recorder()
        runner = FieldRunner(self.config, self.simulation, inputs, recorder)
        step = step or self.config.interval
        now = start
        while now <= end:
            runner.run_tick(now)
            now += step
        return recorder
