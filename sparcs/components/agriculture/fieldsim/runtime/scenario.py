# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.fieldsim.runtime.scenario
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Offline driver: the production ``FieldRunner`` over ``FrameInputs`` and a
``Recorder``, with a synthetic clock stepping ``run_tick`` from ``start`` to
``end``. The entry point for parameter sweeps, model-vs-tensiometer benches
and dt-sensitivity runs.
"""

from __future__ import annotations

import datetime as dt
from typing import Mapping, Optional

import pandas as pd

from ..core.config import FieldSetup
from ..core.simulation import Simulation
from ..core.state import SoilState
from .memory import FrameInputs, Recorder
from .ports import InputKey
from .runner import FieldRunner


class ScenarioRunner:
    def __init__(self, setup: FieldSetup, simulation: Simulation) -> None:
        self.setup = setup
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
        runner = FieldRunner(self.setup, self.simulation, inputs, recorder)
        step = step or self.setup.field.interval_td
        now = start
        while now <= end:
            runner.run_tick(now)
            now += step
        return recorder
