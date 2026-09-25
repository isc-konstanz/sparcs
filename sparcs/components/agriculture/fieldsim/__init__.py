# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.fieldsim
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Skeleton of the target architecture for the soil-simulation chain
(sparcs_agri_sim_doc/uml/target_architecture_4.mmd). Placeholder bodies only.

Three layers, one dependency direction, lories -> application -> core:

    core/        config, state, engine, shading, evapotranspiration, plots,
                 chain, assimilator, planner, simulation (the session and
                 scenario API). Config sections are lories Configurators
                 with Parameter() declarations; no channels, no components,
                 no threads, no I/O.
    runtime/     ports (Inputs / Outputs), runner, scheduler, memory
                 (FrameInputs / Recorder), scenario (ScenarioRunner)
    components   FieldSimulation, the four ChannelNamespace subclasses,
                 ChannelInputs, ChannelOutputs: the lories layer.

This package is NOT wired into ``sparcs.components.agriculture`` and registers
no component type. It exists to be read next to the live ``simulation``
package, not to run.
"""

from . import core  # noqa: F401
from . import runtime  # noqa: F401
from . import components  # noqa: F401
