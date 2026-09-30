# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The soil-simulation chain of an agricultural field, laid out as
sparcs_agri_sim_doc/uml/target_architecture_4.mmd describes.

Three layers, one dependency direction, lories -> application -> core:

    core/        config, state, engine, shading, evapotranspiration, plots,
                 chain, assimilator, planner, simulation (the session and
                 scenario API). Config sections are lories Configurators
                 with Parameter() declarations; no channel I/O, no threads.
    runtime/     ports (Inputs / Outputs), runner, scheduler, memory
                 (FrameInputs / Recorder), scenario (ScenarioRunner)
    components   FieldSimulation, the four ChannelNamespace subclasses,
                 ChannelInputs, ChannelOutputs: the lories layer.

``AgriculturalField`` builds ``components.FieldSimulation`` from its
``field_simulation`` member; no component type is registered.
"""

from . import core  # noqa: F401
from . import runtime  # noqa: F401
from . import components  # noqa: F401
