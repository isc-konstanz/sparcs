# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.fieldsim
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Skeleton of the target architecture for the soil-simulation chain
(sparcs_agri_sim_doc/uml/target_architecture_3.mmd). Placeholder bodies only.

Three layers, one dependency direction, lories -> application -> core:

    core         config, state, engine, chain, assimilator, planner
                 pure Python + FiPy. No lories, no channels, no threads, no I/O.
    application  io (the FieldIO protocol), runner, scheduler
    lories       components (FieldSimulation, SoilSimulation, SoilPredictor,
                 ChannelIO), the only modules that import lories.

This package is NOT wired into ``sparcs.components.agriculture`` and registers
no component type. It exists to be read next to the live ``simulation``
package, not to run.
"""

from . import config, state, engine, chain, assimilator, planner  # noqa: F401  core
from . import io, runner, scheduler  # noqa: F401  application
from . import components  # noqa: F401  lories
