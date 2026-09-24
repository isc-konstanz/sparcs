# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.fieldsim
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Skeleton of the target architecture for the soil-simulation chain
(sparcs_agri_sim_doc/uml/target_architecture_3.mmd). Placeholder bodies only.

Three layers, one dependency direction, lories -> application -> core:

    core/        config, state, engine, chain, assimilator, planner
                 pure Python + FiPy. No lories, no channels, no threads, no I/O.
    base/        io (the FieldIO protocol), runner, scheduler
    components   FieldSimulation, SoilSimulation, SoilPredictor, ChannelIO:
                 the lories layer, the only module that imports lories.

This package is NOT wired into ``sparcs.components.agriculture`` and registers
no component type. It exists to be read next to the live ``simulation``
package, not to run.
"""

from . import core  # noqa: F401
from . import base  # noqa: F401
from . import components  # noqa: F401
