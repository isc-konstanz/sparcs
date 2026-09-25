# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.fieldsim.core
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pure layer: config (lories Parameter declarations), state, engine, shading,
evapotranspiration, plots, chain, assimilator, planner, and the simulation
session that composes them. Imports lories' config machinery only.
No channels, no components, no threads, no I/O.
"""

from . import (  # noqa: F401
    config,
    state,
    engine,
    shading,
    evapotranspiration,
    plots,
    chain,
    anchor,
    assimilator,
    planner,
)
from . import simulation  # noqa: F401
