# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.fieldsim.core
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Config sections (lories Configurators), state, engine, shading,
evapotranspiration, plots, chain, assimilator, planner, and the simulation
session that composes them. Nothing here reads or writes channels or starts a
thread; the one file it writes is the Gmsh mesh, and ``pde`` also holds
``SoilBase``, the component base the ``simulation`` package builds on.
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
