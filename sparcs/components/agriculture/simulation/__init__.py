# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The soil-simulation chain of an agricultural field in three layers: ``components`` -> ``runtime`` -> ``core``.
``AgriculturalField`` builds ``components.FieldSimulation`` from its ``field_simulation`` member; no type is registered.
"""

from . import core  # noqa: F401
from . import runtime  # noqa: F401
from . import components  # noqa: F401
