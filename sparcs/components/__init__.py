# -*- coding: utf-8 -*-
"""
sparcs.components
~~~~~~~~~~~~~~~~~


"""

from . import weather  # noqa: F401

from . import storage  # noqa: F401
from .storage import (  # noqa: F401
    ElectricalEnergyStorage,
    ThermalEnergyStorage,
)

from . import solar  # noqa: F401
from .solar import (  # noqa: F401
    SolarArray,
    SolarInverter,
    SolarSystem,
)

from . import agriculture  # noqa: F401
from .agriculture import (  # noqa: F401
    AgriculturalArea,
    AgriculturalField,
    Irrigation,
)

import importlib

CONNECTORS = [
    "openems",
]

for import_connector in CONNECTORS:
    try:
        importlib.import_module(f".{import_connector}", "sparcs.components")

    except ModuleNotFoundError:
        # TODO: Implement meaningful logging here
        pass

del importlib
