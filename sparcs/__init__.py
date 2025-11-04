# -*- coding: utf-8 -*-
"""
sparcs
~~~~~~

This repository provides a set of python functions and scripts to calculate the
energy generation and provide further utility for photovoltaic systems.

"""

from . import _version

__version__ = _version.get_versions().get("version")
del _version

from .location import Location  # noqa: F401

from . import components  # noqa: F401
from .components import (  # noqa: F401
    SolarArray,
    SolarSystem,
    AgriculturalArea,
    AgriculturalField,
    ElectricalEnergyStorage,
    ThermalEnergyStorage,
)

from . import devices # noqa: F401

from . import system  # noqa: F401
from .system import System  # noqa: F401

from . import application  # noqa: F401
from .application import Application


def load(name: str = "SPARCS", factory=System, **kwargs) -> Application:
    return Application.load(name, factory=factory, **kwargs)
