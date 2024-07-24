# -*- coding: utf-8 -*-
"""
    penguin.components
    ~~~~~~~~~~~~~~~~~~


"""

from . import weather  # noqa: F401

from .current import (  # noqa: F401
    DirectCurrent,
    AlternatingCurrent
)
from .storage import (  # noqa: F401
    ElectricalEnergyStorage,
    ThermalEnergyStorage
)
from .vehicle import ElectricVehicle  # noqa: F401

from . import pv  # noqa: F401
from .pv import PVArray, PVSystem  # noqa: F401

from loris.components import Weather, registry

registry.register(Weather, Weather.TYPE, factory=weather.load, replace=True)

registry.register(PVArray, PVArray.TYPE)
registry.register(PVSystem, PVSystem.TYPE, *PVSystem.ALIAS)

registry.register(ElectricVehicle, ElectricVehicle.TYPE)
registry.register(ElectricalEnergyStorage, ElectricalEnergyStorage.TYPE)

registry.register(ThermalEnergyStorage, ThermalEnergyStorage.TYPE)
del registry
