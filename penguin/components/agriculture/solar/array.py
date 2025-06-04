# -*- coding: utf-8 -*-
"""
penguin.components.agriculture.solar.array
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""

from __future__ import annotations

from penguin.components.agriculture import AgriculturalField
from penguin.components.solar import SolarArray


class AgriSolarArray(AgriculturalField, SolarArray):
    pass
