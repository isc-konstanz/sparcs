# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.solar.system
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""

from __future__ import annotations

from lories.components import register_component_type
from sparcs.components.agriculture import AgriculturalArea
from sparcs.components.solar import SolarSystem


# noinspection SpellCheckingInspection
@register_component_type("agripv", "agri_pv", "agrisolar", "agri_solar", "agrivoltaics")
class AgriSolarSystem(AgriculturalArea, SolarSystem):
    pass
