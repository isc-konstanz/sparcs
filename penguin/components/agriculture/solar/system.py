# -*- coding: utf-8 -*-
"""
penguin.components.agriculture.solar.system
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""

from __future__ import annotations

from lori.components import register_component_type
from penguin.components.agriculture import AgriculturalArea
from penguin.components.solar import SolarSystem


# noinspection SpellCheckingInspection
@register_component_type("agripv", "agri_pv", "agrisolar", "agri_solar", "agrivoltaics")
class AgriSolarSystem(AgriculturalArea, SolarSystem):
    pass
