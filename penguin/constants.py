# -*- coding: utf-8 -*-
"""
penguin.constants
~~~~~~~~~~~~~~~~~


"""
# fmt: off

from lori.components.weather.constants import WEATHER
from penguin.components.constants import ENERGY as COMPONENT_ENERGY
from penguin.components.constants import POWER as COMPONENT_POWER
from penguin.components.constants import STATES
from penguin.system import System

SYSTEM_POWER = {
    System.POWER_EL:     "Total Electrical Power [W]",
    System.POWER_EL_IMP: "Imported Electrical Power [W]",
    System.POWER_EL_EXP: "Exported Electrical Power [W]",
    System.POWER_TH:     "Total Thermal Power [W]",
    System.POWER_TH_HT:  "Heating Water Thermal Power [W]",
    System.POWER_TH_DOM: "Domestic Water Thermal Power [W]",
}

SYSTEM_ENERGY = {
    System.ENERGY_EL:     "Total Electrical Energy [kWh]",
    System.ENERGY_EL_IMP: "Imported Electrical Energy [kWh]",
    System.ENERGY_EL_EXP: "Exported Electrical Energy [kWh]",
    System.ENERGY_TH:     "Total Thermal Energy [kWh]",
    System.ENERGY_TH_HT:  "Heating Water Thermal Energy [kWh]",
    System.ENERGY_TH_DOM: "Domestic Water Thermal Energy [kWh]",
}

SYSTEM = {
    **SYSTEM_POWER,
    **SYSTEM_ENERGY,
}

POWER = {
    **SYSTEM_POWER,
    **COMPONENT_POWER,
}

ENERGY = {
    **SYSTEM_ENERGY,
    **COMPONENT_ENERGY,
}

SOLAR_ANGLES = {
    "solar_elevation": "Solar Elevation [°]",
    "solar_zenith":    "Solar Zenith [°]",
    "solar_azimuth":   "Solar Azimuth [°]",
}

TIME = {
    "hour":        "Hour",
    "day_of_week": "Day of the Week",
    "day_of_year": "Day of the Year",
    "month":       "Month",
    "year":        "Year",
}

COLUMNS = {
    **STATES,
    **POWER,
    **ENERGY,
    **WEATHER,
    **SOLAR_ANGLES,
    **TIME,
}
# fmt: on
