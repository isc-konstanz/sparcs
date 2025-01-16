# -*- coding: utf-8 -*-
"""
penguin.application.view
~~~~~~~~~~~~~~~~~~~~~~~~


"""

from . import irrigation  # noqa: F401
from .irrigation import (  # noqa: F401
    IrrigationSystemPage as IrrigationPage,
    IrrigationSeriesPage,
)

from . import weather  # noqa: F401
from .weather import WeatherPage  # noqa: F401
