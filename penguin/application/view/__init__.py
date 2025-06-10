# -*- coding: utf-8 -*-
"""
penguin.application.view
~~~~~~~~~~~~~~~~~~~~~~~~


"""

from . import agriculture  # noqa: F401
from .agriculture import (  # noqa: F401
    AgriculturalAreaPage as AgriculturePage,
    AgriculturalFieldPage,
)

from . import storage  # noqa: F401
from .storage import WaterStoragePage  # noqa: F401

from . import weather  # noqa: F401
from .weather import WeatherPage  # noqa: F401
