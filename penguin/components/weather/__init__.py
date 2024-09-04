# -*- coding: utf-8 -*-
"""
penguin.components.weather
~~~~~~~~~~~~~~~~~~~~~~~~~~


"""

from .validation import ValidatedWeather as Weather  # noqa: F401

from .file import (  # noqa: F401
    EPWWeather,
    TMYWeather,
)

try:
    from .view import (  # noqa: F401
        WeatherPage,
        WeatherGroup,
    )
except ImportError:
    pass
