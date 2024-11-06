# -*- coding: utf-8 -*-
"""
penguin.application
~~~~~~~~~~~~~~~~~~~


"""

try:
    from .view import (  # noqa: F401
        IrrigationPage,
        IrrigationGroup,
        WeatherPage,
    )
except ImportError:
    pass
