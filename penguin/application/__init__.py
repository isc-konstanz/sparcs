# -*- coding: utf-8 -*-
"""
penguin.application
~~~~~~~~~~~~~~~~~~~


"""

try:
    from .view import (  # noqa: F401
        IrrigationPage,
        WeatherPage,
    )
except ModuleNotFoundError:
    pass

from lori import Application  # noqa: F401
