# -*- coding: utf-8 -*-
"""
sparcs.application
~~~~~~~~~~~~~~~~~~


"""

try:
    from .view import (  # noqa: F401
        AgriculturePage,
        WeatherPage,
    )
except ModuleNotFoundError:
    pass

from lories import Application  # noqa: F401
