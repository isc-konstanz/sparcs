# -*- coding: utf-8 -*-
"""
sparcs.application
~~~~~~~~~~~~~~~~~~


"""

try:
    from .view import (  # noqa: F401
        AgriculturePage,
    )
except ModuleNotFoundError:
    pass

from lories import Application  # noqa: F401
