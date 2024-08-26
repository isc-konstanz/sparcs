# -*- coding: utf-8 -*-
"""
    penguin.components.irrigation
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""

from .array import IrrigationArray  # noqa: F401
from .system import IrrigationSystem  # noqa: F401

try:
    from .view import (  # noqa: F401
        IrrigationPage,
        IrrigationGroup,
    )
except ImportError:
    pass
