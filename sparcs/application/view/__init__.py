# -*- coding: utf-8 -*-
"""
sparcs.application.view
~~~~~~~~~~~~~~~~~~~~~~~


"""

from . import agriculture  # noqa: F401
from .agriculture import (  # noqa: F401
    AgriculturalAreaPage as AgriculturePage,
    AgriculturalFieldPage,
)

from . import storage  # noqa: F401
from .storage import WaterStoragePage  # noqa: F401
