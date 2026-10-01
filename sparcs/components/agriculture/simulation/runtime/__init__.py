# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.simulation.runtime
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Runs the core in real time or offline: ``ports`` the protocols, ``runner`` the tick policy, ``scheduler`` the
``Ticker``, ``memory`` in-memory adapters, ``scenario`` the runner on a synthetic clock. Uses only core and lories.
"""

from . import ports, runner, scheduler, memory, scenario  # noqa: F401
from .scheduler import Ticker  # noqa: F401
