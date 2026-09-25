# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.fieldsim.runtime
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The code that runs the core, live or offline. ``ports`` holds the two
protocols (``Inputs`` reads, ``Outputs`` writes), ``runner`` the live tick
policy, ``scheduler`` the ``Ticker`` around ``lories.scheduler.TickScheduler``,
``memory`` the in-memory adapters and ``scenario`` the offline driver that
reuses the runner with a synthetic clock. Depends on core and lories only.
"""

from . import ports, runner, scheduler, memory, scenario  # noqa: F401
from .scheduler import Ticker  # noqa: F401
