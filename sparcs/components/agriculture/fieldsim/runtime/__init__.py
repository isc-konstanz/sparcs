# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.fieldsim.runtime
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The code that runs the core, live or offline. ``ports`` holds the two
protocols (``Inputs`` reads, ``Outputs`` writes), ``runner`` the live tick
policy, ``scheduler`` the thread, ``memory`` the in-memory adapters and
``scenario`` the offline driver that reuses the runner with a synthetic
clock. Depends on core only.
"""

from . import ports, runner, scheduler, memory, scenario  # noqa: F401
