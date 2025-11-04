# -*- coding: utf-8 -*-
"""
sparcs.devices
~~~~~~~~~~~~~~


"""

import importlib

DEVICES = ["charge_big"]

for import_device in DEVICES:
    try:
        importlib.import_module(f".{import_device}", "sparcs.devices")

    except ModuleNotFoundError:
        # TODO: Implement meaningful logging here, at least log warning
        pass

del importlib
