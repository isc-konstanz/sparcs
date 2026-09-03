# -*- coding: utf-8 -*-
"""
sparcs.connectors
~~~~~~~~~~~~~~~~~


"""

import importlib

CONNECTORS = [
    "openems",
]

for import_connector in CONNECTORS:
    try:
        importlib.import_module(f".{import_connector}", "sparcs.connectors")

    except ModuleNotFoundError:
        # TODO: Implement meaningful logging here
        pass

del importlib
