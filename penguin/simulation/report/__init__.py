# -*- coding: utf-8 -*-
"""
penguin.simulation.report
~~~~~~~~~~~~~~~~~~~~~~~~~


"""

import importlib

for import_report in ["yield"]:
    try:
        importlib.import_module(f".{import_report}", "penguin.simulation.report")

    except ModuleNotFoundError:
        # TODO: Implement meaningful logging here
        pass

del importlib
