# -*- coding: utf-8 -*-
"""
    penguin.connectors
    ~~~~~~~~~~~~~~~~~~


"""

from .papendorf import PapendorfParser

from loris.connectors import registry

registry.register(PapendorfParser, PapendorfParser.TYPE)
del registry
