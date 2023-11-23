# -*- coding: utf-8 -*-
"""
    pvsys.report
    ~~~~~~~~~~~~


"""
from __future__ import annotations
import logging

from corsys import Configurations, Configurable, System

logger = logging.getLogger(__name__)


class Report(Configurable):

    @classmethod
    def read(cls, system, conf_file: str = 'report.cfg') -> Report:
        return cls(Configurations.from_configs(system.configs, conf_file))

    def __init__(self, system: System, name: str = 'Report') -> None:
        super().__init__(system.configs)
        self.system = system
        self.name = name

    def __configure__(self, configs: Configurations) -> None:
        super().__configure__(configs)

    def __call__(self, *args, **kwargs) -> None:
        pass
