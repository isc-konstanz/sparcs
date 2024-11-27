# -*- coding: utf-8 -*-
"""
penguin.components.irrigation.series
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""

from __future__ import annotations

from lori import Configurations
from lori.components import Component


# noinspection SpellCheckingInspection
class IrrigationSeries(Component):
    def configure(self, configs: Configurations) -> None:
        super().configure(configs)

        # noinspection PyShadowingBuiltins
        def _add_channel(key: str, name: str, min: float, max: float, **kwargs) -> None:
            self.data.add(key, name=name, connector="random", type=float, min=min, max=max, **kwargs)

        _add_channel("temperature", "Temperature [°C]", 23, 36)
        _add_channel("humidity", "Humidity [%]", 40, 100)
