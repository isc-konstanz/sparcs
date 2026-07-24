# -*- coding: utf-8 -*-
"""
sparcs.components.vehicle.evse
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""

from lories.components import Component
from lories.core import Constant
from lories.typing import Configurations


class EVSE(Component):
    POWER_MAX = Constant(float, "evse_power_max", "EVSE Power Max", "kW")
    POWER_MIN = Constant(float, "evse_power_min", "EVSE Power Min", "kW")

    STATE = Constant(float, "evse_state", "EVSE State", "")
    LIMIT = Constant(float, "evse_limit", "EVSE Power Limit", "kW")

    TARGET_POWER = Constant(float, "target_power", "EVSE Target Power", "kW")

    def configure(self, configs: Configurations) -> None:
        super().configure(configs)

    def activate(self) -> None:
        pass
