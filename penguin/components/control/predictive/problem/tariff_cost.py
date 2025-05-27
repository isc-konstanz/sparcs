
from typing import Iterable, Union

import numpy as np
from casadi import Opti, DM, MX
from penguin.components.optimization.predictive_optimization import PredictiveOptimization

from lori.components import Component, register_component_type
from lori.core import Configurations, ResourceException
from penguin.components import ElectricalEnergyStorage


@register_component_type("tariff_opti", "tariff_mpc", "tariff_optimization")
class TariffCostOptimization(PredictiveOptimization):
    """
    Base class for tariff cost optimization problems.
    """

    tariff_pos: DM
    tariff_neg: DM
    grid_expected: DM

    grid: MX

    def configure(self, configs: Configurations) -> None:

        super().configure(configs)

        # tariff_config = configs.get_section("tariff", defaults={})
        # if not tariff_config:
        #     raise ResourceException("Missing tariff configuration")
        # opti parameters
        steps = len(self.delta_times)
        self.tariff_pos = self.opti.parameter(steps)
        self.tariff_neg = self.opti.parameter(steps)
        self.grid_expected = self.opti.parameter(steps)

        # opti variables
        self.grid = self.opti.variable(steps)
        # self.grid_pos = opti.variable(self.steps)
        # self.grid_neg = opti.variable(self.steps)

        # self.grid_pos_max = opti.variable(1)



    def activate(self) -> None:
        super().activate()

    def get_required_components(self) -> Union:
        return [ElectricalEnergyStorage]

    def cost_function(self):
        # ----------------------------------------------------------
        def function_atan(x, center, width=1, left=0, right=1):
            # x_threshold = np.tan(0.9 * np.pi / 2) * 2
            x_threshold = 6.313
            return left + (right - left) * (np.arctan((x - center) * x_threshold / width) / np.pi + 0.5)

        def function_exp(x, center, invert=False):
            if invert:
                return np.exp(-x + center) / 1000
            else:
                return np.exp(x - center) / 1000

        # Cost function
        # TODO: add usage cost in models
        # storage usage
        #storage_cost = 0
        #for index, dt in enumerate(self.delta_times):
        #    for key, mpc_component in self.models.items():
        #        power_in = mpc_component.inputs_in[index]  # kW
        #        power_out = mpc_component.inputs_out[index]  # kW
        #        storage_cost += (power_in / 10)  # ** 2
        #        storage_cost += (power_out / 10)  # ** 2

        # grid
        tariff_cost = 0
        for index, dt in enumerate(self.delta_times):
            grid_calculated = self.grid_expected[index]  # kW

            for key, mpc_component in self.models.items():
                power_in = mpc_component.inputs_in[index]  # kW
                power_out = mpc_component.inputs_out[index]  # kW

                grid_calculated += power_in
                grid_calculated -= power_out

            grid_power = self.grid[index]
            self.opti.subject_to(grid_calculated == grid_power)

            # split grid power into positive and negative
            #            pos_grid_power = self.grid_pos[index]
            #            neg_grid_power = self.grid_neg[index]
            #            opti.subject_to(pos_grid_power >= 0)
            #            opti.subject_to(neg_grid_power >= 0)
            #            opti.subject_to(pos_grid_power - neg_grid_power == grid_power)
            #            opti.subject_to(pos_grid_power * neg_grid_power == 0)
            #
            #            # max grid power
            #            opti.subject_to(pos_grid_power <= self.grid_pos_max)
            # opti.subject_to(grid_power <= self.grid_pos_max)

            # add to tariff cost
            # pos_tariff_cost = 0
            # pos_tariff = self.tariff_pos[index]  # in ct/kWh?
            # pos_tariff_cost += pos_tariff * grid_power / 100 / 3600  # in €/Ws
            # pos_tariff_cost += np.exp(grid_power - 100) / 1000

            # neg_tariff = self.tariff_neg[index]  # in ct/kWh?
            # _tariff_cost += neg_tariff * neg_grid_power / 100 / 1000 / 3600  # in €/Ws
            # tariff_cost += _tariff_cost ** 2

            pos_tariff = self.tariff_pos[index]
            neg_tariff = self.tariff_neg[index]
            tariff_cost += function_atan(grid_power, 0, 10, neg_tariff, pos_tariff)
            tariff_cost += function_exp(grid_power, 100, False)
            tariff_cost += function_exp(grid_power, 4, True)

            # TODO: missing square
            pass

        # TODO: add cost for last state x_N
        # for key, mpc_component in self.mpc_components.items():
        #     pass

        cost = 0
        cost += tariff_cost
        #cost += storage_cost

        # cost += max_grid_power * 10000

        self.opti.minimize(cost)

        #self.opti = opti  # TODO: needed?

