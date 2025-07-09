# -*- coding: utf-8 -*-
"""
penguin.components.control.predictive.problems.grid_cost
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""


from typing import Iterable, Union

import casadi
import numpy as np
import pandas as pd
from casadi import Opti, DM, MX

from lori import Configurations
from lori.components import register_component_type
from penguin.components import ElectricalEnergyStorage

from penguin.components.control.predictive.optimization import Optimization
from penguin.components.control.predictive.model import Model

@register_component_type("grid_cost_stochastic_problem", "grid_cost_stochastic_mpc")
class GridCostProblemStochastic(Optimization):
    """
    Component for dynamic tariff cost optimization problem.
    """

    objective_config: Configurations

    import_tariff: casadi.MX
    export_tariff: casadi.MX
    grid_expected: casadi.MX
    grid_expected_std: casadi.MX
    bayes_factors: casadi.MX
    grid_variable: casadi.MX

    def configure(self, configs: Configurations) -> None:
        super().configure(configs)

        # get tariff type
        self.objective_config = configs.get_section("objective")




    @property
    def required_components(self) -> Union:
        return ElectricalEnergyStorage

    def setup(self, model: Model) -> None:
        n = len(model.step_durations)
        self.import_tariff = model.opti.parameter(n)
        self.export_tariff = model.opti.parameter(n)
        self.grid_expected = model.opti.parameter(n)
        self.grid_expected_std = model.opti.parameter(n)  # Standard deviation for stochasticity
        self.grid_variable = model.opti.variable(n)

        self.bayes_factors = model.opti.parameter(5)  # 5 Bayes factors for std_eval -2 to 2
        bf = self._bayes_factors()
        for i, (index, value) in enumerate(bf):
            model.opti.set_value(self.bayes_factors[i], value)


    def set_initials(self, data: pd.DataFrame, model: Model):
        model.opti.set_value(self.import_tariff, data["import"].values)
        model.opti.set_value(self.export_tariff, data["export"].values)
        model.opti.set_value(self.grid_expected, data["forecast"].values / 1000)  # convert to kWh
        model.opti.set_value(self.grid_expected_std, np.sqrt(data["forecast_std"].values / 1000))  # convert to kWh




    def extract_results(self, model: Model, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.loc[:, "import_tariff"] = model.opti.value(self.import_tariff)
        df.loc[:, "export_tariff"] = model.opti.value(self.export_tariff)
        df.loc[:, "grid_expected"] = model.opti.value(self.grid_expected)
        df.loc[:, "grid_expected_std"] = model.opti.value(self.grid_expected_std) + model.opti.value(self.grid_expected)
        df.loc[:, "grid_variable"] = model.opti.value(self.grid_variable)
        return df


    def objective_function(self, grid, import_tariff, export_tariff) -> casadi.Function:
        def function_arctan(x, center, width=1, left=0, right=1):
            # x_threshold = np.tan(0.9 * np.pi / 2) * 2
            x_threshold = 6.313
            return left + (right - left) * (np.arctan((x - center) * x_threshold / width) / np.pi + 0.5)

        # def function_exp(x, center, invert=False):
        #     if invert:
        #         return np.exp(-x + center) / 1000
        #     else:
        #         return np.exp(x - center) / 1000

        def function_tariff(grid,
                            import_tariff,
                            export_tariff,
                            import_limit=None,
                            import_limit_tariff=None,
                            export_limit=None,
                            export_limit_tariff=None):
            tariff = 0
            tariff += function_arctan(grid, 0, 2, export_tariff, import_tariff)
            if import_limit is not None:
                tariff += function_arctan(grid, import_limit, 10, 0, import_limit_tariff)
            if export_limit is not None:
                tariff += function_arctan(grid, export_limit, 2, export_limit_tariff - export_tariff, 0)
            return tariff

        def function_fit_square(a, anchor):
            return a / anchor

        import_limit = self.objective_config.get("import_limit", None)
        export_limit = self.objective_config.get("export_limit", None)
        import_limit_value = self.objective_config.get("import_limit_value", None)
        export_limit_value = self.objective_config.get("export_limit_value", None)

        tariff = function_tariff(
            grid,
            import_tariff,
            export_tariff,
            import_limit=None,#import_limit,
            import_limit_tariff=import_limit_value,
            export_limit=None,#export_limit,
            export_limit_tariff=export_limit_value)

        #return tariff
        return tariff / 100 * grid ** 2


    def cost_function(self, model: Model):



        cost = 0

        if True:
            # plot objective function
            import matplotlib.pyplot as plt
            import numpy as np
            x = np.linspace(-200, 200, 1000)
            import_tariff = 0.25
            export_tariff = -0.10
            tariff_function = self.objective_function(
                x,
                import_tariff,
                export_tariff
            )
            plt.figure(figsize=(10, 5))
            plt.plot(x, tariff_function, label="Tariff Function")
            plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
            plt.axvline(0, color='black', linestyle='--', linewidth=0.5)
            plt.title("Tariff Function")
            plt.xlabel("Grid Power (kW)")
            plt.ylabel("Tariff (€/kWh)")
            plt.grid(True)
            plt.legend()
            #plt.show()




        for index, step_duration in enumerate(model.step_durations):
            grid_calculated = self.grid_expected[index]# * dt_hour # convert to kWh
            pos_tariff = self.import_tariff[index]
            neg_tariff = self.export_tariff[index]
            for std_eval_index in range(-2, 3):
                std_eval = std_eval_index * self.grid_expected_std[index]
                grid_calculated_stochastic = grid_calculated + std_eval

                for component_id, variables in model.variables.items():
                    energy_in = variables["inputs_in"][index]
                    energy_out = variables["inputs_out"][index]

                    grid_calculated_stochastic += energy_in
                    grid_calculated_stochastic -= energy_out

                if std_eval_index == 0:
                    model.opti.subject_to(self.grid_variable[index] == grid_calculated_stochastic)

                dt_hour = step_duration / 3600  # convert to hours
                step_cost = self.objective_function(grid_calculated_stochastic, pos_tariff, neg_tariff)

                step_cost = step_cost *  self.bayes_factors[std_eval_index + 2]  # Adjust for the Bayes factor
                cost += step_cost * dt_hour
                pass

        return cost


    _bayes_factors_buffer: dict[str, np.ndarray] = None
    def _bayes_factors(self) -> np.ndarray:
        bayes_factors_index = [abs(i) for i in range(-2, 3)]
        bayes_factors = [0 for _ in bayes_factors_index]
        for i, bf in enumerate(bayes_factors_index):
            bayes_factors[i] = 1/np.sqrt(2 * np.pi) * np.exp(-0.5 * bf)

        bayes_factors = np.array(bayes_factors) / np.sum(bayes_factors)

        return np.array([bayes_factors_index, bayes_factors]).T

