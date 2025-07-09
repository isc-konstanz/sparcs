# -*- coding: utf-8 -*-
"""
penguin.components.control.predictive.problems.grid_cost
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""


from typing import Iterable, Union

import casadi
import numpy as np
import pandas as pd

from typing import Callable

from lori import Configurations
from lori.components import register_component_type
from penguin.components import ElectricalEnergyStorage

from penguin.components.control.predictive.optimization import Optimization
from penguin.components.control.predictive.model import Model

@register_component_type("grid_cost_problem", "grid_cost_mpc")
class GridCostProblem(Optimization):
    """
    Component for tariff cost optimization problem.
    """

    objective_config: Configurations
    objective_function: Callable

    import_tariff: casadi.MX
    export_tariff: casadi.MX
    grid_expected: casadi.MX
    grid_variable: casadi.MX
    
    grid_expected_std: casadi.MX

    @property
    def required_components(self) -> Union:
        return ElectricalEnergyStorage

    def configure(self, configs: Configurations) -> None:
        super().configure(configs)

        objective_config = configs.get_section("objective")
        
        stochastic_active = objective_config.get("stochastic_active", False)
        bayes_factors = []
        if stochastic_active:
            n_sigma = objective_config.get("stochastic_order", 2)  # Number of standard deviations to consider
            scale_factor = objective_config.get("stochastic_distance", 1.0)  # Scale factor for standard deviation

            bayes_factors = [1/np.sqrt(2 * np.pi) * np.exp(-0.5 * (index * scale_factor)**2) 
                             for index in range(-n_sigma, n_sigma + 1)]
            bayes_factors = np.array(bayes_factors) / np.sum(bayes_factors)
        
        def function_arctan(x, center, width=1, left=0, right=1):
            # x_threshold = np.tan(0.9 * np.pi / 2) * 2
            x_threshold = 6.313
            return left + (right - left) * (np.arctan((x - center) * x_threshold / width) / np.pi + 0.5)

        def tariff_cost_function(grid, import_tariff, export_tariff) -> casadi.Function:
            tariff = 0
            tariff += function_arctan(grid, 0, 2, export_tariff, import_tariff)
            
            if objective_config.get("import_limit_active", False):
                import_limit = objective_config.get("import_limit", 1000)  # Defaults to 1MW
                import_limit_tariff = objective_config.get("import_limit_tariff", 100)  # Defaults to 100 ct/kW

                tariff += function_arctan(grid, import_limit, 10, 0, import_limit_tariff - import_tariff)

            if objective_config.get("export_limit_active", False):
                export_limit = objective_config.get("export_limit", -1000)  # Defaults to -1MW
                export_limit_tariff = objective_config.get("export_limit_tariff", 100)
                
                tariff += function_arctan(grid, export_limit, 2, export_limit_tariff - export_tariff, 0)
                
            if objective_config.get("grid_squared", False):
                cost = tariff / 100 * grid ** 2
            else:
                cost = function_arctan(grid, 0, 2, -1, 1) * tariff / 100 * grid * 100
                
            return cost
            
        def objective_function(grid, grid_std, import_tariff, export_tariff) -> casadi.Function:
            if stochastic_active:
                if grid_std is None:
                    raise ValueError("grid_std must be provided for stochastic optimization")

                # Stochastic optimization
                cost = 0
                for std_eval_index in range(-n_sigma, n_sigma+1):
                    std_eval = std_eval_index * scale_factor * grid_std
                    grid_calculated_stochastic = grid + std_eval
                    cost += tariff_cost_function(grid_calculated_stochastic, import_tariff, export_tariff) \
                            * bayes_factors[std_eval_index + n_sigma]
                return cost
            
            else:
                return tariff_cost_function(grid, import_tariff, export_tariff)
        
        self.objective_config = objective_config
        self.objective_function = objective_function

    def setup(self, model: Model) -> None:
        n = len(model.step_durations)
        self.import_tariff = model.opti.parameter(n)
        self.export_tariff = model.opti.parameter(n)
        self.grid_expected = model.opti.parameter(n)
        self.grid_variable = model.opti.variable(n)
        
        if self.objective_config.get("stochastic_active", False):
            self.grid_expected_std = model.opti.parameter(n)  # Standard deviation for stochasticity

    def set_initials(self, data: pd.DataFrame, model: Model):
        model.opti.set_value(self.import_tariff, data["import"].values)
        model.opti.set_value(self.export_tariff, data["export"].values)
        model.opti.set_value(self.grid_expected, data["forecast"].values / 1000)  # convert to kWh
        
        if self.objective_config.get("stochastic_active", False):
            model.opti.set_value(self.grid_expected_std, data["forecast_std"].values / 1000)  # convert to kWh

    def extract_results(self, model: Model, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.loc[:, "import_tariff"] = model.opti.value(self.import_tariff)
        df.loc[:, "export_tariff"] = model.opti.value(self.export_tariff)
        df.loc[:, "grid_expected"] = model.opti.value(self.grid_expected)
        df.loc[:, "grid_variable"] = model.opti.value(self.grid_variable)

        if self.objective_config.get("plot_results", False):
            self._plot_results(df)

        return df



    

    def cost_function(self, model: Model):
        cost = 0

        for index, step_duration in enumerate(model.step_durations):
            grid_calculated = self.grid_expected[index] # convert to kWh
            pos_tariff = self.import_tariff[index]
            neg_tariff = self.export_tariff[index]

            grid_std = None
            if self.objective_config.get("stochastic_active", False):
                grid_std = self.grid_expected_std[index]

            for component_id, variables in model.variables.items():
                energy_in = variables["inputs_in"][index]
                energy_out = variables["inputs_out"][index]

                grid_calculated += energy_in
                grid_calculated -= energy_out

            model.opti.subject_to(self.grid_variable[index] == grid_calculated)

            step_cost = self.objective_function(grid_calculated,
                                                grid_std,
                                                pos_tariff,
                                                neg_tariff)

            cost += step_cost * step_duration / 3600 # convert to hours

        return cost


    def _plot_tariff_function(self):
        # plot objective function
        import matplotlib.pyplot as plt
        import numpy as np
        x = np.linspace(-200, 200, 10000)
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

    def _plot_results(self, df: pd.DataFrame):
        import matplotlib.pyplot as plt

        if plt.fignum_exists(1):
            fig = plt.figure(1)
            fig.clf()

        else:
            fig = plt.figure(1, figsize=(12, 6))
            plt.waitforbuttonpress()


        ax = fig.add_subplot(111)
        ax.plot(df.index, df["grid_expected"], label="Grid predicted (kWh)")
        ax.plot(df.index, df["grid_variable"], label="Grid MPC solution (kWh)")
        ax.plot(df.index, df["import_tariff"]*100, label="Import Tariff (ct/kWh)", linestyle='--')
        ax.plot(df.index, df["export_tariff"]*100, label="Export Tariff (ct/kWh)", linestyle='--')

        ax.plot(df.index, df["isc.ees_rct_state"], label="EES RCT State (kWh)", linestyle=':')
        ax.plot(df.index, df["isc.ees_rct_input_in"] - df["isc.ees_rct_input_out"],
                label="EES RCT Input (kW)")

        ax.set_xlabel("Time")
        ax.set_ylabel("Power (kW)")
        ax.set_title("Grid power predicted vs MPC solution")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()

        #show 0.2s
        plt.pause(0.1)
