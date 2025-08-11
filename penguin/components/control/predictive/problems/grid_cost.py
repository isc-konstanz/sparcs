# -*- coding: utf-8 -*-
"""
penguin.components.control.predictive.problems.grid_cost
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""


from typing import Iterable, Union

import casadi
import numpy as np
import pandas as pd

from typing import Callable, Dict

from lori import Configurations, Constant
from lori.components import Tariff, register_component_type
from penguin.components import ElectricalEnergyStorage

from penguin.components.control.predictive.optimization import Optimization
from penguin.components.control.predictive.model import Model

@register_component_type("grid_cost_problem", "grid_cost_mpc")
class GridCostProblem(Optimization):
    """
    Component for tariff cost optimization problem.
    """
    IMPORT_TARIFF = "import_tariff"
    EXPORT_TARIFF = "export_tariff"
    GRID_EXPECTED = "grid_expected"
    GRID_EXPECTED_STD = "grid_expected_std"
    GRID_VARIABLE = "grid_variable"

    GRID_EXPECTED_ = Constant(float, "grid_expected", "Expected Grid Power", "kWh")
    # GRID_STANDARD = Constant(float, "grid_expected_std", "Expected Grid Power Standard Deviation", "kWh")
    GRID_SOLUTION = Constant(float, "grid_solution", "Grid Power Solution", "kWh")

    model_variables: Dict[Model, Dict[str, casadi.MX]]

    objective_config: Configurations
    objective_function: Callable


    import_tariff: casadi.MX
    export_tariff: casadi.MX
    grid_expected: casadi.MX
    grid_variable: casadi.MX

    is_stochastic: bool
    grid_expected_std: casadi.MX

    @property
    def required_components(self) -> Iterable[type]:
        return [Tariff]

    @property
    def controlled_components(self) -> Iterable[type]:
        return [ElectricalEnergyStorage]

    #TODO: do i want to create channels from that???
    # or only from the solution?
    @property
    def channels(self) -> Iterable[dict]:
        # channels = [
        #     GridCostProblem.GRID_EXPECTED,
        #     GridCostProblem.GRID_SOLUTION
        # ]
        # if self.is_stochastic:
        #     channels.append(GridCostProblem.GRID_STANDARD)



        channels = [GridCostProblem.GRID_SOLUTION, GridCostProblem.GRID_EXPECTED_]
        return [channel.to_dict() for channel in channels]

    def configure(self, configs: Configurations) -> None:
        super().configure(configs)

        self.model_variables = {}

        objective_config = configs.get_section("objective")

        self.is_stochastic = objective_config.get("stochastic_active", False)
        bayes_factors = []
        if self.is_stochastic:
            n_sigma = objective_config.get_int("stochastic_order", 2)  # Number of standard deviations to consider
            scale_factor = objective_config.get_float("stochastic_distance", 1.0)  # Scale factor for standard deviation

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
                import_limit = objective_config.get("import_limit", default=-10)  # Defaults to 1MW
                import_limit_tariff = objective_config.get("import_limit_tariff", default=100)  # Defaults to 100 ct/kW

                tariff += function_arctan(grid, import_limit, 10, 0, import_limit_tariff - import_tariff)

            if objective_config.get("export_limit_active", False):
                export_limit = objective_config.get("export_limit", default=-1000)  # Defaults to -1MW
                export_limit_tariff = objective_config.get("export_limit_tariff", default=100)
                
                tariff += function_arctan(grid, export_limit, 2, export_limit_tariff - export_tariff, 0)
                
            if objective_config.get("grid_cost_squared", True):
                cost = tariff * grid ** 2
            else:
                cost = function_arctan(grid, 0, 2, -1, 1) * tariff * grid * 100
                
            return cost
            
        def objective_function(grid, grid_std, import_tariff, export_tariff) -> casadi.Function:
            if self.is_stochastic:
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

        self.model_variables[model] = {
            GridCostProblem.IMPORT_TARIFF: model.opti.parameter(n),
            GridCostProblem.EXPORT_TARIFF: model.opti.parameter(n),
            GridCostProblem.GRID_EXPECTED: model.opti.parameter(n),
            GridCostProblem.GRID_EXPECTED_STD: model.opti.variable(n) if self.is_stochastic else None,
            GridCostProblem.GRID_VARIABLE: model.opti.variable(n),
        }

        # self.import_tariff = model.opti.parameter(n)
        # self.export_tariff = model.opti.parameter(n)
        # self.grid_expected = model.opti.parameter(n)
        # self.grid_variable = model.opti.variable(n)
        #
        if self.is_stochastic:
            self.model_variables[model][GridCostProblem.GRID_EXPECTED_STD] = model.opti.parameter(n)
            #self.grid_expected_std = model.opti.parameter(n)  # Standard deviation for stochasticity

    def set_initials(self, data: pd.DataFrame, model: Model):
        #TODO: use predictors here+
        params = self.model_variables[model]

        model.opti.set_value(params[GridCostProblem.IMPORT_TARIFF], data[Tariff.PRICE_IMPORT].values)
        model.opti.set_value(params[GridCostProblem.EXPORT_TARIFF], data[Tariff.PRICE_IMPORT].values)
        model.opti.set_value(params[GridCostProblem.GRID_EXPECTED], data["forecast"].values / 1000)  # convert to kWh
        
        if self.objective_config.get("stochastic_active", False):
            model.opti.set_value(params[GridCostProblem.GRID_EXPECTED_STD], data["forecast_std"].values / 1000)  # convert to kWh

    def extract_results(self, model: Model, df: pd.DataFrame) -> pd.DataFrame:
        results = df.copy()
        params = self.model_variables[model]

        column = self.data[GridCostProblem.GRID_SOLUTION].key
        #results[column] = model.opti.value(self.grid_variable)
        results[column] = model.opti.value(params[GridCostProblem.GRID_VARIABLE]) * 1000
        results["grid_expected"] = model.opti.value(params[GridCostProblem.GRID_EXPECTED]) * 1000

        if self.objective_config.get("plot_results", False):
            plot_df = df.copy()
            # plot_df.loc[:, "import_tariff"] = model.opti.value(self.import_tariff)
            # plot_df.loc[:, "export_tariff"] = model.opti.value(self.export_tariff)
            # plot_df.loc[:, "grid_expected"] = model.opti.value(self.grid_expected)
            # plot_df.loc[:, "grid_solution"] = model.opti.value(self.grid_variable)
            # plot_df.loc[:, "grid_standard"] = model.opti.value(self.grid_expected_std) if self.is_stochastic else None

            plot_df.loc[:, "import_tariff"] = model.opti.value(params[GridCostProblem.IMPORT_TARIFF])
            plot_df.loc[:, "export_tariff"] = model.opti.value(params[GridCostProblem.EXPORT_TARIFF])
            plot_df.loc[:, "grid_expected"] = model.opti.value(params[GridCostProblem.GRID_EXPECTED])
            plot_df.loc[:, "grid_solution"] = model.opti.value(params[GridCostProblem.GRID_VARIABLE])
            if self.is_stochastic:
                plot_df.loc[:, "grid_standard"] = model.opti.value(params[GridCostProblem.GRID_EXPECTED_STD])
            else:
                plot_df.loc[:, "grid_standard"] = None

            self._plot_results(plot_df)

        return results

    def cost_function(self, model: Model):
        cost = 0
        params = self.model_variables[model]


        for index, step_duration in enumerate(model.step_durations):
            # grid_calculated = self.grid_expected[index]
            # pos_tariff = self.import_tariff[index]
            # neg_tariff = self.export_tariff[index]

            grid_calculated = params[GridCostProblem.GRID_EXPECTED][index]
            pos_tariff = params[GridCostProblem.IMPORT_TARIFF][index]
            neg_tariff = params[GridCostProblem.EXPORT_TARIFF][index]

            grid_std = None
            if self.objective_config.get("stochastic_active", False):
                #grid_std = self.grid_expected_std[index]
                grid_std = params[GridCostProblem.GRID_EXPECTED_STD][index]

            for component_id, variables in model.variables.items():
                energy_in = variables["inputs_in"][index]
                energy_out = variables["inputs_out"][index]

                grid_calculated += energy_in
                grid_calculated -= energy_out

            grid_variable = params[GridCostProblem.GRID_VARIABLE][index]
            # model.opti.subject_to(self.grid_variable[index] == grid_calculated)
            model.opti.subject_to(grid_variable == grid_calculated)

            step_cost = self.objective_function(
                grid_calculated,
                grid_std,
                pos_tariff,
                neg_tariff
            )

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

        def plot():
            ax = fig.add_subplot(111)
            ax.plot(df.index, df["grid_expected"], label="Grid expected (kWh)")
            ax.plot(df.index, df["grid_solution"], label="Grid MPC solution (kWh)")
            ax.plot(df.index, df["import_tariff"], label="Import Tariff (ct/kWh)", linestyle='--')
            ax.plot(df.index, df["export_tariff"], label="Export Tariff (ct/kWh)", linestyle='--')
            
            if self.is_stochastic:
                ax.fill_between(df.index, df["grid_expected"] - df["grid_standard"],
                                df["grid_expected"] + df["grid_standard"],
                                color='gray', alpha=0.2, label="Stochastic Range (± std)")

            for component_id, component in self.models[0]._components.items():
                if isinstance(component, ElectricalEnergyStorage):
                    soc_column = f"mpc_{component.data[component.STATE_OF_CHARGE].column}"
                    ax.plot(df.index, df[soc_column], label=f"SoC (%) ({component_id})", linestyle=':')
                    charge_power_column = f"mpc_{component.data[component.POWER_CHARGE].column}"
                    ax.plot(df.index, df[charge_power_column] / 1000, label=f"Charge Power (kW) ({component_id})")

            ax.set_xlabel("Time")
            ax.set_ylabel("Power (kW)")
            ax.set_xlim(df.index[0], df.index[-1])
            ax.set_title("Grid power predicted vs MPC solution")
            ax.legend(loc='upper right')
            ax.grid(True)
            plt.tight_layout()

            plt.pause(0.1)

        if plt.fignum_exists(1):
            fig = plt.figure(1)
            fig.clf()
            plot()

        else:
            fig = plt.figure(1, figsize=(12, 6))
            plot()
            plt.waitforbuttonpress()


