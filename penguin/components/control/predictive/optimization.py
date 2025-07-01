# -*- coding: utf-8 -*-
"""
penguin.components.control.predictive.optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""

from abc import ABC, abstractmethod

import casadi
import numpy as np
import pandas as pd
from absl.testing.parameterized import parameters
from lori.util import to_timedelta, parse_freq
from scipy.linalg import expm

from typing import Iterable, Union

import casadi as ca

from lori.components import Component, register_component_type
from lori.core import Configurations, ResourceException, Activator
from lori.typing import TimestampType

from penguin.components import ElectricalEnergyStorage

from .model import Model


class Optimization(Component, ABC): #TODO: or _Component?
    """
    Base class for predictive optimization problems.
    """

    total_duration: int = 0
    step_durations_list: list[list[int]] = []

    models: list[Model] = []

    @property
    @abstractmethod
    def required_components(self) -> Union[type]:
        pass

    @abstractmethod
    def setup(self, model: Model):
        pass

    @abstractmethod
    def set_initials(self, data: pd.DataFrame, model: Model) -> None:
        pass

    @abstractmethod
    def cost_function(self, model: Model) -> ca.MX:
        pass

    @abstractmethod
    def extract_results(self, model: Model, results: pd.DataFrame) -> pd.DataFrame:
        pass


    def configure(self, configs: Configurations) -> None:
        super().configure(configs)

        timing_config = configs.get_section("timing", defaults={})
        if not timing_config:
            raise ResourceException("Missing timing configuration")
        self._configure_timing(timing_config)

        model_configs = configs.get_section("model", defaults={})
        casadi_configs = configs.get_section("casadi", defaults={})

        # init models
        for index, step_durations in enumerate(self.step_durations_list):
            self.models.append(Model(
                step_durations,
                model_configs = model_configs,
                casadi_configs = casadi_configs
            ))
        pass




    def _configure_timing(self, configs: Configurations) -> None:
        timing_keys = configs.get("timing_keys")

        def is_list_of_list_of_str(obj):
            return (
                isinstance(obj, list) and
                all(isinstance(inner, list) and all(isinstance(s, str) for s in inner) for inner in obj)
            )

        if not is_list_of_list_of_str(timing_keys):
            raise ResourceException(
                f"Invalid timing configuration: {timing_keys}, expected a list of list of strings"
            )

        for key_set in timing_keys:
            step_durations = []
            for key in key_set:
                #TODO: automatically raise exception if not all keys are present?
                t_config = configs.get_section(key)
                if not t_config:
                    raise ResourceException(f"Missing timing configuration for {key}")

                duration = to_timedelta(parse_freq(t_config.get("duration")))
                freq = to_timedelta(parse_freq(t_config.get("freq")))

                if duration % freq != pd.Timedelta(0):
                    raise ResourceException(
                        f"Invalid timing configuration for {key}: duration {duration} is not a multiple of frequency {freq}"
                    )
                [step_durations.append(int(freq.total_seconds())) for _ in range(int(duration / freq))]

            self.step_durations_list.append(step_durations)

        if len(self.step_durations_list) == 0:
            raise ResourceException("No step durations configured, please check your timing configuration")

        elif len(self.step_durations_list) > 1:
            for index in range(len(self.step_durations_list) - 1):
                if sum(self.step_durations_list[index + 1]) not in np.cumsum(self.step_durations_list[index]):
                    raise ResourceException(
                        self,
                        f"Invalid timing configuration: {index + 1}th timing key(s) does not match the previous one"
                    )

        self.total_duration = sum(self.step_durations_list[0])


    def activate(self) -> None:
        super().activate()

        #TODO: can be better (2 line)
        #TODO: check if multi type or list of types
        required_components = self.context.components.get_all(self.required_components)
        if len(required_components) == 0:
            raise ResourceException("No components found in system, required for optimization problem")

        for component in required_components:
            for model in self.models:
                model.add_component(component)


    def solve(
            self,
            data:pd.DataFrame,
            start_time: TimestampType = None
    ) -> pd.DataFrame:
        results = []
        for index, model in enumerate(self.models):
            interval_data = self._df_to_intervals(data, model.step_durations, start_time)

            self.setup(model)
            self.set_initials(interval_data, model)

            model.set_initials(interval_data)
            if index > 0:

                model.set_finals(results[index-1])

            cost = self.cost_function(model)
            for component_id, component_costs in model.costs.items():
                for cost_id, component_cost in component_costs.items():
                    cost += component_cost

            model.opti.minimize(cost)
            model.opti.solve()

            timestamps = self._steps_to_datetime(model.step_durations, start_time)
            result = pd.DataFrame(index=timestamps)
            result = self.extract_results(model, result)
            result = model.extract_results(result)
            results.append(result)
            pass

        results = pd.concat(results)
        results = results.loc[~results.index.duplicated(keep='last')]
        results = results.sort_index()


        plot_df = results.copy()
        plot_solution = True
        if plot_solution:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 6))

            for col in plot_df.columns:
                plt.plot(plot_df.index, plot_df[col], marker='x', linestyle='-', label=col)

            plt.legend()
            plt.xlabel("Time")
            plt.ylabel("Value")
            plt.title("Scatter Plot of All Columns")
            plt.grid(True)
            plt.tight_layout()
            plt.show()


        return results

    def _df_to_intervals(
            self,
            data: pd.DataFrame,
            step_durations: list[int],
            start: TimestampType,
    ) -> pd.DataFrame:
        target_times = self._steps_to_datetime(step_durations, start)
        data = data.iloc[[data.index.get_indexer([ts], method='nearest')[0] for ts in target_times]]
        data.reset_index(drop=True, inplace=True)
        return data

    def _steps_to_datetime(self, step_durations: list[int], start: pd.Timestamp) -> list[pd.Timestamp]:
        step_durations = [0] + step_durations[:-1]
        target_times = [start + pd.Timedelta(seconds=time_difference) for time_difference in np.cumsum(step_durations)]
        return target_times