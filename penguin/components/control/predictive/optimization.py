# -*- coding: utf-8 -*-
"""
penguin.components.control.predictive.optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""

from abc import ABC, abstractmethod
from typing import Optional, Iterable, List

import numpy as np
import pandas as pd
import casadi as ca

from lori.components import Component, ComponentAccess
from lori.core import Configurations, ResourceException
from lori.util import to_timedelta, parse_freq
from lori.typing import TimestampType
from penguin.components.control.predictive.model import Model



class Optimization(Component, ABC):
    """
    Base class for predictive optimization problems.
    """

    total_duration: int
    step_durations_list: List[List[int]]

    models: List[Model]


    @property
    @abstractmethod
    def required_components(self) -> Iterable[type]:
        pass

    @property
    @abstractmethod
    def controlled_components(self) -> Iterable[type]:
        pass

    @property
    @abstractmethod
    def channels(self) -> Iterable[dict]:
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

        self.models = []
        for index, step_durations in enumerate(self.step_durations_list):
            self.models.append(
                Model(
                    step_durations,
                    configs=configs.get_section("casadi", defaults={}),
                )
            )

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

        self.step_durations_list = []
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
                    raise ResourceException((
                        f"Invalid timing configuration for {key}:"
                        f"duration {duration} is not a multiple of frequency {freq}"
                    ))
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

        for channel in self.channels:
            self._add_channel(channel, aggregate="last")
            # self.data.add(**channel)

        all_components: ComponentAccess = self.context.components

        required_components = all_components.get_all(*self.required_components)
        if len(required_components) < len(self.required_components):
            pass
            #raise ResourceException(f"No required component found in system: {self.required_components}")

        controlled_components = all_components.get_all(*self.controlled_components)
        if len(controlled_components) == 0:
            pass
            #raise ResourceException(f"No controlled component found in system: {self.controlled_components}")

        model_configs = self.configs.get_section("models", defaults={})
        #constants = []
        for model in self.models:
            for controllable in controlled_components:
                model_config = model_configs.get_section(controllable.id.replace(".", "_"), defaults={})
                model.add_component(controllable, model_config)

            self.setup(model)
            cost = self.cost_function(model)
            for component_id, component_costs in model.costs.items():
                for cost_id, component_cost in component_costs.items():
                    cost += component_cost
            model.opti.minimize(cost)

        for channel in self.models[0]._channels.values():
            self._add_channel(channel, aggregate="mean")

    def solve(
            self,
            data:pd.DataFrame,
            start_time: TimestampType = None,
            prior: pd.DataFrame = pd.DataFrame(),
    ) -> pd.DataFrame:
        results = []
        
        try:
            for model_index, model in enumerate(self.models):
                interval_data = _df_to_intervals(data, model.step_durations, start_time)

                self.set_initials(interval_data, model)

                model.set_initials(prior)
                
                #TODO: fix this! / nessessary?
                if model_index != 0:
                    model.set_finals(results[-1])

                model.opti.solve()

                timestamps = _steps_to_datetime(list(model.step_durations), start_time)
                result = pd.DataFrame(index=timestamps)
                result = model.extract_results(result)
                result = self.extract_results(model, result)
                results.append(result)

            results = pd.concat(results)
            results = results.loc[~results.index.duplicated(keep='last')].sort_index()
            results.index = results.index - pd.Timedelta(hour=1)
            results = results.resample("1min").ffill()
            self.results_buffer = results
        except Exception as e:
            print(f"Unable to solve optimization problem @ {start_time}: {e}")
            if self.results_buffer is not None:
                results = self.results_buffer
            else:
                results = pd.DataFrame(index=pd.date_range(start_time, periods=self.total_duration, freq="1min"))
                

        return results

    def _add_channel(self, channel: dict, aggregate: str = "mean", **custom) -> None:
        channel["aggregate"] = aggregate
        channel["connector"] = None
        channel.update(custom)
        self.data.add(**channel)

def _df_to_intervals(
        data: pd.DataFrame,
        step_durations: list[int],
        start: TimestampType,
) -> pd.DataFrame:
    target_times = _steps_to_datetime(step_durations, start)
    data = data.iloc[[data.index.get_indexer([ts], method='nearest')[0] for ts in target_times]]
    data.reset_index(drop=True, inplace=True)
    return data

def _steps_to_datetime(
        step_durations: list[int],
        start: pd.Timestamp,
) -> list[pd.Timestamp]:
    step_durations = [0] + step_durations[:-1]
    target_times = [start + pd.Timedelta(seconds=time_difference) for time_difference in np.cumsum(step_durations)]
    return target_times

