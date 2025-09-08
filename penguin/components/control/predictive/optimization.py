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
    freq: str
    total_duration: int
    step_durations: list[int]
    model: Model

    results_buffer: Optional[pd.DataFrame]


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

        self.model = Model(self.step_durations, configs=configs.get_section("casadi", defaults={}))

        self.results_buffer = None

    def _configure_timing(self, configs: Configurations) -> None:
        self.freq = configs.get("freq", default="1h")
        timing_keys = configs.get("timing_keys")

        def is_list_of_str(obj):
            return (
                isinstance(obj, list) and
                all(isinstance(o, str) for o in obj)
            )

        if not is_list_of_str(timing_keys):
            raise ResourceException(
                f"Invalid timing configuration: {timing_keys}, expected a list of strings"
            )

        self.step_durations = []
        for timing in timing_keys:
            #TODO: automatically raise exception if not all keys are present?
            t_config = configs.get_section(timing)
            if not t_config:
                raise ResourceException(f"Missing timing configuration for {timing}")

            duration = to_timedelta(parse_freq(t_config.get("duration")))
            freq = to_timedelta(parse_freq(t_config.get("freq")))

            if duration % freq != pd.Timedelta(0):
                raise ResourceException((
                    f"Invalid timing configuration for {timing}:"
                    f"duration {duration} is not a multiple of frequency {freq}"
                ))

            for _ in range(int(duration / freq)):
                self.step_durations.append(int(freq.total_seconds()))

            #self.step_durations_list.append(step_durations)

        if len(self.step_durations) == 0:
            raise ResourceException("No step durations configured, please check your timing configuration")

        self.total_duration = sum(self.step_durations)


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

        for controllable in controlled_components:
            model_config = model_configs.get_section(controllable.id.replace(".", "_"), defaults={})
            self.model.add_component(controllable, model_config)

        self.setup(self.model)
        cost = self.cost_function(self.model)
        for component_id, component_costs in self.model.costs.items():
            for cost_id, component_cost in component_costs.items():
                cost += component_cost
        self.model.opti.minimize(cost)

        for channel in self.model._channels.values():
            self._add_channel(channel, aggregate="mean")

    def solve(
            self,
            data:pd.DataFrame,
            start_time: TimestampType = None,
            prior: pd.DataFrame = pd.DataFrame(),
    ) -> pd.DataFrame:
        data = data.copy()
        results = []
        
        try:
            interval_data = self._df_to_intervals(data, start_time)

            self.set_initials(interval_data, self.model)

            self.model.set_initials(prior)


            self.model.opti.solve()

            timestamps = self._steps_to_datetime(start_time - pd.Timedelta("15min"))

            result = pd.DataFrame(index=timestamps)
            result = self.model.extract_results(result)
            result = self.extract_results(self.model, result)

            result = result.resample("1min").ffill().bfill()
            # results.index = results.index# - pd.Timedelta("15min")
            self.result_buffer = result
        except Exception as e:
            print(f"Unable to solve optimization problem @ {start_time}: {e}")
            if self.results_buffer is not None:
                result = self.results_buffer
            else:
                result = pd.DataFrame(index=pd.date_range(start_time, start_time + pd.Timedelta("7days"), freq="1min"))
                

        return result

    def _add_channel(self, channel: dict, aggregate: str = "mean", **custom) -> None:
        channel["aggregate"] = aggregate
        channel["connector"] = None
        channel.update(custom)
        self.data.add(**channel)

    def _df_to_intervals(
            self,
            data: pd.DataFrame,
            start: TimestampType,
    ) -> pd.DataFrame:
        target_times = self._steps_to_datetime(start)
        data = data.iloc[[data.index.get_indexer([ts], method='nearest')[0] for ts in target_times]]
        data.reset_index(drop=True, inplace=True)
        return data

    def _steps_to_datetime(
            self,
            start: TimestampType,
    ) -> list[pd.Timestamp]:
        step_durations = [0] + self.step_durations[:-1]
        target_times = [start + pd.Timedelta(seconds=time_difference) for time_difference in np.cumsum(step_durations)]
        return target_times

