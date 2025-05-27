from abc import abstractmethod, abstractproperty
from collections import OrderedDict

import numpy as np
import pandas as pd
from typing import Iterable, Union

from casadi import Opti
from lori.components import Component, register_component_type
from lori.core import Configurations, ResourceException, Activator
from penguin.components import ElectricalEnergyStorage

from .predictive_model import PredictiveModel

class PredictiveOptimization(Component):
    """
    Base class for predictive optimization problems.
    """

    SOLVERS = ["ipopt"]#, "osqp"]


    delta_times: list[int]

    opti: Opti

    model_config: Configurations
    models: dict[Component, PredictiveModel] = OrderedDict()

    @abstractmethod
    def get_required_components(self) -> Union[type]:
        pass

    @abstractmethod
    def cost_function(self):
        pass


    def configure(self, configs: Configurations) -> None:
        super().configure(configs)

        # optimizer
        self.opti = Opti()
        #TODO: better subsection handling (remove default for raise? do configs first???)
        opti_config = configs.get_section("optimizer", defaults={})
        if not opti_config:
            raise ResourceException("Missing optimizer configuration")
        self._configure_opti(opti_config)

        # model
        self.model_config = configs.get_section("model", defaults={})

        # timing
        timing_config = configs.get_section("timing", defaults={})
        if not timing_config:
            raise ResourceException("Missing timing configuration")
        self._configure_timing(timing_config)

    def _configure_opti(self, configs: Configurations) -> None:
        solver = configs.get("solver", default=PredictiveOptimization.SOLVERS[0])

        #TODO: better validation
        if not isinstance(solver, str):
            raise ResourceException(f"Invalid optimizer configuration: {solver}, expected string")
        if solver not in PredictiveOptimization.SOLVERS:
            raise ResourceException(f"Invalid optimizer configuration: {solver}, expected one of {PredictiveOptimization.SOLVERS}")

        #TODO: Similar to electric mode_parameters
        # IPOPT
        if solver == "ipopt":
            max_iter = configs.get_int("max_iter", default=100)
            tol = configs.get_float("tol", default=1e-6)
            print_level = configs.get_int("print_level", default=0)

            # self.opti.solver("ipopt", {"ipopt.max_iter": max_iter, "ipopt.tol": tol, "ipopt.print_level": print_level})
            p_opts = {"expand": True}
            s_opts = {"max_iter": 100000}  # max_iter}#, "tol": tol, "print_level": print_level}
            self.opti.solver("ipopt", p_opts, s_opts)

    def _configure_timing(self, configs: Configurations) -> None:
        timing_keys = configs.get("timing_keys")
        #TODO: better validation
        if not isinstance(timing_keys, list):
            raise ResourceException(f"Invalid timing configuration: {timing_keys}, expected list")

        delta_times = []
        for key in timing_keys:
            t_config = configs.get_section(key)
            if not t_config:
                raise ResourceException(f"Missing timing configuration for {key}")

            N = t_config.get_int("N")
            dt = t_config.get_int("dt")
            unit = t_config.get("unit", "hour")

            # TODO: unit convertion?
            if unit == "second":
                pass
            elif unit == "minute":
                dt = dt * 60
            elif unit == "hour":
                dt = dt * 3600
            elif unit == "day":
                dt = dt * 3600 * 24
            elif unit == "week":
                dt = dt * 3600 * 24 * 7
            else:
                raise ResourceException(f"Invalid timing unit: {unit}, expected hour, minute, second, day or week")

            [delta_times.append(dt) for _ in range(N)]

        self.delta_times = delta_times


    def activate(self) -> None:
        super().activate()

        #TODO: can be better (2 line)
        #TODO: check if multi type or list of types
        #TODO: implement abstract property in subclass (self.get_required_components)
        required_components = self.context.components.get_all(ElectricalEnergyStorage)
        if len(required_components) == 0:
            raise ResourceException("No required components found in system")

        #TODO: check if all data are available
        from penguin.system import System
        required_data = [System.ENERGY_EL]
        required_channels = self.context.data[["ees_charge_power", "el_power"]]

        for component in required_components:
            model = PredictiveModel(component,
                                    self.model_config,
                                    self.delta_times,
                                    self.opti)


            self.models[component] = model


    def solve(self, data:pd.DataFrame, start_time: pd.Timestamp = None) -> None:
        """
        Solve the optimization problem.
        """
        pass
        #self.context.data[["ees_charge_power", "el_power"]]
        #TODO: setup channels
        #TODO:


        #set initials
        #shift old result if available or just set 0
        #solve
        #extract ts 
        #...


