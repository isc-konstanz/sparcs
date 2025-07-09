# -*- coding: utf-8 -*-
"""
penguin.components.control.predictive.model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""

from typing import Optional

import numpy as np
import pandas as pd
import casadi as ca
from scipy.linalg import expm

from lori import Component
from lori.core import Configurations, Constant, ResourceException, Activator

from penguin.components.storage import ElectricalEnergyStorage


class Model:
    SOLVERS = ["ipopt"]#, "osqp"]

    step_durations: list[int]
    epsilon: float


    opti: ca.Opti
    parameters: dict
    variables: dict
    costs: dict

    _system_matrices: dict = {}


    def __init__(
        self,
        step_durations: list[int],
        model_configs: Configurations,
        casadi_configs: Configurations
    ):
        self.step_durations = step_durations
        self.epsilon = model_configs.get_float("epsilon", default=1e-6)

        self.opti = ca.Opti()
        self.parameters = {}
        self.variables = {}
        self.costs = {}

        solver = casadi_configs.get("solver", default=Model.SOLVERS[0])

        # TODO: better validation
        if solver not in Model.SOLVERS:
            raise ResourceException(
                f"Invalid optimizer configuration: {solver}, expected one of {Model.SOLVERS}")

        # TODO: Similar to electric mode_parameters
        # IPOPT
        if solver == "ipopt":
            max_iter = casadi_configs.get_int("max_iter", default=100)
            tol = casadi_configs.get_float("tol", default=1e-6)
            print_level = casadi_configs.get_int("print_level", default=0)

            # self.opti.solver("ipopt", {"ipopt.max_iter": max_iter, "ipopt.tol": tol, "ipopt.print_level": print_level})
            p_opts = {"expand": True}
            s_opts = {"max_iter": 100000}  # max_iter}#, "tol": tol, "print_level": print_level}
            self.opti.solver("ipopt", p_opts, s_opts)

    def set_initials(self, data: pd.DataFrame) -> None:
        #TODO: higher order components
        for component_id, params in self.parameters.items():
            if component_id in data.columns:
                state_0 = data[component_id].values[0]
            else:
                state_0 = 0
            self.opti.set_value(params["state_0"], state_0)

    def set_finals(self, data: pd.DataFrame) -> None:
        #TODO: higher order components
        time_duration = np.sum(self.step_durations)
        row = data.loc[data.index[0] + pd.Timedelta(seconds=time_duration)]
        for component_id, variables in self.variables.items():
            # mostly will lead to unsolvable problem...
            self.opti.subject_to((variables["states"][-1] - row[component_id + "_state"]) ** 2 <= self.epsilon*100)

        pass


    def add_component(self, component: Component):
        if isinstance(component, ElectricalEnergyStorage):
            self._add_electrical_energy_storage(component)
        else:
            raise ResourceException(f"Unsupported component type: {type(component)}")

    def _add_electrical_energy_storage(self, component: ElectricalEnergyStorage):
        order = 1
        a = np.array([[0]])
        b_in = np.array([component.efficiency / 3600])
        b_out = np.array([component.efficiency / 3600])

        state_0 = self.opti.parameter(order)

        num_steps = len(self.step_durations)
        states = self.opti.variable(order, num_steps)
        inputs_in = self.opti.variable(num_steps)
        inputs_out = self.opti.variable(num_steps)

        self.parameters[component.id] = {}
        self.variables[component.id] = {}
        self.costs[component.id] = {}

        self.parameters[component.id]["state_0"] = state_0

        self.variables[component.id]["states"] = states
        self.variables[component.id]["inputs_in"] = inputs_in
        self.variables[component.id]["inputs_out"] = inputs_out

        x_k = state_0
        costs = 0
        for index, step_duration in enumerate(self.step_durations):
            u_k_in = inputs_in[index]
            u_k_out = inputs_out[index]
            x_k1 = states[:, index]

            # limit state
            self.opti.subject_to(self.opti.bounded([0], states[:, index], [component.capacity]))

            # limit input
            self.opti.subject_to(self.opti.bounded(0, u_k_in, component.power_max / 1000))
            self.opti.subject_to(self.opti.bounded(0, u_k_out, component.power_max / 1000))

            # only one input at a time
            self.opti.subject_to((u_k_out * u_k_in) ** 2 <= self.epsilon)

            # discretize system dynamics
            phi, h_in, h_out = self._get_system_matrices(a, b_in, b_out, step_duration)

            # explicit euler
            self.opti.subject_to(((phi * x_k + h_in * u_k_in - h_out * u_k_out) - x_k1) ** 2 <= self.epsilon)

            # # runge kutta 4
            # k1 = a_dt @ x_k + b_dt @ u_k
            # k2 = a_dt @ (x_k + 0.5 * dt * k1) + b_dt @ u_k
            # k3 = a_dt @ (x_k + 0.5 * dt * k2) + b_dt @ u_k
            # k4 = a_dt @ (x_k + dt * k3) + b_dt @ u_k
            # opti.subject_to(x_k1 == x_k + x_k + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4))

            x_k = x_k1

            t_hour = step_duration / 3600
            #TODO: replace placeholders
            charge_cost = 0.001 / (component.power_max)
            discharge_cost = 0.001 / (component.power_max)
            #TODO: square cost
            costs += charge_cost * (inputs_in[index] ** 2) * t_hour
            costs += discharge_cost * (inputs_out[index] ** 2) * t_hour

        #costs = 0
        self.costs[component.id]["usage"] = costs

    def extract_results(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for component_id, variables in self.variables.items():
            # TODO: higher order components
            #states = [self.opti.value(self.parameters[component_id]["state_0"])] + \
            #    self.opti.value(variables["states"])[:-1]
            states = np.insert(
                self.opti.value(variables["states"])[:-1],
                0,
                self.opti.value(self.parameters[component_id]["state_0"]))
            df.loc[:, component_id + "_state"] = states
            df.loc[:, component_id + "_input_in"] = self.opti.value(variables["inputs_in"])
            df.loc[:, component_id + "_input_out"] = self.opti.value(variables["inputs_out"])

        return df


    def _get_system_matrices(
        self,
        a: np.ndarray,
        b_in: np.ndarray,
        b_out: np.ndarray,
        step_duration: int
    ) -> list[np.ndarray]:
        """
        Computes the discretized system matrices.
        Phi = e^(A * dt)
        h_in = integral(0, dt) e^(A * (dt - tau)) * b_in * dtau
        """
        key = "_".join([str(a), str(b_in), str(b_out), str(step_duration)])

        if key not in self._system_matrices:
            order = a.shape[0]
            phi = expm(a * step_duration)

            h_in = np.zeros((order, 1))
            h_out = np.zeros((order, 1))
            for i in range(step_duration):
                h_in += expm(a * (step_duration - i - 1)) @ b_in
                h_out += expm(a * (step_duration - i - 1)) @ b_out

            self._system_matrices[key] = [phi, h_in, h_out]
        return self._system_matrices[key]

