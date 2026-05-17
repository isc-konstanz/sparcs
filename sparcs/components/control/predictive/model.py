# -*- coding: utf-8 -*-
"""
penguin.components.control.predictive.model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""

import numpy as np
import pandas as pd
import casadi as ca
from scipy.linalg import expm
from typing import List, Dict, AnyStr, Sequence, Tuple

from lories import Component, Constant, Channel
from lories.core import Configurations, ResourceError

from sparcs.components.storage import ElectricalEnergyStorage


class Model:

    IPOPT = "ipopt"
    OSQP = "osqp"
    SOLVERS = [IPOPT, OSQP]

    step_durations: list[int]
    epsilon: float


    opti: ca.Opti
    parameters: Dict[str, Dict[str, ca.SX]]
    variables: Dict[str, Dict[str, ca.MX]]
    costs: Dict[str, Dict[str, ca.MX]]

    _system_matrices: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]
    _components: Dict[str, Component]
    _channels: Dict
    _state_bounds: Dict[str, Tuple[float, float]]


    def __init__(
        self,
        step_durations: list[int],
        configs: Configurations
    ):
        self.step_durations = step_durations
        self.epsilon = configs.get_float("epsilon", default=1e-6)

        self.opti = ca.Opti()
        self.parameters = {}
        self.variables = {}
        self.costs = {}

        self._system_matrices = {}
        self._components = {}
        self._channels = {}
        self._state_bounds = {}

        solver = configs.get("solver", default=Model.IPOPT)
        solver_configs = configs.get_member(solver, defaults={})

        if solver not in Model.SOLVERS:
            raise ResourceError(
                f"Invalid optimizer configuration: {solver}, expected one of {Model.SOLVERS}")

        if solver == Model.IPOPT:
            expand = solver_configs.get_bool("expand", default=True)

            max_iter = solver_configs.get_int("max_iter", default=1000)
            tol = solver_configs.get_float("tol", default=1e-3)
            print_level = solver_configs.get_int("print_level", default=3)

            plugin_options = {
                "expand": expand,
                "print_time":False if print_level == -1 else None,
                "verbose":False if print_level == -1 else None,
            }

            # Many options, see https://coin-or.github.io/Ipopt/OPTIONS.html
            solver_options = {
                "max_iter": max_iter,  # Maximum number of iterations before stopping
                "tol": tol,  # Relative optimality tolerance; main convergence target
                "print_level": max(print_level, 0),  # Verbosity / print level
            }

            self.opti.solver("ipopt", plugin_options, solver_options)


    def set_initials(self, prior: pd.DataFrame) -> None:
        for component_id, component in self._components.items():
            if isinstance(component, ElectricalEnergyStorage):
                soc_column = component.data[ElectricalEnergyStorage.STATE_OF_CHARGE].column
                if prior is not None and not prior.empty and soc_column in prior.columns:
                    state_0 = prior[soc_column].values[-1] / 100 * component.capacity
                else:
                    state_0 = component.capacity / 2 # 50% initial state
                # Clip into the (possibly margin-tightened) hard bounds so IPOPT starts feasible.
                lo, hi = self._state_bounds[component_id]
                state_0 = float(np.clip(state_0, lo, hi))
                self.opti.set_value(self.parameters[component_id]["state_0"], state_0)

            else:
                raise ResourceError(f"Unsupported component type for initial state: {type(component)}")

    def set_finals(self, data: pd.DataFrame) -> None:
        time_duration = np.sum(self.step_durations)
        row = data.loc[data.index[0] + pd.Timedelta(seconds=time_duration)]
        for component_id, component in self._components.items():
            if isinstance(component, ElectricalEnergyStorage):
                soc_column = f"mpc_{component.data[ElectricalEnergyStorage.STATE_OF_CHARGE].column}"
                energy = row[soc_column] / 100 * component.capacity
                print("Set finals:")
                print(energy)
                print("-")
                print("-")
                print("-")
                print("-")
                state_var = self.variables[component_id]["states"][-1]
                self.opti.subject_to((state_var - energy) <= self.epsilon)
                self.opti.subject_to((state_var - energy) >= -self.epsilon)

            else:
                raise ResourceError(f"Unsupported component type for final state: {type(component)}")

    def add_component(
            self,
            component: Component,
            configs: Configurations,
    ) -> None:
        self._components[component.id] = component

        if isinstance(component, ElectricalEnergyStorage):
            self._add_electrical_energy_storage(component, configs=configs)
        else:
            raise ResourceError(f"Unsupported component type: {type(component)}")

    def _add_electrical_energy_storage(
            self,
            component: ElectricalEnergyStorage,
            configs: Configurations,
    ) -> None:
        component_id = component.id

        order = 1
        a = np.array([[0]])
        # Round-trip efficiency split symmetrically: sqrt(eff) on each leg so
        # charge→discharge returns η of the energy. b_in scales grid→battery energy
        # gain; b_out scales battery→grid energy drain (grid-side power u_out
        # corresponds to battery drain of u_out / sqrt(eff)).
        sqrt_eff = np.sqrt(component.efficiency)
        b_in = np.array([sqrt_eff / 3600])
        b_out = np.array([1 / (sqrt_eff * 3600)])

        state_0 = self.opti.parameter(order)

        num_steps = len(self.step_durations)
        states = self.opti.variable(order, num_steps)
        inputs_in = self.opti.variable(num_steps)
        inputs_out = self.opti.variable(num_steps)

        # Initialize states strictly interior so the log barrier is well-defined at iteration 0.
        self.opti.set_initial(states, component.capacity / 2.0)

        self.parameters[component_id] = {}
        self.variables[component_id] = {}
        self.costs[component_id] = {}

        self.parameters[component_id]["state_0"] = state_0

        self.variables[component_id]["states"] = states
        self.variables[component_id]["inputs_in"] = inputs_in
        self.variables[component_id]["inputs_out"] = inputs_out

        charge_cost = configs.get_float("charge_cost", default=0.001) * 3 / (2 * component.power_max)
        discharge_cost = configs.get_float("discharge_cost", default=0.001) * 3 / (2 * component.power_max)

        # Optional ramp-rate limit on net battery power (kW per hour).
        # Configured per-EES. Default None disables the constraint.
        power_change_max = configs.get_float("power_change_max", default=None)

        # Soft safety margin: log barrier on SoC inside [soc_margin, 100 - soc_margin] %.
        # Hard bound is tightened to the same band so the log is always defined.
        soc_margin = configs.get_float("soc_margin", default=0.0)
        soc_margin_weight = configs.get_float("soc_margin_weight", default=10.0)
        soc_lo = soc_margin / 100.0 * component.capacity
        soc_hi = (1.0 - soc_margin / 100.0) * component.capacity
        self._state_bounds[component_id] = (soc_lo, soc_hi)
        print(
            f"[MPC EES] {component_id}: soc_margin={soc_margin}%, "
            f"weight={soc_margin_weight}, hard bound=[{soc_lo:.2f}, {soc_hi:.2f}] kWh "
            f"(capacity={component.capacity})"
        )

        x_k = state_0
        costs = 0
        for index, step_duration in enumerate(self.step_durations):
            u_k_in = inputs_in[index]
            u_k_out = inputs_out[index]
            x_k1 = states[:, index]

            # limit state
            self.opti.subject_to(self.opti.bounded([soc_lo], states[:, index], [soc_hi]))

            # limit input
            self.opti.subject_to(self.opti.bounded(0, u_k_in, component.power_max / 1000))
            self.opti.subject_to(self.opti.bounded(0, u_k_out, component.power_max / 1000))

            # ramp-rate limit on net battery power (kW per hour → scaled to step)
            if power_change_max is not None and index > 0:
                delta = (inputs_in[index] - inputs_out[index]) - (inputs_in[index - 1] - inputs_out[index - 1])
                delta_max = power_change_max * step_duration / 3600
                self.opti.subject_to(self.opti.bounded(-delta_max, delta, delta_max))

            # only one input at a time
            self.opti.subject_to((u_k_out * u_k_in) ** 2 <= self.epsilon)

            # discretize system dynamics
            phi, h_in, h_out = self._get_system_matrices(a, b_in, b_out, step_duration)

            # calculate next state
            x_k1_calculated = phi * x_k + h_in * u_k_in - h_out * u_k_out
            self.opti.subject_to((x_k1_calculated - x_k1) ** 2 <= self.epsilon)


            t_hour = step_duration / 3600
            costs += charge_cost * (inputs_in[index] ** 2) * t_hour
            costs += discharge_cost * (inputs_out[index] ** 2) * t_hour

            if soc_margin > 0:
                # Small additive eps so a transient line-search step at/past the bound
                # cannot evaluate log of a non-positive number. Inside the band the eps
                # is negligible (log((0.45 + 1e-3)/1) ≈ log(0.45)).
                eps_log = 1e-3
                soc = states[:, index] / component.capacity * 100
                costs += soc_margin_weight * t_hour * (
                    -ca.log((soc - soc_margin) / 100.0 + eps_log)
                    - ca.log((100.0 - soc_margin - soc) / 100.0 + eps_log)
                )

            x_k = x_k1
        self.costs[component_id]["usage"] = costs


        def mpc_constant(channel: Channel) -> dict:
            component_configs = channel.to_configs()
            config = {
                "type": component_configs["type"],
                "key": f"mpc_{component_configs['column']}",
                "name": f"MPC {component_configs['name']}",
                "unit": component_configs["unit"],
            }
            return config

        for constant in [ElectricalEnergyStorage.STATE_OF_CHARGE, ElectricalEnergyStorage.POWER_CHARGE]:
            channel = component.data[constant]
            self._channels[channel.id] = mpc_constant(channel)



    # noinspection PyShadowingBuiltins
    def extract_results(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for component_id, component in self._components.items():
            parameters = self.parameters[component_id]
            variables = self.variables[component_id]

            if isinstance(component, ElectricalEnergyStorage):
                state_0 = self.opti.value(parameters["state_0"])
                states = np.array([state_0, *self.opti.value(variables["states"])[:-1]])
                socs = states / component.capacity * 100
                soc_column = f"mpc_{component.data[ElectricalEnergyStorage.STATE_OF_CHARGE].column}"
                df.loc[:, soc_column] = socs

                input_in = self.opti.value(variables["inputs_in"])
                input_out = self.opti.value(variables["inputs_out"])
                input = (input_in - input_out) * 1000
                power_column = f"mpc_{component.data[ElectricalEnergyStorage.POWER_CHARGE].column}"
                df.loc[:, power_column] = input

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

            self._system_matrices[key] = (phi, h_in, h_out)
        return self._system_matrices[key]