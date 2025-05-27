
from casadi import *

from lori import Component
from lori.core import Configurations, Constant, ResourceException, Activator
import numpy as np
from scipy.linalg import expm

from penguin.components.storage import ElectricalEnergyStorage

"""
https://web.casadi.org/docs/

MX: Symbolic expressions with a static computational graph; ideal for scalar operations and fast code generation.
DM: Dense numerical matrices (non-symbolic); used mainly for storing data and interfacing with functions, not for symbolic computations.
MX: Symbolic expressions with a dynamic graph; supports complex, high-level operations like function calls and matrix-valued expressions.

Mixing:
You can't directly mix MX and MX in the same expression.
But you can use MX-based functions inside MX expressions (via Function), which combines MX's speed with MX's flexibility.

"""


class PredictiveModel:
    order: int

    # Parameters
    state_0: DM         # initial state
    state_min: DM       # minimum state
    state_max: DM       # maximum state
    input_in_max: DM    # maximum in for input
    input_out_max: DM   # maximum out for input
    epsilon: DM         # zero tolerance for numerical stability
    fixed_inputs: DM    # fixed inputs (e.g. constant inputs over the time horizon)

    states: MX
    inputs_in: MX
    inputs_out: MX

    _a: np.ndarray
    _b_in: np.ndarray
    _b_out: np.ndarray
    _b_fixed: np.ndarray
    _system_matrices: dict[str, list] = {}


    def __init__(self,
                 component: Component,
                 model_configs: Configurations,
                 delta_times: list[int],
                 opti: Opti,
                 ) -> None:



        #TODO: also get max min state and input range from component
        order, a, b_in, b_out, b_fixed = _component_to_model(component)

        order = a.shape[0]
        self.order = order

        self._a = a
        self._b_in = b_in
        self._b_out = b_out
        self._b_fixed = b_fixed

        # initial
        self.state_0 = opti.parameter(order)

        # states range
        self.state_min = opti.parameter(order)
        self.state_max = opti.parameter(order)
        # opti.set_value(self.state_min
        # opti.set_value(self.state_max

        # inputs range
        self.input_in_max = opti.parameter(1)
        self.input_out_max =  opti.parameter(1)
        # opti.set_value(self.input_in_max
        # opti.set_value(self.input_out_max

        self.fixed_inputs = opti.parameter(1, len(delta_times))

        # epsilon (zero tolerance)
        self.epsilon = opti.parameter(1)
        opti.set_value(self.epsilon, model_configs.get_float("epsilon", default=1e-6))


        # variables
        self.states = opti.variable(order, len(delta_times) + 1)
        self.inputs_in = opti.variable(1, len(delta_times))
        self.inputs_out = opti.variable(1, len(delta_times))



        # ---------------------------------------
        # Initial condition
        opti.subject_to(self.states[:, 0] == self.state_0)

        # Markov chain
        for index, dt in enumerate(delta_times):
            # parameters
            u_k_fixed = self.fixed_inputs[index]
            # variables
            x_k1 = self.states[:, index + 1]
            x_k = self.states[:, index]
            u_k_in = self.inputs_in[index]
            u_k_out = self.inputs_out[index]

            # limit input u
            opti.subject_to(u_k_in >= 0)
            opti.subject_to(u_k_in <= self.input_in_max)

            opti.subject_to(u_k_out >= 0)
            opti.subject_to(u_k_out <= self.input_out_max)

            opti.subject_to(u_k_in * u_k_out <= self.epsilon)

            # limit state x
            opti.subject_to(self.state_min <= x_k1)
            opti.subject_to(self.state_max >= x_k1)

            # system dynamics
            phi, h_in, h_out, h_fixed = self._get_system_matrices(dt)

            #TODO: make configurable

            # explicit euler
            opti.subject_to((phi * x_k + h_in * u_k_in - h_out * u_k_out + h_fixed * u_k_fixed) - x_k1 <= self.epsilon)
            opti.subject_to((phi * x_k + h_in * u_k_in - h_out * u_k_out + h_fixed * u_k_fixed) - x_k1 >= -self.epsilon)

            # # runge_kutta
            # k1 = a_dt @ x_k + b_dt @ u_k
            # k2 = a_dt @ (x_k + 0.5 * dt * k1) + b_dt @ u_k
            # k3 = a_dt @ (x_k + 0.5 * dt * k2) + b_dt @ u_k
            # k4 = a_dt @ (x_k + dt * k3) + b_dt @ u_k
            # opti.subject_to(x_k1 == x_k + x_k + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4))




    def _get_system_matrices(self, dt: int):
        if str(dt) not in self._system_matrices:
            # Phi = e^(A * dt)
            # h_in = integral(0, dt) e^(A * (dt - tau)) * b_in * dtau
            phi = expm(self._a * dt)

            #use integral
            h_in = np.zeros((self.order, 1))
            h_out = np.zeros((self.order, 1))
            h_fixed = np.zeros((self.order, 1))
            for i in range(dt):
                h_in += expm(self._a * (dt - i - 1)) @ self._b_in
                h_out += expm(self._a * (dt - i - 1)) @ self._b_out
                h_fixed += expm(self._a * (dt - i - 1)) @ self._b_fixed

            #a_inv = np.linalg.pinv(self._a)
            #phi_min_i = phi - np.eye(self.order)
            #h_in = a_inv @ phi_min_i @ self._b_in
            #h_out = a_inv @ phi_min_i @ self._b_out

            self._system_matrices[str(dt)] = [phi, h_in, h_out, h_fixed]
        return self._system_matrices[str(dt)]

def _component_to_model(component: Component) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    #TODO: make part of class and set self parameters
    #TODO: rename to extract_components or so
    if isinstance(component, ElectricalEnergyStorage):
        # TODO: Extract parameters from component
        order = 1
        a = np.array([[0]])
        b_in = np.array([1 / 3600])
        b_out = np.array([1 / 3600])
        b_fixed = np.array([1 / 3600])

        return order, a, b_in, b_out, b_fixed

    else:
        raise ResourceException(f"Unsupported component type: {type(component)}")