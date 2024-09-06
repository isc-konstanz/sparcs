# -*- coding: utf-8 -*-
"""
penguin.components.solar.array
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This module provides the :class:`penguin.SolarArray`, containing information about orientation
and datasheet parameters of a specific photovoltaic installation.

"""

from __future__ import annotations

import os
import re
from copy import deepcopy
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional

import pvlib as pv
from pvlib import temperature
from pvlib.pvsystem import FixedMount, SingleAxisTrackerMount

# noinspection PyProtectedMember
from pvlib.tools import _build_kwargs

import pandas as pd
from loris import ConfigurationException, Configurations, Context
from loris.components import Component, register_component_type
from penguin.components.solar.db import ModuleDatabase


class Orientation(Enum):
    PORTRAIT = "portrait"
    LANDSCAPE = "landscape"


# noinspection SpellCheckingInspection
@register_component_type
class SolarArray(pv.pvsystem.Array, Component):
    TYPE: str = "pv_array"

    SECTIONS = ["rows", "mounting", "tracking"]

    POWER_AC: str = "p_ac"
    POWER_DC: str = "p_dc"

    CURRENT_SC: str = "i_sc"
    CURRENT_MP: str = "i_mp"

    VOLTAGE_OC: str = "v_oc"
    VOLTAGE_MP: str = "v_mp"

    mount: pv.pvsystem.AbstractMount

    albedo: float = 0.25

    _module_configured: bool = False
    modules_stacked: int = 1
    module_stack_gap: float = 0
    module_row_gap: float = 0
    module_transmission: Optional[float] = None
    module_orientation: Orientation = Orientation.PORTRAIT
    module_width: Optional[int] = None
    module_length: Optional[int] = None
    module_parameters: dict = {}
    modules_per_string: int = 1
    strings: int = 1

    row_pitch: Optional[float] = None

    array_losses_parameters: dict = {}
    shading_losses_parameters: dict = {}
    temperature_model_parameters: dict = {}

    def __init__(  # noqa
        self,
        context: Component | Context,
        configs: Optional[Configurations] = None,
        key: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        super(pv.pvsystem.Array, self).__init__(context, configs=configs, key=key, name=name)

    def __repr__(self) -> str:
        return Component.__repr__(self)

    def __str__(self) -> str:
        return Component.__str__(self)

    def is_configured(self) -> bool:
        return self._module_configured and super().is_configured()

    def configure(self, configs: Configurations) -> None:
        super().configure(configs)
        self.mount = self._new_mount(configs)

        self.surface_type = configs.get("surface_type", default=configs.get("ground_type", default=None))
        if "albedo" not in configs:
            self.albedo = pv.albedo.SURFACE_ALBEDOS.get(self.surface_type, SolarArray.albedo)
        else:
            self.albedo = configs.get_float("albedo")

        self.strings = configs.get_int("strings", default=configs.get_int("count", default=SolarArray.strings))
        self.modules_per_string = configs.get_int("modules_per_string", default=SolarArray.modules_per_string)
        self.module_type = configs.get("module_type", default=configs.get("construct_type"))
        self.module = configs.get("module", default=None)

        self.module_parameters = self._infer_module_params(configs)
        self._module_configured = self._validate_module_params()
        if not self._module_configured:
            # raise ConfigurationException("Unable to configure module parameters")
            self._logger.debug("Unable to configure module parameters of array ", self.name)
            return

        rows = configs.get_section("rows", defaults={})
        self.modules_stacked = rows.get_int("stack", default=SolarArray.modules_stacked)
        self.module_stack_gap = rows.get_float("stack_gap", default=SolarArray.module_stack_gap)
        self.module_row_gap = rows.get_float("row_gap", default=SolarArray.module_row_gap)

        self.module_transmission = rows.get_float("module_transmission", default=SolarArray.module_transmission)

        _module_orientation = configs.get("orientation", default="portrait").upper()
        self.module_orientation = Orientation[_module_orientation]
        if self.module_orientation == Orientation.PORTRAIT:
            self.module_width = self.module_parameters["Width"] + self.module_row_gap
            self.module_length = (
                self.modules_stacked * self.module_parameters["Length"]
                + (self.modules_stacked - 1) * self.module_stack_gap
            )
        elif self.module_orientation == Orientation.LANDSCAPE:
            self.module_width = self.module_parameters["Length"] + self.module_row_gap
            self.module_length = (
                self.modules_stacked * self.module_parameters["Width"]
                + (self.modules_stacked - 1) * self.module_stack_gap
            )
        else:
            raise ValueError(f"Invalid module orientation to calculate length: {str(self.module_orientation)}")

        if self.module_transmission is None:
            module_gaps = self.module_row_gap + self.module_stack_gap * (self.modules_stacked - 1)
            module_area = self.module_length * self.module_width
            self.module_transmission = module_gaps / module_area

        self.row_pitch = rows.get_float("pitch", default=SolarArray.row_pitch)
        if (
            self.row_pitch
            and isinstance(self.mount, SingleAxisTrackerMount)
            and self.mount.gcr == SingleAxisTrackerMount.gcr
        ):
            self.mount.gcr = self.module_length / self.row_pitch

        self.array_losses_parameters = self._infer_array_losses_params(configs)
        self.shading_losses_parameters = self._infer_shading_losses_params(configs)
        self.temperature_model_parameters = self._infer_temperature_model_params(configs)

    @staticmethod
    def _new_mount(configs: Configurations) -> pv.pvsystem.AbstractMount:
        mounting = configs.get_section("mounting", defaults={})
        if configs.has_section("tracking") and configs.get_section("tracking").enabled:
            tracking = configs.get_section("tracking")

            cross_tilt = tracking.get("cross_axis_tilt", default=SingleAxisTrackerMount.cross_axis_tilt)
            # TODO: Implement cross_axis_tilt for sloped ground surface
            # if cross_tilt == SingleAxisTrackerMount.cross_axis_tilt:
            #     from pvlib.tracking import calc_cross_axis_tilt
            #     cross_tilt = calc_cross_axis_tilt(slope_azimuth, slope_tilt, axis_azimuth, axis_tilt)

            return SingleAxisTrackerMount(
                axis_azimuth=mounting.get_float("module_azimuth", default=SingleAxisTrackerMount.axis_azimuth),
                axis_tilt=mounting.get_float("module_tilt", default=SingleAxisTrackerMount.axis_tilt),
                max_angle=tracking.get_float("max_angle", default=SingleAxisTrackerMount.max_angle),
                backtrack=tracking.get("backtrack", default=SingleAxisTrackerMount.backtrack),
                gcr=tracking.get_float("ground_coverage", default=SingleAxisTrackerMount.gcr),
                cross_axis_tilt=cross_tilt,
                racking_model=mounting.get("racking_model", default=SingleAxisTrackerMount.racking_model),
                module_height=mounting.get_float("module_height", default=SingleAxisTrackerMount.module_height),
            )
        else:
            return FixedMount(
                surface_azimuth=mounting.get_float("module_azimuth", default=FixedMount.surface_azimuth),
                surface_tilt=mounting.get_float("module_tilt", default=FixedMount.surface_tilt),
                racking_model=mounting.get("racking_model", default=FixedMount.racking_model),
                module_height=mounting.get_float("module_height", default=FixedMount.module_height),
            )

    def _infer_module_params(self, configs: Configurations) -> dict:
        params = {}
        self._module_parameters_override = False
        if not self._read_module_params(configs, params):
            self._read_module_database(configs, params)

        module_params_exist = len(params) > 0
        if self._read_module_configs(configs, params) and module_params_exist:
            self._module_parameters_override = True

        return params

    # noinspection PyTypeChecker
    def _validate_module_params(self) -> bool:
        if len(self.module_parameters) == 0:
            return False

        def denormalize_coeff(key: str, ref: str) -> float:
            self._logger.debug(f"Denormalized %/C temperature coefficient {key}: ")
            return self.module_parameters[key] / 100 * self.module_parameters[ref]

        if "noct" not in self.module_parameters.keys():
            if "T_NOCT" in self.module_parameters.keys():
                self.module_parameters["noct"] = self.module_parameters["T_NOCT"]
                del self.module_parameters["T_NOCT"]
            else:
                self.module_parameters["noct"] = 45

        if "pdc0" not in self.module_parameters:
            if all(p in self.module_parameters for p in ["I_mp_ref", "V_mp_ref"]):
                self.module_parameters["pdc0"] = self.module_parameters["I_mp_ref"] * self.module_parameters["V_mp_ref"]
            else:
                self.module_parameters["pdc0"] = 0

        if "module_efficiency" not in self.module_parameters.keys():
            if "Efficiency" in self.module_parameters.keys():
                self.module_parameters["module_efficiency"] = self.module_parameters["Efficiency"]
                del self.module_parameters["Efficiency"]
            elif all([k in self.module_parameters for k in ["pdc0", "Width", "Length"]]):
                self.module_parameters["module_efficiency"] = float(self.module_parameters["pdc0"]) / (
                    float(self.module_parameters["Width"]) * float(self.module_parameters["Length"]) * 1000.0
                )

        if self.module_parameters["module_efficiency"] > 1:
            self.module_parameters["module_efficiency"] /= 100.0
            self._logger.debug(
                "Module efficiency configured in percent and will be adjusted: "
                f"{self.module_parameters['module_efficiency']*100.}"
            )

        if "module_transparency" not in self.module_parameters.keys():
            if "Transparency" in self.module_parameters.keys():
                self.module_parameters["module_transparency"] = self.module_parameters["Transparency"]
                del self.module_parameters["Transparency"]
            else:
                self.module_parameters["module_transparency"] = 0
        if self.module_parameters["module_transparency"] > 1:
            self.module_parameters["module_transparency"] /= 100.0
            self._logger.debug(
                "Module transparency configured in percent and will be adjusted: "
                f"{self.module_parameters['module_transparency']*100.}"
            )

        try:
            params_iv = [
                "I_L_ref",
                "I_o_ref",
                "R_s",
                "R_sh_ref",
                "a_ref",
            ]
            params_cec = [
                "Technology",
                "V_mp_ref",
                "I_mp_ref",
                "V_oc_ref",
                "I_sc_ref",
                "alpha_sc",
                "beta_oc",
                "gamma_mp",
                "N_s",
            ]
            params_desoto = [
                "V_mp_ref",
                "I_mp_ref",
                "V_oc_ref",
                "I_sc_ref",
                "alpha_sc",
                "beta_oc",
                "N_s",
            ]
            if self._module_parameters_override or not all(k in self.module_parameters.keys() for k in params_iv):

                def param_values(keys) -> List[float | int]:
                    params_slice = {k: self.module_parameters[k] for k in keys}
                    params_slice["alpha_sc"] = denormalize_coeff("alpha_sc", "I_sc_ref")
                    params_slice["beta_oc"] = denormalize_coeff("beta_oc", "V_oc_ref")

                    return list(params_slice.values())

                if all(k in self.module_parameters.keys() for k in params_cec):
                    params_iv.append("Adjust")
                    params_cec.remove("Technology")
                    params_fit_result = pv.ivtools.sdm.fit_cec_sam(self._infer_cell_type(), *param_values(params_cec))
                    params_fit = dict(zip(params_iv, params_fit_result))
                elif all(k in self.module_parameters.keys() for k in params_desoto):
                    params_fit, params_fit_result = pv.ivtools.sdm.fit_desoto(*param_values(params_desoto))
                elif "gamma_pdc" not in self.module_parameters and "gamma_mp" in self.module_parameters:
                    params_iv.append("gamma_pdc")
                    params_fit = {"gamma_pdc": self.module_parameters["gamma_mp"] / 100.0}
                else:
                    raise RuntimeError("Unable to estimate parameters due to incomplete variables")

                self.module_parameters.update({k: v for k, v in params_fit.items() if k in params_iv})

        except RuntimeError as e:
            self._logger.warning(str(e))

            if "gamma_pdc" not in self.module_parameters and "gamma_mp" in self.module_parameters:
                self.module_parameters["gamma_pdc"] = self.module_parameters["gamma_mp"] / 100.0

        return True

    def _read_module_params(self, configs: Configurations, params: dict) -> bool:
        if configs.has_section("module"):
            module_params = dict(configs["module"])
            _update_parameters(params, module_params)
            self._logger.debug("Extracted module from config file")
            return True
        return False

    def _read_module_database(self, configs: Configurations, params: dict) -> bool:
        if self.module is not None:
            try:
                modules = ModuleDatabase(configs)
                module_params = modules.read(self.module)
                _update_parameters(params, module_params)
            except IOError as e:
                self._logger.warning(f"Error reading module '{self.module}' from database: ", str(e))
                return False
            self._logger.debug(f"Read module '{self.module}' from database")
            return True
        return False

    def _read_module_configs(self, configs: Configurations, params: dict) -> bool:
        module_file = self.key.replace(re.split(r"[^a-zA-Z0-9\s]", self.key)[0], "module") + ".conf"
        if not os.path.isfile(os.path.join(configs.dirs.conf, module_file)):
            module_file = "module.conf"
        if os.path.isfile(os.path.join(configs.dirs.conf, module_file)):
            _update_parameters(params, Configurations.load(module_file, **configs.dirs.to_dict()))
            self._logger.debug("Read module file: %s", module_file)
            return True
        return False

    @staticmethod
    def _read_temperature_model_params(configs: Configurations) -> Optional[Dict[str, Any]]:
        params = {}
        if configs.has_section("losses"):
            temperature_model_keys = ["u_c", "u_v"]
            for key, value in configs["losses"].items():
                if key in temperature_model_keys:
                    params[key] = float(value)

        return params

    # noinspection PyProtectedMember, PyUnresolvedReferences
    def _infer_temperature_model_params(self, configs: Configurations) -> dict:  # noqa
        params = self._read_temperature_model_params(configs)
        if len(params) > 0:
            self._logger.debug("Extracted temperature model parameters from config file")
            return params

        # try to infer temperature model parameters from from racking_model
        # and module_type
        # params = super()._infer_temperature_model_params()
        if self.mount.racking_model is not None:
            param_set = self.mount.racking_model.lower()
            if param_set in ["open_rack", "close_mount", "insulated_back"]:
                param_set += f"_{self.module_type}"
            if param_set in temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"]:
                params.update(temperature._temperature_model_params("sapm", param_set))
            elif "freestanding" in param_set:
                params.update(temperature._temperature_model_params("pvsyst", "freestanding"))
            elif "insulated" in param_set:  # after SAPM to avoid confusing keys
                params.update(temperature._temperature_model_params("pvsyst", "insulated"))
        if len(params) == 0 and len(self.module_parameters) > 0:
            if "noct" in self.module_parameters.keys():
                params["noct"] = self.module_parameters["noct"]

            if "module_efficiency" in self.module_parameters.keys():
                params["module_efficiency"] = self.module_parameters["module_efficiency"]

        return params

    @staticmethod
    def _read_array_losses_params(configs: Configurations) -> Optional[Dict[str, Any]]:
        params = {}
        if "losses" in configs:
            losses_configs = dict(configs["losses"])
            for param in [
                "soiling",
                "shading",
                "snow",
                "mismatch",
                "wiring",
                "connections",
                "lid",
                "age",
                "nameplate_rating",
                "availability",
            ]:
                if param in losses_configs:
                    params[param] = float(losses_configs.pop(param))
            if "dc_ohmic_percent" in losses_configs:
                params["dc_ohmic_percent"] = float(losses_configs.pop("dc_ohmic_percent"))

            # Remove temperature model losses before verifying unknown parameters
            for param in ["u_c", "u_v"]:
                losses_configs.pop(param, None)

            if len(losses_configs) > 0:
                raise ConfigurationException(f"Unknown losses parameters: {', '.join(losses_configs.keys())}")
        return params

    # noinspection PyProtectedMember, PyUnresolvedReferences
    def _infer_array_losses_params(self, configs: Configurations) -> dict:
        params = self._read_array_losses_params(configs)
        if len(params) > 0:
            self._logger.debug("Extracted array losses model parameters from config file")

        return params

    def _infer_shading_losses_params(self, configs: Configurations) -> Optional[Dict[str, Any]]:
        shading = {}
        shading_file = os.path.join(configs.dirs.conf, self.key.replace("array", "shading") + ".conf")
        if not os.path.isfile(shading_file):
            shading_file = os.path.join(configs.dirs.conf, "shading.conf")
        if os.path.isfile(shading_file):
            shading = Configurations.load(shading_file, **configs.dirs.to_dict())
        return shading

    def pvwatts_losses(self, solar_position: pd.DataFrame) -> dict:
        params = _build_kwargs(
            [
                "soiling",
                "shading",
                "snow",
                "mismatch",
                "wiring",
                "connections",
                "lid",
                "nameplate_rating",
                "age",
                "availability",
            ],
            self.array_losses_parameters,
        )
        if "shading" not in params:
            shading_losses = self.shading_losses(solar_position)
            if not (shading_losses.empty or shading_losses.isna().any()):
                params["shading"] = shading_losses
        return params

    def shading_losses(self, solar_position) -> pd.Series:
        shading_losses = deepcopy(solar_position)
        for loss, shading in self.shading_losses_parameters.items():
            shading_loss = shading_losses[shading["column"]]
            if "condition" in shading:
                shading_loss = shading_loss[shading_losses.query(shading["condition"]).index]

            shading_none = float(shading["none"])
            shading_full = float(shading["full"])
            if shading_none > shading_full:
                shading_loss = (1.0 - (shading_loss - shading_full) / (shading_none - shading_full)) * 100
                shading_loss[shading_losses[shading["column"]] > shading_none] = 0
                shading_loss[shading_losses[shading["column"]] < shading_full] = 100
            else:
                shading_loss = (shading_loss - shading_none) / (shading_full - shading_none) * 100
                shading_loss[shading_losses[shading["column"]] < shading_none] = 0
                shading_loss[shading_losses[shading["column"]] > shading_full] = 100

            shading_losses[loss] = shading_loss
        shading_losses = shading_losses.fillna(0)[self.shading_losses_parameters.keys()].max(axis=1)
        shading_losses.name = "shading"
        return shading_losses


def _update_parameters(parameters: Dict, update: Mapping):
    for key, value in update.items():
        try:
            parameters[key] = float(value)
        except ValueError:
            parameters[key] = value

    return parameters
