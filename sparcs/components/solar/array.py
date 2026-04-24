# -*- coding: utf-8 -*-
"""
sparcs.components.solar.array
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This module provides the :class:`sparcs.SolarArray`, containing information about orientation
and datasheet parameters of a specific photovoltaic installation.

"""

from __future__ import annotations

import os
import re
from copy import deepcopy
from enum import Enum
from functools import partial
from math import ceil, radians, sin
from typing import Any, Dict, List, Mapping, Optional

import pvlib as pv
from pvlib import atmosphere
from pvlib.pvsystem import AbstractMount, FixedMount, SingleAxisTrackerMount

# noinspection PyProtectedMember
from pvlib.tools import _build_args, _build_kwargs
from pvlib.tracking import calc_cross_axis_tilt

import pandas as pd
from lories.components import Component
from lories.core import ConfigurationError, Configurations
from lories.typing import ContextArgument
from sparcs.components.solar.bifacial import irradiance, temperature
from sparcs.components.solar.db import ModuleDatabase


# noinspection SpellCheckingInspection
class SolarArray(Component, pv.pvsystem.Array):
    INCLUDES = ["rows", "mounting", "tracking", "transposition", "losses"]

    POWER_AC: str = "p_ac"
    POWER_DC: str = "p_dc"
    POWER_DC_FRONT: str = "p_dc_f"
    POWER_DC_BACK: str = "p_dc_b"

    CURRENT_SC: str = "i_sc"
    CURRENT_MP: str = "i_mp"

    VOLTAGE_OC: str = "v_oc"
    VOLTAGE_MP: str = "v_mp"

    TEMPERATURE_CELL: str = "t_cell"

    rows: Rows
    mount: AbstractMount

    albedo: float = 0.25

    _module_parametrized: bool = False
    module_parameters: dict = {}
    modules_per_string: int = 1
    strings: int = 1

    array_losses_parameters: dict = {}
    shading_losses_parameters: dict = {}
    transposition_model_parameters: dict = {}
    temperature_model_parameters: dict = {}

    def __init__(
        self,
        context: ContextArgument,
        configs: Configurations,
        mount: Optional[AbstractMount] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            context=context,
            configs=configs,
            mount=mount,
            array_losses_parameters={},  # avoid parameter inferring by passing not None
            temperature_model_parameters={},  # avoid parameter inferring by passing not None
            **kwargs,
        )
        self._configure(configs)

    def __repr__(self) -> str:
        return Component.__repr__(self)

    def __str__(self) -> str:
        return Component.__str__(self)

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: Optional[str]) -> None:
        if name is None:
            return
        self._name = name

    @property
    def module_type(self) -> Optional[str]:
        if self._module_type is None:
            module_types = ["Front_type", "Back_type"]
            if all(t in self.module_parameters.keys() for t in module_types):
                if all(self.module_parameters[t].lower().startswith("glass") for t in module_types):
                    return "glass_glass"
                else:
                    return "glass_polymer"
        return self._module_type

    @module_type.setter
    def module_type(self, module_type) -> None:
        self._module_type = module_type

    def configure(self, configs: Configurations) -> None:
        super().configure(configs)
        # Initial configuration will happen in constructor.
        # Only update configs if already parametrized
        if self.is_configured():
            self._configure(configs)

    def _configure(self, configs: Configurations) -> None:
        self.surface_type = configs.get("surface_type", default=configs.get("ground_type", default=None))
        if "albedo" not in configs:
            self.albedo = pv.albedo.SURFACE_ALBEDOS.get(self.surface_type, SolarArray.albedo)
        else:
            self.albedo = configs.get_float("albedo")

        self.module_type = configs.get("module_type", default=configs.get("construct_type"))
        self.module = configs.get("module", default=None)

        self.module_parameters = self._infer_module_params(configs)
        self._module_parametrized = self._validate_module_params()
        if not self._module_parametrized:
            # raise ConfigurationException("Unable to configure module parameters")
            self._logger.debug("Unable to configure module parameters of array ", self.name)
            return

        self.rows = self._create_rows(configs, self.module_parameters)
        self.mount = self._create_mount(configs, self.rows)

        self.strings = configs.get_int("strings", default=self.rows.strings)
        self.modules_per_string = configs.get_int("modules_per_string", default=self.rows.modules_per_string)

        self.array_losses_parameters = self._infer_array_losses_params(configs)
        self.shading_losses_parameters = self._infer_shading_losses_params(configs)
        self.transposition_model_parameters = self._infer_transposition_model_params(configs)
        self.temperature_model_parameters = self._infer_temperature_model_params(configs)

    def is_parametrized(self) -> bool:
        return self._module_parametrized

    def is_bifacial(self) -> bool:
        return (
            "module_bifaciality" in self.module_parameters.keys() and self.module_parameters["module_bifaciality"] > 0
        )

    def has_rows(self) -> bool:
        return self.rows.is_configured()

    @staticmethod
    def _create_rows(configs: Configurations, module_parameters: Dict[str, Any]) -> Optional[Rows]:
        rows = configs.get_member("rows", defaults={})

        modules_stacked = rows.get_int("stack", default=Rows.modules_stacked)
        module_stack_gap = rows.get_float("stack_gap", default=Rows.module_stack_gap)
        module_row_gap = rows.get_float("row_gap", default=Rows.module_row_gap)

        module_orientation = Orientation.from_str(configs.get("orientation", default="portrait"))
        if module_orientation == Orientation.PORTRAIT:
            surface_width = module_parameters["Width"] + module_row_gap
            surface_length = modules_stacked * module_parameters["Length"] + (modules_stacked - 1) * module_stack_gap
        elif module_orientation == Orientation.LANDSCAPE:
            surface_width = module_parameters["Length"] + module_row_gap
            surface_length = modules_stacked * module_parameters["Width"] + (modules_stacked - 1) * module_stack_gap
        else:
            raise ValueError(f"Invalid module orientation to calculate length: {str(module_orientation)}")

        module_transmission = rows.get_float("module_transmission", default=None)
        if module_transmission is None:
            module_gaps = module_row_gap + module_stack_gap * (modules_stacked - 1)
            module_area = surface_length * surface_width
            module_transmission = module_gaps / module_area

        return Rows(
            surface_width,
            surface_length,
            modules_stacked,
            module_stack_gap,
            module_row_gap,
            module_transmission,
            module_orientation,
            rows.get_int("modules", default=None),
            rows.get_int("count", default=None),
            rows.get_float("pitch", default=None),
        )

    @staticmethod
    def _create_mount(configs: Configurations, rows: Rows) -> AbstractMount:
        mounting = configs.get_member("mounting", defaults={})

        def _get_module_height(tilt: float) -> Optional[float]:
            if "module_height" in mounting:
                return mounting["module_height"]
            elif "module_height_clearance" in mounting:
                center_offset = rows.surface_length / 2 * sin(radians(tilt))
                module_height_clearance = mounting.get_float("module_height_clearance")
                return module_height_clearance + center_offset
            else:
                return None

        if configs.has_member("tracking") and configs.get_member("tracking").enabled:
            tracking = configs.get_member("tracking")

            if rows.pitch is None:
                gcr = SingleAxisTrackerMount.gcr
            else:
                gcr = rows.surface_length / rows.pitch

            axis_azimuth = mounting.get_float("module_azimuth", default=SingleAxisTrackerMount.axis_azimuth)
            axis_tilt = mounting.get_float("module_tilt", default=SingleAxisTrackerMount.axis_tilt)
            max_angle = tracking.get_float("max_angle", default=SingleAxisTrackerMount.max_angle)

            if "slope_azimuth" in tracking and "slope_tilt" in tracking:
                cross_axis_tilt = calc_cross_axis_tilt(
                    tracking.get_float("slope_azimuth"),
                    tracking.get_float("slope_tilt"),
                    axis_azimuth,
                    axis_tilt,
                )
            else:
                cross_axis_tilt = SingleAxisTrackerMount.cross_axis_tilt

            return SingleAxisTrackerMount(
                axis_azimuth=axis_azimuth,
                axis_tilt=axis_tilt,
                max_angle=max_angle,
                backtrack=tracking.get("backtrack", default=SingleAxisTrackerMount.backtrack),
                gcr=tracking.get_float("ground_coverage_ratio", default=gcr),
                cross_axis_tilt=tracking.get_float("cross_axis_tilt", default=cross_axis_tilt),
                racking_model=mounting.get("racking_model", default=SingleAxisTrackerMount.racking_model),
                module_height=_get_module_height(max_angle),
            )
        else:
            surface_tilt = mounting.get_float("module_tilt", default=FixedMount.surface_tilt)
            return FixedMount(
                surface_azimuth=mounting.get_float("module_azimuth", default=FixedMount.surface_azimuth),
                surface_tilt=surface_tilt,
                racking_model=mounting.get("racking_model", default=FixedMount.racking_model),
                module_height=_get_module_height(surface_tilt),
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
            self._logger.debug(f"Denormalized %/°C temperature coefficient {key}: ")
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

        if "module_bifaciality" not in self.module_parameters.keys():
            if "Bifaciality" in self.module_parameters.keys():
                self.module_parameters["module_bifaciality"] = self.module_parameters["Bifaciality"]
                del self.module_parameters["Bifaciality"]
            else:
                self.module_parameters["module_bifaciality"] = 0
        if self.module_parameters["module_bifaciality"] > 0:
            if "rho_front" not in self.module_parameters.keys() and "Front_type" in self.module_parameters.keys():
                if self.module_parameters["Front_type"].lower() == "glass":
                    self.module_parameters["rho_front"] = 0.3
                elif self.module_parameters["Front_type"].lower() in ["glass-ar", "glass_ar"]:
                    self.module_parameters["rho_front"] = 0.1

            if "rho_back" not in self.module_parameters.keys() and "Back_type" in self.module_parameters.keys():
                if self.module_parameters["Back_type"].lower() == "glass":
                    self.module_parameters["rho_back"] = 0.5
                elif self.module_parameters["Back_type"].lower() in ["glass-ar", "glass_ar"]:
                    self.module_parameters["rho_back"] = 0.3

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
        if configs.has_member("module"):
            module_params = dict(configs["module"])
            _update_parameters(params, module_params)
            self._logger.debug("Extracted module from member configuration")
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

        module_path = os.path.join(configs.dirs.conf, module_file)
        if module_path != str(configs.path) and os.path.isfile(module_path):
            _update_parameters(params, Configurations.load(module_file, **configs.dirs.to_dict()))
            self._logger.debug("Read module file: %s", module_file)
            return True
        return False

    @staticmethod
    def _read_temperature_model_params(configs: Configurations) -> Optional[Dict[str, Any]]:
        params = {}
        if configs.has_member("losses"):
            temperature_model_keys = ["u_c", "u_v"]
            for key, value in configs["losses"].items():
                if key in temperature_model_keys:
                    params[key] = float(value)

        return params

    # noinspection PyProtectedMember, PyUnresolvedReferences
    def _infer_temperature_model_params(self, configs: Optional[Configurations] = None) -> Dict[str, Any]:
        params = {}

        if configs is not None:
            params.update(self._read_temperature_model_params(configs))
            if len(params) > 0:
                self._logger.debug("Extracted temperature model parameters from config file")
                return params

        # Try to infer temperature model parameters from the racking_model and module_type
        # params = super()._infer_temperature_model_params()
        if self.mount is not None:
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

    def _infer_transposition_model_params(self, configs: Configurations) -> Optional[Dict[str, Any]]:
        transposition = configs.get_member("transposition", defaults={})
        transposition_file = os.path.join(configs.dirs.conf, self.key.replace("array", "transposition") + ".conf")
        if not os.path.isfile(transposition_file):
            transposition_file = os.path.join(configs.dirs.conf, "transposition.conf")
        if os.path.isfile(transposition_file):
            transposition.update(Configurations.load(transposition_file, **configs.dirs.to_dict()))

        rows = min(transposition.get_int("rows", default=7), self.rows.count)
        row_modules = transposition.get_int("row_modules", default=21)
        return {
            "rows": rows,
            "row_index": transposition.get_int("row_index", default=int(ceil(rows / 2))),
            "row_modules": row_modules,
            "row_module_index": transposition.get_int("row_module_index", default=int(ceil(row_modules / 2))),
            "rows_mesh": transposition.get_int("rows_mesh", default=9),
        }

    def _infer_shading_losses_params(self, configs: Configurations) -> Optional[Dict[str, Any]]:
        shading = {}
        shading_file = os.path.join(configs.dirs.conf, self.key.replace("array", "shading") + ".conf")
        if not os.path.isfile(shading_file):
            shading_file = os.path.join(configs.dirs.conf, "shading.conf")
        if os.path.isfile(shading_file):
            shading = Configurations.load(shading_file, **configs.dirs.to_dict())
        return shading

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
            if "mismatch_bifaciality" in losses_configs:
                params["mismatch_bifaciality"] = float(losses_configs.pop("mismatch_bifaciality"))
            if "dc_ohmic_percent" in losses_configs:
                params["dc_ohmic_percent"] = float(losses_configs.pop("dc_ohmic_percent"))

            # Remove temperature model losses before verifying unknown parameters
            for param in ["u_c", "u_v"]:
                losses_configs.pop(param, None)

            if len(losses_configs) > 0:
                raise ConfigurationError(f"Unknown losses parameters: {', '.join(losses_configs.keys())}")
        return params

    # noinspection PyProtectedMember, PyUnresolvedReferences
    def _infer_array_losses_params(self, configs: Configurations) -> dict:
        params = self._read_array_losses_params(configs)
        if len(params) > 0:
            self._logger.debug("Extracted array losses model parameters from config file")

        return params

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

    # noinspection PyUnresolvedReferences
    def run_irradiance_model(
        self,
        solar_zenith,
        solar_azimuth,
        solar_elevation,
        dni,
        ghi,
        dhi,
        dni_extra=None,
        airmass=None,
        albedo=None,
        model="viewfactor",
        **kwargs,
    ):
        """
        Get plane of array irradiance components.

        Uses :py:func:`pvlib.irradiance.get_total_irradiance` to
        calculate the plane of array irradiance components for a surface
        defined by ``self.surface_tilt`` and ``self.surface_azimuth``.

        Parameters
        ----------
        solar_zenith : float or Series.
            Solar zenith angle.
        solar_azimuth : float or Series.
            Solar azimuth angle.
        solar_elevation : float or Series.
            Solar elevation angle.
        dni : float or Series
            Direct normal irradiance. [W/m2]
        ghi : float or Series. [W/m2]
            Global horizontal irradiance
        dhi : float or Series
            Diffuse horizontal irradiance. [W/m2]
        dni_extra : float or Series, optional
            Extraterrestrial direct normal irradiance. [W/m2]
        airmass : float or Series, optional
            Airmass. [unitless]
        albedo : float or Series, optional
            Ground surface albedo. [unitless]
        model : String, default 'haydavies'
            Irradiance model.

        kwargs
            Extra parameters passed to
            :py:func:`pvlib.irradiance.get_total_irradiance`.

        Returns
        -------
        poa_irradiance : DataFrame
            Column names are: ``'poa_global', 'poa_direct', 'poa_diffuse',
            'poa_sky_diffuse', 'poa_ground_diffuse'``.

        Notes
        -----
        Some sky irradiance models require ``dni_extra``. For these models,
        if ``dni_extra`` is not provided and ``solar_zenith`` has a
        ``DatetimeIndex``, then ``dni_extra`` is calculated.
        Otherwise, ``dni_extra=1367`` is assumed.

        See also
        --------
        :py:func:`pvlib.irradiance.get_total_irradiance`
        """
        if albedo is None:
            albedo = self.albedo

        if airmass is None:
            airmass = atmosphere.get_relative_airmass(solar_zenith)

        orientation = self.mount.get_orientation(solar_zenith, solar_azimuth)

        model = re.sub("[^A-Za-z0-9]+", "", model).lower()
        if model in ["solarfactor", "pvfactor"]:
            # return irradiance.get_pvfactors_irradiance()
            raise ValueError(f"Not yet implemented pvfactor bifacial irradiance model")
        elif model == "raytracing":
            # return irradiance.get_ray_tracing_irradiance()
            raise ValueError(f"Not yet implemented raytracing bifacial irradiance model")

        return super().get_irradiance(
            solar_zenith,
            solar_azimuth,
            dni,
            ghi,
            dhi,
            dni_extra=dni_extra,
            airmass=airmass,
            albedo=albedo,
            model=model,
            **kwargs,
        )

    def run_cell_temperature_model(
        self,
        poa_front,
        poa_back,
        temp_air,
        wind_speed,
        model,
        effective_irradiance=None,
        alpha_absorption=0.85,
    ):
        """
        Determine cell temperature using the method specified by ``model``.

        Parameters
        ----------
        poa_front : numeric
            Total front side irradiance in W/m^2.

        poa_back : numeric
            Total back side irradiance in W/m^2.

        temp_air : numeric
            Ambient dry bulb temperature [C]

        wind_speed : numeric
            Wind speed [m/s]

        model : str
            Supported models include ``'sapm'``, ``'pvsyst'``,
            ``'faiman'``, ``'fuentes'``, and ``'noct_sam'``

        effective_irradiance : numeric, optional
            The irradiance that is converted to photocurrent in W/m^2.
            Only used for some models.

        alpha_absorption : numeric, default 0.85
            Absorption coefficient. Parameter :math:`\alpha` in :eq:`pvsyst`.

        Returns
        -------
        numeric
            Values in degrees C.

        See Also
        --------
        pvlib.temperature.sapm_cell, pvlib.temperature.pvsyst_cell,
        pvlib.temperature.faiman, pvlib.temperature.fuentes,
        pvlib.temperature.noct_sam

        Notes
        -----
        Some temperature models have requirements for the input types;
        see the documentation of the underlying model function for details.
        """
        # convenience wrapper to avoid passing args 2 and 3 every call
        _build_tcell_args = partial(
            _build_args, input_dict=self.temperature_model_parameters, dict_name="temperature_model_parameters"
        )

        if model == "pvsyst" and self.module_parameters["module_bifaciality"] > 0:
            required = tuple()
            optional = {
                **_build_kwargs(["module_bifaciality", "module_efficiency", "alpha_absorption"], self.module_parameters)
            }
            if "alpha_absorption" not in optional:
                optional["alpha_absorption"] = alpha_absorption

            optional.update(**_build_kwargs(["u_c", "u_v"], self.temperature_model_parameters))

            return temperature.pvsyst_cell(poa_front, poa_back, temp_air, wind_speed, *required, **optional)

        return self.get_cell_temperature(
            poa_front + poa_back * self.module_parameters["module_bifaciality"],
            temp_air,
            wind_speed,
            model,
            effective_irradiance=effective_irradiance,
        )


class Orientation(Enum):
    PORTRAIT = "portrait"
    LANDSCAPE = "landscape"

    @classmethod
    def from_str(cls, s) -> Orientation:
        s = s.upper()
        if s == "PORTRAIT":
            return cls.PORTRAIT
        elif s == "LANDSCAPE":
            return cls.LANDSCAPE
        else:
            raise NotImplementedError


class Rows:
    surface_width: float
    surface_length: float

    modules_stacked: int = 1
    module_stack_gap: float = 0
    module_row_gap: float = 0
    module_transmission: float = 0
    module_orientation: Orientation = Orientation.PORTRAIT

    modules: Optional[int]
    count: Optional[int]
    pitch: Optional[float]

    def __init__(
        self,
        surface_width: float,
        surface_length: float,
        modules_stacked: int = 1,
        module_stack_gap: float = 0,
        module_row_gap: float = 0,
        module_transmission: float = 0,
        module_orientation: Orientation = Orientation.PORTRAIT,
        modules: Optional[int] = None,
        count: Optional[int] = None,
        pitch: Optional[float] = None,
    ) -> None:
        self.surface_width = surface_width
        self.surface_length = surface_length

        self.modules_stacked = modules_stacked
        self.module_stack_gap = module_stack_gap
        self.module_row_gap = module_row_gap
        self.module_transmission = module_transmission
        self.module_orientation = module_orientation

        self.modules = modules
        self.count = count
        self.pitch = pitch

    def is_configured(self) -> bool:
        return all(a is not None for a in [self.pitch, self.count, self.modules])

    @property
    def strings(self) -> int:
        return 1

    @property
    def modules_per_string(self) -> int:
        return self.modules if self.modules is not None else SolarArray.modules_per_string

    @property
    def gcr(self) -> float:
        return (self.surface_length / self.pitch) if self.pitch is not None and self.pitch > 0 else 0


def _update_parameters(parameters: Dict, update: Mapping):
    for key, value in update.items():
        try:
            parameters[key] = float(value)
        except ValueError:
            parameters[key] = value

    return parameters
