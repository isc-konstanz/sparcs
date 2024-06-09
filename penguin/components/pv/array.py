# -*- coding: utf-8 -*-
"""
    penguin.components.pv.array
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~

    This module provides the :class:`penguin.PVArray`, containing information about orientation
    and datasheet parameters of a specific photovoltaic installation.

"""
from __future__ import annotations
from typing import Optional, Dict, Mapping, List, Any

import os

import pandas as pd
import pvlib as pv

# noinspection PyProtectedMember
from pvlib.tools import _build_kwargs
from pvlib.pvsystem import FixedMount, SingleAxisTrackerMount
from pvlib import temperature
from enum import Enum
from copy import deepcopy
from loris import Configurations, ConfigurationException
from loris.components import Component, ComponentContext
from penguin.components.pv.db import ModuleDatabase


# noinspection SpellCheckingInspection
class PVArray(pv.pvsystem.Array, Component):
    TYPE: str = "pv_array"

    row_pitch: Optional[float] = None

    module_stack_gap: float = 0
    module_row_gap: float = 0
    module_transmission: Optional[float] = None

    def __init__(self, context: ComponentContext, configs: Configurations) -> None:
        super(pv.pvsystem.Array, self).__init__(context, configs)

    def __configure__(self, configs: Configurations) -> None:
        super().__configure__(configs)
        self.mount = self._new_mount(configs)

        self.surface_type = configs.get('surface_type', default=configs.get("ground_type", default=None))
        if 'albedo' not in configs:
            self.albedo = pv.irradiance.SURFACE_ALBEDOS.get(self.surface_type, 0.25)
        else:
            self.albedo = configs.get_float("albedo")

        self.strings = configs.get_int('strings', default=configs.get_int('count', default=1))
        self.modules_per_string = configs.get_int("modules_per_string", default=1)

        self.module = configs.get('module', default=None)
        self.module_type = configs.get('module_type', default=configs.get('construct_type'))
        self.module_parameters = self._infer_module_params()
        self.module_parameters = self._fit_module_params()

        rows = configs.get_section('rows', default={})
        self.modules_stacked = rows.get_int('stack', default=1)
        self.module_stack_gap = rows.get_float('stack_gap', default=PVArray.module_stack_gap)
        self.module_row_gap = rows.get_float('row_gap', default=PVArray.module_row_gap)

        self.module_transmission = rows.get_float('module_transmission', default=PVArray.module_transmission)

        _module_orientation = configs.get('orientation', default='portrait').upper()
        self.module_orientation = Orientation[_module_orientation]
        if self.module_orientation == Orientation.PORTRAIT:
            self.module_width = self.module_parameters['Width'] + self.module_row_gap
            self.module_length = (self.module_parameters['Length'] * self.modules_stacked +
                                  self.module_stack_gap * (self.modules_stacked - 1))

        elif self.module_orientation == Orientation.LANDSCAPE:
            self.module_width = self.module_parameters['Length'] + self.module_row_gap
            self.module_length = (self.module_parameters['Width'] * self.modules_stacked +
                                  self.module_stack_gap * (self.modules_stacked - 1))
        else:
            raise ValueError(f"Invalid module orientation to calculate length: {str(self.module_orientation)}")

        if self.module_transmission is None:
            self.module_transmission = (self.module_row_gap + self.module_stack_gap * (self.modules_stacked - 1)) / \
                                       (self.module_length * self.module_width)

        self.row_pitch = rows.get_float('pitch', default=PVArray.row_pitch)
        if self.row_pitch and isinstance(self.mount, SingleAxisTrackerMount) and \
                self.mount.gcr == SingleAxisTrackerMount.gcr:
            self.mount.gcr = self.module_length / self.row_pitch

        self.array_losses_parameters = self._infer_array_losses_params()
        self.shading_losses_parameters = self._infer_shading_losses_params()
        self.temperature_model_parameters = self._infer_temperature_model_params()

    @staticmethod
    def _new_mount(configs: Configurations) -> pv.pvsystem.AbstractMount:
        mounting = configs.get_section('mounting')
        module_azimuth = mounting.get_float('module_azimuth')
        module_tilt = mounting.get_float('module_tilt')

        tracking = configs.get_section('tracking', default={'enabled': False})
        if tracking.enabled:
            max_angle = tracking.get_float('max_angle', default=SingleAxisTrackerMount.max_angle)
            backtrack = tracking.get('backtrack', default=SingleAxisTrackerMount.backtrack)
            ground_coverage = tracking.get_float('ground_coverage', default=SingleAxisTrackerMount.gcr)

            cross_tilt = tracking.get('cross_axis_tilt', default=SingleAxisTrackerMount.cross_axis_tilt)
            # TODO: Implement cross_axis_tilt for sloped ground surface
            # if cross_tilt == SingleAxisTrackerMount.cross_axis_tilt:
            #     from pvlib.tracking import calc_cross_axis_tilt
            #     cross_tilt = calc_cross_axis_tilt(slope_azimuth, slope_tilt, axis_azimuth, axis_tilt)

            racking_model = mounting.get('racking_model', default=SingleAxisTrackerMount.racking_model)
            module_height = mounting.get_float('module_height', default=SingleAxisTrackerMount.module_height)

            return SingleAxisTrackerMount(axis_azimuth=module_azimuth,
                                          axis_tilt=module_tilt,
                                          max_angle=max_angle,
                                          backtrack=backtrack,
                                          gcr=ground_coverage,
                                          cross_axis_tilt=cross_tilt,
                                          racking_model=racking_model,
                                          module_height=module_height)
        else:
            racking_model = mounting.get('racking_model', default=FixedMount.racking_model)
            module_height = mounting.get_float('module_height', default=FixedMount.module_height)

            return FixedMount(surface_azimuth=module_azimuth,
                              surface_tilt=module_tilt,
                              racking_model=racking_model,
                              module_height=module_height)

    def _infer_module_params(self) -> dict:
        params = {}
        self.module_parameters_override = False
        if not self._read_module_params(params):
            self._read_module_database(params)

        module_params_exist = len(params) > 0
        if self._read_module_configs(params) and module_params_exist:
            self.module_parameters_override = True

        module_params_exist = len(params) > 0
        if not module_params_exist:
            raise ConfigurationException("Unable to find module parameters")

        return params

    # noinspection PyTypeChecker
    def _fit_module_params(self) -> dict:
        params = self.module_parameters

        def denormalize_coeff(key: str, ref: str) -> float:
            self._logger.debug(f"Denormalized %/C temperature coefficient {key}: ")
            return params[key] / 100 * params[ref]

        if 'noct' not in params.keys():
            if 'T_NOCT' in params.keys():
                params['noct'] = params['T_NOCT']
                del params['T_NOCT']
            else:
                params['noct'] = 45

        if 'pdc0' not in params and all(p in params for p in ['I_mp_ref', 'V_mp_ref']):
            params['pdc0'] = params['I_mp_ref'] \
                           * params['V_mp_ref']

        if 'module_efficiency' not in params.keys():
            if 'Efficiency' in params.keys():
                params['module_efficiency'] = params['Efficiency']
                del params['Efficiency']
            else:
                params['module_efficiency'] = float(self.module_parameters['pdc0']) / \
                                              (float(self.module_parameters['Width']) *
                                               float(self.module_parameters['Length']) * 1000.0)
        if params['module_efficiency'] > 1:
            params['module_efficiency'] /= 100.0
            self._logger.debug("Module efficiency configured in percent and will be adjusted: "
                               f"{params['module_efficiency']*100.}")

        if 'module_transparency' not in params.keys():
            if 'Transparency' in params.keys():
                params['module_transparency'] = params['Transparency']
                del params['Transparency']
            else:
                params['module_transparency'] = 0
        if params['module_transparency'] > 1:
            params['module_transparency'] /= 100.0
            self._logger.debug("Module transparency configured in percent and will be adjusted: "
                               f"{params['module_transparency']*100.}")

        try:
            params_iv = ['I_L_ref', 'I_o_ref', 'R_s', 'R_sh_ref', 'a_ref']
            params_cec = ['Technology',
                          'V_mp_ref', 'I_mp_ref', 'V_oc_ref', 'I_sc_ref', 'alpha_sc', 'beta_oc', 'gamma_mp', 'N_s']
            params_desoto = ['V_mp_ref', 'I_mp_ref', 'V_oc_ref', 'I_sc_ref', 'alpha_sc', 'beta_oc', 'N_s']
            if self.module_parameters_override or not all(k in params.keys() for k in params_iv):

                def param_values(keys) -> List[float | int]:
                    params_slice = {k: params[k] for k in keys}
                    params_slice['alpha_sc'] = denormalize_coeff('alpha_sc', 'I_sc_ref')
                    params_slice['beta_oc'] = denormalize_coeff('beta_oc', 'V_oc_ref')

                    return list(params_slice.values())

                if all(k in params.keys() for k in params_cec):
                    params_iv.append('Adjust')
                    params_cec.remove('Technology')
                    params_fit_result = pv.ivtools.sdm.fit_cec_sam(self._infer_cell_type(), *param_values(params_cec))
                    params_fit = dict(zip(params_iv, params_fit_result))
                elif all(k in params.keys() for k in params_desoto):
                    params_fit, params_fit_result = pv.ivtools.sdm.fit_desoto(*param_values(params_desoto))
                elif 'gamma_pdc' not in params and 'gamma_mp' in params:
                    params_iv.append('gamma_pdc')
                    params_fit = {'gamma_pdc': params['gamma_mp'] / 100.}
                else:
                    raise RuntimeError("Unable to estimate parameters due to incomplete variables")

                params.update({k: v for k, v in params_fit.items() if k in params_iv})

        except RuntimeError as e:
            self._logger.warning(str(e))

            if 'gamma_pdc' not in params and 'gamma_mp' in params:
                params['gamma_pdc'] = params['gamma_mp'] / 100.

        return params

    def _read_module_params(self, params: dict) -> bool:
        if self.configs.has_section('Module'):
            module_params = dict(self.configs['Module'])
            _update_parameters(params, module_params)
            self._logger.debug('Extracted module from config file')
            return True
        return False

    def _read_module_database(self, params: dict) -> bool:
        if self.module is not None:
            try:
                modules = ModuleDatabase(self.configs)
                module_params = modules.read(self.module)
                _update_parameters(params, module_params)
            except IOError as e:
                self._logger.warning(f"Error reading module '{self.module}' from database: ", str(e))
                return False
            self._logger.debug(f"Read module '{self.module}' from database")
            return True
        return False

    def _read_module_configs(self, params: dict) -> bool:
        module_file = os.path.join(self.configs.dirs.conf, self.name.replace('array', 'module') + '.conf')
        if not os.path.isfile(module_file):
            module_file = os.path.join(self.configs.dirs.conf, 'module.conf')
        if os.path.exists(module_file):
            _update_parameters(params, Configurations.load(module_file, **self.configs.dirs.encode()))
            self._logger.debug('Read module file: %s', module_file)
            return True
        return False

    @staticmethod
    def _read_temperature_model_params(configs: Configurations) -> Optional[Dict[str, Any]]:
        params = {}
        if configs.has_section('Losses'):
            temperature_model_keys = [
                'u_c',
                'u_v'
            ]
            for key, value in configs['Losses'].items():
                if key in temperature_model_keys:
                    params[key] = float(value)

        return params

    # noinspection PyProtectedMember, PyUnresolvedReferences
    def _infer_temperature_model_params(self) -> dict:
        params = self._read_temperature_model_params(self.configs)
        if len(params) > 0:
            self._logger.debug('Extracted temperature model parameters from config file')
            return params

        # try to infer temperature model parameters from from racking_model
        # and module_type
        # params = super()._infer_temperature_model_params()
        if self.mount.racking_model is not None:
            param_set = self.mount.racking_model.lower()
            if param_set in ['open_rack', 'close_mount', 'insulated_back']:
                param_set += f'_{self.module_type}'
            if param_set in temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']:
                params.update(temperature._temperature_model_params('sapm', param_set))
            elif 'freestanding' in param_set:
                params.update(temperature._temperature_model_params('pvsyst',
                                                                    'freestanding'))
            elif 'insulated' in param_set:  # after SAPM to avoid confusing keys
                params.update(temperature._temperature_model_params('pvsyst',
                                                                    'insulated'))
        if len(params) == 0 and len(self.module_parameters) > 0:
            if 'noct' in self.module_parameters.keys():
                params['noct'] = self.module_parameters['noct']

            if 'module_efficiency' in self.module_parameters.keys():
                params['module_efficiency'] = self.module_parameters['module_efficiency']

        return params

    @staticmethod
    def _read_array_losses_params(configs: Configurations) -> Optional[Dict[str, Any]]:
        params = {}
        if 'losses' in configs:
            losses_configs = dict(configs['Losses'])
            for param in ['soiling', 'shading', 'snow', 'mismatch',
                          'wiring', 'connections', 'lid', 'age',
                          'nameplate_rating', 'availability']:

                if param in losses_configs:
                    params[param] = float(losses_configs.pop(param))
            if 'dc_ohmic_percent' in losses_configs:
                params['dc_ohmic_percent'] = float(losses_configs.pop('dc_ohmic_percent'))

            # Remove temperature model losses before verifying unknown parameters
            for param in ['u_c', 'u_v']:
                losses_configs.pop(param, None)

            if len(losses_configs) > 0:
                raise ConfigurationException(f"Unknown losses parameters: {', '.join(losses_configs.keys())}")
        return params

    # noinspection PyProtectedMember, PyUnresolvedReferences
    def _infer_array_losses_params(self) -> dict:
        params = self._read_array_losses_params(self.configs)
        if len(params) > 0:
            self._logger.debug('Extracted array losses model parameters from config file')

        return params

    def _infer_shading_losses_params(self) -> Optional[Dict[str, Any]]:
        shading = {}
        shading_file = os.path.join(self.configs.dirs.conf, self.name.replace('array', 'shading') + '.conf')
        if not os.path.isfile(shading_file):
            shading_file = os.path.join(self.configs.dirs.conf, 'shading.conf')
        if os.path.isfile(shading_file):
            shading = Configurations.load(shading_file, **self.configs.dirs.encode())
        return shading

    def pvwatts_losses(self, solar_position: pd.DataFrame) -> dict:
        params = _build_kwargs(['soiling', 'shading', 'snow', 'mismatch',
                                'wiring', 'connections', 'lid',
                                'nameplate_rating', 'age', 'availability'],
                               self.array_losses_parameters)
        if 'shading' not in params:
            shading_losses = self.shading_losses(solar_position)
            if not (shading_losses.empty or
                    shading_losses.isna().any()):
                params['shading'] = shading_losses
        return params

    def shading_losses(self, solar_position) -> pd.Series:
        shading_losses = deepcopy(solar_position)
        for loss, shading in self.shading_losses_parameters.items():
            shading_loss = shading_losses[shading['column']]
            if 'condition' in shading:
                shading_loss = shading_loss[shading_losses.query(shading['condition']).index]

            shading_none = float(shading['none'])
            shading_full = float(shading['full'])
            if shading_none > shading_full:
                shading_loss = (1. - (shading_loss - shading_full)/(shading_none - shading_full))*100
                shading_loss[shading_losses[shading['column']] > shading_none] = 0
                shading_loss[shading_losses[shading['column']] < shading_full] = 100
            else:
                shading_loss = (shading_loss - shading_none)/(shading_full - shading_none)*100
                shading_loss[shading_losses[shading['column']] < shading_none] = 0
                shading_loss[shading_losses[shading['column']] > shading_full] = 100

            shading_losses[loss] = shading_loss
        shading_losses = shading_losses.fillna(0)[self.shading_losses_parameters.keys()].max(axis=1)
        shading_losses.name = 'shading'
        return shading_losses

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        if name is None:
            return
        self._name = name

    def get_type(self) -> str:
        return self.TYPE


class Orientation(Enum):

    PORTRAIT = 'portrait'
    LANDSCAPE = 'landscape'


def _update_parameters(parameters: Dict, update: Mapping):
    for key, value in update.items():
        try:
            parameters[key] = float(value)
        except ValueError:
            parameters[key] = value

    return parameters
