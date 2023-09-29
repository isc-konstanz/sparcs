# -*- coding: utf-8 -*-
"""
    pvsys.pv.system
    ~~~~~~~~~~~~~~~
    
    This module provides the :class:`pvsys.pv.system.Photovoltaics`, containing information about orientation
    and datasheet parameters of a specific photovoltaic installation.
    
"""
from __future__ import annotations
from typing import Optional, Dict, List, Any

import os
import glob
import logging

import pandas as pd
import pvlib as pv
from pvlib.pvsystem import FixedMount, SingleAxisTrackerMount
from enum import Enum
from copy import deepcopy
from corsys.io import DatabaseException
from corsys.configs import Configurations, Configurable, ConfigurationException
from corsys.cmpt import Photovoltaic
from corsys import System
from . import ModuleDatabase, InverterDatabase

logger = logging.getLogger(__name__)


class PVSystem(Photovoltaic, pv.pvsystem.PVSystem):

    def __init__(self, system: System, configs: Configurations) -> None:
        super().__init__(system, configs, arrays=self.__arrays__(configs), name=configs.get('General', 'id'))

    def __configure__(self, configs: Configurations) -> None:
        super().__configure__(configs)

        self.modules_per_inverter = sum([array.modules_per_string * array.strings for array in self.arrays])
        self.inverters_per_system = configs.getfloat('Inverter', 'count', fallback=1)
        self.inverter_parameters = self._infer_inverter_params()
        self.inverter_parameters = self._fit_inverter_params()

        if all(['pdc0' in a.module_parameters for a in self.arrays]):
            self.power_max = round(sum([array.modules_per_string * array.strings * array.module_parameters['pdc0']
                                        for array in self.arrays])) * self.inverters_per_system

    @staticmethod
    def __arrays__(configs: Configurations) -> List[PVArray]:
        arrays = []
        array_dir = os.path.join(configs.dirs.conf,
                                 configs.get('General', 'id') + '.d')

        if 'Mounting' in configs.sections():
            # TODO: verify parameter availability in 'General' by keys
            array_configs = deepcopy(configs)
            array_configs.set(Configurations.GENERAL, 'override_dir', array_dir)
            array_configs.set(Configurations.GENERAL, 'id', 'array')
            array_override = os.path.join(array_dir, 'array.cfg')
            if os.path.isfile(array_override):
                array_configs.read(array_override)
            arrays.append(PVArray(array_configs))

        for array in glob.glob(os.path.join(array_dir, 'array*.cfg')):
            array_file = os.path.basename(array)
            array_id = array_file.rsplit('.', maxsplit=1)[0]
            if any([array_id == a.name for a in arrays]):
                continue

            array_configs = Configurations.from_configs(configs, conf_dir=array_dir, conf_file=array_file)
            array_configs.set(Configurations.GENERAL, 'override_dir', array_dir)
            array_configs.set(Configurations.GENERAL, 'id', array_id)
            arrays.append(PVArray(array_configs))

        return arrays

    @staticmethod
    def __array__(configs: Configurations) -> PVArray:
        return PVArray(configs)

    def _infer_inverter_params(self) -> dict:
        params = {}
        self._inverter_parameters_override = False
        if not self._read_inverter_params(params):
            self._read_inverter_database(params)

        inverter_params_exist = len(params) > 0
        if self._read_inverter_configs(params) and inverter_params_exist:
            self._inverter_parameters_override = True

        return params

    def _fit_inverter_params(self) -> dict:
        params = self.inverter_parameters

        if 'pdc0' not in params and all(['pdc0' in a.module_parameters for a in self.arrays]):
            params['pdc0'] = round(sum([array.modules_per_string * array.strings * array.module_parameters['pdc0']
                                        for array in self.arrays]))

        if 'eta_inv_nom' not in params and 'Efficiency' in params:
            if params['Efficiency'] > 1:
                params['Efficiency'] /= 100.0
                logger.debug("Inverter efficiency configured in percent and will be adjusted: ",
                             params['Efficiency'] * 100.)
            params['eta_inv_nom'] = params['Efficiency']

        return params

    def _read_inverter_params(self, params: dict) -> bool:
        if self.configs.has_section('Inverter'):
            module_params = dict({k: v for k, v in self.configs['Inverter'].items() if k not in ['count', 'model']})
            if len(module_params) > 0:
                _update_parameters(params, module_params)
                logger.debug('Extract inverter from config file')
                return True
        return False

    def _read_inverter_database(self, params: dict) -> bool:
        if self.inverter is not None:
            try:
                inverters = InverterDatabase(self.configs)
                inverter_params = inverters.read(self.inverter)
                _update_parameters(params, inverter_params)
            except DatabaseException:
                # TODO:
                pass
            logger.debug('Read inverter "%s" from database', self.inverter)
            return True
        return False

    def _read_inverter_configs(self, params: dict) -> bool:
        inverter_file = os.path.join(self.configs.dirs.conf,
                                     self.configs.get('General', 'id') + '.d', 'inverter.cfg')
        if os.path.exists(inverter_file):
            with open(inverter_file) as f:
                inverter_str = '[Inverter]\n' + f.read()

            from configparser import ConfigParser
            inverter_configs = ConfigParser()
            inverter_configs.optionxform = str
            inverter_configs.read_string(inverter_str)
            inverter_params = dict(inverter_configs['Inverter'])
            _update_parameters(params, inverter_params)
            logger.debug('Read inverter file: %s', inverter_file)
            return True
        return False

    # noinspection SpellCheckingInspection
    def pvwatts_losses(self, solar_position: pd.DataFrame):
        # noinspection SpellCheckingInspection
        def _pvwatts_losses(array: PVArray):
            return pv.pvsystem.pvwatts_losses(
                **array.pvwatts_losses(solar_position)
            )
        if self.num_arrays > 1:
            return tuple(_pvwatts_losses(array) for array in self.arrays)
        else:
            return _pvwatts_losses(self.arrays[0])

    @property
    def type(self) -> str:
        return 'pv'


class PVArray(Configurable, pv.pvsystem.Array):

    ROWS = 'Rows'

    row_pitch: Optional[float] = None

    module_stack_gap_height: float = 0
    module_stack_gap_width: float = 0
    module_stack_gap_ratio: Optional[float] = None

    def __init__(self, configs: Configurations) -> None:
        super().__init__(configs, mount=self.__mount__(configs), **self._infer_params(configs))

    def __configure__(self, configs: Configurations) -> None:
        super().__configure__(configs)
        self._override_dir = configs.get(Configurations.GENERAL, 'override_dir')
        self.module_parameters = self._infer_module_params()
        self.module_parameters = self._fit_module_params()

        self.modules_stacked = configs.getint(PVArray.ROWS, 'stack', fallback=1)
        self.module_stack_gap_height = configs.getfloat('Rows', 'gap_height', fallback=PVArray.module_stack_gap_height)
        self.module_stack_gap_width = configs.getfloat('Rows', 'gap_width', fallback=PVArray.module_stack_gap_width)
        self.module_stack_gap_ratio = configs.getfloat('Rows', 'gap_ratio', fallback=PVArray.module_stack_gap_ratio)

        self.module_orientation = Orientation[configs.get('Module', 'orientation', fallback='portrait').upper()]
        if self.module_orientation == Orientation.PORTRAIT:
            self.module_length = self.module_parameters['Length'] * self.modules_stacked

        elif self.module_orientation == Orientation.LANDSCAPE:
            self.module_length = self.module_parameters['Width'] * self.modules_stacked

        else:
            raise ValueError(f"Invalid module orientation to calculate length: {str(self.module_orientation)}")

        if self.modules_stacked > 1:
            self.module_length += self.module_stack_gap_height * (self.modules_stacked - 1)

            self.module_stack_gap_ratio = (self.module_stack_gap_height *
                                           (self.modules_stacked - 1) / self.module_length)
        else:
            self.module_stack_gap_ratio = 0

        self.row_pitch = configs.get(PVArray.ROWS, 'pitch', fallback=PVArray.row_pitch)
        if self.row_pitch and isinstance(self.mount, SingleAxisTrackerMount) and \
                self.mount.gcr == SingleAxisTrackerMount.gcr:
            self.mount.gcr = self.module_length / self.row_pitch

        self.temperature_model_parameters = self._infer_temperature_model_params()
        self.shading_losses_parameters = self._infer_shading_losses_params()
        self.array_losses_parameters = self._infer_array_losses_params()

    @staticmethod
    def __mount__(configs: Configurations) -> pv.pvsystem.AbstractMount:
        module_azimuth = configs.getfloat('Mounting', 'module_azimuth')
        module_tilt = configs.getfloat('Mounting', 'module_tilt')

        if configs.has_section('Tracking') and \
                configs.getboolean('Tracking', 'enabled', fallback=False):
            max_angle = configs.get('Tracking', 'max_angle', fallback=SingleAxisTrackerMount.max_angle)
            backtrack = configs.get('Tracking', 'backtrack', fallback=SingleAxisTrackerMount.backtrack)
            ground_coverage = configs.get('Tracking', 'ground_coverage', fallback=SingleAxisTrackerMount.gcr)

            cross_tilt = configs.get('Tracking', 'cross_axis_tilt', fallback=SingleAxisTrackerMount.cross_axis_tilt)
            # TODO: Implement cross_axis_tilt for sloped ground surface
            # if cross_tilt == SingleAxisTrackerMount.cross_axis_tilt:
            #     from pvlib.tracking import calc_cross_axis_tilt
            #     cross_tilt = calc_cross_axis_tilt(slope_azimuth, slope_tilt, axis_azimuth, axis_tilt)

            racking_model = configs.get('Mounting', 'racking_model', fallback=SingleAxisTrackerMount.racking_model)
            module_height = configs.get('Mounting', 'module_height', fallback=SingleAxisTrackerMount.module_height)

            return SingleAxisTrackerMount(axis_azimuth=module_azimuth,
                                          axis_tilt=module_tilt,
                                          max_angle=max_angle,
                                          backtrack=backtrack,
                                          gcr=ground_coverage,
                                          cross_axis_tilt=cross_tilt,
                                          racking_model=racking_model,
                                          module_height=module_height)
        else:
            return FixedMount(surface_azimuth=module_azimuth,
                              surface_tilt=module_tilt,
                              module_height=configs.get('Mounting', 'module_height', fallback=FixedMount.module_height),
                              racking_model=configs.get('Mounting', 'racking_model', fallback=FixedMount.racking_model))

    @staticmethod
    def _infer_params(configs: Configurations, **kwargs) -> Dict[str, Any]:
        params = {}

        def add_param(key: str, conv=None, alias: str = None) -> bool:
            val = configs.get('General', key, fallback=None)
            if val is None and alias is not None:
                val = configs.get('General', alias, fallback=None)
            if val is None:
                return False
            if conv is not None:
                val = conv(val)
            params[key] = val
            return True

        add_param('name', alias='id')

        if not add_param('albedo', conv=float):
            add_param('surface_type', alias='ground_type')

        add_param('module')
        add_param('module_type', alias='construct_type')
        add_param('modules_per_string', alias='count', conv=int)
        add_param('strings', conv=int)

        return params

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

    def _fit_module_params(self) -> dict:
        params = self.module_parameters

        def denormalize_coeff(key: str, ref: str) -> float:
            logger.debug(f"Denormalized %/C temperature coefficient {key}: ")
            return params[key] / 100 * params[ref]

        if 'pdc0' not in params and all(p in params for p in ['I_mp_ref', 'V_mp_ref']):
            params['pdc0'] = params['I_mp_ref'] \
                           * params['V_mp_ref']

        if 'Efficiency' not in params.keys():
            params['Efficiency'] = float(self.module_parameters['pdc0']) / \
                                   (float(self.module_parameters['Width']) *
                                    float(self.module_parameters['Length']) * 1000.0)
        elif params['Efficiency'] > 1:
            params['Efficiency'] /= 100.0
            logger.debug(f"Module efficiency configured in percent and will be adjusted: {params['Efficiency']*100.}")

        if 'T_NOCT' not in params.keys():
            params['T_NOCT'] = 45

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
                    params['gamma_pdc'] = params['gamma_mp'] / 100.
                else:
                    raise RuntimeError("Unable to estimate parameters due to incomplete variables")

                params.update({k: v for k, v in params_fit.items() if k in params_iv})

        except RuntimeError as e:
            logger.warning(str(e))

            if 'gamma_pdc' not in params and 'gamma_mp' in params:
                params['gamma_pdc'] = params['gamma_mp'] / 100.

        return params

    def _read_module_params(self, params: dict) -> bool:
        if self.configs.has_section('Module'):
            module_params = dict(self.configs['Module'])
            _update_parameters(params, module_params)
            logger.debug('Extract module from config file')
            return True
        return False

    def _read_module_database(self, params: dict) -> bool:
        if self.module is not None:
            try:
                modules = ModuleDatabase(self.configs)
                module_params = modules.read(self.module)
                _update_parameters(params, module_params)
            except DatabaseException:
                # TODO:
                pass
            logger.debug('Read module "%s" from database', self.module)
            return True
        return False

    def _read_module_configs(self, params: dict) -> bool:
        module_file = os.path.join(self._override_dir, self.name.replace('array', 'module') + '.cfg')
        if os.path.exists(module_file):
            with open(module_file) as f:
                module_str = '[Module]\n' + f.read()

            from configparser import ConfigParser
            module_configs = ConfigParser()
            module_configs.optionxform = str
            module_configs.read_string(module_str)
            module_params = dict(module_configs['Module'])
            _update_parameters(params, module_params)
            logger.debug('Read module file: %s', module_file)
            return True
        return False

    # TODO: Verify if default sapm model performance is good enough
    def _infer_temperature_model_params(self) -> dict:
        params = super()._infer_temperature_model_params()

        if len(self.module_parameters) > 0:
            if 'T_NOCT' in self.module_parameters.keys():
                params['noct'] = self.module_parameters['T_NOCT']

            if 'Efficiency' in self.module_parameters.keys():
                params['module_efficiency'] = self.module_parameters['Efficiency']

        return params

    def _infer_array_losses_params(self) -> dict:
        params = {}
        if 'Losses' in self.configs:
            losses_configs = dict(self.configs['Losses'])
            for param in ['soiling', 'shading', 'snow', 'mismatch',
                          'wiring', 'connections', 'lid', 'age',
                          'nameplate_rating', 'availability']:

                if param in losses_configs:
                    params[param] = float(losses_configs.pop(param))
            if 'dc_ohmic_percent' in losses_configs:
                params['dc_ohmic_percent'] = float(losses_configs.pop('dc_ohmic_percent'))
            if len(losses_configs) > 0:
                raise ConfigurationException(f"Unknown losses parameters for array {self.name}")
        return params

    def _infer_shading_losses_params(self) -> dict:
        shading = {}
        shading_file = os.path.join(self._override_dir, self.name.replace('array', 'shading') + '.cfg')
        if os.path.isfile(shading_file):
            shading_configs = Configurations(shading_file)
            for section in shading_configs.sections():
                shading[section] = dict(shading_configs.items(section))
        return shading

    # noinspection SpellCheckingInspection, PyProtectedMember
    def pvwatts_losses(self, solar_position: pd.DataFrame) -> dict:
        params = pv.tools._build_kwargs(['soiling', 'shading', 'snow', 'mismatch',
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


class Orientation(Enum):

    PORTRAIT = 'portrait'
    LANDSCAPE = 'landscape'


def _update_parameters(parameters: dict, update: dict):
    for key, value in update.items():
        try:
            parameters[key] = float(value)
        except ValueError:
            parameters[key] = value

    return parameters
