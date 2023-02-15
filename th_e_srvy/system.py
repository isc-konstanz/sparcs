# -*- coding: utf-8 -*-
"""
    th-e-srvy.system
    ~~~~~~~~~~~~~~~~
    
    
"""
from __future__ import annotations
from typing import Callable, Dict

import os
import json
import logging
import pandas as pd
import traceback
import th_e_core
from th_e_core import Component
from th_e_core.io import DatabaseUnavailableException
from th_e_core.configs import Configurations
from pvlib import solarposition
from .pv import PVSystem
from .model import Model
from .location import Location

logger = logging.getLogger(__name__)

CMPTS = {
    'tes': 'Buffer Storage',
    'ees': 'Battery Storage',
    'ev': 'Electric Vehicle',
    'pv': 'Photovoltaics'
}

AC_E = 'Energy yield [kWh]'
AC_Y = 'Specific yield [kWh/kWp]'


class System(th_e_core.System):

    def __configure__(self, configs):
        super().__configure__(configs)
        data_dir = configs.dirs.data
        if not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)

        self._results_json = os.path.join(data_dir, 'results.json')
        self._results_excel = os.path.join(data_dir, 'results.xlsx')
        self._results_csv = os.path.join(data_dir, 'results.csv')
        self._results_dir = os.path.join(data_dir, 'results')
        if not os.path.exists(self._results_dir):
            os.makedirs(self._results_dir)

    def __location__(self, configs: Configurations) -> Location:
        # FIXME: location necessary for for weather instantiation, but called afterwards here
        # if isinstance(self.weather, TMYWeather):
        #     return Location.from_tmy(self.weather.meta)
        # elif isinstance(self.weather, EPWWeather):
        #     return Location.from_epw(self.weather.meta)

        return Location(configs.getfloat('Location', 'latitude'),
                        configs.getfloat('Location', 'longitude'),
                        timezone=configs.get('Location', 'timezone', fallback='UTC'),
                        altitude=configs.getfloat('Location', 'altitude', fallback=None),
                        country=configs.get('Location', 'country', fallback=None),
                        state=configs.get('Location', 'state', fallback=None))

    def __cmpt_types__(self):
        return super().__cmpt_types__('solar', 'array')

    # noinspection PyShadowingBuiltins
    def __cmpt__(self, configs: Configurations, type: str) -> Component:
        if type in ['pv', 'solar', 'array']:
            return PVSystem(self, configs)

        return super().__cmpt__(configs, type)

    def __call__(self,
                 results=None,
                 results_json=None) -> pd.DataFrame:
        progress = Progress(len(self) + 1, file=results_json)

        weather = self._get_result(results, f"{self.id}/input", self._get_weather)
        progress.update()

        result = pd.DataFrame(columns=['pv_power', 'dc_power'], index=weather.index).fillna(0)
        result.index.name = 'time'
        for cmpt in self.values():
            if cmpt.type == 'pv':
                result_pv = self._get_result(results, f"{self.id}/{cmpt.id}/output", self._get_solar_yield, cmpt, weather)
                result[['pv_power', 'dc_power']] += result_pv[['pv_power', 'dc_power']].abs()

            progress.update()

        return pd.concat([result, weather], axis=1)

    # noinspection PyUnresolvedReferences
    @staticmethod
    def _get_result(results, key: str, func: Callable, *args, **kwargs) -> pd.DataFrame:
        from th_e_core.tools import to_bool
        if results is None or key not in results:
            concat = to_bool(kwargs.pop('concat', False))
            result = func(*args, **kwargs)
            if results is not None:
                results.set(key, result, concat=concat)
            return result

        return results.get(key)

    def _get_weather(self) -> pd.DataFrame:
        weather = self.weather.get()
        if 'precipitable_water' not in weather.columns or weather['precipitable_water'].sum() == 0:
            from pvlib.atmosphere import gueymard94_pw
            weather['precipitable_water'] = gueymard94_pw(weather['temp_air'], weather['relative_humidity'])
        if 'albedo' in weather.columns and weather['albedo'].sum() == 0:
            weather.drop('albedo', axis=1, inplace=True)

        solar_position = self._get_solar_position(weather.index)
        return pd.concat([weather, solar_position], axis=1)

    def _get_solar_position(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        data = pd.DataFrame(index=index)
        try:
            # TODO: use weather pressure for solar position
            data = solarposition.get_solarposition(index,
                                                   self.location.latitude,
                                                   self.location.longitude,
                                                   altitude=self.location.altitude)
            data = data.loc[:, ['azimuth', 'apparent_zenith', 'apparent_elevation']]
            data.columns = ['solar_azimuth', 'solar_zenith', 'solar_elevation']

        except ImportError as e:
            logger.warning("Unable to generate solar position: {}".format(str(e)))

        return data

    # noinspection PyMethodMayBeStatic
    def _get_solar_yield(self, pv: PVSystem, weather: pd.DataFrame) -> pd.DataFrame:
        model = Model.read(pv)
        return model(weather).rename(columns={'p_ac': 'pv_power',
                                              'p_dc': 'dc_power'})

    # noinspection PyProtectedMember
    def evaluate(self, **kwargs):
        from th_e_data import Results
        from th_e_core.io._var import COLUMNS
        from th_e_data.io import write_csv, write_excel
        logger.info("Starting evaluation for system: %s", self.name)

        results = Results(self)
        results.durations.start('Evaluation')
        results_key = f"{self.id}/output"
        try:
            if results_key not in results:
                results.durations.start('Prediction')
                result = self(results, self._results_json)
                results.set(results_key, result)
                results.durations.stop('Prediction')
            else:
                # If this component was simulated already, load the results and skip the calculation
                results.load(results_key)
            try:
                reference = self._get_result(results, f"{self.id}/reference", self.database.read, **kwargs)

                def add_reference(type: str, unit: str = 'power'):
                    cmpts = self.get_type(type)
                    if len(cmpts) > 0:
                        if all([f'{cmpt.id}_{unit}' in reference.columns for cmpt in cmpts]):
                            results.data[f'{type}_{unit}_ref'] = 0
                            for cmpt in self.get_type(f'{type}'):
                                results.data[f'{type}_{unit}_ref'] += reference[f'{cmpt.id}_{unit}']
                        elif f'{type}_{unit}' in reference.columns:
                            results.data[f'{type}_{unit}_ref'] = reference[f'{type}_{unit}']

                        results.data[f'{type}_{unit}_err'] = (results.data[f'{type}_{unit}'] -
                                                              results.data[f'{type}_{unit}_ref'])

                add_reference('pv')

            except DatabaseUnavailableException as e:
                reference = None
                logger.debug("Unable to retrieve reference values for system %s: %s", self.name, str(e))

            def prepare_data(data: pd.DataFrame) -> pd.DataFrame:
                data = data.tz_convert(self.location.timezone).tz_localize(None)
                data = data[[column for column in COLUMNS.keys() if column in data.columns]]
                data.rename(columns=COLUMNS, inplace=True)
                data.index.name = 'Time'
                return data

            summary = pd.DataFrame(columns=pd.MultiIndex.from_tuples((), names=['System', '']))
            summary_json = {
                'status': 'success'
            }
            self._evaluate(summary_json, summary, results.data, reference)

            summary_data = {
                self.name: prepare_data(results.data)
            }
            if len(self.system) > 1:
                for cmpt in self.system.values():
                    cmpt_key = f"{self.system.id}/{cmpt.id}/output"
                    if cmpt_key in results:
                        results_suffix = cmpt.name
                        for cmpt_type in self.system.get_types():
                            results_suffix = results_suffix.replace(cmpt_type, '')
                        if len(results_suffix) < 1 and len(self.system.get_type(cmpt.type)) > 1:
                            results_suffix += str(list(self.system.values()).index(cmpt) + 1)
                        results_name = CMPTS[cmpt.type] if cmpt.type in CMPTS else cmpt.type.upper()
                        results_name = f"{results_name} {results_suffix}".strip().title()
                        summary_data[results_name] = prepare_data(results[cmpt_key])

            write_csv(self, summary, self._results_csv)
            write_excel(summary, summary_data, file=self._results_excel)

            with open(self._results_json, 'w', encoding='utf-8') as f:
                json.dump(summary_json, f, ensure_ascii=False, indent=4)

        except Exception as e:
            logger.error("Error evaluating system %s: %s", self.name, str(e))
            logger.debug("%s: %s", type(e).__name__, traceback.format_exc())

            with open(self._results_json, 'w', encoding='utf-8') as f:
                results_json = {
                    'status': 'error',
                    'message': str(e),
                    'error': type(e).__name__,
                    'trace': traceback.format_exc()
                }
                json.dump(results_json, f, ensure_ascii=False, indent=4)

            raise e

        finally:
            results.durations.stop('Evaluation')
            results.close()

        logger.info("Evaluation complete")
        logger.debug('Evaluation complete in %i minutes', results.durations['Evaluation'])

        return results

    def _evaluate(self,
                  summary_json: Dict,
                  summary: pd.DataFrame,
                  results: pd.DataFrame,
                  reference: pd.DataFrame = None) -> None:
        summary_json.update(self._evaluate_yield(summary, results, reference))
        summary_json.update(self._evaluate_weather(summary, results))

    def _evaluate_yield(self, summary: pd.DataFrame, results: pd.DataFrame, reference: pd.DataFrame = None) -> Dict:
        hours = pd.Series(results.index, index=results.index)
        hours = (hours - hours.shift(1)).fillna(method='bfill').dt.total_seconds() / 3600.
        results_kwp = 0
        for system in self.values():
            results_kwp += system.power_max / 1000.

        results['pv_energy'] = results['pv_power'] / 1000. * hours
        results['pv_yield'] = results['pv_energy'] / results_kwp

        results['dc_energy'] = results['dc_power'] / 1000. * hours

        results.dropna(axis='index', how='all', inplace=True)

        yield_specific = round(results['pv_yield'].sum(), 2)
        yield_energy = round(results['pv_energy'].sum(), 2)

        summary.loc[self.name, ('Yield', AC_E)] = yield_energy
        summary.loc[self.name, ('Yield', AC_Y)] = yield_specific

        return {'yield_energy': yield_energy,
                'yield_specific': yield_specific}

    def _evaluate_weather(self, summary: pd.DataFrame, results: pd.DataFrame) -> Dict:
        hours = pd.Series(results.index, index=results.index)
        hours = (hours - hours.shift(1)).fillna(method='bfill').dt.total_seconds() / 3600.
        ghi = round((results['ghi'] / 1000. * hours).sum(), 2)
        dhi = round((results['dhi'] / 1000. * hours).sum(), 2)

        summary.loc[self.name, ('Weather', 'GHI [kWh/m^2]')] = ghi
        summary.loc[self.name, ('Weather', 'DHI [kWh/m^2]')] = dhi

        return {}


class Progress:

    def __init__(self, total, value=0, file=None):
        self._file = file
        self._total = total + 1
        self._value = value

    def update(self):
        self._value += 1
        self._update(self._value)

    def _update(self, value):
        progress = value / self._total * 100
        if progress % 1 <= 1 / self._total * 100 and self._file is not None:
            with open(self._file, 'w', encoding='utf-8') as f:
                results = {
                    'status': 'running',
                    'progress': int(progress)
                }
                json.dump(results, f, ensure_ascii=False, indent=4)

