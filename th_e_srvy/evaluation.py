# -*- coding: utf-8 -*-
"""
    th-e-yield.evaluation
    ~~~~~~~~~~~~~~~~~


"""
from __future__ import annotations
from typing import Dict
import os
import json
import logging
import pandas as pd
import traceback

# noinspection PyProtectedMember
from th_e_core.io._var import COLUMNS
from th_e_core.io import DatabaseUnavailableException
from th_e_core.tools import to_bool
from th_e_core import Configurations, Configurable, System
from th_e_data.io import write_csv, write_excel
from th_e_data import Results

logger = logging.getLogger(__name__)

CMPTS = {
    'tes': 'Buffer Storage',
    'ees': 'Battery Storage',
    'ev': 'Electric Vehicle',
    'pv': 'Photovoltaics'
}

AC_E = 'Energy yield [kWh]'
AC_Y = 'Specific yield [kWh/kWp]'
DC_E = 'Energy yield (DC) [kWh]'


class Evaluation(Configurable):

    def __init__(self, system: System) -> None:
        super().__init__(system.configs)
        self.system = system

    def __configure__(self, configs: Configurations) -> None:
        super().__configure__(configs)
        data_dir = configs.dirs.data
        if not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)

        self._results_json = os.path.join(data_dir, 'results.json')
        self._results_excel = os.path.join(data_dir, 'results.xlsx')
        self._results_csv = os.path.join(data_dir, 'results.csv')
        self._results_pdf = os.path.join(data_dir, 'results.pdf')
        self._results_dir = os.path.join(data_dir, 'results')
        if not os.path.exists(self._results_dir):
            os.makedirs(self._results_dir)

    # noinspection PyProtectedMember
    def __call__(self, **kwargs) -> Results:
        logger.info("Starting evaluation for system: %s", self.system.name)
        progress = Progress(len(self.system) + 1, file=self._results_json)

        results = Results(self.system)
        results.durations.start('Evaluation')
        results_key = f"{self.system.id}/output"
        try:
            if results_key not in results:
                results.durations.start('Prediction')
                weather = _get(results, f"{self.system.id}/input", self.system._get_weather)
                progress.update()

                result = pd.DataFrame(columns=['pv_power', 'dc_power'], index=weather.index).fillna(0)
                result.index.name = 'time'
                for cmpt in self.system.values():
                    if cmpt.type == 'pv':
                        cmpt_key = f"{self.system.id}/{cmpt.id}/output"
                        result_pv = _get(results, cmpt_key, self.system._get_solar_yield, cmpt, weather)
                        result[['pv_power', 'dc_power']] += result_pv[['pv_power', 'dc_power']].abs()

                    progress.update()

                result = pd.concat([result, weather], axis=1)
                results.set(results_key, result)
                results.durations.stop('Prediction')
            else:
                # If this component was simulated already, load the results and skip the calculation
                results.load(results_key)
            try:
                reference = _get(results, f"{self.system.id}/reference", self.system.database.read, **kwargs)

                # noinspection SpellCheckingInspection, PyShadowingBuiltins, PyShadowingNames
                def add_reference(type: str, unit: str = 'power'):
                    cmpts = self.system.get_type(type)
                    if len(cmpts) > 0:
                        if all([f'{cmpt.id}_{unit}' in reference.columns for cmpt in cmpts]):
                            results.data[f'{type}_{unit}_ref'] = 0
                            for cmpt in self.system.get_type(f'{type}'):
                                results.data[f'{type}_{unit}_ref'] += reference[f'{cmpt.id}_{unit}']
                        elif f'{type}_{unit}' in reference.columns:
                            results.data[f'{type}_{unit}_ref'] = reference[f'{type}_{unit}']

                        results.data[f'{type}_{unit}_err'] = (results.data[f'{type}_{unit}'] -
                                                              results.data[f'{type}_{unit}_ref'])

                add_reference('pv')

            except DatabaseUnavailableException as e:
                reference = None
                logger.debug("Unable to retrieve reference values for system %s: %s", self.system.name, str(e))

            def prepare_data(data: pd.DataFrame) -> pd.DataFrame:
                data = data.tz_convert(self.system.location.timezone).tz_localize(None)
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
                self.system.name: prepare_data(results.data)
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

            write_csv(self.system, summary, self._results_csv)
            write_excel(summary, summary_data, file=self._results_excel)

            with open(self._results_json, 'w', encoding='utf-8') as f:
                json.dump(summary_json, f, ensure_ascii=False, indent=4)

        except Exception as e:
            logger.error("Error evaluating system %s: %s", self.system.name, str(e))
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
        for system in self.system.values():
            results_kwp += system.power_max / 1000.

        results['pv_energy'] = results['pv_power'] / 1000. * hours
        results['pv_yield'] = results['pv_energy'] / results_kwp

        results['dc_energy'] = results['dc_power'] / 1000. * hours

        results.dropna(axis='index', how='all', inplace=True)

        yield_specific = round(results['pv_yield'].sum(), 2)
        yield_energy = round(results['pv_energy'].sum(), 2)

        summary.loc[self.system.name, ('Yield', AC_E)] = yield_energy
        summary.loc[self.system.name, ('Yield', AC_Y)] = yield_specific

        return {'yield_energy': yield_energy,
                'yield_specific': yield_specific}

    def _evaluate_weather(self, summary: pd.DataFrame, results: pd.DataFrame) -> Dict:
        hours = pd.Series(results.index, index=results.index)
        hours = (hours - hours.shift(1)).fillna(method='bfill').dt.total_seconds() / 3600.
        ghi = round((results['ghi'] / 1000. * hours).sum(), 2)
        dhi = round((results['dhi'] / 1000. * hours).sum(), 2)

        summary.loc[self.system.name, ('Weather', 'GHI [kWh/m^2]')] = ghi
        summary.loc[self.system.name, ('Weather', 'DHI [kWh/m^2]')] = dhi

        return {}


# noinspection PyUnresolvedReferences
def _get(results, key: str, func: Callable, *args, **kwargs) -> pd.DataFrame:
    if results is None or key not in results:
        concat = to_bool(kwargs.pop('concat', False))
        result = func(*args, **kwargs)
        if results is not None:
            results.set(key, result, concat=concat)
        return result

    return results.get(key)


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
