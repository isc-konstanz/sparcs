# -*- coding: utf-8 -*-
"""
    pvsys.evaluation
    ~~~~~~~~~~~~~~~~


"""
from __future__ import annotations
from typing import Dict
import os
import json
import logging
import pandas as pd
import calendar
import traceback

# noinspection PyProtectedMember
from corsys.io._var import COLUMNS
from corsys.io import DatabaseUnavailableException
from corsys import Configurations, Configurable, System
from scisys.io import excel, plot
from scisys import Results, Progress
from .pv import PVSystem

logger = logging.getLogger(__name__)

CMPTS = {
    'tes': 'Buffer Storage',
    'ees': 'Battery Storage',
    'ev': 'Electric Vehicle',
    'pv': 'Photovoltaics'
}

AC_Y = 'Specific yield [kWh/kWp]'
AC_E = 'Energy yield [kWh]'
DC_E = 'Energy yield (DC) [kWh]'


class Evaluation(Configurable):

    def __init__(self, system: System, name: str = 'Evaluation') -> None:
        super().__init__(system.configs)
        self.system = system
        self.name = name

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

        self._plots_dir = os.path.join(data_dir, 'plots')
        if not os.path.exists(self._plots_dir):
            os.makedirs(self._plots_dir)

    # noinspection PyProtectedMember
    def __call__(self, *args, **kwargs) -> Results:
        logger.info("Starting evaluation for system: %s", self.system.name)
        progress = Progress(desc=self.name, total=len(self.system.get_type(PVSystem.TYPE))*2+1, file=self._results_json)

        results = Results(self.system)
        results.durations.start('Evaluation')
        results_key = f"{self.system.id}/output"
        try:
            if results_key not in results:
                results.durations.start('Prediction')
                input = _get(results, f"{self.system.id}/input", self.system._get_input, *args, **kwargs)
                progress.update()

                result = pd.DataFrame()
                result.index.name = 'time'
                for cmpt in self.system.get_type(PVSystem.TYPE):
                    cmpt_key = f"{self.system.id}/{cmpt.id}/output"
                    result_pv = _get(results, cmpt_key, self.system._get_solar_yield, cmpt, input)
                    if result.empty:
                        result = result_pv
                    else:
                        result += result_pv
                    # result[f'{cmpt.id}_power'] = result_pv[PVSystem.POWER]

                    progress.update()

                result = pd.concat([result, input], axis=1)
                results.set(results_key, result, how='combine')
                results.durations.stop('Prediction')
            else:
                # If this component was simulated already, load the results and skip the calculation
                results.load(results_key)
                progress.complete()
            try:
                reference = _get(results, f"{self.system.id}/reference", self.system.database.read, **kwargs)

                # noinspection PyShadowingBuiltins, PyShadowingNames
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

                add_reference(PVSystem.TYPE)

            except DatabaseUnavailableException as e:
                reference = None
                logger.debug("Unable to retrieve reference values for system %s: %s", self.system.name, str(e))

            def prepare_data(data: pd.DataFrame) -> pd.DataFrame:
                data = data.tz_convert(self.system.location.timezone).tz_localize(None)
                data = data[[column for column in COLUMNS.keys() if column in data.columns]]
                data.rename(columns=COLUMNS, inplace=True)
                data.index.name = 'Time'
                return data

            hours = pd.Series(results.data.index, index=results.data.index)
            hours = (hours - hours.shift(1)).fillna(method='bfill').dt.total_seconds() / 3600.

            if PVSystem.ENERGY not in results.data and PVSystem.POWER in results.data:
                results.data[PVSystem.ENERGY] = results.data[PVSystem.POWER] / 1000. * hours
            if f'{PVSystem.ENERGY}_ref' not in results.data and f'{PVSystem.POWER}_ref' in results.data:
                results.data[f'{PVSystem.ENERGY}_ref'] = results.data[f'{PVSystem.POWER}_ref'] / 1000. * hours
                results.data[f'{PVSystem.ENERGY}_err'] = (results.data[PVSystem.ENERGY] -
                                                          results.data[f'{PVSystem.ENERGY}_ref'])

            if PVSystem.ENERGY_DC not in results.data and PVSystem.POWER_DC in results.data:
                results.data[PVSystem.ENERGY_DC] = results.data[PVSystem.POWER_DC] / 1000. * hours

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
                        results_name = f"{results_name.strip()} {results_suffix.strip()}".title()
                        summary_data[results_name] = prepare_data(results[cmpt_key])

            summary.to_csv(self._results_csv, encoding='utf-8-sig')
            excel.write(summary, summary_data, file=self._results_excel, index=False)
            progress.complete(summary_json)

        except Exception as e:
            logger.error("Error evaluating system %s: %s", self.system.name, str(e))
            logger.debug("%s: %s", type(e).__name__, traceback.format_exc())
            progress.complete({
                    'status': 'error',
                    'message': str(e),
                    'error': type(e).__name__,
                    'trace': traceback.format_exc()
                })
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
        results_kwp = 0
        for system in self.system.values():
            results_kwp += system.power_max / 1000.

        results[PVSystem.YIELD_SPECIFIC] = results[PVSystem.ENERGY] / results_kwp

        results = results.dropna(axis='index', how='all')

        plot_data = results[[PVSystem.ENERGY]].groupby(results.index.month).sum()
        plot.bar(x=plot_data.index, y=PVSystem.ENERGY, data=plot_data,
                 xlabel='Month', ylabel='Energy [kWh]', title='Monthly Yield',
                 colors=list(reversed(plot.COLORS)), file=os.path.join(self._plots_dir, 'yield_months.png'))

        plot_data = pd.concat([pd.Series(data=results.loc[results.index.month == m, PVSystem.POWER]/1000.,
                                         name=calendar.month_name[m]) for m in range(1, 13)], axis='columns')
        plot_data['hour'] = plot_data.index.hour + plot_data.index.minute/60.
        plot_melt = plot_data.melt(id_vars='hour', var_name='Months')
        plot.line(x='hour', y='value', data=plot_melt,
                  xlabel='Hour of the Day', ylabel='Power [kW]', title='Yield Profile', hue='Months',  # style='Months',
                  colors=list(reversed(plot.COLORS)), file=os.path.join(self._plots_dir, 'yield_months_profile.png'))

        yield_specific = round(results[PVSystem.YIELD_SPECIFIC].sum(), 2)
        yield_energy = round(results[PVSystem.ENERGY].sum(), 2)

        summary.loc[self.system.name, ('Yield', AC_Y)] = yield_specific
        summary.loc[self.system.name, ('Yield', AC_E)] = yield_energy

        summary_dict = {'yield_specific': yield_specific,
                        'yield_energy': yield_energy}

        if PVSystem.ENERGY_DC in results:
            dc_energy = round(results[PVSystem.ENERGY_DC].sum(), 2)

            summary.loc[self.system.name, ('Yield', DC_E)] = dc_energy
            summary_dict['yield_energy_dc'] = dc_energy

        return summary_dict

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
        result = func(*args, **kwargs)
        if results is not None:
            results.set(key, result)
        return result

    return results.get(key)
