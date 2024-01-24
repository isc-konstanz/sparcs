# -*- coding: utf-8 -*-
"""
    pvsys.evaluation
    ~~~~~~~~~~~~~~~~


"""
from __future__ import annotations
from typing import Dict, Tuple
import os
import logging
import numpy as np
import pandas as pd
import traceback

# noinspection PyProtectedMember
from corsys.io._var import COLUMNS
from corsys.io import DatabaseUnavailableException
from corsys import Configurations, Configurable, System
from corsys.weather import Weather
from scisys.io import excel, plot
from scisys import Results, Progress
from .pv import PVSystem

logger = logging.getLogger(__name__)

COLUMNS_DC = {
    PVSystem.POWER_DC: 'Generated PV DC Power [W]'
}

COLUMNS_IV = {
    PVSystem.CURRENT_DC_MP: 'Current at MPP (A)',
    PVSystem.VOLTAGE_DC_MP: 'Voltage at MPP (V)',
    PVSystem.CURRENT_DC_SC: 'Short-circuit current (A)',
    PVSystem.VOLTAGE_DC_OC: 'Open-circuit voltage (V)'
}

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

    PV_SPECIFIC_YIELD = PVSystem.YIELD_SPECIFIC
    PV_ENERGY_YIELD = f"{PVSystem.ENERGY}_yield"
    PV_ENERGY_DC_YIELD = f"{PVSystem.ENERGY_DC}_yield"
    PV_ENERGY_MONTHS = f"{PVSystem.ENERGY}_months"
    GHI_TOTAL = f"{Weather.GHI}_total"
    DHI_TOTAL = f"{Weather.DHI}_total"

    def __init__(self, system: System, name: str = 'Evaluation') -> None:
        super().__init__(system.configs)
        self._columns = self.__columns__()
        self.system = system
        self.name = name

    @staticmethod
    def __columns__() -> Dict[str, str]:
        return {**COLUMNS, **COLUMNS_DC, **COLUMNS_IV}

    def __configure__(self, configs: Configurations) -> None:
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

        self._plots_dir = os.path.join(data_dir, 'plots')
        if not os.path.exists(self._plots_dir):
            os.makedirs(self._plots_dir)

        self._targets = {
            PVSystem.TYPE: configs.get('References', PVSystem.TYPE, fallback=PVSystem.POWER)
        }

    # noinspection PyProtectedMember, PyTypeChecker, PyShadowingBuiltins
    def __call__(self, *args, **kwargs) -> Results:
        logger.debug("Starting evaluation for system: %s", self.system.name)
        progress = Progress.instance(desc=f"{self.name}: {self.system.name}",
                                     total=len(self.system)*2+1,
                                     file=self._results_json)

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
                        result = result_pv.loc[:, result_pv.columns.str.contains('pv.*_power')]
                    else:
                        result += result_pv.loc[:, result_pv.columns.str.contains('pv.*_power')]

                    progress.update()

                result = pd.concat([result, input], axis=1)
                results.set(results_key, result, how='combine')
                results.durations.stop('Prediction')
            else:
                # If this component was simulated already, load the results and skip the calculation
                results.load(results_key)
                progress.complete()
            try:
                references = _get(results, f"{self.system.id}/reference", self.system.database.read, **kwargs)

                # noinspection PyTypeChecker, PyShadowingBuiltins, PyShadowingNames
                def add_reference(cmpt_type: str, target: str):
                    cmpts = self.system.get_type(cmpt_type)
                    if len(cmpts) > 0:
                        if all([target.replace(cmpt_type, cmpt.id) in references.columns or
                                f'{target.replace(cmpt_type, cmpt.id)}_ref' in references.columns
                                for cmpt in cmpts]):
                            results.data[f'{cmpt_type}_ref'] = 0
                            for cmpt in self.system.get_type(f'{cmpt_type}'):
                                cmpt_target = target.replace(cmpt_type, cmpt.id)
                                cmpt_reference = (references[f'{cmpt_target}'] if cmpt_target in references.columns else
                                                  references[f'{cmpt_target}_ref'])
                                results.data[f'{target}_ref'] += cmpt_reference

                        elif target in references.columns:
                            results.data[f'{target}_ref'] = references[target]

                        elif f'{target}_ref' in references.columns:
                            results.data[f'{target}_ref'] = references[f'{target}_ref']

                        results.data[f'{target}_err'] = (results.data[target] -
                                                         results.data[f'{target}_ref'])

                for cmpt_type, target in self._targets.items():
                    add_reference(cmpt_type, target)

            except DatabaseUnavailableException as e:
                references = None
                logger.debug("Unable to retrieve reference values for system %s: %s", self.system.name, str(e))

            def prepare_data(data: pd.DataFrame) -> pd.DataFrame:
                data = data.tz_convert(self.system.location.timezone).tz_localize(None)
                data = data[[column for column in self._columns.keys() if column in data.columns]]
                data = data.rename(columns=self._columns)
                data.index.name = 'Time'
                return data

            hours = pd.Series(results.data.index, index=results.data.index)
            hours = (hours - hours.shift(1)).bfill().dt.total_seconds() / 3600.

            if '_power' in self._targets[PVSystem.TYPE]:
                pv_power_target = self._targets[PVSystem.TYPE]
                pv_energy_target = pv_power_target.replace('_power', '_energy')
                if pv_energy_target not in results.data and pv_power_target in results.data:
                    results.data[pv_energy_target] = results.data[pv_power_target] / 1000. * hours
                if f'{pv_energy_target}_ref' not in results.data and f'{pv_power_target}_ref' in results.data:
                    results.data[f'{pv_energy_target}_ref'] = results.data[f'{pv_power_target}_ref'] / 1000. * hours
                    results.data[f'{pv_energy_target}_err'] = (results.data[pv_energy_target] -
                                                               results.data[f'{pv_energy_target}_ref'])

            if PVSystem.ENERGY not in results.data and PVSystem.POWER in results.data:
                results.data[PVSystem.ENERGY] = results.data[PVSystem.POWER] / 1000. * hours
            if PVSystem.ENERGY_DC not in results.data and PVSystem.POWER_DC in results.data:
                results.data[PVSystem.ENERGY_DC] = results.data[PVSystem.POWER_DC] / 1000. * hours

            summary = pd.DataFrame(columns=pd.MultiIndex.from_tuples((), names=['System', '']))

            self._evaluate(summary, results.summary, results.images, results.data, references)

            summary_data = {
                self.system.name: prepare_data(results.data)
            }
            if len(self.system) > 1:
                for cmpt in self.system.values():
                    cmpt_key = f"{self.system.id}/{cmpt.id}/output"
                    if cmpt_key in results:
                        results_name = cmpt.name
                        results_prefix = ''
                        for cmpt_type in self.system.get_types():
                            if results_name.startswith(cmpt_type):
                                results_prefix = cmpt_type
                                break

                        if len(results_prefix) > 0 or len(np.unique([c.type for c in self.system.values()])) > 1:
                            if len(results_prefix) > 0:
                                results_name = results_name.replace(results_prefix, '')
                            if len(results_name) < 1 < len(self.system.get_type(cmpt.type)):
                                results_name += str(list(self.system.get_type(cmpt.type)).index(cmpt) + 1)

                            results_prefix = CMPTS[cmpt.type] if cmpt.type in CMPTS else cmpt.type.upper()
                            results_name = f"{results_prefix.strip()} {results_name}".strip().title()

                        summary_data[results_name] = prepare_data(results[cmpt_key])

            summary.to_csv(self._results_csv, encoding='utf-8-sig')
            excel.write(summary, summary_data, file=self._results_excel)

            results.durations.complete('Evaluation')
            results.summary['status'] = 'success'
            progress.complete(results.summary)

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
            results.close()
            progress.reset()

        logger.debug('Evaluation complete in %i minutes', results.durations['Evaluation'])

        return results

    def _evaluate(self,
                  summary_frame: pd.DataFrame,
                  summary: Dict[str, str | float | int],
                  images: Dict[str, str],
                  results: pd.DataFrame,
                  reference: pd.DataFrame = None) -> None:
        self._evaluate_yield(summary_frame, summary, images, results, reference)
        self._evaluate_weather(summary_frame, summary, images, results, reference)

    def _evaluate_yield(self,
                        summary_frame: pd.DataFrame,
                        summary_dict: Dict[str, str | float | int],
                        images_dict: Dict[str, str],
                        results: pd.DataFrame,
                        reference: pd.DataFrame = None) -> None:
        if PVSystem.ENERGY not in results.columns:
            return

        results_kwp = 0
        for system in self.system.values():
            results_kwp += system.power_max / 1000.

        results[PVSystem.YIELD_SPECIFIC] = results[PVSystem.ENERGY] / results_kwp

        results = results.dropna(axis='index', how='all')

        # profile_months = os.path.join(self._image_dir, 'pv_profile_months.png')
        # plot_data = pd.concat([pd.Series(data=results.loc[results.index.month == m, PVSystem.POWER]/1000.,
        #                                  name=calendar.month_name[m]) for m in range(1, 13)], axis='columns')
        # plot_data['hour'] = plot_data.index.hour + plot_data.index.minute/60.
        # plot_melt = plot_data.melt(id_vars='hour', var_name='Months')
        # plot.line(x='hour', y='value', data=plot_melt,
        #           xlabel='Hour of the Day', ylabel='Power [kW]', title='Yield Profile', hue='Months',
        #           colors=list(reversed(plot.COLORS)), file=profile_months)
        # images_dict["pv_profile_months"] = profile_months

        yield_months = os.path.join(self._plots_dir, f"{Evaluation.PV_ENERGY_MONTHS}.png")
        plot_data = results[[PVSystem.ENERGY]].groupby(results.index.month).sum()
        plot.bar(x=plot_data.index, y=PVSystem.ENERGY, data=plot_data,
                 xlabel='Month', ylabel='Energy [kWh]', title='Monthly Yield',
                 colors=list(reversed(plot.COLORS)), file=yield_months)
        images_dict[Evaluation.PV_ENERGY_MONTHS] = yield_months

        yield_specific = round(results[PVSystem.YIELD_SPECIFIC].sum(), 2)
        yield_energy, yield_energy_column = self._scale_yield(results[PVSystem.ENERGY].sum(), AC_E)

        summary_frame.loc[self.system.name, ('Yield', AC_Y)] = yield_specific
        summary_dict[Evaluation.PV_SPECIFIC_YIELD] = yield_specific

        summary_frame.loc[self.system.name, ('Yield', yield_energy_column)] = yield_energy
        summary_dict[Evaluation.PV_ENERGY_YIELD] = yield_energy

        if PVSystem.ENERGY_DC in results:
            dc_energy, dc_energy_column = self._scale_yield(results[PVSystem.ENERGY_DC].sum(), DC_E)

            summary_frame.loc[self.system.name, ('Yield', dc_energy_column)] = dc_energy
            summary_dict[Evaluation.PV_ENERGY_DC_YIELD] = dc_energy

    def _evaluate_weather(self,
                          summary_frame: pd.DataFrame,
                          summary_dict: Dict[str, str | float | int],
                          images_dict: Dict[str, str],
                          results: pd.DataFrame,
                          reference: pd.DataFrame = None) -> None:
        hours = pd.Series(results.index, index=results.index)
        hours = (hours - hours.shift(1)).bfill().dt.total_seconds() / 3600.
        ghi = round((results[Weather.GHI] / 1000. * hours).sum(), 2)
        dhi = round((results[Weather.DHI] / 1000. * hours).sum(), 2)

        summary_frame.loc[self.system.name, ('Weather', 'GHI [kWh/m^2]')] = ghi
        summary_dict[Evaluation.GHI_TOTAL] = ghi

        summary_frame.loc[self.system.name, ('Weather', 'DHI [kWh/m^2]')] = dhi
        summary_dict[Evaluation.DHI_TOTAL] = dhi

    @staticmethod
    def _scale_yield(energy: float, column: str) -> Tuple[float, str]:
        if energy >= 1e7:
            energy = round(energy / 1e6)
            column = column.replace('kWh', 'GWh')
        elif energy >= 1e4:
            energy = round(energy / 1e3)
            column = column.replace('kWh', 'MWh')
        return round(energy, 2), column


# noinspection PyUnresolvedReferences
def _get(results, key: str, func: Callable, *args, **kwargs) -> pd.DataFrame:
    if results is None or key not in results:
        result = func(*args, **kwargs)
        if results is not None:
            results.set(key, result)
        return result

    return results.get(key)
