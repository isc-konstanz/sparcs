# -*- coding: utf-8 -*-
"""
    th-e-yield.system
    ~~~~~~~~~~~~~~~~~
    
    
"""
import os
import json
import logging
import pandas as pd
import datetime as dt

from pvlib.location import Location
from configparser import ConfigParser
from th_e_core import Forecast, ConfigUnavailableException
from th_e_core.weather import Weather, TMYWeather, EPWWeather
from th_e_core.pvsystem import PVSystem
from th_e_core.system import System as SystemCore
from th_e_yield.database import ModuleDatabase, InverterDatabase
from th_e_yield.model import Model

logger = logging.getLogger(__name__)

AC_P = 'Power [W]'
AC_E = 'Energy yield [kWh]'
AC_Y = 'Specific yield [kWh/kWp]'
DC_P = 'Power (DC) [W]'
DC_E = 'Energy yield (DC) [kWh]'


class System(SystemCore):

    def _configure(self, configs, **kwargs):
        super()._configure(configs, **kwargs)

        data_dir = configs.get('General', 'data_dir')
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        self._results_json = os.path.join(data_dir, 'results.json')
        self._results_excel = os.path.join(data_dir, 'results.xlsx')
        self._results_csv = os.path.join(data_dir, 'results.csv')
        self._results_dir = os.path.join(data_dir, 'results')
        if not os.path.exists(self._results_dir):
            os.makedirs(self._results_dir)

    def _activate(self, components, configs, **kwargs):
        super()._activate(components, configs, **kwargs)
        try:
            self.weather = Forecast.read(self, **kwargs)
            
        except ConfigUnavailableException:
            # Use weather instead of forecast, if forecast.cfg not present
            self.weather = Weather.read(self, **kwargs)
        
        if isinstance(self.weather, TMYWeather):
            self.location = Location.from_tmy(self.weather.meta)
        elif isinstance(self.weather, EPWWeather):
            self.location = Location.from_epw(self.weather.meta)
        else:
            self.location = self._location_read(configs, **kwargs)

    def _location_read(self, configs, **kwargs):
        return Location(configs.getfloat('Location', 'latitude'), 
                        configs.getfloat('Location', 'longitude'), 
                        tz=configs.get('Location', 'timezone', fallback='UTC'), 
                        altitude=configs.getfloat('Location', 'altitude', fallback=0), 
                        name=self.name, 
                        **kwargs)

    @property
    def _forecast(self):
        if isinstance(self.weather, Forecast):
            return self.weather
        
        raise AttributeError("System forecast not configured")

    @property
    def _component_types(self):
        return super()._component_types + ['solar', 'modules', 'configs']

    def _component(self, configs, type, **kwargs):
        if type in ['pv', 'solar', 'modules', 'configs']:
            return Configurations(configs, self, **kwargs)
        
        return super()._component(configs, type, **kwargs)

    def run(self, *args, **kwargs):

        logger.info("Starting simulation for system: %s", self.name)
        start = dt.datetime.now()

        weather = self.weather.get(*args, **kwargs)
        result = pd.DataFrame(columns=['p_ac', 'p_dc'], index=weather.index).fillna(0)
        result.index.name = 'time'
        results = {}
        try:
            progress = Progress(len(self) + 1, file=self._results_json)

            for key, configs in self.items():
                if configs.type == 'pv':
                    model = Model.read(self, configs, **kwargs)
                    data = model.run(weather, **kwargs)

                    if self._database is not None:
                        self._database.persist(data, **kwargs)

                    result[['p_ac', 'p_dc']] += data[['p_ac', 'p_dc']].abs()
                    results[key] = data
                    progress.update()

            results_json = self.evaluate(results, weather)

            with open(self._results_json, 'w', encoding='utf-8') as f:
                json.dump(results_json, f, ensure_ascii=False, indent=4)

        except Exception as e:
            with open(self._results_json, 'w', encoding='utf-8') as f:
                import traceback
                results_json = {
                    'status': 'error',
                    'message': str(e),
                    'error': type(e).__name__,
                    'trace': traceback.format_exc()
                }
                json.dump(results_json, f, ensure_ascii=False, indent=4)

            raise e

        logger.info('Simulation complete')
        logger.debug('Simulation complete in %i seconds', (dt.datetime.now() - start).total_seconds())

        return pd.concat([result, weather], axis=1)

    def evaluate(self, results, weather):
        json = {}
        json.update(self._evaluate_yield(results, weather))

        return json

    def _evaluate_yield(self, results, weather):
        hours = pd.Series(weather.index, index=weather.index)
        hours = round((hours - hours.shift(1)).fillna(method='bfill').dt.total_seconds() / 3600)

        results_total_columns = [AC_P, AC_E, AC_Y, DC_P]
        results_total = pd.DataFrame(index=weather.index, columns=[AC_P, DC_P]).fillna(0)
        results_total.index.name = 'Time'
        results_kwp = 0
        for key, configs in self.items():
            result = results[key]
            results_kwp += float(configs.module_parameters['pdc0']) * \
                                 configs.inverters_per_system*configs.modules_per_inverter / 1000

            results_total[[AC_P, DC_P]] += result[['p_ac', 'p_dc']].abs().values

        results_total[AC_E] = results_total[AC_P] / 1000 * hours
        results_total[AC_Y] = results_total[AC_E] / results_kwp

        results_total[DC_E] = results_total[DC_P] / 1000 * hours

        results_total = results_total.dropna(axis='index', how='all')

        yield_specific = round(results_total[AC_Y].sum(), 2)
        yield_energy = round(results_total[AC_E].sum(), 2)

        ghi = round((weather['ghi'] / 1000 * hours).sum(), 2)
        dhi = round((weather['dhi'] / 1000 * hours).sum(), 2)

        results_total = results_total[results_total_columns]

        results_summary_columns = [AC_Y, AC_E]
        results_summary_data = [yield_specific, yield_energy]

        results_summary_columns.append('GHI [kWh/m^2]')
        results_summary_data.append(ghi)

        results_summary_columns.append('DHI [kWh/m^2]')
        results_summary_data.append(dhi)

        results_summary = pd.DataFrame(data=[results_summary_data], columns=results_summary_columns)

        self._write_csv(results_summary, results_total, results)
        self._write_excel(results_summary, results_total, results)
        return {
                   'status': 'success',
                   'yield_energy': yield_energy,
                   'yield_specific': yield_specific
               }

    def _write_csv(self, results_summary, results_total, results):
        for key, configs in self.items():
            configs_name = configs.name
            for configs_type in self._component_types:
                configs_name = configs_name.replace(configs_type, '')
            if len(configs_name) < 1:
                configs_name += str(list(self.values()).index(configs) + 1)

            results[key].to_csv(os.path.join(self._results_dir, 'results_'+configs_name+'.csv'), encoding='utf-8-sig')

        results_total.to_csv(os.path.join(self._results_dir, 'results.csv'), encoding='utf-8-sig')
        results_summary.to_csv(self._results_csv, encoding='utf-8-sig', index=False)

        return results

    def _write_excel(self, results_summary, results_total, results):
        try:
            from openpyxl import Workbook
            from openpyxl.utils import get_column_letter
            from openpyxl.styles import Border, Side

            border_side = Side(border_style=None)
            border = Border(top=border_side,
                            right=border_side,
                            bottom=border_side,
                            left=border_side)

            results_book = Workbook()
            results_book.remove_sheet(results_book.active)
            results_writer = pd.ExcelWriter(self._results_excel, engine='openpyxl', options={'encoding': 'utf-8'})
            results_writer.book = results_book

            for key, configs in self.items():
                configs_name = configs.name
                for configs_type in self._component_types:
                    configs_name = configs_name.replace(configs_type, '')
                if len(configs_name) < 1:
                    configs_name += str(list(self.values()).index(configs) + 1)

                results[key].tz_localize(None).to_excel(results_writer,
                                                        sheet_name=self.name+' '+configs_name,
                                                        encoding='utf-8-sig')

            results_summary.to_excel(results_writer, sheet_name='Summary', float_format="%.2f", encoding='utf-8-sig')
            results_total.tz_localize(None).to_excel(results_writer, sheet_name=self.name, encoding='utf-8-sig')
            results_book['Summary'].delete_cols(1, 1)
            results_sheets = results_book._sheets
            results_sheets.insert(0, results_sheets.pop(len(results_sheets)-1))
            results_sheets.insert(0, results_sheets.pop(len(results_sheets)-1))

            for result_sheet in results_sheets:
                for result_column in range(1, len(result_sheet[1])):
                    result_sheet[1][result_column].border = border
                    result_sheet.column_dimensions[get_column_letter(result_column+1)].width = len(result_sheet[1][result_column].value)+2

                result_index_width = 0
                for result_row in result_sheet:
                    result_row[0].border = border
                    result_index_width = max(result_index_width, len(str(result_row[0].value)))

                result_sheet.column_dimensions[get_column_letter(1)].width = result_index_width+2

            results_book.save(self._results_excel)
            results_writer.close()

        except ImportError as e:
            logger.debug("Unable to plot boxplot for {} of system {}: {}".format(os.path.abspath(self._results_excel),
                                                                                 self.name, str(e)))


class Configurations(PVSystem):

    def _configure(self, configs, **kwargs):
        super()._configure(configs, **kwargs)

        self.module_parameters = self._configure_module(configs)
        self.modules_per_inverter = self.strings_per_inverter * self.modules_per_string

        self.inverter_parameters = self._configure_inverter(configs)
        self.inverters_per_system = configs.getfloat('Inverter', 'count', fallback=1)

    def _configure_module(self, configs):
        module = {}
        
        if 'type' in configs['Module']:
            type = configs['Module']['type']
            modules = ModuleDatabase(configs)
            module = modules.get(type)
        
        def module_update(items):
            for key, value in items:
                try:
                    module[key] = float(value)
                except ValueError:
                    module[key] = value
        
        if configs.has_section('Parameters'):
            module_update(configs.items('Parameters'))
        
        module_file = os.path.join(configs['General']['config_dir'], 
                                   configs['General']['id']+'.d', 'module.cfg')
        
        if os.path.exists(module_file):
            with open(module_file) as f:
                module_str = '[Module]\n' + f.read()
            
            module_configs = ConfigParser()
            module_configs.optionxform = str
            module_configs.read_string(module_str)
            module_update(module_configs.items('Module'))
        
        if 'pdc0' not in module:
            module['pdc0'] = module['I_mp_ref'] \
                           * module['V_mp_ref']
        
        return module

    def _configure_inverter(self, configs):
        inverter = {}
        
        if 'type' in configs['Inverter']:
            type = configs['Inverter']['type']
            inverters = InverterDatabase(configs)
            # TODO: Test and verify inverter CEC parameters
            # inverter = inverters.get(type)
        
        def inverter_update(items):
            for key, value in items:
                try:
                    inverter[key] = float(value)
                except ValueError:
                    inverter[key] = value
        
        inverter_file = os.path.join(configs['General']['config_dir'], 
                                     configs['General']['id']+'.d', 'inverter.cfg')
        
        if os.path.exists(inverter_file):
            with open(inverter_file) as f:
                inverter_str = '[Inverter]\n' + f.read()
            
            inverter_configs = ConfigParser()
            inverter_configs.optionxform = str
            inverter_configs.read_string(inverter_str)
            inverter_update(inverter_configs.items('Inverter'))
        
        if 'pdc0' not in inverter and 'pdc0' in self.module_parameters and \
                configs.has_section('Inverter'):
            total = configs.getfloat('Inverter', 'strings') \
                  * configs.getfloat('Module', 'count')
            
            inverter['pdc0'] = self.module_parameters['pdc0']*total
        
        return inverter


class Progress:

    def __init__(self, total, value=0, file=None):
        self._file = file
        self._total = total
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

