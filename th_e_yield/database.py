# -*- coding: utf-8 -*-
"""
    th-e-yield.database
    ~~~~~~~~~~~~~~~~~~~
    
    
"""
import os
import io
import json
import shutil
import logging
import time
import pandas as pd
import numpy as np

from collections import OrderedDict

logger = logging.getLogger(__name__)


class JsonDatabase:

    def __init__(self, configs, key):
        self.data_dir = configs.get('General', 'lib_dir')
        
        if configs.has_option('General', 'cec_dir'):
            self.cec_dir = configs.get('General', 'cec_dir')
        
        self.key = key
        
        self.sam_url = 'https://raw.githubusercontent.com/pvlib/pvlib-python/master/pvlib/data'
        self.sam_db = None

    def get(self, path):
        path = path.split('/')
        filename = os.path.join(self.data_dir, self.key, path[0], path[1]+'.json')
        
        with open(filename, encoding='utf-8') as file:
            return json.load(file)

    def clean(self):
        module_dir = os.path.join(self.data_dir, self.key)
        
        if os.path.exists(module_dir):
            shutil.rmtree(module_dir)
            
            while os.path.exists(module_dir):
                time.sleep(.1)
        
        os.makedirs(module_dir)

    def _write_json(self, manufacturer, module, data):
        json_dir = os.path.join(self.data_dir, self.key, manufacturer)
        if not os.path.exists(json_dir):
            os.makedirs(json_dir)
        
        json_file = os.path.join(json_dir, module+'.json')
        with open(json_file, 'w', encoding='UTF-8') as file:
            file.write(json.dumps(data, separators=(',', ':')))  # , indent=2)

    def _write_meta(self, data):
        count = 0
        for manufacturer, modules in data.items():
            count += len(modules.keys())
            # Sort the meta data by module name before writing them as JSON
            modules = OrderedDict(sorted(modules.items(), key=lambda m: m[1]['Name']))
            
            meta_file = os.path.join(self.data_dir, self.key, manufacturer+'.json')
            with open(meta_file, 'w', encoding='UTF-8') as file:
                file.write(json.dumps(modules, separators=(',', ':')))  # , indent=2)
        
        return count

    def _load_cec(self, sep=','):
        csv = os.path.join(self.cec_dir, self.key + '_cec.csv')
        return pd.read_csv(csv, sep=sep, skiprows=[1, 2], encoding="ISO-8859-1", low_memory=False)

    def _load_cec_custom(self):
        csv = os.path.join(self.cec_dir, self.key + '_cec_custom.csv')
        return pd.read_csv(csv, skiprows=[1, 2], encoding="ISO-8859-1", low_memory=False)

    def _load_cec_sam(self, download=False):
        try:
            from urllib2 import urlopen
        except ImportError:
            from urllib.request import urlopen
        
        if download:
            response = urlopen(self.sam_url + '/' + self.sam_db + '.csv')
            csv = io.StringIO(response.read().decode(errors='ignore'))
        else:
            csv = os.path.join(self.cec_dir, self.key + '_cec_sam.csv')
        
        return pd.read_csv(csv, skiprows=[1, 2], encoding = "ISO-8859-1", low_memory=False)


class ModuleDatabase(JsonDatabase):

    def __init__(self, configs):
        super().__init__(configs, 'modules')
        
        self.sam_db = 'sam-library-cec-modules-2017-6-5'

    def build(self):
        db_cec = self._load_cec()
        db_sam = self._load_cec_sam()
        db_custom = self._load_cec_custom()
        
        db_meta = {}
        
        for _, module in pd.concat([db_cec, db_custom], sort=True).iterrows():
            path, meta = self._parse_module_meta(module)
            
            module_sam = db_sam.loc[db_sam['Name'] == module['Manufacturer'] + ' ' + module['Model Number']]
            if len(module_sam) > 0:
                db_sam = db_sam.drop(module_sam.iloc[0].name)
                self._write_module_singlediode(path, module_sam.iloc[0].combine_first(module))
                
            elif not module.loc[['a_ref', 'I_L_ref', 'I_o_ref', 'R_sh_ref', 'R_s']].isnull().any():
                self._write_module_singlediode(path, module)
                
            else:
                self._write_module_pvwatts(path, module)
            
            self._build_module_meta(db_meta, meta, *path)
            
            logger.debug("Successfully built Module: %s %s", meta['Manufacturer'], meta['Name'])
        
        db_count = self._write_meta(db_meta)
        
        file_remain = os.path.join(self.cec_dir, self.key + '_cec_sam_remain.csv')
        db_sam.to_csv(file_remain, encoding = "ISO-8859-1")
        
        logger.info("Complete module library built for %i entries", db_count)
        logger.debug("Unable to build %i SAM modules", len(db_sam))

    def _build_module_meta(self, database, meta, manufacturer, module):
        if manufacturer not in database:
            database[manufacturer] = {}
        
        database[manufacturer]['/'.join([manufacturer, module])] = meta

    def _parse_module_meta(self, module):
        meta = OrderedDict()
        meta['Name']         = module['Model Number']
        meta['Manufacturer'] = module['Manufacturer']
        meta['Description']  = module['Description']
        meta['BIPV']         = module['BIPV']
        
        manufacturer = meta['Manufacturer'].lower().replace(' ', '_').replace('/', '-').replace('&', 'n') \
                                           .replace(',', '').replace('.', '').replace('!', '') \
                                           .replace('(', '').replace(')', '')
        
        name = meta['Name'].lower().replace(' ', '_').replace('/', '-').replace('&', 'n') \
                                   .replace(',', '').replace('.', '').replace('!', '') \
                                   .replace('(', '').replace(')', '')
        
        path = [manufacturer, name]
        return path, meta

    def _write_module_singlediode(self, path, cec):
        module = OrderedDict()
        module['Date']          = cec['Date']
        module['Version']       = cec['Version']
        module['Technology']    = cec['Technology']
        module['BIPV']          = cec['BIPV']
        module['A_c']           = float(cec['A_c'])
        module['N_s']           = float(cec['N_s'])
        module['T_NOCT']        = float(cec['T_NOCT'])
        module['I_sc_ref']      = float(cec['I_sc_ref'])
        module['V_oc_ref']      = float(cec['V_oc_ref'])
        module['I_mp_ref']      = float(cec['I_mp_ref'])
        module['V_mp_ref']      = float(cec['V_mp_ref'])
        module['alpha_sc']      = float(cec['alpha_sc'])
        module['beta_oc']       = float(cec['beta_oc'])
        module['a_ref']         = float(cec['a_ref'])
        module['I_L_ref']       = float(cec['I_L_ref'])
        module['I_o_ref']       = float(cec['I_o_ref'])
        module['R_s']           = float(cec['R_s'])
        module['R_sh_ref']      = float(cec['R_sh_ref'])
        module['Adjust']        = float(cec['Adjust'])
        module['PTC']           = float(cec['PTC'])
        module['pdc0']          = float(cec['pdc0']) if not np.isnan(cec['pdc0']) else float(cec['Nameplate Pmax'])
        module['gamma_pdc']     = float(cec['gamma_pdc']) if not np.isnan(cec['gamma_pdc']) else float(cec['gamma_r'])/100.0
        module['gamma_r']       = float(cec['gamma_r'])
        
        self._write_json(*path, module)

    def _write_module_pvwatts(self, path, cec):
        module = OrderedDict()
        module['Technology']    = cec['Technology']
        module['BIPV']          = cec['BIPV']
        module['A_c']           = float(cec['A_c'])
        module['N_s']           = float(cec['N_s'])
        module['T_NOCT']        = float(cec['Average NOCT'])
        module['PTC']           = float(cec['PTC'])
        module['pdc0']          = float(cec['pdc0']) if not np.isnan(cec['pdc0']) else float(cec['Nameplate Pmax'])
        module['gamma_pdc']     = float(cec['gamma_pdc']) if not np.isnan(cec['gamma_pdc']) else float(cec['?Pmax'])/100.0
        
        self._write_json(*path, module)


class InverterDatabase(JsonDatabase):

    def __init__(self, configs):
        super().__init__(configs, 'inverters')
        self.sam_db = 'sam-library-cec-inverters-2018-3-18'

    def build(self):
        db_cec = self._load_cec(';')
        db_sam = self._load_cec_sam()
        #db_custom = self._load_cec_custom()
        db_meta = {}
        
        for _, inverter in db_cec.iterrows():
            path, meta = self._parse_inverter_meta(inverter)
            
            inverter_sam = db_sam.loc[db_sam['Name'].str.startswith(inverter['Manufacturer'] + ': ' + inverter['Model Number'])]
            if len(inverter_sam) > 0:
                db_sam = db_sam.drop(inverter_sam.iloc[0].name)
                self._write_inverter_sandia(path, inverter_sam.iloc[0].combine_first(inverter))
                
            else:
                self._write_inverter_pvwatts(path, inverter)
               
            self._build_inverter_meta(db_meta, meta, *path)
            
            logger.debug("Successfully built Inverter: %s %s", meta['Manufacturer'], meta['Name'])
        
        db_count = self._write_meta(db_meta)
        
        file_remain = os.path.join(self.cec_dir, self.key + '_cec_sam_remain.csv')
        db_sam.to_csv(file_remain, encoding = "ISO-8859-1")
        
        logger.info("Complete inverter library built for %i entries", db_count)
        logger.debug("Unable to build %i SAM inverters", len(db_sam))

    def _build_inverter_meta(self, database, meta, manufacturer, inverter):
        if manufacturer not in database:
            database[manufacturer] = {}
        
        database[manufacturer]['/'.join([manufacturer, inverter])] = meta

    def _parse_inverter_meta(self, inverter):
        meta = OrderedDict()
        meta['Name']           = inverter['Model Number']
        meta['Manufacturer']   = inverter['Manufacturer']
        meta['Description']    = inverter['Description']
        meta['Built-In Meter'] = inverter['Built-In Meter']
        meta['Microinverter']  = inverter['Microinverter']
        
        manufacturer = meta['Manufacturer'].lower().replace(' ', '_').replace('/', '-').replace('&', 'n') \
                                           .replace(',', '').replace('.', '').replace('!', '') \
                                           .replace('(', '').replace(')', '')
        
        name = meta['Name'].lower().replace(' ', '_').replace('/', '-').replace('&', 'n') \
                                   .replace(',', '').replace('.', '').replace('!', '') \
                                   .replace('(', '').replace(')', '')
        
        path = [manufacturer, name]
        return path, meta

    def _write_inverter_sandia(self, path, cec):
        inverter = OrderedDict()
        inverter['Vac']       = float(cec['Vac'])
        inverter['Paco']      = float(cec['Paco'])
        inverter['Pdco']      = float(cec['Pdco'])
        inverter['Vdco']      = float(cec['Vdco'])
        inverter['Pso']       = float(cec['Pso'])
        inverter['C0']        = float(cec['C0'])
        inverter['C1']        = float(cec['C1'])
        inverter['C2']        = float(cec['C2'])
        inverter['C3']        = float(cec['C3'])
        inverter['Pnt']       = float(cec['Pnt'])
        inverter['Vdcmax']    = float(cec['Vdcmax'])
        inverter['Mppt_low']  = float(cec['Mppt_low'])
        inverter['Mppt_high'] = float(cec['Mppt_high'])
        
        self._write_json(*path, inverter)

    def _write_inverter_pvwatts(self, path, cec):
        inverter = OrderedDict()
        inverter['pdc0'] = float(cec['Maximum Continuous Output Power at Unity Power Factor']) * 1000.0
        
        self._write_json(*path, inverter)

