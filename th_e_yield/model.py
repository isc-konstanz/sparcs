# -*- coding: utf-8 -*-
"""
    th-e-yield.model
    ~~~~~~~~~~~~~~~~
    
    
"""
import os
import logging
import pandas as pd

from pvlib.modelchain import ModelChain
from th_e_core import Model as ModelCore

logger = logging.getLogger(__name__)


class Model(ModelChain, ModelCore):

    @classmethod
    def read(cls, context, system, **kwargs):
        configs = cls.read_configs(system, **kwargs)
        configs_override = os.path.join(configs['General']['config_dir'], 
                                        system.id+'.d', 'model.cfg')
        
        if os.path.isfile(configs_override):
            configs.read(configs_override)
        
        type = configs.get('General', 'type', fallback='default')  #@ReservedAssignment
        if type.lower() in ['default', 'optical', 'pvlib']:
            return Model(system, context.location, configs, **kwargs)
        
        return cls.from_configs(system, configs, **kwargs)

    def __init__(self, system, location, configs, **kwargs):
        ModelChain.__init__(self, system, location, **dict(configs.items('Model')), **kwargs)
        ModelCore.__init__(self, configs, system, **kwargs)

    def run(self, weather, **_):
        self.run_model(weather)

        result = pd.concat([self.ac, self.dc], axis=1)
        result = result[['p_ac', 'p_dc', 'i_mp', 'v_mp', 'i_sc', 'v_oc']]

        return pd.concat([result, weather], axis=1)

    def pvwatts_dc(self):
        self.dc = self.system.pvwatts_dc(self.effective_irradiance,
                                         self.cell_temperature)

        self.dc *= self.system.modules_per_string * self.system.strings_per_inverter

        return self

    def pvwatts_inverter(self):
        # Scale the nameplate power rating to enable compatibility with other models
        self.system.inverter_parameters['pdc0'] *= self.system.modules_per_string*self.system.strings_per_inverter
        
        if isinstance(self.dc, pd.Series):
            pdc = self.dc
        elif 'p_mp' in self.dc:
            self.dc.rename(columns={'p_mp': 'p_dc'}, inplace=True)

            pdc = self.dc['p_dc']
        else:
            raise ValueError('Unknown error while calculating PVWatts AC model')
        
        self.ac = self.system.pvwatts_ac(pdc).fillna(0)
        self.ac.name = 'p_ac'
        
        return self

