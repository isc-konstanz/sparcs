# -*- coding: utf-8 -*-
"""
    th-e-yield.model
    ~~~~~~~~~~~~~~~~
    
    
"""
from __future__ import annotations

import os
import logging
import pandas as pd

from pvlib.modelchain import ModelChain
from th_e_core import Model as ModelCore
from th_e_core import System
from th_e_core.cmpt import Photovoltaics

logger = logging.getLogger(__name__)


class Model(ModelChain, ModelCore):

    # noinspection PyShadowingBuiltins
    @classmethod
    def read(cls, system: System, array: PVSystem, **kwargs) -> Model:
        configs = cls._read_configs(array, **kwargs)
        configs_override = os.path.join(configs['General']['config_dir'], 
                                        array.id+'.d', 'model.cfg')

        if os.path.isfile(configs_override):
            configs.read(configs_override)

        return cls(system, array, configs, **kwargs)

    def __init__(self, system, array, configs, section='Model', **kwargs):
        ModelChain.__init__(self, array, system.location, **dict(configs.items(section)), **kwargs)
        ModelCore.__init__(self, system, configs, **kwargs)

    def run(self, weather, **_):
        self.run_model(weather)

        result = pd.concat([self.ac, self.dc], axis=1)
        result = result[['p_ac', 'p_dc', 'i_mp', 'v_mp', 'i_sc', 'v_oc']]

        return pd.concat([result, weather], axis=1)

    def infer_losses_model(self):
        pass

    def pvwatts_inverter(self):
        if isinstance(self.dc, pd.Series):
            pdc = self.dc
        elif 'p_mp' in self.dc:
            self.dc.rename(columns={'p_mp': 'p_dc'}, inplace=True)

            pdc = self.dc['p_dc']
        else:
            raise ValueError('Unknown error while calculating PVWatts AC model')

        self.ac = self.system.pvwatts_ac(pdc).fillna(0).abs()
        self.ac.name = 'p_ac'

        return self
