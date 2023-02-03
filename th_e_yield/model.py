# -*- coding: utf-8 -*-
"""
    th-e-yield.model
    ~~~~~~~~~~~~~~~~
    
    
"""
from __future__ import annotations
import os
import logging
from abc import ABC

import pandas as pd
from typing import Dict, Any
from pvlib.modelchain import ModelChain
from th_e_core.model import Model as ModelCore
from th_e_core.configs import Configurations
from th_e_core import System
from . import PVSystem

logger = logging.getLogger(__name__)


class Model(ModelCore, ModelChain):

    # noinspection PyShadowingBuiltins
    @classmethod
    def read(cls, system: PVSystem, config_file: str = 'model.cfg') -> Model:
        configs = Configurations.from_configs(system.configs, config_file)
        configs_override = os.path.join(configs.dirs.conf,
                                        system.id+'.d', 'model.cfg')

        if os.path.isfile(configs_override):
            configs.read(configs_override)

        return cls(system.context, system, configs)

    def __init__(self, context: System, system: PVSystem, configs: Configurations, section: str = 'Model', **kwargs):
        super().__init__(context, configs, system, context.location, **self._infer_params(configs, section, **kwargs))

    def __call__(self, weather, **_):
        self.run_model(weather)

        results_dc = self.results.dc.rename(columns={'p_mp': 'p_dc'})
        results_ac = self.results.ac.rename(columns={'p_mp': 'p_ac'})['p_ac']
        result = pd.concat([results_ac, results_dc], axis=1)
        result = result[['p_ac', 'p_dc', 'i_mp', 'v_mp', 'i_sc', 'v_oc']]

        if not isinstance(self.results.losses, float):
            losses = self.results.losses
            result = pd.concat([result, losses], axis=1)

        return pd.concat([result, weather], axis=1)

    @staticmethod
    def _infer_params(configs: Configurations, section: str, **kwargs) -> Dict[str, Any]:
        params = dict(configs.items(section))
        params.update(kwargs)

        return params

    def infer_losses_model(self):
        raise NotImplementedError
