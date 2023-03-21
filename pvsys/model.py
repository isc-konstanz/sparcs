# -*- coding: utf-8 -*-
"""
    pvsys.model
    ~~~~~~~~~~~
    
    
"""
from __future__ import annotations
from typing import Dict, Any
import os
import logging
import pandas as pd

from copy import deepcopy
from pvlib.modelchain import ModelChain
from corsys.model import Model as ModelCore
from corsys.configs import Configurations
from corsys import System
from . import PVSystem

logger = logging.getLogger(__name__)


class Model(ModelCore, ModelChain):

    # noinspection PyShadowingBuiltins
    @classmethod
    def read(cls, system: PVSystem, override_file: str = 'forecast.cfg') -> Model:
        configs = deepcopy(system.configs)
        configs_override = os.path.join(configs.dirs.conf, system.id+'.d', override_file)
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
        return 'pvwatts'

    # noinspection SpellCheckingInspection
    def pvwatts_losses(self):
        if isinstance(self.results.dc, tuple):
            self.results.losses = tuple((100 - losses) / 100. for losses in
                                        self.system.pvwatts_losses(self.results.solar_position))

            for dc, losses in zip(self.results.dc, self.results.losses):
                dc[:] = dc.mul(losses, axis='index')
        else:
            self.results.losses = (100 - self.system.pvwatts_losses(self.results.solar_position)) / 100.
            self.results.dc[:] = self.results.dc.mul(self.results.losses, axis='index')
        return self
