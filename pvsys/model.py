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
from corsys import Model as ModelCore
from corsys import Location, Configurations
from . import PVSystem

logger = logging.getLogger(__name__)


class Model(ModelCore, ModelChain):

    # noinspection SpellCheckingInspection, PyTypeChecker, PyShadowingBuiltins
    @classmethod
    def read(cls, pvsystem: PVSystem, override_file: str = 'forecast.cfg') -> Model:
        configs = deepcopy(pvsystem.configs)
        configs_override = os.path.join(configs.dirs.conf, pvsystem.id+'.d', override_file)
        if os.path.isfile(configs_override):
            configs.read(configs_override)

        return cls(pvsystem, pvsystem.system.location, configs)

    # noinspection SpellCheckingInspection
    def __init__(self, pvsystem: PVSystem, location: Location, configs: Configurations, section: str = 'Model', **kwargs):
        super().__init__(configs, pvsystem, location, **self._infer_params(configs, section, **kwargs))

    def __call__(self, weather, **_):
        self.run_model(weather)
        results = deepcopy(self.results)
        results_dc = results.dc.to_frame() if isinstance(results.dc, pd.Series) else results.dc
        results_dc = results_dc.rename(columns={'p_mp': 'p_dc'})
        results_ac = results.ac.to_frame() if isinstance(results.ac, pd.Series) else results.ac
        results_ac = results_ac.rename(columns={'p_mp': 'p_ac'})['p_ac']

        result = pd.concat([results_ac, results_dc], axis=1)
        result = result[[c for c in ['p_ac', 'p_dc', 'i_mp', 'v_mp', 'i_sc', 'v_oc'] if c in result.columns]]
        result.loc[:, result.columns.str.startswith(('p_', 'i_'))] *= self.system.inverters_per_system

        if not isinstance(results.losses, float):
            losses = results.losses
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
