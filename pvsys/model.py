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
    def read(cls, pvsystem: PVSystem, override_file: str = 'model.cfg') -> Model:
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

        results = self.results
        losses = self.results.losses

        if isinstance(results.dc, tuple):
            results_dc = pd.concat([dc['p_mp'] for dc in results.dc], axis='columns').sum(axis='columns')
        elif isinstance(results.dc, pd.DataFrame):
            results_dc = results.dc['p_mp']
        else:
            results_dc = results.dc
        results_dc.name = 'p_dc'

        results_ac = results.ac.to_frame() if isinstance(results.ac, pd.Series) else results.ac
        results_ac = results_ac.rename(columns={'p_mp': 'p_ac'})

        results = pd.concat([results_ac, results_dc], axis='columns')
        results = results[[c for c in ['p_ac', 'p_dc', 'i_x', 'i_xx', 'i_mp', 'v_mp', 'i_sc', 'v_oc']
                           if c in results.columns]]

        results.loc[:, results.columns.str.startswith(('p_', 'i_'))] *= self.system.inverters_per_system

        if not isinstance(losses, float) and not \
                (isinstance(losses, tuple) and any([isinstance(loss, float) for loss in losses])):
            if isinstance(losses, tuple):
                losses = pd.concat(list(losses), axis='columns').mean(axis='columns')
                losses.name = 'losses'
            results = pd.concat([results, losses], axis='columns')
        return results

    @staticmethod
    def _infer_params(configs: Configurations, section: str, **kwargs) -> Dict[str, Any]:
        params = dict()

        if configs.has_section(section):
            params.update(configs.items(section))
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
