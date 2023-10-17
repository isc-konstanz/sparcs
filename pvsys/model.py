# -*- coding: utf-8 -*-
"""
    pvsys.model
    ~~~~~~~~~~~
    
    
"""
from __future__ import annotations

import os
import logging
import pandas as pd

from copy import deepcopy
from pvlib.modelchain import ModelChain
from corsys import Model as ModelCore
from corsys import Location, Configurations
from . import PVSystem

logger = logging.getLogger(__name__)

# noinspection SpellCheckingInspection
DEFAULTS = dict(
    # ac_model='pvwatts',
    # dc_model='pvwatts',
    temperature_model='sapm',
    aoi_model='physical',
    spectral_model='no_loss',
    dc_ohmic_model='no_loss',
    losses_model='pvwatts'
)


# noinspection SpellCheckingInspection, PyAbstractClass
class Model(ModelCore, ModelChain):

    @classmethod
    def read(cls, pvsystem: PVSystem, override_file: str = 'model.cfg', section: str = 'Model') -> Model:
        configs = deepcopy(pvsystem.configs)
        configs_override = os.path.join(configs.dirs.conf, pvsystem.id+'.d', override_file)
        if os.path.isfile(configs_override):
            configs.read(configs_override)

        params = DEFAULTS
        if configs.has_section(section):
            params.update(configs.items(section))

        return cls(configs, pvsystem, pvsystem.system.location, **params)

    def __init__(self, configs: Configurations, pvsystem: PVSystem, location: Location, **kwargs):
        super().__init__(configs, pvsystem, location, **kwargs)

    def __call__(self, weather, **_):
        self.run_model(weather)

        results = self.results
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

        losses = self.results.losses
        if not isinstance(losses, float) and not \
                (isinstance(losses, tuple) and any([isinstance(loss, float) for loss in losses])):
            if isinstance(losses, tuple):
                losses = pd.concat(list(losses), axis='columns').mean(axis='columns')
                losses.name = 'losses'
            results = pd.concat([results, losses], axis='columns')
        return results

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
