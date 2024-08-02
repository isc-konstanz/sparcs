# -*- coding: utf-8 -*-
"""
    penguin.system
    ~~~~~~~~~~~~~~


"""
from __future__ import annotations

import datetime as dt
from typing import Optional

import loris
import pandas as pd
from loris import ComponentException, Configurations
from penguin import Location, PVSystem


class System(loris.System):
    POWER_EL:      str = "el_power"
    POWER_EL_IMP:  str = "el_import_power"
    POWER_EL_EXP:  str = "el_export_power"
    POWER_TH:      str = "th_power"
    POWER_TH_HT:   str = "th_ht_power"
    POWER_TH_DOM:  str = "th_dom_power"

    ENERGY_EL:      str = "el_energy"
    ENERGY_EL_IMP:  str = "el_import_energy"
    ENERGY_EL_EXP:  str = "el_export_energy"
    ENERGY_TH:      str = "th_energy"
    ENERGY_TH_HT:   str = "th_ht_energy"
    ENERGY_TH_DOM:  str = "th_dom_energy"

    _location: Optional[Location] = None

    def configure(self, configs: Configurations) -> None:
        super().configure(configs)
        if self.has_type(PVSystem.TYPE):
            from penguin.constants import COLUMNS

            self.data.add(id=PVSystem.POWER_EST, name=COLUMNS[PVSystem.POWER_EST], connector=None, value_type=float)

    def localize(self, configs: Configurations) -> None:
        if configs.enabled:
            self._location = Location(
                configs.get_float("latitude"),
                configs.get_float("longitude"),
                timezone=configs.get("timezone", default="UTC"),
                altitude=configs.get_float("altitude", default=None),
                country=configs.get("country", default=None),
                state=configs.get("state", default=None),
            )
        else:
            self._location = None

    # # noinspection PyShadowingBuiltins
    # def evaluate(self, **kwargs):
    #     from .evaluation import Evaluation
    #     eval = Evaluation(self)
    #     return eval(**kwargs)

    # noinspection PyShadowingBuiltins, PyUnresolvedReferences
    def run(
        self,
        start: pd.Timestamp | dt.datetime = None,
        end: pd.Timestamp | dt.datetime = None,
        **kwargs
    ) -> pd.DataFrame:
        try:
            weather = self.weather.get(start, end, validate=True, **kwargs)
            result = pd.DataFrame(columns=[], index=weather.index)
            result.index.name = "timestamp"
            if self.has_type(PVSystem.TYPE):
                result.loc[:, [PVSystem.POWER, PVSystem.POWER_DC]] = 0.0
                for pv in self.get_all(PVSystem.TYPE):
                    pv_result = pv.run(weather)
                    result[[PVSystem.POWER, PVSystem.POWER_DC]] += pv_result[[PVSystem.POWER, PVSystem.POWER_DC]].abs()

                pv_power_channel = self.data[PVSystem.POWER_EST]
                pv_power = result[PVSystem.POWER]
                if not pv_power.empty:
                    pv_power_channel.set(pv_power.index[0], pv_power)
                else:
                    pv_power_channel.state = ChannelState.NOT_AVAILABLE

            return pd.concat([result, weather], axis="columns")

        except ComponentException as e:
            self._logger.warning(f"Unable to run system '{self.name}': {str(e)}")
