# -*- coding: utf-8 -*-
"""
penguin.system
~~~~~~~~~~~~~~


"""

from __future__ import annotations

import datetime as dt
from typing import Optional

import lori
import pandas as pd
from lori import ChannelState, ComponentException, Configurations
from penguin import Location, SolarSystem


class System(lori.System):
    # fmt: off
    POWER_EL:       str = "el_power"
    POWER_EL_IMP:   str = "el_import_power"
    POWER_EL_EXP:   str = "el_export_power"
    POWER_TH:       str = "th_power"
    POWER_TH_HT:    str = "th_ht_power"
    POWER_TH_DOM:   str = "th_dom_power"

    ENERGY_EL:      str = "el_energy"
    ENERGY_EL_IMP:  str = "el_import_energy"
    ENERGY_EL_EXP:  str = "el_export_energy"
    ENERGY_TH:      str = "th_energy"
    ENERGY_TH_HT:   str = "th_ht_energy"
    ENERGY_TH_DOM:  str = "th_dom_energy"
    # fmt: on

    _location: Optional[Location] = None

    def configure(self, configs: Configurations) -> None:
        super().configure(configs)
        if self.has_type(SolarSystem.TYPE):
            from penguin.constants import COLUMNS

            self.data.add(key=SolarSystem.POWER_EST, name=COLUMNS[SolarSystem.POWER_EST], connector=None, type=float)

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
        **kwargs,
    ) -> pd.DataFrame:
        try:
            weather = self.weather.get(start, end, validate=True, **kwargs)
            result = pd.DataFrame(columns=[], index=weather.index)
            result.index.name = "timestamp"
            if self.has_type(SolarSystem.TYPE):
                result.loc[:, [SolarSystem.POWER, SolarSystem.POWER_DC]] = 0.0
                for pv in self.get_all(SolarSystem.TYPE):
                    pv_result = pv.run(weather)
                    pv_columns = [SolarSystem.POWER, SolarSystem.POWER_DC]
                    result[pv_columns] += pv_result[pv_columns].abs()

                pv_power_channel = self.data[SolarSystem.POWER_EST]
                pv_power = result[SolarSystem.POWER]
                if not pv_power.empty:
                    pv_power_channel.set(pv_power.index[0], pv_power)
                else:
                    pv_power_channel.state = ChannelState.NOT_AVAILABLE

            return pd.concat([result, weather], axis="columns")

        except ComponentException as e:
            self._logger.warning(f"Unable to run system '{self.name}': {str(e)}")
