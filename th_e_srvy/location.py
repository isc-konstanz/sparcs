# -*- coding: utf-8 -*-
"""
    th-e-srvy.location
    ~~~~~~~~~~~~~~~~~~


"""
from __future__ import annotations
import datetime as dt
import pytz as tz
import pvlib as pv
import th_e_core as core


class Location(core.Location, pv.location.Location):

    def __init__(self,
                 latitude: float,
                 longitude: float,
                 timezone: str | dt.tzinfo = tz.utc,
                 altitude: float = None,
                 country: str = None,
                 state: str = None,
                 name: str = None):
        super().__init__(latitude,
                         longitude,
                         timezone=timezone,
                         altitude=altitude,
                         country=country,
                         state=state)
        self.name = name

        if self._altitude is None:
            self._altitude = pv.location.lookup_altitude(self.latitude, self.longitude)

    def __repr__(self):
        attrs = ['name', 'latitude', 'longitude', 'altitude', 'timezone']
        return ('Location:\n\t' + '\n\t'.join(
            f'{attr}: {str(getattr(self, attr))}' for attr in attrs))

    # noinspection PyUnresolvedReferences
    @property
    def tz(self) -> str:
        return self.timezone.zone

    @property
    def pytz(self) -> dt.tzinfo:
        return self.timezone
