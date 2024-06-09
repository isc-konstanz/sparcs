# -*- coding: utf-8 -*-
"""
    penguin.location
    ~~~~~~~~~~~~~~~~


"""
from __future__ import annotations

from typing import Optional

import datetime as dt
import loris
import pytz
import pvlib as pv


class Location(loris.Location, pv.location.Location):

    def __init__(
         self,
         latitude: float,
         longitude: float,
         timezone: str | dt.tzinfo = pytz.UTC,
         tz: Optional[str] = None,
         altitude: Optional[float] = None,
         country: Optional[str] = None,
         state: Optional[str] = None,
         name: Optional[str] = None
    ) -> None:
        super().__init__(
            latitude,
            longitude,
            timezone=timezone if tz is None else tz,
            altitude=altitude,
            country=country,
            state=state
        )
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
