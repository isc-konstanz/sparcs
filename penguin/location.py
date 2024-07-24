# -*- coding: utf-8 -*-
"""
    penguin.location
    ~~~~~~~~~~~~~~~~


"""
from __future__ import annotations

import datetime as dt
from typing import Optional

import pvlib as pv

import loris
import pytz


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
        name: Optional[str] = None,
    ) -> None:
        super().__init__(
            latitude,
            longitude,
            timezone=timezone if tz is None else tz,
            altitude=altitude,
            country=country,
            state=state,
        )
        self.name = name

        if self._altitude is None:
            self._altitude = pv.location.lookup_altitude(self.latitude, self.longitude)

    # noinspection PyUnresolvedReferences
    @property
    def tz(self) -> str:
        return self.timezone.zone

    @property
    def pytz(self) -> dt.tzinfo:
        return self.timezone
