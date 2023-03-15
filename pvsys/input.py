# -*- coding: utf-8 -*-
"""
    pvsys.input
    ~~~~~~~~~~~


"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pvlib import atmosphere


def precipitable_water_from_relative_humidity(temperature: pd.Series, relative_humidity: pd.Series) -> pd.Series:
    r"""
    Calculates precipitable water (cm) from ambient air temperature (C)
    and relatively humidity (%) using an empirical model. The
    accuracy of this method is approximately 20% for moderate PW (1-3
    cm) and less accurate otherwise.

    Parameters
    ----------
    temperature : `pd.Series`
        Ambient air temperature :math:`T` at the surface. [C]
    relative_humidity : `pd.Series`
        Relative humidity :math:`R_H` at the surface. [%]

    Returns
    -------
        `pd.Series`
            Precipitable water [cm]
    """
    return atmosphere.gueymard94_pw(temperature, relative_humidity)


def relative_humidity_from_dewpoint(temperature: pd.Series, dewpoint: pd.Series) -> pd.Series:
    r"""Calculate the relative humidity.
    Uses temperature and dewpoint to calculate relative humidity as the ratio of vapor
    pressure to saturation vapor pressures.
    Parameters
    ----------
    temperature : `pd.Series`
        Air temperature
    dewpoint : `pd.Series`
        Dewpoint temperature
    Returns
    -------
    `pd.Series`
        Relative humidity
    Notes
    -----
    .. math:: rh = \frac{e(T_d)}{e_s(T)}
    """
    e = saturation_vapor_pressure(dewpoint)
    e_s = saturation_vapor_pressure(temperature)
    return e / e_s


def saturation_vapor_pressure(temperature: pd.Series) -> pd.Series:
    r"""Calculate the saturation water vapor (partial) pressure.
    Parameters
    ----------
    temperature : `pd.Series`
        Air temperature
    Returns
    -------
    `pd.Series`
        Saturation water vapor (partial) pressure
    Notes
    -----
    Instead of temperature, dewpoint may be used in order to calculate
    the actual (ambient) water vapor (partial) pressure.
    The formula used is that from Bolton, D., 1980: The computation of
    equivalent potential temperature for T in degrees Celsius:
    .. math:: 6.112 e^\frac{17.67T}{T + 243.5}
    """
    return 6.112 * np.exp(
        17.67 * (temperature - 273.15) / (temperature - 29.65)
    )
