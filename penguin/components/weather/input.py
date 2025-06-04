# -*- coding: utf-8 -*-
"""
penguin.input
~~~~~~~~~~~~~


"""

from __future__ import annotations

from pvlib import atmosphere
from pvlib.irradiance import complete_irradiance, dirint, disc, dni

import numpy as np
import pandas as pd
from lori import Constant, Location, Weather, WeatherException

SOLAR_ELEVATION = Constant(float, "solar_elevation", "Solar Elevation", "°")
SOLAR_ZENITH = Constant(float, "solar_zenith", "Solar Zenith", "°")
SOLAR_AZIMUTH = Constant(float, "solar_azimuth", "Solar Azimuth", "°")

validated_meteo_inputs = [
    Weather.GHI,
    Weather.DHI,
    Weather.DNI,
    Weather.TEMP_AIR,
    Weather.TEMP_DEW_POINT,
    Weather.HUMIDITY_REL,
    Weather.PRECIPITABLE_WATER,
    Weather.PRESSURE_SEA,
    Weather.WIND_SPEED,
    SOLAR_ELEVATION,
    SOLAR_ZENITH,
    SOLAR_AZIMUTH,
]


# noinspection PyUnresolvedReferences, PyTypeChecker
def validate_meteo_inputs(weather: pd.DataFrame, location: Location) -> pd.DataFrame:
    def assert_columns(*columns: str) -> bool:
        if any(column for column in columns if column not in weather.columns or weather[column].isna().any()):
            return False
        return True

    def insert_column(column: str, data: pd.DataFrame) -> None:
        if column not in weather.columns or weather[column].dropna().empty:
            weather[column] = data
        else:
            weather[column] = weather[column].combine_first(data)

    if not assert_columns(Weather.PRESSURE_SEA):
        pressure = pd.Series(index=weather.index, data=atmosphere.alt2pres(location.altitude))
        insert_column(Weather.PRESSURE_SEA, pressure)

    solar_position = location.get_solarposition(weather.index, pressure=weather[Weather.PRESSURE_SEA])

    if not assert_columns(Weather.GHI, Weather.DHI, Weather.DNI):
        if not assert_columns(Weather.GHI):
            if not assert_columns(Weather.CLOUD_COVER):
                raise WeatherException(
                    f"Unable to estimate missing '{Weather.GHI}' data with "
                    f"missing or invalid column: {Weather.CLOUD_COVER}"
                )
            ghi = global_irradiance_from_cloud_cover(location, weather[Weather.CLOUD_COVER], solar_position)
            insert_column(Weather.GHI, ghi)

        if not assert_columns(Weather.DHI, Weather.DNI):
            if not assert_columns(Weather.DHI):
                if not assert_columns(Weather.GHI):
                    raise WeatherException(
                        f"Unable to estimate missing '{Weather.DHI}' and '{Weather.DNI}' data with "
                        f"missing or invalid columns: {', '.join([Weather.GHI])}"
                    )
                kwargs = {}
                if assert_columns(Weather.TEMP_DEW_POINT):
                    kwargs["temp_dew"] = weather[Weather.TEMP_DEW_POINT]
                if assert_columns(Weather.PRESSURE_SEA):
                    kwargs["pressure"] = weather[Weather.PRESSURE_SEA]

                dhi, dni = direct_diffuse_from_global_irradiance(solar_position, weather[Weather.GHI], **kwargs)
                insert_column(Weather.DHI, dhi)
                insert_column(Weather.DNI, dni)
            else:
                if not assert_columns(Weather.GHI, Weather.DHI):
                    raise WeatherException(
                        f"Unable to estimate missing '{Weather.DNI}' data with "
                        f"missing or invalid columns: {', '.join([Weather.GHI, Weather.DHI])}"
                    )
                dni = direct_normal_from_global_diffuse_irradiance(
                    solar_position, weather[Weather.GHI], weather[Weather.DHI]
                )
                insert_column(Weather.DNI, dni)

    if not assert_columns(Weather.HUMIDITY_REL):
        if not assert_columns(Weather.TEMP_AIR, Weather.TEMP_DEW_POINT):
            raise WeatherException(
                f"Unable to estimate missing '{Weather.HUMIDITY_REL}' data with "
                f"missing or invalid columns: {', '.join([Weather.TEMP_AIR, Weather.TEMP_DEW_POINT])}"
            )
        else:
            insert_column(
                Weather.HUMIDITY_REL,
                relative_humidity_from_dewpoint(weather[Weather.TEMP_AIR], weather[Weather.TEMP_DEW_POINT]),
            )
    if not assert_columns(Weather.PRECIPITABLE_WATER):
        if not assert_columns(Weather.TEMP_AIR, Weather.HUMIDITY_REL):
            raise WeatherException(
                f"Unable to estimate missing '{Weather.PRECIPITABLE_WATER}' data with "
                f"missing or invalid columns: {', '.join([Weather.TEMP_AIR, Weather.HUMIDITY_REL])}"
            )
        else:
            insert_column(
                Weather.PRECIPITABLE_WATER,
                precipitable_water_from_relative_humidity(weather[Weather.TEMP_AIR], weather[Weather.HUMIDITY_REL]),
            )

    insert_column(SOLAR_AZIMUTH, solar_position["azimuth"])
    insert_column(SOLAR_ZENITH, solar_position["apparent_zenith"])
    insert_column(SOLAR_ELEVATION, solar_position["apparent_elevation"])

    return weather


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
    e = _saturation_vapor_pressure(dewpoint)
    e_s = _saturation_vapor_pressure(temperature)
    return e / e_s * 100


def _saturation_vapor_pressure(temperature: pd.Series) -> pd.Series:
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
    kelvin = temperature + 273.15
    return 6.112 * np.exp(17.67 * (kelvin - 273.15) / (kelvin - 29.65))


def direct_normal_from_global_diffuse_irradiance(solar_position, ghi, dhi):
    """
    Determine DNI from GHI and DHI.

    When calculating the DNI from GHI and DHI the calculated DNI may be
    unreasonably high or negative for zenith angles close to 90 degrees
    (sunrise/sunset transitions). This function identifies unreasonable DNI
    values and sets them to NaN. If the clearsky DNI is given unreasonably high
    values are cut off.

    Parameters
    ----------
    solar_position : DataFrame
        Solar positions in decimal degrees
    ghi : Series
        Global horizontal irradiance.
    dhi : Series
        Diffuse horizontal irradiance.

    Returns
    -------
    dni : Series
        The modeled direct normal irradiance.
    """

    # Use true (not refraction-corrected) zenith angles in decimal
    irr = dni(ghi, dhi, solar_position["zenith"])
    return _fill_irradiance(irr)


# noinspection PyShadowingNames
def direct_diffuse_from_global_irradiance(solar_position, ghi, **kwargs):
    """
    Estimates DNI using the DIRINT or DISC model and calculate DHI from DNI and GHI.

    Parameters
    ----------
    solar_position : DataFrame
        Solar positions in decimal degrees
    ghi : Series
        Global horizontal irradiance.
    **kwargs
        Passed to the method that does the conversion

    Returns
    -------
    dhi, dni : Series
        Estimated DHI and DNI.
    """
    if "temp_dew" in kwargs:
        dni = dirint(ghi, solar_position["zenith"], ghi.index, **kwargs).fillna(0)
    else:
        dni = disc(ghi, solar_position["zenith"], ghi.index, **kwargs)["dni"]

    dhi = complete_irradiance(solar_position["zenith"], ghi=ghi, dni=dni)["dhi"]
    return (
        _fill_irradiance(dhi),
        _fill_irradiance(dni),
    )


# noinspection PyShadowingNames
def global_irradiance_from_cloud_cover(location, cloud_cover, solar_position=None, method="linear", **kwargs):
    """
    Estimates irradiance from cloud cover in the following steps:
    1. Determine clear sky GHI using Ineichen model and
       climatological turbidity.
    2. Estimate cloudy sky GHI using a function of cloud_cover

    Parameters
    ----------
    location : Location
        Location to estimate irradiance for.
    cloud_cover : Series
        Cloud cover in %.
    solar_position : DataFrame
        Solar positions in decimal degrees
    method : str, default 'linear'
        Method for converting cloud cover to GHI.
        'linear' is currently the only option.
    **kwargs
        Passed to the method that does the conversion

    Returns
    -------
    ghi : Series
        The modeled global horizontal irradiance.
    """
    if solar_position is None:
        solar_position = location.get_solarposition(cloud_cover.index)
    cs = location.get_clearsky(cloud_cover.index, model="ineichen", solar_position=solar_position)

    method = method.lower()
    if method == "linear":
        ghi = _cloud_cover_to_ghi_linear(cloud_cover, cs["ghi"], **kwargs)
    else:
        raise ValueError("invalid method argument")

    return _fill_irradiance(ghi)


def _cloud_cover_to_ghi_linear(cloud_cover, ghi_clear, offset=35):
    """
    Convert cloud cover to GHI using a linear relationship.
    0% cloud cover returns ghi_clear.
    100% cloud cover returns offset*ghi_clear.
    Parameters
    ----------
    cloud_cover: numeric
        Cloud cover in %.
    ghi_clear: numeric
        GHI under clear sky conditions.
    offset: numeric, default 35
        Determines the minimum GHI.
    Returns
    -------
    ghi: numeric
        Estimated GHI.
    References
    ----------
    Larson et. al. "Day-ahead forecasting of solar power output from
    photovoltaic plants in the American Southwest" Renewable Energy
    91, 11-20 (2016).
    """
    offset = offset / 100.0
    cloud_cover = cloud_cover / 100.0
    ghi = (offset + (1 - offset) * (1 - cloud_cover)) * ghi_clear
    return ghi


def _fill_irradiance(irradiance):
    if irradiance.isna().any():
        irradiance = irradiance.interpolate(method="akima")
        irradiance[irradiance < 1e-3] = 0
    return irradiance.abs()
