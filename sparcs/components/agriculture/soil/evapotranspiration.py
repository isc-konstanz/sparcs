# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.soil.evapotranspiration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""

from __future__ import annotations


import numpy as np
import pandas as pd

from lories import Component, Constant, Configurations
from lories.components.weather import Weather


class Evapotranspiration(Component):
    TYPE: str = "evapotranspiration"

    # Inputs # TODO: Move these constants to weather
    # TEMP_AIR = Constant(float, "temp_air", "Air Temperature", "°C")
    TEMP_GROUND = Constant(float, "temp_ground", "Ground Temperature", "°C")
    # HUM_REL = Constant(float, "humidity_relative", "Relative Air Humidity", "%")
    # GHI = Constant(float, "ghi", "Global Horizontal Irradiance", "W/m^2")
    # WIND_SPEED = Constant(float, "wind_speed", "Wind Speed", "m/s")
    LAI = Constant(float, "lai", "Leaf Area Index", "m^2/m^2")
    ROUGHNESS = Constant(float, "roughness", "Roughness", "-")
    PLANT_HEIGHT = Constant(float, "plant_height", "Plant Height", "m")
    NDVI = Constant(float, "ndvi", "Normalized Difference Vegetation Index", "-")
    CSI = Constant(float, "csi", "Clear Sky Index", "-")

    REQUIRED = [
        Weather.TEMP_AIR,
        TEMP_GROUND,
        Weather.HUMIDITY_REL,
        Weather.GHI,
        Weather.WIND_SPEED,
        LAI,
        ROUGHNESS,
        PLANT_HEIGHT,
        NDVI,
        CSI,
    ]

    # Calculated
    SVP = Constant(float, "sat_vapor_pressure", "Saturation Vapor Pressure", "kPa")
    GVP = Constant(float, "ground_vapor_pressure", "Vapor Pressure on the Ground Surface", "kPa")
    VAP_HEAT = Constant(float, "vaporization_heat", "Latent Heat of Vaporization", "J/kg")
    SVP_SLOPE = Constant(float, "slope_sat_vapor_pressure", "Saturation Vapor Pressure Slope", "kPa/K")
    NET_IRR = Constant(float, "net_irradiance", "Net Irradiance", "W/m^2")
    AIR_RES = Constant(float, "aerodynamic_resistance", "Aerodynamic Resistance", "s/m")
    SOIL_HEAT_FLOW = Constant(float, "soil_heat_flow", "Soil Heat Flow", "W/m^2")
    SURFACE_RES = Constant(float, "resistance_surface", "Surface Resistance", "s/m")
    RAD_TERM = Constant(float, "radiation_term", "Radiation Term", "(kPa*W)/(K*m^2)")
    AER_TERM = Constant(float, "aerodynamic_term", "Aerodynamic Term", "(kPa*J)/(m^2*K*s)")
    EVAPOTRANSPIRATION = Constant(float, "evapotranspiration", "Evapotranspiration", "kg/(m^2*s)")

    def configure(self, configs: Configurations) -> None:
        super().configure(configs)


    def evapotranspiration(
        self,
        df: pd.DataFrame,
    ) -> pd.Series | pd.DataFrame:
        """
        Compute evapotranspiration (ET) with the Penman-Monteith formulation.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe containing all required meteorological and vegetation
            variables.

        Returns
        -------
        pd.Series | pd.DataFrame
            Evapotranspiration [kg/(m^2*s) ≈ mm/s].

        Raises
        ------
        ValueError
            If one or more required input columns are missing.
        """

        missing_cols = [col.key for col in self.REQUIRED if col.key not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns for evapotranspiration calculation: {missing_cols}")

        df[Evapotranspiration.SVP] = self._sat_vapor_pressure(
            temperature=df[Weather.TEMP_AIR],
            only_pos=True
        )

        df[Evapotranspiration.GVP] = self._ground_vapor_pressure(
            hum_rel=df[Weather.HUMIDITY_REL],
            svp=df[Evapotranspiration.SVP],
        )

        df[Evapotranspiration.VAP_HEAT] = self._vaporization_heat(
            temperature=df[Weather.TEMP_AIR],
        )

        df[Evapotranspiration.SVP_SLOPE] = self._slope_sat_vapor_pressure(
            temperature=df[Weather.TEMP_AIR],
            svp=df[Evapotranspiration.SVP],
            vh=df[Evapotranspiration.VAP_HEAT],
        )

        df[Evapotranspiration.NET_IRR] = self._net_irradiance(
            ghi=df[Weather.GHI],
            gvp=df[Evapotranspiration.GVP],
            temp_air=df[Weather.TEMP_AIR],
            temp_gnd=df[Evapotranspiration.TEMP_GROUND],
            ndvi=df[Evapotranspiration.NDVI],
            csi=df[Evapotranspiration.CSI],
        )

        df[Evapotranspiration.AIR_RES] = self._aerodynamic_resistance(
            wind_speed=df[Weather.WIND_SPEED],
            roughness=df[Evapotranspiration.ROUGHNESS],
            plant_height=df[Evapotranspiration.PLANT_HEIGHT],
            measure_height=2.0,
        )

        df[Evapotranspiration.SOIL_HEAT_FLOW] = self._soil_heat_flow(
            lai=df[Evapotranspiration.LAI],
            net_irradiance=df[Evapotranspiration.NET_IRR]
        )

        df[Evapotranspiration.SURFACE_RES] = self._resistance_surface(
            lai=df[Evapotranspiration.LAI],
        )

        df[Evapotranspiration.RAD_TERM] = self._radiation_term(
            svp_slope=df[Evapotranspiration.SVP_SLOPE],
            net_irradiance=df[Evapotranspiration.NET_IRR],
            soil_heat_flow=df[Evapotranspiration.SOIL_HEAT_FLOW],
        )

        df[Evapotranspiration.AER_TERM] = self._aerodynamic_term(
            svp=df[Evapotranspiration.SVP],
            gvp=df[Evapotranspiration.GVP],
            aerodynamic_resistance=df[Evapotranspiration.AIR_RES],
        )

        df[Evapotranspiration.EVAPOTRANSPIRATION] = self._evapotranspiration(
            radiation_term=df[Evapotranspiration.RAD_TERM],
            aerodynamic_term=df[Evapotranspiration.AER_TERM],
            vaporization_heat=df[Evapotranspiration.VAP_HEAT],
            svp_slope=df[Evapotranspiration.SVP_SLOPE],
            surface_resistance=df[Evapotranspiration.SURFACE_RES],
            aerodynamic_resistance=df[Evapotranspiration.AIR_RES],
        )

        print(df[:50].to_string())

        return df[Evapotranspiration.EVAPOTRANSPIRATION]

    # noinspection PyPep8Naming
    @staticmethod
    def _sat_vapor_pressure(
        temperature: pd.Series,
        only_pos: bool = True,
    ) -> pd.Series:
        """
        Compute saturation vapor pressure [kPa] from air temperature.

        Parameters
        ----------
        temperature : pd.Series
            Air temperature [°C].
        only_pos : bool, default=True
            If True, use the positive-temperature parameterization only.
            If False, blend positive and negative parameterizations smoothly.

        Returns
        -------
        pd.Series
            Saturation vapor pressure [kPa].

        Notes
        -----
        - Uses empirical constants for vapor pressure over water/ice.
        - The blended branch avoids a hard discontinuity around 0 °C.
        """

        # --- Empirical constant ---
        SVP_AT_0C = 0.61078          # Saturation vapor pressure at 0 °C [kPa]
        B_POS, C_POS = 17.270, 237.3  # Positive-temperature constants [-], [°C]
        B_NEG, C_NEG = 21.875, 265.5  # Negative-temperature constants [-], [°C]
        ATAN_WIDTH = 10.0
        ATAN_TRANSITION_SHARPNESS = 6.313  # = np.tan(0.9 * np.pi / 2) * 2

        if only_pos:
            svp = SVP_AT_0C * np.exp((B_POS * temperature) / (temperature + C_POS))
        else:
            svp_n = SVP_AT_0C * ((B_NEG * temperature) / (temperature + C_NEG)).apply(np.exp)
            svp_p = SVP_AT_0C * ((B_POS * temperature) / (temperature + C_POS)).apply(np.exp)

            atan_v = np.arctan(temperature * ATAN_TRANSITION_SHARPNESS / ATAN_WIDTH) / np.pi + 0.5
            svp = svp_n + (svp_p - svp_n) * atan_v

        return svp

    @staticmethod
    def _ground_vapor_pressure(
        hum_rel: pd.Series,
        svp: pd.Series,
    ) -> pd.Series:
        """
        Compute vapor pressure near the ground surface [kPa].

        Parameters
        ----------
        hum_rel : pd.Series
            Relative humidity [%].
        svp : pd.Series
            Saturation vapor pressure [kPa].

        Returns
        -------
        pd.Series
            Ground-level vapor pressure [kPa].

        Notes
        -----
        - Uses the standard relation: e = RH/100 * es.
        """

        return hum_rel / 100 * svp

    # noinspection PyPep8Naming
    @staticmethod
    def _vaporization_heat(
        temperature: pd.Series,
    ) -> pd.Series:
        """
        Latent vaporization heat [J/kg]

        Parameters
        ----------
        temperature : pd.Series
            Air temperature [°C]

        Returns
        -------
        pd.Series
            Latent heat of vaporization [J/kg]

        Notes
        -----
        - Linear approximation of temperature dependence.
        - Commonly used in hydrology and evapotranspiration models.
        - Valid for typical atmospheric temperature ranges.
        """

        # --- Empirical constant ---
        LATENT_HEAT_AT_0C = 2501.0        # [kJ/kg]
        TEMPERATURE_COEFFICIENT = 2.36    # [kJ/(kg °C)]

        # --- Linear temperature dependence ---
        lambda_kj = LATENT_HEAT_AT_0C - TEMPERATURE_COEFFICIENT * temperature

        # --- Unit conversion (kJ/kg → J/kg) ---
        lambda_j = lambda_kj * 1000.0

        return lambda_j

    # noinspection PyPep8Naming
    @staticmethod
    def _slope_sat_vapor_pressure(
        temperature: pd.Series,
        svp: pd.Series,
        vh: pd.Series,
    ) -> pd.Series:
        """
        Slope of the saturation vapor pressure curve [kPa/K]
        using the Clausius-Clapeyron relation.

        Parameters
        ----------
        temperature : pd.Series
            Air temperature [°C]
        svp : pd.Series
            Saturation vapor pressure [kPa]
        vh : pd.Series
            Latent heat of vaporization [J/kg]

        Returns
        -------
        pd.Series
            Slope of saturation vapor pressure curve [kPa/K]

        Notes
        -----
        - Based on the Clausius-Clapeyron equation.
        - Gas constant for water vapor is assumed constant.
        """

        # --- Physical constant ---
        GAS_CONSTANT_WATER_VAPOR = 461.0  # [J kg⁻¹ K⁻¹]

        # --- Unit conversion ---
        temperature_k = _celsius_to_kelvin(temperature)

        # --- Clausius–Clapeyron slope ---
        delta = (vh * svp) / (GAS_CONSTANT_WATER_VAPOR * temperature_k**2)

        return delta

    # noinspection PyPep8Naming
    @staticmethod
    def _net_irradiance(
        ghi: pd.Series,
        gvp: pd.Series,
        temp_air: pd.Series,
        temp_gnd: pd.Series,
        ndvi: pd.Series,
        csi: pd.Series,
    ) -> pd.Series:
        """
        Net irradiance (Rn) [W/m^2] as the balance of shortwave and longwave radiation.

        Parameters
        ----------
        ghi : pd.Series
            Global Horizontal Irradiance [W/m^2]
        gvp : pd.Series
            Actual vapor pressure [kPa] (required for Brutsaert equation)
        temp_air : pd.Series
            Air temperature [°C]
        temp_gnd : pd.Series
            Ground/surface temperature [°C]
        ndvi : pd.Series
            Normalized Difference Vegetation Index [-]
        csi : pd.Series
            Clear Sky Index [-]

        Returns
        -------
        pd.Series
            Net irradiance Rn [W/m^2]

        Notes
        -----
        - Longwave radiation is computed using the Stefan-Boltzmann law.
        - Atmospheric emissivity follows Brutsaert (1975).
        - Cloud correction is an empirical parameterization (Idso/Monteith-type).
        - Surface emissivity is adjusted using NDVI as a vegetation proxy.
        """

        # --- Empirical constant ---
        STEFAN_BOLTZMANN = 5.67e-8   # [W m^-2 K^-4]
        SURFACE_EMISSIVITY_BASE = 0.9585
        NDVI_EMISSIVITY_FACTOR = 0.0357
        CLOUD_TYPE_FACTOR = 0.22     # empirical (e.g. stratocumulus)
        ALBEDO = 0.2                 # typical for grass

        # --- Unit conversions ---
        temp_air_k = _celsius_to_kelvin(temp_air)
        temp_gnd_k = _celsius_to_kelvin(temp_gnd)

        # --- Atmospheric emissivity (Brutsaert, 1975) ---
        epsilon_atm = 1.24 * (gvp * 10 / temp_air_k) ** (1 / 7)

        # --- Surface emissivity (NDVI-adjusted) ---
        epsilon_surface = SURFACE_EMISSIVITY_BASE + NDVI_EMISSIVITY_FACTOR * ndvi
        epsilon_surface = epsilon_surface.clip(upper=1.0)  # physically bounded

        # --- Cloud correction (empirical) ---
        epsilon_atm_cloud = epsilon_atm * (1 + CLOUD_TYPE_FACTOR * csi**2)

        # --- Radiation components ---
        shortwave_net = ghi * (1 - ALBEDO)
        longwave_in = epsilon_atm_cloud * STEFAN_BOLTZMANN * temp_air_k**4
        longwave_out = epsilon_surface * STEFAN_BOLTZMANN * temp_gnd_k**4

        rn = shortwave_net + longwave_in - longwave_out

        return rn

    # noinspection PyPep8Naming
    @staticmethod
    def _aerodynamic_resistance(
        wind_speed: pd.Series,
        roughness: pd.Series,
        plant_height: pd.Series,
        measure_height: float,
    ) -> pd.Series:
        """
        Aerodynamic resistance [s/m] using a logarithmic wind profile
        based on Monin-Obukhov similarity theory (neutral conditions).

        Parameters
        ----------
        wind_speed : pd.Series
            Wind speed at measurement height [km/h]
        roughness : pd.Series
            Dimensionless roughness scaling factor [-]
        plant_height : pd.Series
            Vegetation height [m]
        measure_height : float
            Height of wind measurement [m]

        Returns
        -------
        pd.Series
            Aerodynamic resistance ra [s/m]

        Notes
        -----
        - Assumes neutral atmospheric stability.
        - Based on logarithmic wind profile.
        """

        # --- Empirical constant ---
        VON_KARMAN = 0.41  # [-]

        # --- Surface geometry ---
        displacement_height = (2.0 / 3.0) * plant_height        # [m]
        roughness_momentum = roughness * plant_height           # [m]
        roughness_heat = 0.1 * roughness_momentum               # [m]

        # --- Effective measurement height ---
        z_eff = measure_height - displacement_height
        z_eff = z_eff.clip(lower=1e-6)  # avoid log(0) / division issues

        # --- Log wind profile (neutral) ---
        ra = (
            np.log(z_eff / roughness_momentum)
            * np.log(z_eff / roughness_heat)
            / (VON_KARMAN**2 * _kmh_to_ms(wind_speed))
        )

        return pd.Series(ra, index=wind_speed.index)

    # noinspection PyPep8Naming
    @staticmethod
    def _soil_heat_flow(
        lai: pd.Series,
        net_irradiance: pd.Series,
    ) -> pd.Series:
        """
        Estimate soil heat flux (G) [W/m^2] from net irradiance using an
        exponential attenuation with vegetation (LAI).

        Parameters
        ----------
        lai : pd.Series
            Leaf Area Index [-]
        net_irradiance : pd.Series
            Net irradiance Rn [W/m^2]

        Returns
        -------
        pd.Series
            Soil heat flux G [W/m^2]

        Notes
        -----
        - Based on Choudhury-type parameterization (FAO context).
        - Negative Rn (nighttime) will produce negative soil heat flux.
        """

        # --- Empirical constant ---
        ATTENUATION_COEFF = 0.5
        FRACTION_BARE_SOIL = 0.4

        # --- Soil heat flux ---
        g = FRACTION_BARE_SOIL * np.exp(-ATTENUATION_COEFF * lai) * net_irradiance

        return g

    # noinspection PyPep8Naming
    @staticmethod
    def _resistance_surface(
        lai: pd.Series,
    ) -> pd.Series:
        """
        Compute bulk surface resistance (rs) [s/m] from Leaf Area Index.

        Parameters
        ----------
        lai : pd.Series
            Leaf Area Index [-]

        Returns
        -------
        pd.Series
            Surface resistance rs [s/m]

        Notes
        -----
        - Based on FAO formulation for well-watered vegetation.
        - Assumes uniform stomatal resistance.
        """

        # --- Empirical constant ---
        STOMATAL_RESISTANCE = 100.0  # [s/m] per leaf

        # --- Avoid division by zero ---
        lai_safe = lai.clip(lower=1e-6)

        # --- Surface resistance ---
        rs = STOMATAL_RESISTANCE / lai_safe

        return rs

    # noinspection PyPep8Naming
    @staticmethod
    def _radiation_term(
        svp_slope: pd.Series,
        net_irradiance: pd.Series,
        soil_heat_flow: pd.Series,
    ) -> pd.Series:
        """
        Radiation term of Penman-Monteith.

        Parameters
        ----------
        svp_slope : pd.Series
            Slope of saturation vapor pressure curve [kPa/K].
        net_irradiance : pd.Series
            Net irradiance [W/m^2].
        soil_heat_flow : pd.Series
            Soil heat flow [W/m^2].

        Returns
        -------
        pd.Series
            Radiation term [(kPa*W)/(K*m^2)].

        Notes
        -----
        - Represents the radiative energy available for latent heat flux.
        """

        return svp_slope * (net_irradiance - soil_heat_flow)

    # noinspection PyPep8Naming
    @staticmethod
    def _aerodynamic_term(
        svp: pd.Series,
        gvp: pd.Series,
        aerodynamic_resistance: pd.Series,
    ) -> pd.Series:
        """
        Aaerodynamic term of Penman-Monteith.

        Parameters
        ----------
        svp : pd.Series
            Saturation vapor pressure [kPa].
        gvp : pd.Series
            Ground-level vapor pressure [kPa].
        aerodynamic_resistance : pd.Series
            Aerodynamic resistance [s/m].

        Returns
        -------
        pd.Series
            Aerodynamic term [(kPa*J)/(m^2*K*s)].

        Notes
        -----
        - Uses bulk air properties with constant heat capacity and density.
        """

        # --- Empirical constant ---
        HEAT_CAPACITY_AIR = 1010.0  # Heat capacity of air [J/(kg*K)]
        AIR_DENSITY = 1.2           # Air density [kg/m^3]

        return AIR_DENSITY * HEAT_CAPACITY_AIR * (svp - gvp) / aerodynamic_resistance

    # noinspection PyPep8Naming
    @staticmethod
    def _evapotranspiration(
        radiation_term: pd.Series,
        aerodynamic_term: pd.Series,
        vaporization_heat: pd.Series,
        svp_slope: pd.Series,
        surface_resistance: pd.Series,
        aerodynamic_resistance: pd.Series,
    ) -> pd.Series:
        """
        Evapotranspiration (ET) using the Penman-Monteith equation

        Parameters
        ----------
        radiation_term : pd.Series
            Radiation component of Penman-Monteith [(kPa·W)/(K·m²)]
        aerodynamic_term : pd.Series
            Aerodynamic component [(kPa·J)/(K·m²·s)]
        vaporization_heat : pd.Series
            Latent heat of vaporization λ [J/kg]
        svp_slope : pd.Series
            Slope of saturation vapor pressure curve Δ [kPa/K]
        surface_resistance : pd.Series
            Surface (stomatal) resistance rs [s/m]
        aerodynamic_resistance : pd.Series
            Aerodynamic resistance ra [s/m]

        Returns
        -------
        pd.Series
            Evapotranspiration ET [kg/m²/s ≈ mm/s]

        Notes
        -----
        - Psychrometric constant γ is assumed 0.067 kPa/K.
        - Follows FAO-56 Penman-Monteith formulation.
        - Numerator and denominator units are consistent with SI.
        """

        # --- Physical constant ---
        PSYCHROMETRIC_CONSTANT = 0.067  # [kPa/K]

        # --- Denominator ---
        resistance_factor = 1.0 + surface_resistance / aerodynamic_resistance
        denominator = vaporization_heat * (svp_slope + PSYCHROMETRIC_CONSTANT * resistance_factor)

        # --- Numerator ---
        numerator = radiation_term + aerodynamic_term  # note: sum, not product

        # --- Penman-Monteith evapotranspiration ---
        et = numerator / denominator

        return et


def _celsius_to_kelvin(temp_celsius: pd.Series | float) -> pd.Series | float:
    """Convert temperature from Celsius to Kelvin."""
    return temp_celsius + 273.15


def _kmh_to_ms(speed_kmh: pd.Series | float) -> pd.Series | float:
    """Converts speed from km/h to m/s."""
    return speed_kmh * 3.6