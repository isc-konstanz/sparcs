# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.simulation.core.evapotranspiration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Penman-Monteith evapotranspiration per soil segment, as a pure model. This is
today's ``evapotranspiration.py`` without the Component base and the channel
registration; the per-term helpers keep their names so the move is a cut.
``evaluate`` is the ``publish=False`` branch of the live ``evaluate`` verbatim:
no channel writes, no ``self.data``/``self.context`` reads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from lories.components.weather import Weather


@dataclass(frozen=True)
class SegmentProperties:
    """Per-segment vegetation + radiation state for one ET evaluation.

    ``shade_factor`` in [0, 1] scales bulk GHI to local incoming shortwave.
    ``face_length`` [m] weights bulk means across segments.
    """

    name: str
    lai: float
    plant_height: float = 0.1
    ndvi: float = 0.25
    roughness: float = 0.002
    shade_factor: float = 1.0
    face_length: float = 0.0
    is_canopy: bool = False


class ETModel:
    # Beer-Lambert canopy extinction coefficient [-].
    BEER_K: float = 0.6

    # T_gnd = T_air + LIFT * GHI * shade_factor [K/(W/m^2)].
    TEMP_GROUND_LIFT: float = 0.012

    # Internal computation in kg/(m^2*s); the bulk frame publishes in kg/(m^2*h).
    _KG_PER_S_TO_KG_PER_H: float = 3600.0

    REQUIRED_WEATHER_COLUMNS: tuple[str, ...] = (
        Weather.TEMP_AIR,
        Weather.HUMIDITY_REL,
        Weather.GHI,
        Weather.WIND_SPEED,
        Weather.CLEAR_SKY_INDEX,
    )

    # Bulk-frame column names, matching the live Constant.key values exactly
    # (Constant is a str subclass, so these are the same strings today's
    # ``df[Evapotranspiration.NET_IRR]`` etc. resolve to).
    SVP = "sat_vapor_pressure"
    GVP = "ground_vapor_pressure"
    VAP_HEAT = "vaporization_heat"
    SVP_SLOPE = "slope_sat_vapor_pressure"
    NET_IRR = "net_irradiance"
    AIR_RES = "aerodynamic_resistance"
    SOIL_HEAT_FLOW = "soil_heat_flow"
    SURFACE_RES = "resistance_surface"
    RAD_TERM = "radiation_term"
    AER_TERM = "aerodynamic_term"
    EVAPOTRANSPIRATION = "evapotranspiration"

    def evaluate(
        self, weather: pd.DataFrame, segments: Sequence[SegmentProperties]
    ) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
        """Compute Penman-Monteith ET per segment.

        Returns ``(bulk, seg_et)``: ``bulk`` is ``weather`` augmented with the
        weather-only terms (``SVP``/``GVP``/``VAP_HEAT``/``SVP_SLOPE``) and the
        face-length-weighted mean of every per-segment term (``NET_IRR`` ...
        ``EVAPOTRANSPIRATION``, the last in kg/(m^2*h)); ``seg_et`` maps
        segment name to a frame with columns ``("et", "evap", "transp")`` in
        kg/(m^2*s).
        """
        seg_list = list(segments)
        if not seg_list:
            raise ValueError("ETModel.evaluate requires at least one segment")

        missing_cols = [k for k in self.REQUIRED_WEATHER_COLUMNS if k not in weather.columns or weather[k].isna().any()]
        if missing_cols:
            raise ValueError(f"Missing or NaN required columns for evapotranspiration: {missing_cols}")

        # Weather-only terms, segment-independent, computed once.
        svp = self._sat_vapor_pressure(temperature=weather[Weather.TEMP_AIR])
        gvp = self._ground_vapor_pressure(hum_rel=weather[Weather.HUMIDITY_REL], svp=svp)
        vh = self._vaporization_heat(temperature=weather[Weather.TEMP_AIR])
        svp_slope = self._slope_sat_vapor_pressure(temperature=weather[Weather.TEMP_AIR], svp=svp, vh=vh)

        # Per-segment Penman-Monteith. Vegetation properties and the local
        # radiation scaling come from the segment; everything weather-only
        # is reused as-is.
        seg_et: dict[str, pd.DataFrame] = {}
        seg_terms: dict[str, dict[str, pd.Series]] = {}
        ones = pd.Series(1.0, index=weather.index)
        ghi_pos = weather[Weather.GHI].clip(lower=0.0)
        for seg in seg_list:
            shade = float(seg.shade_factor)
            ghi_local = weather[Weather.GHI] * shade
            temp_gnd_local = weather[Weather.TEMP_AIR] + self.TEMP_GROUND_LIFT * ghi_pos * shade
            net_irr = self._net_irradiance(
                ghi=ghi_local,
                gvp=gvp,
                temp_air=weather[Weather.TEMP_AIR],
                temp_gnd=temp_gnd_local,
                ndvi=ones * float(seg.ndvi),
                csi=weather[Weather.CLEAR_SKY_INDEX],
            )
            air_res = self._aerodynamic_resistance(
                wind_speed=weather[Weather.WIND_SPEED],
                roughness=ones * float(seg.roughness),
                plant_height=ones * float(seg.plant_height),
                measure_height=2.0,
            )
            soil_heat = self._soil_heat_flow(lai=ones * float(seg.lai), net_irradiance=net_irr)
            surf_res = self._resistance_surface(lai=ones * float(seg.lai))
            rad_term = self._radiation_term(svp_slope=svp_slope, net_irradiance=net_irr, soil_heat_flow=soil_heat)
            aer_term = self._aerodynamic_term(svp=svp, gvp=gvp, aerodynamic_resistance=air_res)
            et = self._evapotranspiration(
                radiation_term=rad_term,
                aerodynamic_term=aer_term,
                vaporization_heat=vh,
                svp_slope=svp_slope,
                surface_resistance=surf_res,
                aerodynamic_resistance=air_res,
            )
            # Beer-Lambert evap/transp split; bare-soil segments have evap_frac=1 (no transpiration).
            if seg.is_canopy:
                evap_frac = float(np.exp(-self.BEER_K * float(seg.lai)))
            else:
                evap_frac = 1.0
            seg_et[seg.name] = pd.DataFrame(
                {
                    "et": et,
                    "evap": et * evap_frac,
                    "transp": et * (1.0 - evap_frac),
                }
            )
            seg_terms[seg.name] = {
                self.NET_IRR: net_irr,
                self.AIR_RES: air_res,
                self.SOIL_HEAT_FLOW: soil_heat,
                self.SURFACE_RES: surf_res,
                self.RAD_TERM: rad_term,
                self.AER_TERM: aer_term,
                self.EVAPOTRANSPIRATION: et,
            }

        bulk = weather.copy()
        bulk[self.SVP] = svp
        bulk[self.GVP] = gvp
        bulk[self.VAP_HEAT] = vh
        bulk[self.SVP_SLOPE] = svp_slope

        weights = np.array([max(s.face_length, 0.0) for s in seg_list], dtype=float)
        if weights.sum() <= 0:
            weights = np.ones(len(seg_list), dtype=float)
        weights /= weights.sum()

        for c in (
            self.NET_IRR,
            self.AIR_RES,
            self.SOIL_HEAT_FLOW,
            self.SURFACE_RES,
            self.RAD_TERM,
            self.AER_TERM,
            self.EVAPOTRANSPIRATION,
        ):
            stacked = pd.concat([seg_terms[s.name][c] for s in seg_list], axis=1)
            bulk[c] = stacked.to_numpy().dot(weights)

        # The bulk column carries the declared unit [kg/(m^2*h)]; the
        # per-segment seg_et decomposition stays in kg/(m^2*s) for the PDE.
        bulk[self.EVAPOTRANSPIRATION] *= self._KG_PER_S_TO_KG_PER_H

        return bulk, seg_et

    # noinspection PyPep8Naming
    @staticmethod
    def _sat_vapor_pressure(
        temperature: pd.Series,
    ) -> pd.Series:
        """Saturation vapor pressure [kPa] from air temperature [°C] (Magnus, above-freezing constants)."""
        SVP_AT_0C = 0.61078  # [kPa] at 0 °C
        B_POS, C_POS = 17.270, 237.3  # positive-temperature constants [-], [°C]

        return SVP_AT_0C * np.exp((B_POS * temperature) / (temperature + C_POS))

    @staticmethod
    def _ground_vapor_pressure(
        hum_rel: pd.Series,
        svp: pd.Series,
    ) -> pd.Series:
        """Ground-level vapor pressure [kPa]: e = RH/100 * SVP."""

        return hum_rel / 100 * svp

    # noinspection PyPep8Naming
    @staticmethod
    def _vaporization_heat(
        temperature: pd.Series,
    ) -> pd.Series:
        """Latent heat of vaporization [J/kg]; linear in temperature [°C]."""
        LATENT_HEAT_AT_0C = 2501.0  # [kJ/kg]
        TEMPERATURE_COEFFICIENT = 2.36  # [kJ/(kg*°C)]

        lambda_kj = LATENT_HEAT_AT_0C - TEMPERATURE_COEFFICIENT * temperature
        lambda_j = lambda_kj * 1000.0  # kJ/kg -> J/kg

        return lambda_j

    # noinspection PyPep8Naming
    @staticmethod
    def _slope_sat_vapor_pressure(
        temperature: pd.Series,
        svp: pd.Series,
        vh: pd.Series,
    ) -> pd.Series:
        """Slope of the saturation vapor pressure curve [kPa/K] via Clausius-Clapeyron."""
        GAS_CONSTANT_WATER_VAPOR = 461.0  # [J kg^-1 K^-1]

        temperature_k = _celsius_to_kelvin(temperature)
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
        """Net irradiance [W/m^2]: shortwave (GHI, albedo) minus longwave (Stefan-Boltzmann).

        Atmospheric emissivity: Brutsaert (1975). Surface emissivity: NDVI-adjusted. Cloud correction: empirical.
        """
        STEFAN_BOLTZMANN = 5.67e-8  # [W m^-2 K^-4]
        SURFACE_EMISSIVITY_BASE = 0.9585
        NDVI_EMISSIVITY_FACTOR = 0.0357
        CLOUD_TYPE_FACTOR = 0.22  # empirical
        ALBEDO = 0.2  # typical for grass

        temp_air_k = _celsius_to_kelvin(temp_air)
        temp_gnd_k = _celsius_to_kelvin(temp_gnd)

        epsilon_atm = 1.24 * (gvp * 10 / temp_air_k) ** (1 / 7)  # Brutsaert (1975)
        epsilon_surface = SURFACE_EMISSIVITY_BASE + NDVI_EMISSIVITY_FACTOR * ndvi
        epsilon_surface = epsilon_surface.clip(upper=1.0)
        # Clouds boost downward longwave: ~(1 + 0.22*cloudiness^2) with
        # cloudiness ~= 1 - clear-sky index (TVA-type correction).
        epsilon_atm_cloud = epsilon_atm * (1 + CLOUD_TYPE_FACTOR * (1.0 - csi) ** 2)

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
        """Aerodynamic resistance [s/m] via log wind profile (neutral stability, Monin-Obukhov)."""
        VON_KARMAN = 0.41  # [-]
        # FAO-56 floor for calm conditions; wind = 0 would drive ra to infinity
        # and silently zero the aerodynamic term.
        WIND_FLOOR_MS = 0.5  # [m/s]

        displacement_height = (2.0 / 3.0) * plant_height  # [m]
        roughness_momentum = (roughness * plant_height).clip(lower=1e-4)  # [m]
        roughness_heat = 0.1 * roughness_momentum  # [m]

        z_eff = measure_height - displacement_height
        z_eff = z_eff.clip(lower=1e-6)

        wind_ms = np.maximum(wind_speed, WIND_FLOOR_MS)
        ra = np.log(z_eff / roughness_momentum) * np.log(z_eff / roughness_heat) / (VON_KARMAN**2 * wind_ms)

        return pd.Series(ra, index=wind_speed.index)

    # noinspection PyPep8Naming
    @staticmethod
    def _soil_heat_flow(
        lai: pd.Series,
        net_irradiance: pd.Series,
    ) -> pd.Series:
        """Soil heat flux [W/m^2] from net irradiance via Beer-Lambert LAI attenuation (Choudhury-type)."""
        ATTENUATION_COEFF = 0.5
        FRACTION_BARE_SOIL = 0.4

        g = FRACTION_BARE_SOIL * np.exp(-ATTENUATION_COEFF * lai) * net_irradiance

        return g

    # noinspection PyPep8Naming
    @staticmethod
    def _resistance_surface(
        lai: pd.Series,
    ) -> pd.Series:
        """Bulk surface resistance [s/m] from LAI (FAO stomatal resistance formula)."""
        STOMATAL_RESISTANCE = 100.0  # [s/m] per leaf

        lai_safe = lai.clip(lower=1e-6)
        rs = STOMATAL_RESISTANCE / lai_safe

        return rs

    # noinspection PyPep8Naming
    @staticmethod
    def _radiation_term(
        svp_slope: pd.Series,
        net_irradiance: pd.Series,
        soil_heat_flow: pd.Series,
    ) -> pd.Series:
        """Radiation term of Penman-Monteith [(kPa*W)/(K*m^2)]: svp_slope * (Rn - G)."""

        return svp_slope * (net_irradiance - soil_heat_flow)

    # noinspection PyPep8Naming
    @staticmethod
    def _aerodynamic_term(
        svp: pd.Series,
        gvp: pd.Series,
        aerodynamic_resistance: pd.Series,
    ) -> pd.Series:
        """Aerodynamic term of Penman-Monteith [(kPa*J)/(m^2*K*s)]: rho*Cp*(SVP-GVP)/ra."""
        HEAT_CAPACITY_AIR = 1010.0  # [J/(kg*K)]
        AIR_DENSITY = 1.2  # [kg/m^3]

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
        """FAO-56 Penman-Monteith evapotranspiration [kg/(m^2*s)]."""
        PSYCHROMETRIC_CONSTANT = 0.067  # [kPa/K]

        resistance_factor = 1.0 + surface_resistance / aerodynamic_resistance
        denominator = vaporization_heat * (svp_slope + PSYCHROMETRIC_CONSTANT * resistance_factor)
        numerator = radiation_term + aerodynamic_term
        et = numerator / denominator

        return et


def _celsius_to_kelvin(temp_celsius: pd.Series | float) -> pd.Series | float:
    """Convert temperature from Celsius to Kelvin."""
    return temp_celsius + 273.15
