# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.evapotranspiration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from lories import Component, Configurations, Constant
from lories.components.weather import Weather


@dataclass
class SegmentProperties:
    """Per-segment vegetation + radiation state for one ET evaluation.

    Carries the vegetation properties Penman-Monteith needs locally
    (``lai``, ``plant_height``, ``ndvi``, ``roughness``) and the radiation
    scaling that turns bulk GHI into the segment's local incoming shortwave
    (``shade_factor`` ∈ [0, 1]; 0 = full PV shade, 1 = open sky).

    ``face_length`` is the segment's top-boundary face length [m] in the soil
    mesh, kept here so the ET component can publish face-length-weighted
    means on its bulk channels without reaching back into the mesh.
    """

    name: str
    lai: float
    plant_height: float = 0.1
    ndvi: float = 0.25
    roughness: float = 0.002
    shade_factor: float = 1.0
    face_length: float = 0.0
    is_canopy: bool = False


class Evapotranspiration(Component):
    TYPE: str = "evapotranspiration"

    # Penman-Monteith no longer needs any pre-computed inputs from the
    # DataFrame: ``temp_ground`` is now derived per-segment inside the
    # evaluation loop from ``T_air`` and the segment's shade-scaled GHI
    # (see ``TEMP_GROUND_LIFT``). Vegetation state flows in via
    # ``SegmentProperties``.
    REQUIRED_INPUT_KEYS: tuple[str, ...] = ()

    REQUIRED_WEATHER_CHANNELS = [
        Weather.TEMP_AIR,
        Weather.HUMIDITY_REL,
        Weather.GHI,
        Weather.WIND_SPEED,
        Weather.CLEAR_SKY_INDEX,
    ]

    # Output channels
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
    EVAPOTRANSPIRATION = Constant(float, "evapotranspiration", "Evapotranspiration", "kg/(m^2*h)")

    CHANNELS = [
        SVP,
        GVP,
        VAP_HEAT,
        SVP_SLOPE,
        NET_IRR,
        AIR_RES,
        SOIL_HEAT_FLOW,
        SURFACE_RES,
        RAD_TERM,
        AER_TERM,
        EVAPOTRANSPIRATION,
    ]

    def configure(self, configs: Configurations) -> None:
        super().configure(configs)

        for c in self.CHANNELS:
            self.data.add(c, aggregate="mean", logger={"enabled": False})

        # Per-segment ET channels live on FieldSimulation (the parent owns
        # the soil mesh, so segment-named channels are wired in one place).
        # ``evaluate`` writes into them via ``self.context.set_segment_values``.

    # Beer-Lambert canopy extinction. Shared with the soil PDE, but lives
    # here because the evap/transp split is part of the ET model.
    BEER_K: float = 0.6

    # Internal computation runs in kg/(m²·s) (what the soil PDE consumes
    # via ``seg_et``). Channels publish in kg/(m²·h) so the displayed
    # values aren't 0.0e+00 in the UI and read in a familiar hourly rate.
    _KG_PER_S_TO_KG_PER_H: float = 3600.0

    # Algebraic surface heating: T_gnd = T_air + LIFT · GHI · shade_factor.
    # 0.012 K/(W/m²) lands an open-sky segment ~12 K above air at noon
    # clear-sky (GHI≈1000) and a fully-shaded segment at T_air. Refine
    # once a real surface energy balance couples to the soil PDE.
    TEMP_GROUND_LIFT: float = 0.012

    def evaluate(
        self,
        df: pd.DataFrame,
        segments: Iterable[SegmentProperties],
        *,
        publish: bool = True,
    ) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
        """
        Compute Penman-Monteith evapotranspiration for each soil-mesh top
        segment and return both the bulk-augmented frame and the per-segment
        decomposition.

        Parameters
        ----------
        df : pd.DataFrame
            Weather frame. Must include the ``REQUIRED_WEATHER_CHANNELS``
            columns. Per-segment ground temperature is derived from
            ``Weather.TEMP_AIR`` and the shade-scaled GHI inside the loop.
        segments : Iterable[SegmentProperties]
            One entry per segment that should evaporate. The list also
            drives the face-length-weighted means published on the bulk
            output channels — pass a single representative segment when
            no soil mesh is wired.

        Returns
        -------
        (df, seg_et) : (pd.DataFrame, dict[str, pd.DataFrame])
            ``df`` is augmented in place with the eleven bulk output
            columns (face-length-weighted across segments for the
            segment-dependent terms; weather-only terms are direct).
            ``seg_et`` maps each segment name to a DataFrame indexed by
            ``df.index`` with columns ``("et", "evap", "transp")`` in
            kg/(m²·s); ``evap = et · exp(-k·LAI)`` and
            ``transp = et · (1 - exp(-k·LAI))``.

        Raises
        ------
        ValueError
            If required weather/ground columns are missing or no segments
            are provided.
        """
        seg_list = list(segments)
        if not seg_list:
            raise ValueError("Evapotranspiration.evaluate requires at least one segment")

        required_keys = list(self.REQUIRED_INPUT_KEYS) + [c.key for c in self.REQUIRED_WEATHER_CHANNELS]
        missing_cols = [k for k in required_keys if k not in df.columns or df[k].isna().any()]
        if missing_cols:
            raise ValueError(f"Missing or NaN required columns for evapotranspiration: {missing_cols}")

        # Weather-only terms — segment-independent, computed once.
        svp = self._sat_vapor_pressure(temperature=df[Weather.TEMP_AIR], only_pos=True)
        gvp = self._ground_vapor_pressure(hum_rel=df[Weather.HUMIDITY_REL], svp=svp)
        vh = self._vaporization_heat(temperature=df[Weather.TEMP_AIR])
        svp_slope = self._slope_sat_vapor_pressure(temperature=df[Weather.TEMP_AIR], svp=svp, vh=vh)

        # Per-segment Penman-Monteith. Vegetation properties and the local
        # radiation scaling come from the segment; everything weather-only
        # is reused as-is.
        seg_et: dict[str, pd.DataFrame] = {}
        seg_terms: dict[str, dict[Constant, pd.Series]] = {}
        seg_temp_gnd: dict[str, pd.Series] = {}
        ones = pd.Series(1.0, index=df.index)
        ghi_pos = df[Weather.GHI].clip(lower=0.0)
        for seg in seg_list:
            shade = float(seg.shade_factor)
            ghi_local = df[Weather.GHI] * shade
            temp_gnd_local = df[Weather.TEMP_AIR] + self.TEMP_GROUND_LIFT * ghi_pos * shade
            seg_temp_gnd[seg.name] = temp_gnd_local
            net_irr = self._net_irradiance(
                ghi=ghi_local,
                gvp=gvp,
                temp_air=df[Weather.TEMP_AIR],
                temp_gnd=temp_gnd_local,
                ndvi=ones * float(seg.ndvi),
                csi=df[Weather.CLEAR_SKY_INDEX],
            )
            air_res = self._aerodynamic_resistance(
                wind_speed=df[Weather.WIND_SPEED],
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
            # Beer-Lambert evap/transp split. Only canopy segments transpire;
            # bare-soil segments dump all their ET to the surface boundary
            # (``evap = et``). Without this gate, every bare strip would
            # silently feed water through the plant-cell sink in the soil PDE.
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
                Evapotranspiration.NET_IRR: net_irr,
                Evapotranspiration.AIR_RES: air_res,
                Evapotranspiration.SOIL_HEAT_FLOW: soil_heat,
                Evapotranspiration.SURFACE_RES: surf_res,
                Evapotranspiration.RAD_TERM: rad_term,
                Evapotranspiration.AER_TERM: aer_term,
                Evapotranspiration.EVAPOTRANSPIRATION: et,
            }

        # Bulk channel publishing: weather-only terms direct, segment-dependent
        # terms as face-length-weighted means (or simple mean when no face
        # lengths are supplied — single-segment fallback case).
        df[Evapotranspiration.SVP] = svp
        df[Evapotranspiration.GVP] = gvp
        df[Evapotranspiration.VAP_HEAT] = vh
        df[Evapotranspiration.SVP_SLOPE] = svp_slope

        weights = np.array([max(s.face_length, 0.0) for s in seg_list], dtype=float)
        if weights.sum() <= 0:
            weights = np.ones(len(seg_list), dtype=float)
        weights /= weights.sum()

        for c in (
            Evapotranspiration.NET_IRR,
            Evapotranspiration.AIR_RES,
            Evapotranspiration.SOIL_HEAT_FLOW,
            Evapotranspiration.SURFACE_RES,
            Evapotranspiration.RAD_TERM,
            Evapotranspiration.AER_TERM,
            Evapotranspiration.EVAPOTRANSPIRATION,
        ):
            stacked = pd.concat([seg_terms[s.name][c] for s in seg_list], axis=1)
            df[c] = stacked.to_numpy().dot(weights)

        # Publish the latest row to each bulk output channel so subscribers
        # (SoilSimulation, UI widgets, loggers) see live values. The
        # bulk EVAPOTRANSPIRATION channel is published in kg/(m²·h) — the
        # internal DataFrame stays in kg/(m²·s) for SoilSimulation.
        # ``publish=False`` (forecast / dry-run) skips every channel
        # write so the live dashboard isn't polluted with future values.
        if publish:
            ts = df.index[-1]
            for c in self.CHANNELS:
                value = float(df[c].iloc[-1])
                if c is Evapotranspiration.EVAPOTRANSPIRATION:
                    value *= self._KG_PER_S_TO_KG_PER_H
                self.data[c].set(ts, value)

            # Bulk ground temperature: face-length-weighted mean of per-segment
            # T_gnd derived above. The channel is owned by FieldSimulation
            # (parent); write through ``self.context`` so dashboards see the
            # physics-derived value instead of the placeholder.
            temp_gnd_stack = pd.concat([seg_temp_gnd[s.name] for s in seg_list], axis=1)
            temp_gnd_bulk_last = float(temp_gnd_stack.to_numpy().dot(weights)[-1])
            ctx_data = getattr(self.context, "data", None)
            if ctx_data is not None and "temp_ground" in ctx_data:
                ctx_data["temp_ground"].set(ts, temp_gnd_bulk_last)

            # Per-segment ET and ground temperature — written into the parent
            # FieldSimulation's bundled channels. Skipped on the single
            # ``_bulk`` fallback (no soil mesh wired) since the parent
            # registers no segment channels in that case.
            field = self.context
            if field.top_segment_names:
                et_mapping = {
                    s.name: float(seg_et[s.name]["et"].iloc[-1]) * self._KG_PER_S_TO_KG_PER_H for s in seg_list
                }
                tg_mapping = {s.name: float(seg_temp_gnd[s.name].iloc[-1]) for s in seg_list}
                field.set_segment_values(field.SEG_EVAPOTRANSPIRATION, ts, et_mapping)
                field.set_segment_values(field.SEG_TEMP_GROUND, ts, tg_mapping)

        return df, seg_et

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
        SVP_AT_0C = 0.61078  # Saturation vapor pressure at 0 °C [kPa]
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
        LATENT_HEAT_AT_0C = 2501.0  # [kJ/kg]
        TEMPERATURE_COEFFICIENT = 2.36  # [kJ/(kg °C)]

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
            Net irradiance [W/m^2]

        Notes
        -----
        - Longwave radiation is computed using the Stefan-Boltzmann law.
        - Atmospheric emissivity follows Brutsaert (1975).
        - Cloud correction is an empirical parameterization (Idso/Monteith-type).
        - Surface emissivity is adjusted using NDVI as a vegetation proxy.
        """

        # --- Empirical constant ---
        STEFAN_BOLTZMANN = 5.67e-8  # [W m^-2 K^-4]
        SURFACE_EMISSIVITY_BASE = 0.9585
        NDVI_EMISSIVITY_FACTOR = 0.0357
        CLOUD_TYPE_FACTOR = 0.22  # empirical (e.g. stratocumulus)
        ALBEDO = 0.2  # typical for grass

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
            Aerodynamic resistance [s/m]

        Notes
        -----
        - Assumes neutral atmospheric stability.
        - Based on logarithmic wind profile.
        """

        # --- Empirical constant ---
        VON_KARMAN = 0.41  # [-]

        # --- Surface geometry ---
        displacement_height = (2.0 / 3.0) * plant_height  # [m]
        roughness_momentum = roughness * plant_height  # [m]
        roughness_heat = 0.1 * roughness_momentum  # [m]

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
            Net irradiance [W/m^2]

        Returns
        -------
        pd.Series
            Soil heat flux [W/m^2]

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
        Compute bulk surface resistance [s/m] from Leaf Area Index.

        Parameters
        ----------
        lai : pd.Series
            Leaf Area Index [-]

        Returns
        -------
        pd.Series
            Surface resistance [s/m]

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
        AIR_DENSITY = 1.2  # Air density [kg/m^3]

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
            Latent heat of vaporization [J/kg]
        svp_slope : pd.Series
            Slope of saturation vapor pressure [kPa/K]
        surface_resistance : pd.Series
            Surface (stomatal) resistance [s/m]
        aerodynamic_resistance : pd.Series
            Aerodynamic resistance [s/m]

        Returns
        -------
        pd.Series
            Evapotranspiration [kg/m²/s ≈ mm/s]

        Notes
        -----
        - Psychrometric constant is assumed 0.067 kPa/K.
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
    return speed_kmh / 3.6
