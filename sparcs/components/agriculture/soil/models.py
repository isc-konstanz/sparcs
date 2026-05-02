# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.soil.model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class SoilModel(ABC):
    TYPE: str = "model"

    @abstractmethod
    def psi_from_theta(self, theta: Any) -> Any:
        """
        Calculate the soil water tension ψ from the volumetric water content θ.

        Parameters
        ----------
        theta : float or array-like
            Volumetric water content θ [cm³ cm⁻³].

        Returns
        -------
        float or array-like
            Soil water tension ψ [hPa].
        """
        ...

    @abstractmethod
    def theta_from_psi(self, psi: Any) -> Any:
        """
        Calculate the volumetric water content θ from the soil water tension ψ.

        Parameters
        ----------
        psi : float or array-like
            Soil water tension ψ [hPa].

        Returns
        -------
        float or array-like
            Volumetric water content θ [cm³ cm⁻³].
        """
        ...

    @abstractmethod
    def psi_from_se(self, se: Any) -> Any:
        """
        Calculate the soil water tension ψ from the effective saturation Sₑ.

        Parameters
        ----------
        se : float or array-like
            Effective saturation Sₑ [-], between 0 and 1.

        Returns
        -------
        float or array-like
            Soil water tension ψ [hPa].
        """
        ...

    @abstractmethod
    def se_from_psi(self, psi: Any) -> Any:
        """
        Calculate the effective saturation Sₑ from the soil water tension ψ.

        Parameters
        ----------
        psi : float or array-like
            Soil water tension ψ [hPa].

        Returns
        -------
        float or array-like
            Effective saturation Sₑ [-], between 0 and 1.
        """
        ...

    @abstractmethod
    def k_from_theta(self, theta: Any) -> Any:
        """
        Calculate the hydraulic conductivity k from the volumetric water content θ.

        Parameters
        ----------
        theta : float or array-like
            Volumetric water content θ [cm³ cm⁻³].

        Returns
        -------
        float or array-like
            Hydraulic conductivity k [m s⁻¹].
        """
        ...

    @abstractmethod
    def k_from_se(self, se: Any) -> Any:
        """
        Calculate the hydraulic conductivity k from the effective saturation Sₑ.

        Parameters
        ----------
        se : float or array-like
            Effective saturation Sₑ [-], between 0 and 1.

        Returns
        -------
        float or array-like
            Hydraulic conductivity k [m s⁻¹].
        """
        ...

    @staticmethod
    def pf_from_psi(water_tension: Any) -> Any:
        """
        Calculate the pF value from the soil water tension ψ.

        Parameters
        ----------
        water_tension : float or array-like
            Soil water tension ψ [hPa].

        Returns
        -------
        float or array-like
            pF value [-].

        See also
        --------
        https://de.wikipedia.org/wiki/PF-Wert
        """
        return np.log10(np.abs(water_tension))

    @staticmethod
    def psi_from_pf(water_tension: Any) -> Any:
        """
        Calculate the soil water tension ψ from the pF value.

        Parameters
        ----------
        water_tension : float or array-like
            pF value [-].

        Returns
        -------
        float or array-like
            Soil water tension ψ [hPa].

        See also
        --------
        https://de.wikipedia.org/wiki/PF-Wert
        """
        return -(10**water_tension)


# noinspection SpellCheckingInspection
class Genuchten(SoilModel):
    """
    Mualem-van Genuchten soil hydraulic model.

    van Genuchten (1970) water retention curve coupled with the Mualem
    model for unsaturated hydraulic conductivity. Relates pressure head
    (ψ), volumetric water content (θ), effective saturation (S_e), and
    hydraulic conductivity (k).

    Water retention:

        θ(ψ) = θ_r + (θ_s - θ_r) / (1 + |α ψ|^n)^m,    m = 1 - 1/n

    Effective saturation:

        S_e = (θ - θ_r) / (θ_s - θ_r)

    Mualem-van Genuchten relative hydraulic conductivity:

        k_r(S_e) = S_e^{1/2} * [1 - (1 - S_e^{1/m})^m]^2
        k = k_s * k_r

    Parameters
    ----------
    theta_r : float
        Residual volumetric water content [cm³ cm⁻³].
    theta_s : float
        Saturated volumetric water content [cm³ cm⁻³].
    alpha : float
        Inverse air-entry suction, α > 0 [cm⁻¹].
    n : float
        Pore-size distribution parameter, n > 1 [-].
    k_s : float
        Saturated hydraulic conductivity [m s⁻¹].

    Notes
    -----
    Pressure head ψ is expressed internally in centimeters of water column.
    This class provides conversions between θ ↔ ψ, θ ↔ S_e, ψ ↔ S_e, and
    S_e ↔ k.

    References
    ----------
    van Genuchten, M. Th. (1970). A closed-form equation for predicting the
    hydraulic conductivity of unsaturated soils. Soil Science Society of
    America Journal, 44(5), 892-898.

    See also
    --------
    https://en.wikipedia.org/wiki/Water_retention_curve
    https://github.com/martinvonk/pedon
    """

    theta_r: float
    theta_s: float
    alpha: float
    n: float
    k_s: float

    def __init__(
        self,
        theta_r: float,
        theta_s: float,
        alpha: float,
        n: float,
        k_s: float,
    ):
        """
        Mualem-van Genuchten Soil Model

        Parameters
        ----------
        theta_r: float
            Residual water content in cm^3^cm^-3^ or vol. %
        theta_s: float
            Saturated water content in cm^3^cm^-3^ or vol. %
        alpha: float
            Inverse of the air entry suction, with alpha > 0, in cm^-1^
        n: float
            Measure of the pore-size distribution, n > 1
        k_s: float
            Saturated permeability of the soil
        """
        self.theta_r = theta_r
        self.theta_s = theta_s
        self.alpha = alpha
        self.n = n
        self.m = 1 - 1 / n
        self.k_s = k_s

    def psi_from_theta(self, theta: Any) -> Any:
        se = self._se_from_theta(theta)
        return self.psi_from_se(se)

    def theta_from_psi(self, psi: Any) -> Any:
        se = self.se_from_psi(psi)
        return self._theta_from_se(se)

    def psi_from_se(self, se: Any) -> Any:
        water_column = self._water_column_from_se(se)
        return _psi_from_water_column(water_column)

    def se_from_psi(self, psi: Any) -> Any:
        water_column = _water_column_from_psi(psi)
        return self._se_from_water_column(water_column)

    def k_from_theta(self, theta: Any) -> Any:
        se = self._se_from_theta(theta)
        return self.k_from_se(se)

    def k_from_se(self, se: Any) -> Any:
        k_rel = se**0.5 * (1 - (1 - se ** (1 / self.m)) ** self.m) ** 2
        return self.k_s * k_rel

    def dpsi_dse(self, se: Any) -> Any:
        # |dψ/dSe| in hPa per unit Se. psi_from_se returns the suction ψ ≥ 0,
        # which decreases monotonically with Se, so the raw derivative dψ/dSe
        # is negative. The Richards soil-water diffusivity D(Se) = K(Se)·|dψ/dSe|
        # needs the magnitude (equivalently dψ_pressure/dSe under the
        # negative-pressure-head convention), so we return the positive value
        # directly. The 0.980665 cm-water → hPa factor mirrors psi_from_se.
        u = se ** (-1.0 / self.m) - 1.0
        return (
            (0.980665 / (self.n * self.m * self.alpha))
            * u ** ((1.0 / self.n) - 1.0)
            * se ** (-(1.0 / self.m) - 1.0)
        )

    def _se_from_theta(self, theta: Any) -> Any:
        return (theta - self.theta_r) / (self.theta_s - self.theta_r)

    def _theta_from_se(self, se: Any) -> Any:
        return self.theta_r + se * (self.theta_s - self.theta_r)

    def _water_column_from_se(self, se: Any) -> Any:
        # Todo: why using sign * abs here?
        wc = np.sign(se) * np.abs(se) ** (-1 / self.m) - 1
        wc = np.sign(wc) * np.abs(wc) ** (1 / self.n) / self.alpha
        return wc

    def _se_from_water_column(self, wc: Any) -> Any:
        return 1 / ((1 + np.abs(self.alpha * wc) ** self.n) ** self.m)


def _psi_from_water_column(wc: float) -> float:
    return wc * 0.980665


def _water_column_from_psi(psi: float) -> float:
    return psi * 1.019716
