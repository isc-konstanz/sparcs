# -*- coding: utf-8 -*-
"""
penguin.components.agriculture.soil.model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class SoilModel(ABC):
    SECTION: str = "model"

    @abstractmethod
    def water_tension(self, water_content: float | pd.Series) -> float | pd.Series:
        """
        Calculate the soil water tension in hPa
        """
        ...

    @abstractmethod
    def water_content(self, water_tension: float | pd.Series) -> float | pd.Series:
        """
        Calculate the soil water content in cm^3^cm^−3^
        """
        ...

    @staticmethod
    def pf_from_pressure(water_tension: float | pd.Series) -> float:
        """

        See also
        --------
        https://de.wikipedia.org/wiki/PF-Wert
        """
        return np.log10(np.abs(water_tension))

    @staticmethod
    def pf_to_pressure(water_tension: float | pd.Series) -> float:
        """

        See also
        --------
        https://de.wikipedia.org/wiki/PF-Wert
        """
        return 10**water_tension


# noinspection SpellCheckingInspection
class Genuchten(SoilModel):
    """
    Mualem-van Genuchten Soil Model

    van Genuchten, M. Th. (1970):
    A Closed-form Equation for Predicting the Hydraulic Conductivity of Unsaturated Soil

    θ(ψ) = θ_r + ((θ_s − θ_r)/((1 + (α * abs(ψ)) ** n) ** (1−1/n))

    θ(ψ) is the water retention curve expressing the soil moisture ([cm^3^cm^−3^], or vol. %)
    ψ is the suction pressure (cm of water)
    θ_s is the saturated water content [cm^3^cm^−3^]
    θ_r is the residual water content [cm^3^cm^−3^]
    α is related to the inverse of the air entry suction, α > 0 [cm^−1^]
    n is a measure of the pore-size distribution, n > 1 (dimensionless)

    See also
    --------
    https://en.wikipedia.org/wiki/water_retention_curve
    https://github.com/martinvonk/pedon
    """

    theta_r: float
    theta_s: float
    k_s: float
    n: float
    alpha: float

    l: float  # noqa: E741

    def __init__(self, theta_r: float, theta_s: float, alpha: float, n: float, k_s: float, l: float = 0.5):  # noqa: E741
        """
        Mualem-van Genuchten Soil Model

        Parameters
        ----------
        theta_r: float
            Residual water content in cm^3^cm^−3^ or vol. %
        theta_s: float
            Saturated water content in cm^3^cm^−3^ or vol. %
        alpha: float
            Inverse of the air entry suction, with α > 0, in cm^−1^
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
        self.l = l

    def water_tension(self, water_content: float | pd.Series) -> float | pd.Series:
        water_column = self._c(water_content)
        return _water_column_to_hectopascals(water_column)

    def _c(self, theta: float | pd.Series) -> float | pd.Series:
        """
        Calculate the water column from the water content
        """
        c = 1 / ((theta - self.theta_r) / (self.theta_s - self.theta_r))
        c = np.sign(c) * np.abs(c) ** (1 / self.m) - 1
        c = np.sign(c) * np.abs(c) ** (1 / self.n) / self.alpha
        return c

    def water_content(self, water_tension: float | pd.Series) -> float | pd.Series:
        water_column = _hectopascal_to_water_column(water_tension)
        return self._theta(water_column)

    def _theta(self, c: float | pd.Series) -> float | pd.Series:
        """
        Calculate the soil moisture content from the water column
        """
        theta = self.theta_r + (self.theta_s - self.theta_r) / (1 + np.abs(self.alpha * c) ** self.n) ** self.m
        return theta

    def _s(self, c: float | pd.Series) -> float | pd.Series:
        """
        Calculate the effective saturation from the water column
        """
        return (self._theta(c) - self.theta_r) / (self.theta_s - self.theta_r)

    def _k_r(self, c: float | pd.Series, s: float | pd.Series | None = None) -> float | pd.Series:
        """
        Calculate the relative permeability from the water column
        """
        if s is None:
            s = self._s(c)
        return s**self.l * (1 - (1 - s ** (1 / self.m)) ** self.m) ** 2

    def _k(self, c: float | pd.Series, s: float | pd.Series | None = None) -> float | pd.Series:
        """
        Calculate the permeability from the water column
        """
        return self.k_s * self._k_r(c=c, s=s)


def _water_column_to_hectopascals(c: float) -> float:
    return c * 0.980665


def _hectopascal_to_water_column(p: float) -> float:
    return p * 1.019716
