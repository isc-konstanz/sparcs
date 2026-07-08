# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.soil.model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

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
        Calculate the soil matric potential ψ from the effective saturation Sₑ.

        Parameters
        ----------
        se : float or array-like
            Effective saturation Sₑ [-], between 0 and 1.

        Returns
        -------
        float or array-like
            Signed matric potential ψ [hPa], negative: 0 at saturation and more
            negative as the soil dries (the tensiometer / DB convention). Its
            magnitude ``abs(ψ)`` is the suction/tension.
        """
        ...

    @abstractmethod
    def se_from_psi(self, psi: Any) -> Any:
        """
        Calculate the effective saturation Sₑ from the soil matric potential ψ.

        Parameters
        ----------
        psi : float or array-like
            Matric potential ψ [hPa]. Sign-agnostic: the inverse depends only on
            ``abs(ψ)``, so either the signed (negative) potential or its magnitude
            yields the same Sₑ.

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
        bpar: float = 0.5,
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
        bpar: float, optional
            Mualem pore-interaction exponent (L in Mualem 1976; ``BPar`` in
            HYDRUS-1D, where it is configurable per soil). Default 0.5 is
            the classic Mualem value and reproduces the prior behaviour.
        """
        self.theta_r = theta_r
        self.theta_s = theta_s
        self.alpha = alpha
        self.n = n
        self.m = 1 - 1 / n
        self.k_s = k_s
        self.bpar = bpar

    def psi_from_theta(self, theta: Any) -> Any:
        se = self._se_from_theta(theta)
        return self.psi_from_se(se)

    def theta_from_psi(self, psi: Any) -> Any:
        se = self.se_from_psi(psi)
        return self._theta_from_se(se)

    def psi_from_se(self, se: Any) -> Any:
        # Signed matric potential ψ (negative hPa; 0 at saturation, more negative
        # as the soil dries), the convention the real tensiometer and the DB store.
        # _water_column_from_se returns the head MAGNITUDE, so negate it. The
        # inverse se_from_psi is sign-agnostic (abs internally), so the round-trip
        # holds for either sign of ψ.
        water_column = self._water_column_from_se(se)
        return -_psi_from_water_column(water_column)

    def se_from_psi(self, psi: Any) -> Any:
        water_column = _water_column_from_psi(psi)
        return self._se_from_water_column(water_column)

    def k_from_theta(self, theta: Any) -> Any:
        se = self._se_from_theta(theta)
        return self.k_from_se(se)

    def k_from_se(self, se: Any) -> Any:
        k_rel = se**self.bpar * (1 - (1 - se ** (1 / self.m)) ** self.m) ** 2
        return self.k_s * k_rel

    def dh_dse(self, se: Any) -> Any:
        # |dh/dSe| in metres of water column per unit Se. This is the
        # quantity the Richards soil-water diffusivity D(Se) = K(Se)·|dh/dSe|
        # consumes when K is in m/s and the mesh is in metres; the product
        # then lands in m²/s. h here is the magnitude of the (negative)
        # matric pressure head; psi_from_se returns the same magnitude in
        # hPa. Conversion: `alpha` is in cm⁻¹ by van Genuchten convention,
        # so 1/alpha is in cm; the leading 0.01 takes cm → m.
        u = se ** (-1.0 / self.m) - 1.0
        return (0.01 / (self.n * self.m * self.alpha)) * u ** ((1.0 / self.n) - 1.0) * se ** (-(1.0 / self.m) - 1.0)

    def _se_from_theta(self, theta: Any) -> Any:
        return (theta - self.theta_r) / (self.theta_s - self.theta_r)

    def _theta_from_se(self, se: Any) -> Any:
        return self.theta_r + se * (self.theta_s - self.theta_r)

    def _water_column_from_se(self, se: Any) -> Any:
        # Magnitude of the (negative) pressure head in cm of water column.
        # van Genuchten: |h| = (1/α) · (Se^(-1/m) - 1)^(1/n). The previous
        # ``sign(x) * abs(x)`` wrappers were no-ops for physical Se ∈ (0,1)
        # (always ≥ 0) and would have returned garbage for negative inputs;
        # the clip below makes the contract explicit.
        se_clipped = np.clip(np.asarray(se, dtype=float), 1.0e-12, 1.0 - 1.0e-12)
        u = se_clipped ** (-1.0 / self.m) - 1.0
        return (u ** (1.0 / self.n)) / self.alpha

    def _se_from_water_column(self, wc: Any) -> Any:
        return 1 / ((1 + np.abs(self.alpha * wc) ** self.n) ** self.m)


def _psi_from_water_column(wc: float) -> float:
    return wc * 0.980665


def _water_column_from_psi(psi: float) -> float:
    return psi * 1.019716


# noinspection SpellCheckingInspection
class BrooksCorey(SoilModel):
    """
    Brooks–Corey (1964) retention + Mualem (1976) hydraulic conductivity.

    Water retention:

        Se(h) = (|α h|)^(-λ)   for |h| > 1/α (i.e. ψ > h_b = 1/α)
        Se(h) = 1              for |h| <= 1/α   (saturated above air-entry)
        θ(h)  = θ_r + (θ_s - θ_r) · Se(h)

    Mualem-Brooks-Corey relative hydraulic conductivity:

        k_r(Se) = Se^(2/λ + L + 2)
        K(Se)   = K_s · k_r(Se)

    Parameters
    ----------
    theta_r : float
        Residual volumetric water content [cm³ cm⁻³].
    theta_s : float
        Saturated volumetric water content [cm³ cm⁻³].
    alpha : float
        Inverse air-entry suction (1/h_b), α > 0 [cm⁻¹]. h_b = 1/α is the
        bubbling pressure in cm of water column.
    n : float
        Pore-size distribution index λ (called ``n`` in HYDRUS-1D's
        common parameter slot to keep TOML compatible with the
        Genuchten block). λ > 0; larger λ ⇒ more uniform pore sizes.
    k_s : float
        Saturated hydraulic conductivity [m s⁻¹].
    bpar : float, optional
        Mualem pore-interaction exponent L. Default 2.0 (the original
        Brooks–Corey–Mualem value); HYDRUS-1D exposes this as ``BPar``.

    References
    ----------
    Brooks, R. H., & Corey, A. T. (1964). Hydraulic Properties of Porous
    Media. Hydrology Papers, Colorado State University, 3.
    Mualem, Y. (1976). A new model for predicting the hydraulic
    conductivity of unsaturated porous media. Water Resour. Res., 12(3).
    """

    theta_r: float
    theta_s: float
    alpha: float
    n: float  # = λ (pore-size index)
    k_s: float
    bpar: float

    def __init__(
        self,
        theta_r: float,
        theta_s: float,
        alpha: float,
        n: float,
        k_s: float,
        bpar: float = 2.0,
    ):
        self.theta_r = theta_r
        self.theta_s = theta_s
        self.alpha = alpha
        self.n = n  # pore-size index λ
        self.k_s = k_s
        self.bpar = bpar
        # Mualem-Brooks-Corey conductivity exponent: 2/λ + L + 2
        self._k_exp = 2.0 / n + bpar + 2.0

    # -- water content ↔ tension ------------------------------------------------

    def psi_from_theta(self, theta: Any) -> Any:
        se = self._se_from_theta(theta)
        return self.psi_from_se(se)

    def theta_from_psi(self, psi: Any) -> Any:
        se = self.se_from_psi(psi)
        return self._theta_from_se(se)

    def psi_from_se(self, se: Any) -> Any:
        # Signed matric potential ψ (negative hPa; see Genuchten.psi_from_se).
        # _water_column_from_se is the head magnitude, so negate for the physical
        # sign; se_from_psi is the sign-agnostic inverse.
        wc = self._water_column_from_se(se)
        return -_psi_from_water_column(wc)

    def se_from_psi(self, psi: Any) -> Any:
        wc = _water_column_from_psi(psi)
        return self._se_from_water_column(wc)

    # -- hydraulic conductivity ------------------------------------------------

    def k_from_theta(self, theta: Any) -> Any:
        return self.k_from_se(self._se_from_theta(theta))

    def k_from_se(self, se: Any) -> Any:
        # Se is clipped to (0, 1]; above air-entry h_b the model defines
        # K = K_s exactly, which the formula recovers at Se = 1.
        se_eff = np.clip(np.asarray(se, dtype=float), 1.0e-12, 1.0)
        return self.k_s * se_eff**self._k_exp

    # -- diffusivity helper for the PDE ---------------------------------------

    def dh_dse(self, se: Any) -> Any:
        """
        |dh/dSe| in metres of water column per unit Se, for use as the
        diffusion coefficient in the Richards Se-form alongside K in
        m/s on a metre-scaled mesh. Above air-entry the inverse curve is
        flat so dh/dSe is unbounded; we clip Se to (0, 1) and rely on
        the source-term clipper in ``SoilSimulation`` to guard the
        near-saturated band.
        """
        # h = (1/α) · Se^(-1/λ)      (in cm of water column; α in cm⁻¹)
        # d|h|/dSe = (1/(α λ)) · Se^(-1/λ - 1)   (cm/Se)
        # multiply by 0.01 to convert cm/Se → m/Se.
        se_eff = np.clip(np.asarray(se, dtype=float), 1.0e-12, 1.0 - 1.0e-12)
        return (0.01 / (self.alpha * self.n)) * se_eff ** (-(1.0 / self.n) - 1.0)

    # -- internals -------------------------------------------------------------

    def _se_from_theta(self, theta: Any) -> Any:
        return (theta - self.theta_r) / (self.theta_s - self.theta_r)

    def _theta_from_se(self, se: Any) -> Any:
        return self.theta_r + se * (self.theta_s - self.theta_r)

    def _water_column_from_se(self, se: Any) -> Any:
        # Magnitude of the negative pressure head, in cm of water column.
        # Returns the air-entry value h_b = 1/α for Se ≥ 1.
        se_arr = np.asarray(se, dtype=float)
        se_eff = np.clip(se_arr, 1.0e-12, 1.0)
        return (1.0 / self.alpha) * se_eff ** (-1.0 / self.n)

    def _se_from_water_column(self, wc: Any) -> Any:
        # Brooks-Corey: Se = (|α wc|)^(-λ) above air-entry; 1 below.
        wc_arr = np.abs(np.asarray(wc, dtype=float))
        hb = 1.0 / self.alpha
        se = np.where(wc_arr <= hb, 1.0, (self.alpha * wc_arr) ** (-self.n))
        return se


# ---- registry & factory ------------------------------------------------------

_MODEL_ALIASES: dict[str, str] = {
    # canonical
    "van_genuchten": "van_genuchten",
    "brooks_corey": "brooks_corey",
    # common spellings
    "vg": "van_genuchten",
    "vangenuchten": "van_genuchten",
    "genuchten": "van_genuchten",
    "mualem_van_genuchten": "van_genuchten",
    "mualem-van-genuchten": "van_genuchten",
    "bc": "brooks_corey",
    "brookscorey": "brooks_corey",
    "brooks-corey": "brooks_corey",
    "mualem_brooks_corey": "brooks_corey",
}

_MODEL_REGISTRY: dict[str, type[SoilModel]] = {
    "van_genuchten": Genuchten,
    "brooks_corey": BrooksCorey,
}

DEFAULT_SOIL_MODEL: str = "van_genuchten"


def _canonical_model_name(name: Optional[str]) -> str:
    key = (name or DEFAULT_SOIL_MODEL).strip().lower().replace(" ", "_")
    canonical = _MODEL_ALIASES.get(key)
    if canonical is None:
        raise ValueError(f"Unknown soil model {name!r}. Available: {sorted(set(_MODEL_ALIASES.values()))}")
    return canonical


def create_soil_model(model: Optional[str] = None, **params: Any) -> SoilModel:
    """
    Build a :class:`SoilModel` by name, with kwarg filtering.

    The ``model`` argument can be any alias (e.g. ``"vg"``, ``"van_genuchten"``,
    ``"brooks_corey"``, ``"bc"``). Unrecognised kwargs are dropped silently so
    a single TOML ``[pde]`` block can carry both common params (``theta_r``,
    ``theta_s``, ``alpha``, ``n``, ``k_s``) and model-specific knobs
    (``bpar``) without forcing the caller to pre-partition them.

    Parameters
    ----------
    model : str, optional
        Model selector. Defaults to ``"van_genuchten"`` for backwards
        compatibility with existing configs.
    **params : Any
        Hydraulic parameters forwarded to the chosen model. Filtered against
        the model's ``__init__`` signature before instantiation.

    Returns
    -------
    SoilModel
        A configured concrete model instance.
    """
    import inspect

    canonical = _canonical_model_name(model)
    cls = _MODEL_REGISTRY[canonical]
    accepted = set(inspect.signature(cls).parameters)
    kwargs = {k: v for k, v in params.items() if k in accepted}
    missing = [p for p in ("theta_r", "theta_s", "k_s") if p in accepted and p not in kwargs]
    if missing:
        raise ValueError(f"Soil model {canonical!r} missing required parameter(s): {missing}")
    return cls(**kwargs)
