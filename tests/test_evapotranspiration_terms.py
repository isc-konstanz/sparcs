# -*- coding: utf-8 -*-
"""Physics guards for the Penman-Monteith term helpers.

The cloud correction must boost downward longwave with CLOUDINESS
(clear-sky index 1 = clear sky): a clear night radiates away more heat than an
overcast one. The pre-fix formula scaled the boost with *clearness* and
inverted that. Calm wind must yield a finite aerodynamic resistance (FAO-56
floor) instead of silently zeroing the aerodynamic term via ra = inf.
"""

import numpy as np
import pandas as pd
from sparcs.components.agriculture.simulation.evapotranspiration import Evapotranspiration


def _series(value: float) -> pd.Series:
    return pd.Series([value], index=[pd.Timestamp("2026-07-01 00:00")])


def test_net_irradiance_clear_night_loses_more_than_overcast_night():
    common = {
        "ghi": _series(0.0),
        "gvp": _series(1.2),
        "temp_air": _series(15.0),
        "temp_gnd": _series(15.0),
        "ndvi": _series(0.25),
    }
    rn_clear = float(Evapotranspiration._net_irradiance(csi=_series(1.0), **common).iloc[0])
    rn_overcast = float(Evapotranspiration._net_irradiance(csi=_series(0.0), **common).iloc[0])
    assert rn_clear < rn_overcast < 0


def test_aerodynamic_resistance_finite_at_calm_wind():
    ra = Evapotranspiration._aerodynamic_resistance(
        wind_speed=_series(0.0),  # m/s
        roughness=_series(0.002),
        plant_height=_series(0.1),
        measure_height=2.0,
    )
    assert np.isfinite(float(ra.iloc[0]))
    assert float(ra.iloc[0]) > 0


def test_aerodynamic_resistance_finite_at_zero_plant_height():
    ra = Evapotranspiration._aerodynamic_resistance(
        wind_speed=_series(10.0),
        roughness=_series(0.002),
        plant_height=_series(0.0),
        measure_height=2.0,
    )
    assert np.isfinite(float(ra.iloc[0]))
