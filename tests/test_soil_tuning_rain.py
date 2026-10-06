# -*- coding: utf-8 -*-
"""
tests.test_soil_tuning_rain
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The bench reads precipitation as an hourly rate in mm/h, like the live chain, whatever its row spacing.

``soil_tuning`` pulls in the Dash UI stack at import time; ``importorskip`` keeps this out of environments that
lack it.
"""

from types import SimpleNamespace

import pytest

import pandas as pd

soil_tuning = pytest.importorskip("soil_tuning")


class _Stop(Exception):
    pass


def test_precipitation_rate_on_a_minute_row_gives_the_hourly_flux():
    ts = pd.Timestamp("2026-05-01 10:01")
    et_data = pd.DataFrame({"precipitation": [6.0]}, index=[ts])

    rates = soil_tuning._build_flux_rates(ts, 60.0, et_data, {}, pd.Series(dtype=float))

    assert rates.rain_flux == pytest.approx(6.0 / 3600.0)


def test_logged_rain_intensity_reaches_the_chain_as_the_precipitation_rate():
    index = pd.date_range("2026-05-01 10:00", periods=3, freq="1min", tz="UTC")
    logged = pd.DataFrame({"precipitation_intensity": [0.0, 6.0, 6.0], "ghi": 100.0, "temp_air": 15.0}, index=index)
    seen = {}

    def horizon_inputs(frame):
        seen["frame"] = frame
        raise _Stop

    field_sim = SimpleNamespace(
        weather=SimpleNamespace(data=SimpleNamespace(from_logger=lambda **kwargs: logged.copy())),
        simulation=SimpleNamespace(chain=SimpleNamespace(horizon_inputs=horizon_inputs)),
    )

    with pytest.raises(_Stop):
        soil_tuning._load_history(None, field_sim, index[0], index[-1])

    assert seen["frame"]["precipitation"].tolist() == [0.0, 6.0, 6.0]
