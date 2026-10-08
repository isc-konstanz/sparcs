# -*- coding: utf-8 -*-
"""
tests.test_fieldsim_chain
~~~~~~~~~~~~~~~~~~~~~~~~~

``WeatherChain``, ``ETModel`` and ``ShadingModel`` as pure models.
"""

from pathlib import Path

import pytest

import numpy as np
import pandas as pd
from lories.components.weather import Weather
from lories.core.configs.configurations import Configurations
from sparcs.components.agriculture.simulation.core.chain import WeatherChain
from sparcs.components.agriculture.simulation.core.config import FieldConfig, FieldSetup, SoilConfig
from sparcs.components.agriculture.simulation.core.evapotranspiration import ETModel, SegmentProperties
from sparcs.components.agriculture.simulation.core.pde import flow_m3s_per_m, segment_flux_dicts
from sparcs.components.agriculture.simulation.core.shading import ShadingConfig, ShadingModel

_GROUND_SHADING_CONF = Path(
    r"C:\Users\jb\My_Nextcloud\ISC_Share\Software\lories_sparcs\sparcs\data\copperhead\conf\agri_pv.d"
    r"\field_2.d\field_simulation.d\ground_shading.conf"
)


def _weather_frame(hours: int = 24) -> pd.DataFrame:
    """Synthetic hourly rows with every required weather column, daytime GHI centred on noon."""
    idx = pd.date_range("2026-06-21", periods=hours, freq="1h", tz="UTC")
    daylight = np.clip(np.sin(np.pi * (idx.hour - 5) / 14), 0.0, None)
    return pd.DataFrame(
        {
            Weather.GHI: 800.0 * daylight,
            Weather.TEMP_AIR: 20.0 + 5.0 * daylight,
            Weather.HUMIDITY_REL: 55.0,
            Weather.WIND_SPEED: 2.0,
            Weather.CLEAR_SKY_INDEX: 0.8,
            Weather.PRECIPITATION: 0.0,
        },
        index=idx,
    )


def _setup(total_drip_line_length_m: float = 12.6) -> FieldSetup:
    field = FieldConfig.from_dict({"lai_type": "grass", "bay_width": 3.5})
    soil = SoilConfig.from_dict({"mesh": {}, "total_drip_line_length_m": total_drip_line_length_m})
    return FieldSetup(field=field, soil=soil, shading=None, planner=None, location=None)


# --------------------------------------------------------------------------- ShadingConfig / real geometry


def test_shading_config_accepts_real_ground_shading_conf():
    """The real copperhead field_2 geometry conf must parse without error."""
    if not _GROUND_SHADING_CONF.is_file():
        pytest.skip(f"real conf not present: {_GROUND_SHADING_CONF}")
    configs = Configurations.load(_GROUND_SHADING_CONF.name, data_dir=str(_GROUND_SHADING_CONF.parent), flat=True)
    config = ShadingConfig()
    config.configure(configs)
    assert config.mode == "as_is"
    assert config.mirrored is True
    assert config.surface_tilt == 10.0
    assert config.surface_azimuth == 90.0
    assert config.axis_azimuth == 180.0
    assert config.albedo == 0.2


# --------------------------------------------------------------------------- ShadingModel free_field


def test_free_field_shading_all_ones_and_open_sky_ghi_equals_ghi():
    config = ShadingConfig.from_dict({"mode": "free_field"}).derive(bay_width=3.5, segment_ranges={"seg1": (0.0, 1.0)})
    model = ShadingModel(config)
    weather = _weather_frame()

    out = model.evaluate(weather)

    assert (out["seg1"] == 1.0).all()
    assert (out["open_sky_ghi"] == weather[Weather.GHI]).all()
    assert (out["ghi_seg1"] == weather[Weather.GHI]).all()


def test_free_field_shading_factor_is_one():
    config = ShadingConfig.from_dict({"mode": "free_field"}).derive(bay_width=3.5, segment_ranges={"seg1": (0.0, 1.0)})

    out = ShadingModel(config).evaluate(_weather_frame())

    assert (out[ShadingModel.SHADING_FACTOR] == 1.0).all()


def test_bay_mean_factor_is_length_weighted_over_the_middle_bay_and_sun_up_rows():
    """Seven rows 3.4 m apart put the bay at [8.5, 11.9]; the two pieces inside it weight to 650 W/m^2
    (a plain mean gives 500), and the ground outside the bay and the night row do not count."""
    model = ShadingModel(ShadingConfig.from_dict({"mode": "as_is"}).derive(bay_width=3.4))
    ground_day = [
        ((0.0, 0.0), (8.5, 0.0), {"qinc": 1000.0}),
        ((8.5, 0.0), (9.35, 0.0), {"qinc": 200.0}),
        ((9.35, 0.0), (11.9, 0.0), {"qinc": 800.0}),
        ((11.9, 0.0), (30.0, 0.0), {"qinc": 1000.0}),
    ]
    ground_night = [((0.0, 0.0), (30.0, 0.0), {"qinc": 0.0})]

    factor = model._bay_mean_factor([ground_day, ground_night], np.array([1000.0, 0.0]))

    assert factor == pytest.approx(0.65)


# --------------------------------------------------------------------------- ETModel


def test_et_model_positive_at_noon_for_canopy_segment():
    weather = _weather_frame()
    seg = SegmentProperties(
        name="top",
        lai=1.0,
        plant_height=0.5,
        ndvi=0.4,
        roughness=0.05,
        shade_factor=1.0,
        face_length=1.0,
        is_canopy=True,
    )

    bulk, seg_et = ETModel().evaluate(weather, [seg])

    noon = weather.index[12]
    assert seg_et["top"].loc[noon, "et"] > 0.0
    assert bulk.loc[noon, ETModel.EVAPOTRANSPIRATION] > 0.0


def test_et_model_raises_on_missing_required_column():
    weather = _weather_frame().drop(columns=[Weather.WIND_SPEED])
    seg = SegmentProperties(name="top", lai=1.0)

    with pytest.raises(ValueError, match="wind_speed"):
        ETModel().evaluate(weather, [seg])


# --------------------------------------------------------------------------- WeatherChain.forcing_series


def _chain(setup) -> WeatherChain:
    shading_config = ShadingConfig.from_dict({"mode": "free_field"}).derive(bay_width=3.5, segment_ranges=None)
    return WeatherChain(setup, ShadingModel(shading_config), ETModel(), top_segment_names=(), segment_face_length={})


def test_prepare_weather_uses_the_apple_apv_lai_profile():
    field = FieldConfig.from_dict({"lai_type": "apple_apv", "bay_width": 3.5})
    soil = SoilConfig.from_dict({"mesh": {}, "total_drip_line_length_m": 12.6})
    chain = _chain(FieldSetup(field=field, soil=soil, shading=None, planner=None, location=None))
    june = _weather_frame()
    july = june.set_axis(june.index + pd.DateOffset(months=1))
    october = june.set_axis(june.index + pd.DateOffset(months=4))

    assert set(chain._prepare_weather(july)["lai"]) == {4.2}
    assert set(chain._prepare_weather(october)["lai"]) == {3.4}


def test_forcing_series_dt_flow_rain_and_segment_fluxes():
    setup = _setup(total_drip_line_length_m=12.6)
    chain = _chain(setup)
    shading = chain.shading
    et = chain.et

    weather = _weather_frame()
    weather.iloc[5, weather.columns.get_loc(Weather.PRECIPITATION)] = 2.0  # mm/h at hour 5
    irrigation_lpm = pd.Series(0.53, index=weather.index)

    forcings, chain_result = chain.forcing_series(weather, irrigation_lpm)

    assert len(forcings) == len(weather.index)
    for forcing, ts in zip(forcings, weather.index):
        assert pd.Timestamp(forcing.at) == ts
    assert forcings[0].dt_s == 0.0
    assert all(f.dt_s == 3600.0 for f in forcings[1:])

    expected_flow = flow_m3s_per_m(0.53, setup.soil.total_drip_line_length_m)
    assert forcings[0].flow_m3s == pytest.approx(expected_flow)

    assert forcings[5].rain_flux == pytest.approx(2.0 / 3600.0)
    assert forcings[0].rain_flux == 0.0

    df = chain._prepare_weather(weather)
    shading_df = shading.evaluate(df)
    segments = chain._segments(df, shading_df)
    _, seg_et = et.evaluate(df, segments)
    noon = weather.index[12]
    expected_evap, expected_transp = segment_flux_dicts(seg_et, noon)
    assert forcings[12].seg_evap == pytest.approx(expected_evap)
    assert forcings[12].seg_transp == pytest.approx(expected_transp)

    assert chain_result.evapotranspiration.index.equals(weather.index)
    assert chain_result.shading.index.equals(weather.index)
    # free field: nothing to draw but the envelope
    assert (list(chain_result.ground), list(chain_result.pv_rows)) == ([], [])
    assert chain_result.sun_state == (90.0, 0.0, None)
    assert chain_result.envelope.y_min == -setup.soil.mesh.height - 0.5


def test_rain_is_an_hourly_rate_whatever_the_row_spacing():
    """Precipitation is in mm/h, so a minute row carries the same flux as an hourly row."""
    chain = _chain(_setup())
    weather = _weather_frame(hours=2).resample("1min").ffill()
    weather[Weather.PRECIPITATION] = 6.0

    forcings, _ = chain.forcing_series(weather, pd.Series(0.0, index=weather.index))

    assert [f.rain_flux for f in forcings[1:]] == pytest.approx([6.0 / 3600.0] * (len(forcings) - 1))


def test_forcing_windows_reach_backwards_from_their_row():
    chain = _chain(_setup())
    weather = _weather_frame(hours=3)
    irrigation_lpm = pd.Series(0.0, index=weather.index)

    forcings, _ = chain.forcing_series(weather, irrigation_lpm)

    assert [f.dt_s for f in forcings] == [0.0, 3600.0, 3600.0]
    for forcing, ts in zip(forcings, weather.index):
        assert pd.Timestamp(forcing.end) == ts
    assert pd.Timestamp(forcings[1].start) == weather.index[0]
    assert pd.Timestamp(forcings[2].start) == weather.index[1]


def test_forcing_series_drops_rows_at_or_before_the_frontier():
    chain = _chain(_setup())
    weather = _weather_frame(hours=3)
    irrigation_lpm = pd.Series(0.0, index=weather.index)

    forcings, chain_result = chain.forcing_series(weather, irrigation_lpm, frontier=weather.index[0])

    assert [pd.Timestamp(f.at) for f in forcings] == list(weather.index[1:])
    assert forcings[0].dt_s == 3600.0
    assert pd.Timestamp(forcings[0].start) == weather.index[0]
    assert chain_result.shading.index.equals(weather.index)


def test_forcing_series_first_window_spans_a_gapped_frontier():
    chain = _chain(_setup())
    weather = _weather_frame(hours=4)
    irrigation_lpm = pd.Series(0.0, index=weather.index)
    frontier = weather.index[0] - pd.Timedelta(minutes=30)

    forcings, _ = chain.forcing_series(weather, irrigation_lpm, frontier=frontier)

    assert len(forcings) == len(weather.index)
    assert forcings[0].dt_s == 1800.0
    assert pd.Timestamp(forcings[0].start) == frontier
