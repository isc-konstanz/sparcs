# -*- coding: utf-8 -*-
"""tests.test_fieldsim_chain
~~~~~~~~~~~~~~~~~~~~~~~~~~

``WeatherChain``, ``ETModel`` and ``ShadingModel`` as pure models: a weather
frame in, a ``Forcing`` list plus a ``ChainResult`` out, numerically
identical to what the live ``simulation`` Components compute with
``publish=False``. No FiPy import, no Gmsh, no channels.
"""

from pathlib import Path

import pytest

import numpy as np
import pandas as pd
from lories.components.weather import Weather
from lories.core.configs.configurations import Configurations
from sparcs.components.agriculture.fieldsim.core.chain import (
    WeatherChain,
    flow_m3s_per_m,
    rain_flux,
    segment_flux_dicts,
)
from sparcs.components.agriculture.fieldsim.core.config import FieldConfig, FieldSetup, SoilConfig
from sparcs.components.agriculture.fieldsim.core.evapotranspiration import ETModel, SegmentProperties
from sparcs.components.agriculture.fieldsim.core.shading import ShadingConfig, ShadingModel

# Real copperhead field_2 geometry conf (see project_copperhead_field2_config);
# the acceptance criterion is that ShadingConfig parses it without error.
_GROUND_SHADING_CONF = Path(
    r"C:\Users\jb\My_Nextcloud\ISC_Share\Software\lories_sparcs\sparcs\data\copperhead\conf\agri_pv.d"
    r"\field_2.d\field_simulation.d\ground_shading.conf"
)


def _weather_frame(hours: int = 24) -> pd.DataFrame:
    """24 synthetic hourly rows with every ``ETModel.REQUIRED_WEATHER_COLUMNS``
    column, daytime GHI > 0 centred on noon."""
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
    return FieldSetup(field=field, soil=soil, shading=None, planner=None, plots=None, model=None, location=None)


# --------------------------------------------------------------------------- ShadingConfig / real geometry


def test_shading_config_accepts_real_ground_shading_conf():
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


def test_et_model_matches_live_evapotranspiration_publish_false():
    live_mod = pytest.importorskip("sparcs.components.agriculture.simulation.evapotranspiration")
    weather = _weather_frame()
    kwargs = dict(
        name="top",
        lai=1.0,
        plant_height=0.5,
        ndvi=0.4,
        roughness=0.05,
        shade_factor=0.8,
        face_length=1.0,
        is_canopy=True,
    )
    ours_bulk, ours_seg_et = ETModel().evaluate(weather.copy(), [SegmentProperties(**kwargs)])

    live = object.__new__(live_mod.Evapotranspiration)
    live_bulk, live_seg_et = live.evaluate(weather.copy(), [live_mod.SegmentProperties(**kwargs)], publish=False)

    # Constant is a str subclass; rename to plain str so the column Index
    # comparison can't trip on the subclass identity, only the values.
    pd.testing.assert_frame_equal(ours_bulk.rename(columns=str), live_bulk.rename(columns=str))
    pd.testing.assert_frame_equal(ours_seg_et["top"], live_seg_et["top"])


# --------------------------------------------------------------------------- WeatherChain.forcing_series


def test_forcing_series_dt_flow_rain_and_segment_fluxes():
    setup = _setup(total_drip_line_length_m=12.6)
    shading_config = ShadingConfig.from_dict({"mode": "free_field"}).derive(bay_width=3.5, segment_ranges=None)
    shading = ShadingModel(shading_config)
    et = ETModel()
    chain = WeatherChain(setup, shading, et, top_segment_names=(), segment_face_length={})

    weather = _weather_frame()
    weather.iloc[5, weather.columns.get_loc(Weather.PRECIPITATION)] = 2.0  # mm at hour 5
    irrigation_lpm = pd.Series(0.53, index=weather.index)

    forcings, chain_result = chain.forcing_series(weather, irrigation_lpm)

    assert len(forcings) == len(weather.index)
    for forcing, ts in zip(forcings, weather.index):
        assert pd.Timestamp(forcing.start) == ts
    assert all(f.dt_s == 3600.0 for f in forcings)

    expected_flow = flow_m3s_per_m(0.53, setup.soil.total_drip_line_length_m)
    assert forcings[0].flow_m3s == pytest.approx(expected_flow)

    rain_ts = weather.index[5]
    assert forcings[5].rain_flux == pytest.approx(rain_flux(weather, rain_ts, forcings[5].dt_s))
    assert forcings[5].rain_flux > 0.0
    assert forcings[0].rain_flux == 0.0

    # Reconstruct the same per-segment ET the chain computed internally and
    # check the noon Forcing's seg_evap/seg_transp against the shared helper.
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
    assert chain_result.image is None
