# -*- coding: utf-8 -*-
"""
tests.test_fieldsim_plots_wiring
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Progress images through the real wiring: ``ShadingModel`` keeps the current tick's frame inputs,
not the planner's horizon roll, and ``ChannelOutputs`` renders real matplotlib PNGs.
"""

import datetime as dt
from dataclasses import replace
from types import SimpleNamespace

import pytest

import numpy as np
import pandas as pd
from lories.components.weather import Weather
from sparcs.components.agriculture.simulation.components import ChannelOutputs
from sparcs.components.agriculture.simulation.core.chain import WeatherChain
from sparcs.components.agriculture.simulation.core.config import FieldConfig, FieldSetup, PlotConfig, SoilConfig
from sparcs.components.agriculture.simulation.core.engine import SoilEngine
from sparcs.components.agriculture.simulation.core.evapotranspiration import ETModel
from sparcs.components.agriculture.simulation.core.shading import ShadingConfig, ShadingModel
from sparcs.components.agriculture.simulation.core.state import SoilState, StepResult

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_NIGHT = (90.0, 0.0, None)
_SEGMENT_RANGES = {
    "LeftTopSegment_0": (-1.5, -1.0),
    "PlantTopLeftSegment": (-1.0, 0.0),
    "PlantTopRightSegment": (0.0, 1.0),
    "RightTopSegment_0": (1.0, 1.5),
}


def _weather(start: str, *, zenith: float, azimuth: float, hours: int = 2) -> pd.DataFrame:
    """Rows with the solar position given, so no ``Location`` is needed to derive it."""
    idx = pd.date_range(start, periods=hours, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "solar_zenith": zenith,
            "solar_azimuth": azimuth,
            Weather.GHI: 600.0,
            Weather.DNI: 500.0,
            Weather.DHI: 100.0,
            Weather.TEMP_AIR: 22.0,
            Weather.HUMIDITY_REL: 55.0,
            Weather.WIND_SPEED: 2.0,
            Weather.CLEAR_SKY_INDEX: 0.8,
            Weather.PRECIPITATION: 0.0,
        },
        index=idx,
    )


def _chain() -> WeatherChain:
    """The copperhead A-frame (mirrored ``as_is``) over a 3.4 m bay."""
    setup = FieldSetup(
        field=FieldConfig.from_dict({"bay_width": 3.4}), soil=SoilConfig.from_dict({"mesh": {}}), shading=None
    )
    config = ShadingConfig.from_dict(
        {"mode": "as_is", "mirrored": True, "surface_tilt": 10.0, "surface_azimuth": 90.0, "axis_azimuth": 180.0}
    ).derive(bay_width=3.4, segment_ranges=_SEGMENT_RANGES)
    return WeatherChain(
        setup,
        ShadingModel(config),
        ETModel(),
        top_segment_names=tuple(_SEGMENT_RANGES),
        segment_face_length={name: 0.5 for name in _SEGMENT_RANGES},
    )


def _tick(chain: WeatherChain, weather: pd.DataFrame):
    return chain.forcing_series(weather, pd.Series(0.0, index=weather.index))


class _Channel:
    def __init__(self):
        self.calls = []

    def set(self, ts, value):
        self.calls.append((ts, value))


class _Child:
    def __init__(self, name: str, image_key: str):
        self.name = name
        self.plot_config = PlotConfig.from_dict()
        self._last_plot_ts = None
        self._plot_strikes = 0
        self.data = {image_key: _Channel(), "plot_strikes": _Channel(), "shading_factor": _Channel()}
        self.image = self.data[image_key]


class _FakeMesh:
    def __init__(self, cell_centers: np.ndarray) -> None:
        self.cellCenters = cell_centers


# --------------------------------------------------------------------------- ShadingModel memory


def test_the_live_tick_remembers_the_frame_inputs_and_the_horizon_roll_does_not():
    chain = _chain()
    model = chain.shading
    day = _weather("2026-06-21 11:00", zenith=30.0, azimuth=180.0)
    _tick(chain, day)
    ground, pv_rows, sun_state = model.render_inputs_at(day.index[-1])
    assert ground and sun_state == (30.0, 180.0, 180.0)
    assert len(pv_rows) == sum(setup.n_rows for setup in model._setups)  # both halves of the A-frame

    later = _weather("2026-06-21 16:00", zenith=70.0, azimuth=260.0)
    chain.horizon_inputs(later)

    assert model._last_ground is ground
    assert model._last_pv_rows is pv_rows
    assert model._last_sun_state == sun_state

    _tick(chain, later)

    assert model._last_sun_state == (70.0, 260.0, 180.0)
    assert model._last_ground is not ground


def test_a_night_tick_keeps_the_pv_rows_and_drops_ground_and_shadows():
    chain = _chain()
    model = chain.shading
    _tick(chain, _weather("2026-06-21 11:00", zenith=30.0, azimuth=180.0))
    pv_rows = model.pv_rows_at(pd.Timestamp("2026-06-21 12:00", tz="UTC"))

    night = _weather("2026-06-21 23:00", zenith=110.0, azimuth=10.0)
    _, result = _tick(chain, night)

    assert list(result.ground) == []
    assert result.pv_rows is pv_rows
    assert result.sun_state == _NIGHT
    assert result.envelope.center_x == pytest.approx(model._middle_row_x())


# --------------------------------------------------------------------------- real renders


def test_shading_progress_image_renders_a_png_from_the_chain_result():
    chain = _chain()
    day = _weather("2026-06-21 11:00", zenith=30.0, azimuth=180.0)
    _, result = _tick(chain, day)
    child = _Child("ground_shading", "shading_progress_image")
    field = SimpleNamespace(
        setup=replace(chain.setup, location=SimpleNamespace(timezone="Europe/Berlin")),
        simulation=SimpleNamespace(engine=SimpleNamespace(top_segment_names=[])),
    )
    outputs = ChannelOutputs(field, shading=child, et=None, soil=None, predictor=None)

    outputs.chain(day.index[-1], result)

    [(ts, png)] = child.image.calls
    assert ts == day.index[-1]
    assert png.startswith(_PNG_SIGNATURE)
    assert child.data["plot_strikes"].calls == [(ts, 0.0)]
    assert child._last_plot_ts == ts


def _soil_outputs(mesh, *, width: float, height: float):
    child = _Child("soil_simulation", "soil_progress_image")
    field = SimpleNamespace(
        setup=SimpleNamespace(location=None, soil=SimpleNamespace(mesh=SimpleNamespace(width=width, height=height))),
        simulation=SimpleNamespace(engine=SimpleNamespace(mesh=mesh)),
    )
    return child, ChannelOutputs(field, shading=None, et=None, soil=child, predictor=None)


def _step(se: np.ndarray, at: dt.datetime) -> StepResult:
    return StepResult(state=SoilState(se=se, se_old=se.copy(), surface_h={}, at=at), diagnostics={})


def test_soil_progress_image_renders_a_png_per_step():
    rng = np.random.default_rng(0)
    width, height, n = 3.0, 1.5, 200
    mesh = _FakeMesh(np.vstack([rng.uniform(0.0, width, n), rng.uniform(-height, 0.0, n)]))
    child, outputs = _soil_outputs(mesh, width=width, height=height)
    at = dt.datetime(2026, 6, 21, 9, tzinfo=dt.timezone.utc)

    outputs.steps([_step(rng.uniform(0.2, 0.9, n), at)])

    [(ts, png)] = child.image.calls
    assert ts == pd.Timestamp(at)
    assert png.startswith(_PNG_SIGNATURE)
    assert child.data["plot_strikes"].calls == [(ts, 0.0)]


@pytest.mark.slow  # builds a real Gmsh mesh
def test_soil_progress_image_renders_off_the_engines_real_mesh(pde_core_factory):
    core = pde_core_factory("plots_wiring")
    engine = SoilEngine(SimpleNamespace(mesh=core.mesh_config), None, core)
    child, outputs = _soil_outputs(engine.mesh, width=core.mesh_config.width, height=core.mesh_config.height)
    at = dt.datetime(2026, 6, 21, 9, tzinfo=dt.timezone.utc)

    outputs.steps([_step(core.snapshot(), at)])

    [(ts, png)] = child.image.calls
    assert ts == pd.Timestamp(at)
    assert png.startswith(_PNG_SIGNATURE)
