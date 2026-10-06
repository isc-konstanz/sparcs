# -*- coding: utf-8 -*-
"""
tests.test_soil_tuning_replay
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The soil-tuning bench replays a window through the live chain and the live ``SoilEngine``.
``soil_tuning`` pulls in the Dash UI stack at import time; ``importorskip`` keeps this out of
environments that lack it.
"""

import pickle
import queue
import types

import pytest
from conftest import MESH_KW, load_configs

import numpy as np
import pandas as pd
from lories.components.weather import Weather

soil_tuning = pytest.importorskip("soil_tuning")

from sparcs.components.agriculture.simulation.core.chain import WeatherChain  # noqa: E402
from sparcs.components.agriculture.simulation.core.config import FieldConfig, FieldSetup, SoilConfig  # noqa: E402
from sparcs.components.agriculture.simulation.core.evapotranspiration import ETModel  # noqa: E402
from sparcs.components.agriculture.simulation.core.pde import PDEConfig  # noqa: E402
from sparcs.components.agriculture.simulation.core.shading import ShadingConfig, ShadingModel  # noqa: E402
from sparcs.components.agriculture.simulation.core.simulation import Simulation  # noqa: E402
from sparcs.location import Location  # noqa: E402

_DRIP_LINE_M = 12.6


def _weather(start: str, end: str, freq: str = "1min") -> pd.DataFrame:
    idx = pd.date_range(start, end, freq=freq, tz="UTC")
    return pd.DataFrame(
        {
            Weather.GHI: 400.0,
            Weather.TEMP_AIR: 21.0,
            Weather.HUMIDITY_REL: 55.0,
            Weather.WIND_SPEED: 2.0,
            Weather.CLEAR_SKY_INDEX: 0.8,
            Weather.PRECIPITATION: 0.0,
        },
        index=idx,
    )


def _no_flow(begin: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    return pd.Series(dtype=float)


def _chunks(weather: pd.DataFrame, irrigation_for=_no_flow, **kwargs):
    kwargs = {"interval": 30, "offset": 20, "intake_delay": pd.Timedelta("30min"), "cold_start_s": 10800.0, **kwargs}
    return soil_tuning._replay_chunks(weather, irrigation_for, **kwargs)


# --------------------------------------------------------------------------- _replay_chunks


def test_chunk_ends_follow_the_live_cutoff_grid_and_the_midnight_split():
    weather = _weather("2026-06-20 22:00", "2026-06-21 00:45")

    chunks = _chunks(weather)

    ends = [c.weather.index[-1] for c in chunks]
    expected = ["22:20", "22:50", "23:20", "23:50", "00:00", "00:20", "00:45"]
    days = ["2026-06-20"] * 4 + ["2026-06-21"] * 3
    assert ends == [pd.Timestamp(f"{d} {t}", tz="UTC") for d, t in zip(days, expected)]


def test_every_row_lands_in_exactly_one_chunk():
    weather = _weather("2026-06-20 22:00", "2026-06-21 00:45")

    chunks = _chunks(weather)

    rows = pd.DatetimeIndex(np.concatenate([c.weather.index.values for c in chunks])).tz_localize("UTC")
    assert rows.equals(weather.index)
    assert all(len(c.irrigation) == len(c.weather) for c in chunks)


def test_frontier_and_cold_start_propagate_between_chunks():
    weather = _weather("2026-06-20 22:00", "2026-06-21 00:45")

    chunks = _chunks(weather, cold_start_s=10800.0)

    assert chunks[0].frontier is None
    assert chunks[0].first_dt_s == 10800.0
    for previous, chunk in zip(chunks, chunks[1:]):
        assert pd.Timestamp(chunk.frontier) == previous.weather.index[-1]
        assert chunk.first_dt_s == 0.0


def test_a_window_inside_one_tick_is_one_chunk():
    weather = _weather("2026-06-20 22:02", "2026-06-20 22:10")

    chunks = _chunks(weather)

    assert len(chunks) == 1
    assert len(chunks[0].weather) == len(weather)


def test_irrigation_is_read_per_span_so_a_source_switch_is_not_forward_filled():
    weather = _weather("2026-06-21 08:00", "2026-06-21 11:00")
    switch = pd.Timestamp("2026-06-21 09:30", tz="UTC")
    calls = []

    def irrigation_for(begin, end):
        calls.append((begin, end))
        if end <= switch:
            return pd.Series([0.5], index=[begin])
        return pd.Series(dtype=float)

    chunks = _chunks(weather, irrigation_for, interval=60, offset=0, intake_delay=pd.Timedelta(0))

    assert len(calls) == len(chunks)
    assert all(begin < end for begin, end in calls)
    assert [c.weather.index[-1] for c in chunks] == [end for _, end in calls]
    early = [c for c in chunks if c.weather.index[-1] <= switch]
    late = [c for c in chunks if c.weather.index[-1] > switch]
    assert early and late
    assert all((c.irrigation == 0.5).all() for c in early)
    assert all((c.irrigation == 0.0).all() for c in late)


# --------------------------------------------------------------------------- forcing through the worker path


def _chain_setup() -> tuple[FieldSetup, WeatherChain]:
    field = FieldConfig.from_dict({"lai_type": "grass", "bay_width": 3.5})
    soil = SoilConfig.from_dict({"mesh": {}, "total_drip_line_length_m": _DRIP_LINE_M})
    setup = FieldSetup(field=field, soil=soil, shading=None, planner=None, location=None)
    shading = ShadingConfig.from_dict({"mode": "free_field"}).derive(bay_width=3.5, segment_ranges=None)
    chain = WeatherChain(setup, ShadingModel(shading), ETModel(), top_segment_names=(), segment_face_length={})
    return setup, chain


def test_worker_forcing_normalises_the_flow_by_the_drip_line_length(monkeypatch):
    setup, chain = _chain_setup()
    monkeypatch.setattr(soil_tuning, "_W_BASE", types.SimpleNamespace(chain=chain))
    weather = _weather("2026-06-21 08:00", "2026-06-21 08:30")
    flow_lpm = 0.53
    irrigation = pd.Series(0.0, index=weather.index)
    irrigation.iloc[10:] = flow_lpm
    chunk = soil_tuning.ReplayChunk(weather=weather, irrigation=irrigation, frontier=None, first_dt_s=0.0)

    forcing = soil_tuning._worker_forcing(chunk)

    watering = forcing[15]
    assert watering.flow_m3s == pytest.approx(flow_lpm / (60_000 * _DRIP_LINE_M))
    assert forcing[5].flow_m3s == 0.0


# --------------------------------------------------------------------------- overrides


def _ode(tmp_path, **values) -> PDEConfig:
    return PDEConfig(load_configs(tmp_path, "pde.conf", **values))


def test_overrides_accept_the_allowlist_and_leave_the_base_untouched(tmp_path):
    base = _ode(tmp_path, ic_water_table_depth=1.0)
    params = {key: 0.25 for key in soil_tuning.OVERRIDABLE_KEYS}

    ode = soil_tuning._apply_overrides(base, params)

    assert all(getattr(ode, key) == 0.25 for key in soil_tuning.OVERRIDABLE_KEYS)
    assert base.ic_water_table_depth == 1.0
    assert ode is not base


@pytest.mark.parametrize("key", ["ponding", "model", "watering_h_max_mm", "ic_se"])
def test_overrides_reject_keys_outside_the_allowlist(tmp_path, key):
    with pytest.raises(ValueError, match=key):
        soil_tuning._apply_overrides(_ode(tmp_path), {key: 1.0})


@pytest.mark.parametrize("value", ["wet", None, True, float("nan"), float("inf")])
def test_overrides_reject_values_that_are_not_finite_numbers(tmp_path, value):
    with pytest.raises(ValueError, match="alpha"):
        soil_tuning._apply_overrides(_ode(tmp_path), {"alpha": value})


def test_a_water_table_override_needs_a_base_that_has_one(tmp_path):
    with pytest.raises(ValueError, match="ic_water_table_depth"):
        soil_tuning._apply_overrides(_ode(tmp_path), {"ic_water_table_depth": 2.0})


# --------------------------------------------------------------------------- the bench against the live session


def _replay_setup(tmp_path) -> FieldSetup:
    soil = SoilConfig.from_dict(
        {
            "mesh": {**MESH_KW, "filename": str(tmp_path / "replay.msh")},
            "pde": {"dt": "600s", "dt_min": "30s"},
            "probes": {"points": {"strip": {"x_offset": 0.0, "depth": 30.0}}},
            "total_drip_line_length_m": _DRIP_LINE_M,
        }
    )
    soil.mesh.derive(bay_width=3.0)
    shading = ShadingConfig.from_dict(
        {
            "mode": "as_is",
            "mirrored": True,
            "surface_tilt": 10.0,
            "surface_azimuth": 90.0,
            "axis_azimuth": 180.0,
            "height": 3.77,
            "width": 1.134,
            "distance": 3.0,
        }
    )
    location = Location(48.1, 11.6, timezone="Europe/Berlin", altitude=520.0)
    return FieldSetup(field=FieldConfig.from_dict({"bay_width": 3.0}), soil=soil, shading=shading, location=location)


@pytest.mark.slow  # builds a real Gmsh mesh and runs FiPy
def test_job_rows_equal_the_live_simulation_over_the_same_chunks(tmp_path):
    setup = _replay_setup(tmp_path)
    weather = _weather("2026-06-21 08:00", "2026-06-21 11:00", freq="10min")
    weather.iloc[4:6, weather.columns.get_loc(Weather.PRECIPITATION)] = 4.0
    flow = pd.Series(0.0, index=weather.index)
    flow.iloc[8:] = 0.53
    chunks = soil_tuning._replay_chunks(
        weather,
        lambda begin, end: flow.loc[begin:end],
        interval=30,
        offset=0,
        intake_delay=pd.Timedelta(0),
        cold_start_s=3 * 3600.0,
    )
    assert len(chunks) > 2

    live = Simulation.build(setup, rel_sat_name="Se_replay_live")
    whole, _ = live.chain.forcing_series(weather, flow, frontier=None, first_dt_s=3 * 3600.0)
    expected = []
    for chunk in chunks:
        results, _ = live.run(chunk.weather, chunk.irrigation)
        expected += [(pd.Timestamp(r.state.at), r.probe_tension["strip"]) for r in results]

    progress, pngs = queue.Queue(), {}
    soil_tuning._worker_init(setup, list(live.probes), 4, progress, pngs, {})
    forcing = [soil_tuning._worker_forcing(chunk) for chunk in chunks]
    chunked = [f for part in forcing for f in part]
    assert [f.at for f in chunked] == [f.at for f in whole]
    assert any(a.seg_evap != b.seg_evap for a, b in zip(chunked, whole)), "chunking must matter for this test"
    path = str(tmp_path / "forcing.pkl")
    with open(path, "wb") as fh:
        pickle.dump(forcing, fh)

    soil_tuning._worker_run_job("replay1", "replay", {}, path)

    messages = _drain(progress)
    assert messages[-1]["type"] == "done"
    rows = [m["row"] for m in messages if m["type"] == "row"]
    assert [r["timestamp"] for r in rows] == [t for t, _ in expected]
    np.testing.assert_allclose([r["strip__tension"] for r in rows], [v for _, v in expected], rtol=0, atol=1e-9)
    se = np.array([r["strip__se"] for r in rows])
    assert np.all((se > 0.0) & (se <= 1.0))
    assert "replay1" in pngs

    soil_tuning._worker_run_job("replay2", "replay", {"alpha": 0.5}, path)
    altered = [m["row"]["strip__tension"] for m in _drain(progress) if m["type"] == "row"]
    assert altered and altered != [v for _, v in expected]

    soil_tuning._worker_run_job("replay3", "replay", {"ponding": 1.0}, path)
    failed = [m for m in _drain(progress) if m["type"] == "failed"]
    assert len(failed) == 1 and "ponding" in failed[0]["error"]


def _drain(q: "queue.Queue") -> list:
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return out
