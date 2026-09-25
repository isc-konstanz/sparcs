# -*- coding: utf-8 -*-
"""sparcs.tests.test_fieldsim_planner
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``IrrigationPlanner`` over the live roll-out and forecast-table builders.
Heavy (Gmsh + FiPy): marked slow.
"""

import datetime as dt

import pytest

import numpy as np
import pandas as pd

pytestmark = pytest.mark.slow

from lories.components.weather import Weather  # noqa: E402
from lories.core import ConfigurationError  # noqa: E402
from sparcs.components.agriculture.fieldsim.core.candidates import build_candidate_grid  # noqa: E402
from sparcs.components.agriculture.fieldsim.core.config import DripConfig, PlannerConfig, SoilConfig  # noqa: E402
from sparcs.components.agriculture.fieldsim.core.engine import SoilEngine  # noqa: E402
from sparcs.components.agriculture.fieldsim.core.planner import IrrigationPlanner  # noqa: E402
from sparcs.components.agriculture.fieldsim.core.rollout import RolloutEngine  # noqa: E402
from sparcs.components.agriculture.fieldsim.core.state import Forcing  # noqa: E402

UTC = dt.timezone.utc

_MESH_KW = {
    "dl": 0.2,
    "width": 3.0,
    "height": 1.5,
    "plant_width": 1.0,
    "plant_height": 0.5,
    "watering_width": 0.5,
    "d_x": 0.5,
}


def _soil_config(tmp_path, filename: str) -> SoilConfig:
    soil = SoilConfig.from_dict(
        {
            "mesh": {**_MESH_KW, "filename": str(tmp_path / filename)},
            "pde": {"dt": "600s", "dt_min": "30s"},
            "probes": {"points": {"strip": {"x_offset": 0.0, "depth": 30.0}}},
        }
    )
    soil.mesh.derive(bay_width=3.0)
    return soil


def _engine_and_probes(tmp_path, filename: str):
    soil = _soil_config(tmp_path, filename)
    engine = SoilEngine.build(soil, rel_sat_name=f"Se_{filename}")
    probes_block = soil.configs.get_member("probes", defaults={}, ensure_exists=True)
    probes = engine.probes(probes_block)
    return engine, probes


def _weather_and_seg_et(engine: SoilEngine, horizon_start: pd.Timestamp, hours: int = 6, step_hours: int = 2):
    idx = pd.DatetimeIndex(
        [horizon_start + pd.Timedelta(hours=h) for h in range(0, hours + 1, step_hours)], name="timestamp"
    )
    weather = pd.DataFrame({Weather.PRECIPITATION: 0.0}, index=idx)
    seg_et = {name: pd.DataFrame({"evap": 1.0e-6, "transp": 1.0e-6}, index=idx) for name in engine.top_segment_names}
    return weather, seg_et


def _drip() -> DripConfig:
    return DripConfig.from_dict({"nozzle_count": 5, "nozzle_flow_lph": 60.0})


def _planner_config(**overrides) -> dict:
    base = {
        "windows": {
            "w0": {"start": "08:10", "durations": ["0min", "5min"]},
            "w1": {"start": "10:10", "durations": ["0min", "5min"]},
        },
        "grid_mode": "fill_order",
        "decision_probes": ["strip"],
        # Near-saturation target, so more watering always scores better and the
        # recommended candidate is deterministic across both grid modes.
        "threshold_hpa": 5.0,
        "max_windows": 4,
    }
    base.update(overrides)
    return base


def _build_planner(tmp_path, filename: str, **config_overrides) -> tuple[IrrigationPlanner, SoilEngine]:
    engine, probes = _engine_and_probes(tmp_path, filename)
    config = PlannerConfig.from_dict(_planner_config(**config_overrides))
    planner = IrrigationPlanner(config, engine, probes=probes, drip=_drip(), total_drip_line_length_m=1.0)
    return planner, engine


# --------------------------------------------------------------------------- candidate grid


def test_candidate_grid_matches_live_build_candidate_grid(tmp_path):
    planner, _engine = _build_planner(tmp_path, "planner_grid.msh")

    durations = [pd.Timedelta(minutes=m) for m in (0, 5)]
    expected = build_candidate_grid([durations, durations], "fill_order")

    assert planner.has_windows
    assert planner._ladder == expected
    assert len(planner._ladder) == 3  # |D0| + (|D1| - 1) = 2 + 1


# --------------------------------------------------------------------------- plan()


def test_plan_returns_chosen_candidate_and_forecast_tables(tmp_path):
    planner, engine = _build_planner(tmp_path, "planner_plan.msh")

    horizon_start = pd.Timestamp("2026-07-03 08:00", tz=UTC)
    weather, seg_et = _weather_and_seg_et(engine, horizon_start)
    state = engine.initial_state(horizon_start.to_pydatetime())

    plan = planner.plan(
        state,
        weather,
        seg_et,
        weather.index[0],
        weather.index[-1],
        run_timestamp=horizon_start,
        weather_creation=horizon_start,
    )

    assert plan.chosen in planner._ladder
    assert plan.trajectories, "expected one trajectory frame per ladder candidate"
    for candidate, frame in plan.trajectories.items():
        assert candidate in planner._ladder
        assert "strip" in frame.columns
        values = frame["strip"].to_numpy()
        assert np.all(np.isfinite(values))
        assert np.all(values <= 0.0)  # signed matric potential

    assert not plan.header.empty
    expected_header_cols = {
        "forecast_id",
        "is_recommended",
        "total_min",
        "weather_creation",
        "w0_min",
        "w0_start",
        "w1_min",
        "w1_start",
    }
    assert expected_header_cols <= set(plan.header.columns)
    assert plan.header["is_recommended"].sum() == 1
    assert bool(plan.header.loc[plan.header["is_recommended"], "total_min"].iloc[0] is not None)

    assert not plan.detail.empty
    assert {"traj_strip", "traj_strip_timestamp_creation", "traj_strip_forecast_id"} <= set(plan.detail.columns)

    # threshold_hpa is near-saturation (see _planner_config), so the argmin
    # candidate carries watering and the edge-row schedule must be non-empty.
    assert plan.chosen != tuple(pd.Timedelta(0) for _ in plan.chosen)
    assert not plan.irrigation.empty
    assert set(plan.irrigation.columns) == {"irrigation_state", "irrigation_timestamp_creation"}

    assert engine._current is None  # invalidated on the way out


# --------------------------------------------------------------------------- ladder vs full equivalence


def test_ladder_and_full_grid_modes_agree_on_a_shared_candidate(tmp_path):
    horizon_start = pd.Timestamp("2026-07-03 08:00", tz=UTC)

    planner_ladder, engine_ladder = _build_planner(tmp_path, "planner_eq_ladder.msh", grid_mode="fill_order")
    weather_ladder, seg_et_ladder = _weather_and_seg_et(engine_ladder, horizon_start)
    ic_ladder = engine_ladder.initial_state(horizon_start.to_pydatetime())
    ladder_traj = planner_ladder.rollout(
        np.asarray(ic_ladder.se), weather_ladder, seg_et_ladder, weather_ladder.index[0], weather_ladder.index[-1]
    )

    planner_full, engine_full = _build_planner(tmp_path, "planner_eq_full.msh", grid_mode="full")
    weather_full, seg_et_full = _weather_and_seg_et(engine_full, horizon_start)
    ic_full = engine_full.initial_state(horizon_start.to_pydatetime())
    full_traj = planner_full.rollout(
        np.asarray(ic_full.se), weather_full, seg_et_full, weather_full.index[0], weather_full.index[-1]
    )

    shared = (pd.Timedelta(minutes=5), pd.Timedelta(minutes=5))
    assert shared in ladder_traj
    assert shared in full_traj

    ladder_ts, ladder_se = ladder_traj[shared]
    full_ts, full_se = full_traj[shared]
    assert ladder_ts == full_ts
    np.testing.assert_allclose(
        ladder_se["strip"],
        full_se["strip"],
        atol=1e-6,
        err_msg="fill_order (ladder) and full independent rolls must agree for the same candidate",
    )


# --------------------------------------------------------------------------- no windows


def test_plan_without_windows_returns_zero_flow_baseline(tmp_path):
    engine, probes = _engine_and_probes(tmp_path, "planner_nowin.msh")
    config = PlannerConfig.from_dict({})
    planner = IrrigationPlanner(config, engine, probes=probes, drip=_drip(), total_drip_line_length_m=1.0)
    assert not planner.has_windows

    horizon_start = pd.Timestamp("2026-07-03 08:00", tz=UTC)
    weather, seg_et = _weather_and_seg_et(engine, horizon_start)
    state = engine.initial_state(horizon_start.to_pydatetime())

    plan = planner.plan(state, weather, seg_et, weather.index[0], weather.index[-1], run_timestamp=horizon_start)

    assert plan.chosen is None
    assert list(plan.trajectories.keys()) == [()]
    assert not plan.header.empty
    assert len(plan.header) == 1
    assert plan.header["is_recommended"].iloc[0] is np.False_ or plan.header["is_recommended"].iloc[0] is False
    assert engine._current is None


# --------------------------------------------------------------------------- invalidate contract


def test_plan_invalidates_engine_so_advance_matches_a_fresh_engine(tmp_path):
    planner, engine = _build_planner(tmp_path, "planner_invalidate.msh")

    horizon_start = pd.Timestamp("2026-07-03 08:00", tz=UTC)
    weather, seg_et = _weather_and_seg_et(engine, horizon_start)
    state = engine.initial_state(horizon_start.to_pydatetime())

    planner.plan(state, weather, seg_et, weather.index[0], weather.index[-1], run_timestamp=horizon_start)
    assert engine._current is None

    forcing = Forcing(at=state.at, dt_s=3600.0, rain_flux=1.0e-5)
    result = engine.advance(state, forcing)

    fresh_engine, _fresh_probes = _engine_and_probes(tmp_path, "planner_invalidate_fresh.msh")
    fresh_result = fresh_engine.advance(state, forcing)

    np.testing.assert_allclose(result.state.se, fresh_result.state.se, atol=1e-12)
    assert result.state.surface_h.keys() == fresh_result.state.surface_h.keys()
    for name, value in result.state.surface_h.items():
        assert value == pytest.approx(fresh_result.state.surface_h[name], abs=1e-12)


# --------------------------------------------------------------------------- candidate cap


def test_combo_cap_exceeded_raises(tmp_path):
    engine, probes = _engine_and_probes(tmp_path, "planner_cap.msh")
    durations = ["0min", "5min", "10min", "15min", "20min"]
    config = PlannerConfig.from_dict(
        _planner_config(
            windows={
                "w0": {"start": "08:10", "durations": durations},
                "w1": {"start": "10:10", "durations": durations},
            },
            combo_cap=2,
        )
    )
    with pytest.raises(ValueError):
        IrrigationPlanner(config, engine, probes=probes, drip=_drip(), total_drip_line_length_m=1.0)


# --------------------------------------------------------------------------- parallel degrade


def test_parallel_rollout_failure_degrades_to_ladder(tmp_path, monkeypatch, caplog):
    planner, engine = _build_planner(tmp_path, "planner_degrade.msh", parallel=True)

    def _raise(self, *args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(RolloutEngine, "rollout_parallel", _raise)

    horizon_start = pd.Timestamp("2026-07-03 08:00", tz=UTC)
    weather, seg_et = _weather_and_seg_et(engine, horizon_start)
    ic = engine.initial_state(horizon_start.to_pydatetime())

    with caplog.at_level("ERROR", logger="sparcs.components.agriculture.fieldsim.core.planner"):
        traj = planner.rollout(np.asarray(ic.se), weather, seg_et, weather.index[0], weather.index[-1])

    assert set(traj.keys()) == set(planner._ladder)
    assert any("falling back to the sequential caterpillar" in r.message for r in caplog.records)


def test_per_window_durations_are_honoured(tmp_path):
    planner, _ = _build_planner(
        tmp_path,
        "durations.msh",
        windows={
            "morning": {"start": "08:00", "durations": ["0min", "30min", "1h"]},
            "evening": {"start": "20:00", "durations": ["0min", "15min"]},
        },
    )
    assert planner._window_durations[0] == [pd.Timedelta(0), pd.Timedelta("30min"), pd.Timedelta("1h")]
    assert planner._window_durations[1] == [pd.Timedelta(0), pd.Timedelta("15min")]
    with pytest.raises(ValueError, match="missing a '0min' duration"):
        _build_planner(tmp_path, "durations.msh", windows={"m": {"start": "08:00", "durations": ["30min"]}})


def test_window_without_durations_is_a_configuration_error(tmp_path):
    with pytest.raises(ConfigurationError, match=r"\[windows.evening\] is missing its 'durations' list"):
        _build_planner(
            tmp_path,
            "durations_missing.msh",
            windows={
                "morning": {"start": "08:00", "durations": ["0min", "30min"]},
                "evening": {"start": "20:00"},
            },
        )
