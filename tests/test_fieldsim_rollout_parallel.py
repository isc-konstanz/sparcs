# -*- coding: utf-8 -*-
"""sparcs.tests.test_fieldsim_rollout_parallel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The parallel independent-roll executor: ``IrrigationPlanner.rollout``'s routing
and graceful degrade, ``_worker_init``'s one-core pin, ``rollout_parallel``'s
fan-out/gather/worker sizing and env scoping, the pickle-safe column coercion,
and the headline ``parallel == caterpillar`` invariant (slow).
"""

import datetime
import logging
import os
import pickle
from concurrent.futures import Future
from types import SimpleNamespace

import pytest

import numpy as np
import pandas as pd
from lories import Configurations
from lories.components.weather import Weather
from sparcs.components.agriculture.fieldsim.core import rollout as _rollout
from sparcs.components.agriculture.fieldsim.core.candidates import WateringWindow, build_candidate_grid
from sparcs.components.agriculture.fieldsim.core.config import PlannerConfig
from sparcs.components.agriculture.fieldsim.core.pde import (
    MeshConfig,
    PDEConfig,
    ProbeSpec,
    SoilPDECore,
    _coords_to_cell,
    ensure_mesh,
)
from sparcs.components.agriculture.fieldsim.core.planner import IrrigationPlanner
from sparcs.components.agriculture.fieldsim.core.rollout import RolloutEngine

_TZ = "Europe/Berlin"


# --- IrrigationPlanner.rollout: routing + graceful degrade (no PDE) ----------


def _dispatch_planner(parallel, engine) -> IrrigationPlanner:
    planner = object.__new__(IrrigationPlanner)
    planner.config = SimpleNamespace(parallel=parallel)
    planner.name = "test_parallel_executor"
    planner._ladder = [(pd.Timedelta(0),)]
    planner._flow_m3s = 1.0e-5
    planner._rollout_engine = lambda: engine
    return planner


def _spy_engine(calls: list):
    def ladder(*args, **kwargs):
        calls.append(("caterpillar", args))
        return {"cat": True}

    def parallel(*args, **kwargs):
        calls.append(("parallel", args))
        return {"par": True}

    return SimpleNamespace(rollout_ladder=ladder, rollout_parallel=parallel)


def test_rollout_parallel_false_takes_caterpillar():
    calls: list = []
    planner = _dispatch_planner(parallel=False, engine=_spy_engine(calls))

    out = planner.rollout("ic", "et", "seg", "hs", "he")

    assert [c[0] for c in calls] == ["caterpillar"]
    assert out == {"cat": True}
    # The caterpillar is called with the full explicit arg tuple, including the
    # shared ladder and derived flow.
    assert calls[0][1] == ("ic", planner._ladder, "et", "seg", planner._flow_m3s, "hs", "he")


def test_rollout_parallel_true_takes_parallel():
    calls: list = []
    planner = _dispatch_planner(parallel=True, engine=_spy_engine(calls))

    out = planner.rollout("ic", "et", "seg", "hs", "he")

    assert [c[0] for c in calls] == ["parallel"]
    assert out == {"par": True}


def test_rollout_degrades_to_caterpillar_when_parallel_raises(caplog):
    calls: list = []
    engine = _spy_engine(calls)

    def boom(*args, **kwargs):
        raise RuntimeError("pool could not be created")

    engine.rollout_parallel = boom
    planner = _dispatch_planner(parallel=True, engine=engine)

    with caplog.at_level(logging.ERROR):
        out = planner.rollout("ic", "et", "seg", "hs", "he")

    assert [c[0] for c in calls] == ["caterpillar"], "a parallel failure must fall back to the caterpillar"
    assert out == {"cat": True}
    assert "parallel roll-out failed" in caplog.text.lower()


# --- _worker_init: pin one core BEFORE building the PDE (no solver) ----------


_INIT_KWARGS = dict(
    rel_sat_name="predictor relative saturation",
    name="worker",
    flow_m3s=1.0e-5,
    grid_mode="fill_order",
    ic_rel_sat=None,
    et_data=None,
    seg_et={},
    horizon_start=None,
    horizon_end=None,
)


def test_worker_init_pins_one_core_before_building_pde(monkeypatch):
    seen = {}

    class _FakePDE:
        def __init__(self, mesh_config, ode_config, *, rel_sat_name):
            # Capture the env exactly at PDE-construction time: the pin must
            # already be in place when the (real) solver would be built.
            seen["omp"] = os.environ.get("OMP_NUM_THREADS")
            seen["kmp"] = os.environ.get("KMP_DUPLICATE_LIB_OK")
            seen["rel_sat_name"] = rel_sat_name

    monkeypatch.setattr(_rollout, "SoilPDECore", _FakePDE)
    monkeypatch.setattr(_rollout, "ensure_mesh", lambda mesh_config: seen.setdefault("ensure_mesh", True))
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)

    _rollout._worker_init(mesh_config=object(), ode_config=object(), probes=[], windows=[], **_INIT_KWARGS)

    assert seen["omp"] == "1", "worker must pin OMP_NUM_THREADS=1 before building the PDE"
    assert seen["kmp"] == "TRUE", "worker must set KMP_DUPLICATE_LIB_OK before building the PDE"
    assert seen.get("ensure_mesh") is True
    assert seen["rel_sat_name"] == "predictor relative saturation"
    assert _rollout._WORKER["seg_et"] == {}


def test_worker_init_stashes_rollout_engine_with_six_fields(monkeypatch):
    """The spawn worker's independent roll reads a ``RolloutEngine`` carrying
    exactly the six loose rollout fields -- not a component."""

    class _FakePDE:
        def __init__(self, mesh_config, ode_config, *, rel_sat_name):
            self.rel_sat_name = rel_sat_name

    monkeypatch.setattr(_rollout, "SoilPDECore", _FakePDE)
    monkeypatch.setattr(_rollout, "ensure_mesh", lambda mesh_config: None)

    probes = [object()]
    windows = [object()]
    _rollout._worker_init(mesh_config=object(), ode_config=object(), probes=probes, windows=windows, **_INIT_KWARGS)

    engine = _rollout._WORKER["engine"]
    assert isinstance(engine, RolloutEngine)
    assert engine.name == "worker"
    assert isinstance(engine.pde, _FakePDE)
    assert engine.probes is probes
    assert engine.windows is windows
    assert engine.flow_m3s == 1.0e-5
    assert engine.grid_mode == "fill_order"


# --- rollout_parallel: fan-out / gather / max_workers sizing (fake executor) --


class _FakeExecutor:
    """Synchronous in-process stand-in for ProcessPoolExecutor: runs the
    initializer once, and each submit runs the (patched) task immediately."""

    last = None

    def __init__(self, max_workers, mp_context, initializer, initargs):
        _FakeExecutor.last = self
        self.max_workers = max_workers
        self.mp_context = mp_context
        self.initargs = initargs
        initializer(*initargs)

    def submit(self, fn, arg):
        fut = Future()
        fut.set_result(fn(arg))
        return fut

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _parallel_engine(ladder, max_workers) -> RolloutEngine:
    return RolloutEngine(
        pde=None,
        probes=[],
        windows=[],
        flow_m3s=1.0e-5,
        grid_mode="fill_order",
        ladder=ladder,
        max_workers=max_workers,
        name="test_parallel_executor",
        mesh_config=object(),
        ode_config=object(),
        rel_sat_name="Se_test",
    )


def _candidate_value(candidate):
    """A distinct, candidate-derived scalar so a mis-keyed gather is detectable."""
    return float(len(candidate) + sum(d.total_seconds() for d in candidate))


def test_rollout_parallel_fans_out_and_gathers_by_candidate(monkeypatch):
    ladder = [
        (pd.Timedelta(0), pd.Timedelta(0)),
        (pd.Timedelta(minutes=5), pd.Timedelta(0)),
        (pd.Timedelta(minutes=5), pd.Timedelta(minutes=5)),
    ]
    engine = _parallel_engine(ladder, max_workers=2)

    init_seen = {}
    monkeypatch.setattr(_rollout, "_worker_init", lambda *initargs: init_seen.setdefault("initargs", initargs))
    monkeypatch.setattr(
        _rollout,
        "_worker_roll",
        lambda candidate: (candidate, (["ts"], {"probe": [_candidate_value(candidate)]})),
    )
    monkeypatch.setattr(_rollout, "ProcessPoolExecutor", _FakeExecutor)

    out = engine.rollout_parallel("ic", pd.DataFrame(), {}, "hs", "he")

    # Every candidate present and mapped to ITS OWN result (ordering-independent:
    # the task returns its candidate, so the gather never depends on completion order).
    assert set(out.keys()) == set(ladder)
    for candidate in ladder:
        assert out[candidate][1]["probe"] == [_candidate_value(candidate)]
    # max_workers capped to min(max_workers, len(ladder)) == min(2, 3) == 2.
    assert _FakeExecutor.last.max_workers == 2
    # The initializer received the shared configs first; payloads are only the
    # candidate tuples.
    assert init_seen["initargs"][0] is engine.mesh_config
    assert init_seen["initargs"][1] is engine.ode_config


@pytest.mark.parametrize(
    ("max_workers", "n_candidates", "expected_workers"),
    [(2, 3, 2), (8, 3, 3), (1, 3, 1), (4, 0, 1)],
)
def test_rollout_parallel_worker_count_capped_to_ladder(monkeypatch, max_workers, n_candidates, expected_workers):
    ladder = [(pd.Timedelta(minutes=i),) for i in range(n_candidates)]
    engine = _parallel_engine(ladder, max_workers=max_workers)

    monkeypatch.setattr(_rollout, "_worker_init", lambda *initargs: None)
    monkeypatch.setattr(_rollout, "_worker_roll", lambda candidate: (candidate, (["ts"], {"probe": [0.0]})))
    monkeypatch.setattr(_rollout, "ProcessPoolExecutor", _FakeExecutor)

    out = engine.rollout_parallel("ic", pd.DataFrame(), {}, "hs", "he")

    assert _FakeExecutor.last.max_workers == expected_workers
    assert len(out) == n_candidates


def test_rollout_parallel_restores_env_on_success_and_on_raise(monkeypatch):
    """The OMP/KMP pin set before pool creation must be scoped to this call: the
    parent's prior env must be back in place once the pool block exits, whether
    it returns normally or raises -- a degrade to the caterpillar must not
    inherit a leaked pin."""
    engine = _parallel_engine([(pd.Timedelta(0),)], max_workers=1)

    monkeypatch.setenv("OMP_NUM_THREADS", "5")
    monkeypatch.delenv("KMP_DUPLICATE_LIB_OK", raising=False)

    monkeypatch.setattr(_rollout, "_worker_init", lambda *initargs: None)
    monkeypatch.setattr(_rollout, "_worker_roll", lambda candidate: (candidate, (["ts"], {"probe": [0.0]})))
    monkeypatch.setattr(_rollout, "ProcessPoolExecutor", _FakeExecutor)

    engine.rollout_parallel("ic", pd.DataFrame(), {}, "hs", "he")

    assert os.environ.get("OMP_NUM_THREADS") == "5", "the caller's prior OMP_NUM_THREADS must be restored"
    assert "KMP_DUPLICATE_LIB_OK" not in os.environ, "KMP_DUPLICATE_LIB_OK must be unset again, not leaked as 'TRUE'"

    class _RaisingExecutor(_FakeExecutor):
        def __enter__(self):
            raise RuntimeError("pool failed mid-block")

    monkeypatch.setattr(_rollout, "ProcessPoolExecutor", _RaisingExecutor)

    with pytest.raises(RuntimeError):
        engine.rollout_parallel("ic", pd.DataFrame(), {}, "hs", "he")

    assert os.environ.get("OMP_NUM_THREADS") == "5", "env must be restored even when the pool block raises"
    assert "KMP_DUPLICATE_LIB_OK" not in os.environ, "a raising pool must not leave KMP_DUPLICATE_LIB_OK set"


# --- Config: defaults + parse contract ---------------------------------------


def test_default_parallel_is_false():
    config = PlannerConfig.from_dict({})
    assert config.parallel is False
    assert config.max_workers is None


def test_config_parses_parallel_and_max_workers():
    config = PlannerConfig.from_dict({"parallel": True, "max_workers": 3})
    assert config.parallel is True
    assert config.max_workers == 3


# --- Constant-labeled frames survive the spawn worker boundary ---------------


def test_stringify_columns_makes_constant_labels_pickle_safe():
    """Real chain-replay ``et_data`` carries lories ``Constant`` column labels,
    which do NOT survive pickling to a spawn worker -- ``Constant.__new__`` takes
    ``(type, key, ...)``, so pickle's str-subclass reconstruction passes the value
    as ``type`` with ``key=None`` and raises."""
    idx = pd.DatetimeIndex([pd.Timestamp("2026-07-06 12:00", tz=_TZ)], name="timestamp")
    frame = pd.DataFrame({Weather.PRECIPITATION: [0.5]}, index=idx)

    raised = False
    try:
        pickle.loads(pickle.dumps(frame))
    except Exception:  # noqa: BLE001 -- documenting that the raw frame is unpicklable
        raised = True
    assert raised, "a raw lories Constant column label should not round-trip through pickle"

    restored = pickle.loads(pickle.dumps(_rollout._stringify_columns(frame)))  # must not raise
    assert list(restored.columns) == ["precipitation"]
    # Constant-keyed access still resolves (a Constant equals its key str), so the
    # worker's _rain_flux lookup is unaffected.
    assert restored.loc[idx[0], Weather.PRECIPITATION] == 0.5


# --- Headline invariant: parallel == caterpillar within solver tolerance ------

WATERING = "WateringTopSegment"
# ~2000 mm/h over the 0.5 m strip -- far beyond intake, so the strip ponds.
_EXTREME_FLOW = 2000.0e-3 / 3600.0 * 0.5


def _configs(tmp_dir, **values):
    return Configurations.load("test.conf", conf_dir=str(tmp_dir), require=False, **values)


def _build_core(tmp_dir, dt="30s"):
    mesh_config = MeshConfig(
        _configs(
            tmp_dir,
            filename=str(tmp_dir / "soil_test.msh"),
            dl=0.2,
            width=3.0,
            height=1.5,
            plant_width=1.0,
            plant_height=0.5,
            watering_width=0.5,
            d_x=0.5,
        )
    )
    ode_config = PDEConfig(_configs(tmp_dir, dt=dt, dt_min="1s"))
    ensure_mesh(mesh_config)
    return SoilPDECore(mesh_config, ode_config, rel_sat_name="Se_test")


def _strip_probe(core):
    idx = _coords_to_cell(core.mesh, core.mesh_config, x_offset_cm=0.0, depth_cm=5.0)
    return ProbeSpec(
        name="watering strip probe",
        channel_id="strip",
        cell_indices=np.array([idx], dtype=int),
        weights=np.array([1.0]),
    )


@pytest.mark.slow
def test_parallel_equals_caterpillar_solver_backed(tmp_path):
    """For a small fill_order ladder, the parallel executor's
    ``{candidate: trajectory}`` map equals the sequential caterpillar's within
    solver tolerance: a wall-time win, not a change to what is stored."""
    horizon_start = pd.Timestamp("2026-07-03 08:00", tz=_TZ)
    idx = pd.DatetimeIndex(
        [horizon_start + pd.Timedelta(minutes=m) for m in (0, 10, 20, 25)],
        name="timestamp",
    )
    horizon_end = idx[-1]
    # Include a lories Constant column label (zero rain, so physics is unchanged)
    # so this exercises the spawn+pickle path a bare frame would miss.
    et_data = pd.DataFrame({Weather.PRECIPITATION: [0.0, 0.0, 0.0, 0.0]}, index=idx)
    seg_et = {}
    windows = [
        WateringWindow(start=datetime.time(8, 10)),
        WateringWindow(start=datetime.time(8, 20)),
    ]
    window_durations = [
        [pd.Timedelta(0), pd.Timedelta(minutes=5)],
        [pd.Timedelta(0), pd.Timedelta(minutes=5)],
    ]
    ladder = build_candidate_grid(window_durations, grid_mode="fill_order")

    core = _build_core(tmp_path)
    ic_rel_sat = core.snapshot()
    engine = RolloutEngine(
        pde=core,
        probes=[_strip_probe(core)],
        windows=windows,
        window_durations=window_durations,
        flow_m3s=_EXTREME_FLOW,
        grid_mode="fill_order",
        ladder=ladder,
        max_workers=2,
        name="test_parallel_executor",
        mesh_config=core.mesh_config,
        ode_config=core.ode_config,
        rel_sat_name="Se_test",
    )

    # Parallel first (rebuilds the PDE in workers; never touches engine.pde), then
    # the caterpillar from the same IC snapshot.
    parallel_map = engine.rollout_parallel(ic_rel_sat, et_data, seg_et, horizon_start, horizon_end)
    caterpillar_map = engine.rollout_ladder(
        ic_rel_sat, ladder, et_data, seg_et, _EXTREME_FLOW, horizon_start, horizon_end
    )

    assert set(parallel_map.keys()) == set(caterpillar_map.keys()) == set(ladder)
    assert core.surface_h[WATERING] > 0.0, "scenario must pond for a meaningful equivalence check"
    for candidate in ladder:
        par_ts, par_traj = parallel_map[candidate]
        cat_ts, cat_traj = caterpillar_map[candidate]
        assert par_ts == cat_ts
        for probe_id in cat_traj:
            np.testing.assert_allclose(
                par_traj[probe_id],
                cat_traj[probe_id],
                atol=1e-6,
                err_msg=f"parallel != caterpillar for candidate {candidate}, probe {probe_id}",
            )
