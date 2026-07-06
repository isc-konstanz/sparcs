# -*- coding: utf-8 -*-
"""sparcs.tests.test_soil_predictor_parallel_executor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tests for the parallel independent-roll executor (issue 03 of
soil-predictor-mesh): the ``parallel`` execution strategy that rolls every
``fill_order`` ladder candidate independently across an in-component spawn
``ProcessPoolExecutor``, instead of the sequential prefix-shared caterpillar.

Coverage:
- ``_rollout_dispatch`` routing (parallel vs caterpillar) and graceful degrade
  to the caterpillar when the parallel path raises -- no PDE, spy collaborators.
- ``_worker_init`` pins one core (OMP_NUM_THREADS=1, KMP flag) *before* building
  the PDE -- fake ``SoilPDECore``, no solver.
- ``_rollout_parallel`` fan-out/gather and ``max_workers`` sizing -- fake
  in-process executor + fake ``_worker_roll``, no solver.
- Config defaults / parse contract for ``parallel`` / ``max_workers``.
- The headline invariant ``parallel == caterpillar`` within solver tolerance --
  a real spawn pool + real PDE (``pytest.mark.slow``; promotes the issue-02 spike).

The whole module is skipped if the FiPy/PDE stack is absent (import guard),
matching the other predictor test modules.
"""

import datetime
import logging
from concurrent.futures import Future

import pytest

import numpy as np
import pandas as pd

soil_predictor = pytest.importorskip("sparcs.components.agriculture.simulation.soil_predictor")
SoilPredictor = soil_predictor.SoilPredictor
WateringWindow = soil_predictor.WateringWindow

from lories import Configurations  # noqa: E402
from sparcs.components.agriculture.simulation._soil import (  # noqa: E402
    MeshConfig,
    PDEConfig,
    ProbeSpec,
    SoilPDECore,
    _coords_to_cell,
    ensure_mesh,
)

# ---------------------------------------------------------------------------
# _rollout_dispatch: routing + graceful degrade (no PDE)
# ---------------------------------------------------------------------------


def _dispatch_predictor(parallel):
    p = object.__new__(SoilPredictor)
    p._name = "test_parallel_executor"
    p._parallel = parallel
    p._ladder = [(pd.Timedelta(0),)]
    p._flow_m3s = 1.0e-5
    return p


def test_dispatch_parallel_false_takes_caterpillar():
    p = _dispatch_predictor(parallel=False)
    calls = []

    def caterpillar(*args, **kwargs):
        calls.append(("caterpillar", args))
        return {"cat": True}

    def parallel(*args, **kwargs):
        calls.append(("parallel", args))
        return {"par": True}

    p._rollout_ladder = caterpillar
    p._rollout_parallel = parallel

    out = p._rollout_dispatch("ic", "et", "seg", "hs", "he")

    assert [c[0] for c in calls] == ["caterpillar"]
    assert out == {"cat": True}
    # The caterpillar is called with the full explicit arg tuple, including the
    # shared ladder and derived flow (a rename/reorder here would break predict()).
    assert calls[0][1] == ("ic", p._ladder, "et", "seg", p._flow_m3s, "hs", "he")


def test_dispatch_parallel_true_takes_parallel():
    p = _dispatch_predictor(parallel=True)
    calls = []
    p._rollout_ladder = lambda *a, **k: calls.append("caterpillar") or {"cat": True}
    p._rollout_parallel = lambda *a, **k: calls.append("parallel") or {"par": True}

    out = p._rollout_dispatch("ic", "et", "seg", "hs", "he")

    assert calls == ["parallel"]
    assert out == {"par": True}


def test_dispatch_degrades_to_caterpillar_when_parallel_raises(caplog):
    p = _dispatch_predictor(parallel=True)
    calls = []

    def boom(*args, **kwargs):
        raise RuntimeError("pool could not be created")

    def caterpillar(*args, **kwargs):
        calls.append("caterpillar")
        return {"cat": True}

    p._rollout_parallel = boom
    p._rollout_ladder = caterpillar

    with caplog.at_level(logging.ERROR):
        out = p._rollout_dispatch("ic", "et", "seg", "hs", "he")

    assert calls == ["caterpillar"], "a parallel failure must fall back to the caterpillar"
    assert out == {"cat": True}
    assert "parallel roll-out failed" in caplog.text.lower()


# ---------------------------------------------------------------------------
# _worker_init: pin one core BEFORE building the PDE (no solver)
# ---------------------------------------------------------------------------


def test_worker_init_pins_one_core_before_building_pde(monkeypatch):
    seen = {}

    class _FakePDE:
        def __init__(self, mesh_config, ode_config, *, rel_sat_name):
            # Capture the env exactly at PDE-construction time: the pin must
            # already be in place when the (real) solver would be built.
            seen["omp"] = __import__("os").environ.get("OMP_NUM_THREADS")
            seen["kmp"] = __import__("os").environ.get("KMP_DUPLICATE_LIB_OK")
            seen["rel_sat_name"] = rel_sat_name

    monkeypatch.setattr(soil_predictor, "SoilPDECore", _FakePDE)
    monkeypatch.setattr(soil_predictor, "ensure_mesh", lambda mesh_config: seen.setdefault("ensure_mesh", True))
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)

    soil_predictor._worker_init(
        mesh_config=object(),
        ode_config=object(),
        rel_sat_name="predictor relative saturation",
        name="worker",
        probes=[],
        windows=[],
        flow_m3s=1.0e-5,
        grid_mode="fill_order",
        ic_rel_sat=None,
        et_data=None,
        seg_et={},
        horizon_start=None,
        horizon_end=None,
    )

    assert seen["omp"] == "1", "worker must pin OMP_NUM_THREADS=1 before building the PDE"
    assert seen["kmp"] == "TRUE", "worker must set KMP_DUPLICATE_LIB_OK before building the PDE"
    assert seen.get("ensure_mesh") is True
    assert seen["rel_sat_name"] == "predictor relative saturation"
    # The rebuilt predictor + shared inputs are stashed for the task function.
    assert soil_predictor._WORKER["predictor"]._name == "worker"
    assert soil_predictor._WORKER["seg_et"] == {}


# ---------------------------------------------------------------------------
# _rollout_parallel: fan-out / gather / max_workers sizing (fake executor)
# ---------------------------------------------------------------------------


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


def _parallel_predictor(ladder, max_workers):
    p = object.__new__(SoilPredictor)
    p._name = "test_parallel_executor"
    p._parallel = True
    p._ladder = ladder
    p._max_workers = max_workers
    p._mesh_config = object()
    p._ode_config = object()
    p._probes = []
    p._windows = []
    p._flow_m3s = 1.0e-5
    p._grid_mode = "fill_order"
    return p


def _candidate_value(candidate):
    """A distinct, candidate-derived scalar so a mis-keyed gather is detectable."""
    return float(len(candidate) + sum(d.total_seconds() for d in candidate))


def test_rollout_parallel_fans_out_and_gathers_by_candidate(monkeypatch):
    ladder = [
        (pd.Timedelta(0), pd.Timedelta(0)),
        (pd.Timedelta(minutes=5), pd.Timedelta(0)),
        (pd.Timedelta(minutes=5), pd.Timedelta(minutes=5)),
    ]
    p = _parallel_predictor(ladder, max_workers=2)

    init_seen = {}
    monkeypatch.setattr(soil_predictor, "_worker_init", lambda *initargs: init_seen.setdefault("initargs", initargs))
    monkeypatch.setattr(
        soil_predictor,
        "_worker_roll",
        lambda candidate: (candidate, (["ts"], {"probe": [_candidate_value(candidate)]})),
    )
    monkeypatch.setattr(soil_predictor, "ProcessPoolExecutor", _FakeExecutor)

    out = p._rollout_parallel("ic", "et", {}, "hs", "he")

    # Every candidate present and mapped to ITS OWN result (ordering-independent:
    # the task returns its candidate, so the gather never depends on completion order).
    assert set(out.keys()) == set(ladder)
    for candidate in ladder:
        assert out[candidate][1]["probe"] == [_candidate_value(candidate)]
    # max_workers capped to min(_max_workers, len(ladder)) == min(2, 3) == 2.
    assert _FakeExecutor.last.max_workers == 2
    # The initializer received the shared inputs (configs first), payloads are
    # only the candidate tuples.
    assert init_seen["initargs"][0] is p._mesh_config
    assert init_seen["initargs"][1] is p._ode_config


@pytest.mark.parametrize(
    "max_workers,n_candidates,expected_workers",
    [(2, 3, 2), (8, 3, 3), (1, 3, 1), (4, 0, 1)],
)
def test_rollout_parallel_worker_count_capped_to_ladder(monkeypatch, max_workers, n_candidates, expected_workers):
    ladder = [(pd.Timedelta(minutes=i),) for i in range(n_candidates)]
    p = _parallel_predictor(ladder, max_workers=max_workers)

    monkeypatch.setattr(soil_predictor, "_worker_init", lambda *initargs: None)
    monkeypatch.setattr(soil_predictor, "_worker_roll", lambda candidate: (candidate, (["ts"], {"probe": [0.0]})))
    monkeypatch.setattr(soil_predictor, "ProcessPoolExecutor", _FakeExecutor)

    out = p._rollout_parallel("ic", "et", {}, "hs", "he")

    assert _FakeExecutor.last.max_workers == expected_workers
    assert len(out) == n_candidates


# ---------------------------------------------------------------------------
# Config: defaults + parse contract
# ---------------------------------------------------------------------------


def test_default_parallel_is_false():
    assert soil_predictor._DEFAULT_PARALLEL is False


def test_config_parses_parallel_and_max_workers(tmp_path):
    cfg = Configurations.load("t.conf", conf_dir=str(tmp_path), require=False, parallel=True, max_workers=3)
    assert cfg.get_bool("parallel", default=soil_predictor._DEFAULT_PARALLEL) is True
    assert cfg.get_int("max_workers", default=99) == 3

    empty = Configurations.load("t2.conf", conf_dir=str(tmp_path), require=False)
    assert empty.get_bool("parallel", default=soil_predictor._DEFAULT_PARALLEL) is False


# ---------------------------------------------------------------------------
# Headline invariant: parallel == caterpillar within solver tolerance (slow)
# ---------------------------------------------------------------------------

WATERING = "WateringTopSegment"
# ~2000 mm/h over the 0.5 m strip -- far beyond intake, so the strip ponds
# (matches test_soil_predictor_ladder_rollout._EXTREME_FLOW).
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


def _equivalence_predictor(core, probes, windows, window_durations, ladder, max_workers):
    p = object.__new__(SoilPredictor)
    p._name = "test_parallel_executor"
    p._parallel = True
    p._pde = core
    p._mesh_config = core.mesh_config
    p._ode_config = core.ode_config
    p._probes = probes
    p._windows = windows
    p._window_durations = window_durations
    p._ladder = ladder
    p._flow_m3s = _EXTREME_FLOW
    p._grid_mode = "fill_order"
    p._max_workers = max_workers
    return p


@pytest.mark.slow
def test_parallel_equals_caterpillar_solver_backed(tmp_path):
    """The headline invariant: for a small fill_order ladder, the parallel
    independent-roll executor's {candidate: trajectory} map equals the sequential
    caterpillar's within solver tolerance. Real spawn pool + real PDE; promotes
    the issue-02 spike. Parallel must be a pure wall-time win, not a change to
    what is stored."""
    horizon_start = pd.Timestamp("2026-07-03 08:00", tz="Europe/Berlin")
    idx = pd.DatetimeIndex(
        [horizon_start + pd.Timedelta(minutes=m) for m in (0, 10, 20, 25)],
        name="timestamp",
    )
    horizon_end = idx[-1]
    et_data = pd.DataFrame(index=idx)
    seg_et = {}
    windows = [
        WateringWindow(start=datetime.time(8, 10)),
        WateringWindow(start=datetime.time(8, 20)),
    ]
    window_durations = [
        [pd.Timedelta(0), pd.Timedelta(minutes=5)],
        [pd.Timedelta(0), pd.Timedelta(minutes=5)],
    ]
    ladder = SoilPredictor._build_ladder(window_durations, grid_mode="fill_order")

    core = _build_core(tmp_path)
    ic_rel_sat = core.snapshot()
    probes = [_strip_probe(core)]
    p = _equivalence_predictor(core, probes, windows, window_durations, ladder, max_workers=2)

    # Parallel first (rebuilds the PDE in workers; never touches p._pde), then the
    # caterpillar from the same IC snapshot on p._pde.
    parallel_map = p._rollout_parallel(ic_rel_sat, et_data, seg_et, horizon_start, horizon_end)
    caterpillar_map = p._rollout_ladder(ic_rel_sat, ladder, et_data, seg_et, _EXTREME_FLOW, horizon_start, horizon_end)

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
