# -*- coding: utf-8 -*-
"""
tests.test_fieldsim_rollout_walk
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``RolloutEngine.roll_segment``'s walk loop over a stub PDE: the
``sample_on_zero_dt`` parameterization and the interval observers that fire
once per WALKED interval. Also pins that the module's flux helpers are the
shared ``core.pde`` functions, not copies.
"""

from types import SimpleNamespace

import pandas as pd
from sparcs.components.agriculture.fieldsim.core import pde as _pde
from sparcs.components.agriculture.fieldsim.core import rollout as _rollout
from sparcs.components.agriculture.fieldsim.core.pde import ClipDiagnostics
from sparcs.components.agriculture.fieldsim.core.rollout import RolloutEngine

_TZ = "Europe/Berlin"


def _stub_engine(calls: list, flow_m3s: float = 2.0) -> RolloutEngine:
    """RolloutEngine over a stub PDE that logs walk_window calls."""

    def walk_window(*, rates, window_s, accept_at_dt_min, log_name):
        calls.append(("walk", window_s, rates.flow_m3s))
        return SimpleNamespace(clip=ClipDiagnostics())

    pde = SimpleNamespace(sample=lambda p: 0.5, walk_window=walk_window)
    return RolloutEngine(
        pde=pde,
        probes=[SimpleNamespace(channel_id="a")],
        flow_m3s=flow_m3s,
        name="stub",
    )


def _dup_index() -> pd.DatetimeIndex:
    t0 = pd.Timestamp("2026-07-03 08:00", tz=_TZ)
    hour = pd.Timedelta(hours=1)
    # Pairs: (t0, t1) walked, (t1, t1) zero-dt, (t1, t2) walked.
    return pd.DatetimeIndex([t0, t0 + hour, t0 + hour, t0 + 2 * hour], name="timestamp")


def test_sample_on_zero_dt_true_appends_duplicate_sample():
    idx = _dup_index()
    calls: list = []
    engine = _stub_engine(calls)

    timestamps, trajectories = engine.roll_segment(idx, pd.DataFrame(index=idx), {}, [])

    assert timestamps == list(idx)
    assert len(trajectories["a"]) == 4
    assert len([c for c in calls if c[0] == "walk"]) == 2


def test_sample_on_zero_dt_false_skips_and_observers_fire_per_walked_interval():
    idx = _dup_index()
    calls: list = []
    engine = _stub_engine(calls)
    events: list = []

    def interval_begin(ts_prev, ts_next, elapsed_s):
        events.append(("begin", ts_next, elapsed_s))

    def interval_end(ts_next, elapsed_s, seg_evap, seg_transp, rain_flux, clip_total, irrigated_mass):
        assert isinstance(clip_total, ClipDiagnostics)
        events.append(("end", ts_next, elapsed_s, irrigated_mass))

    def snapshot_sink(ts):
        events.append(("snap", ts))

    # Watering covers the last half hour of the first interval and the first
    # half hour of the second: 1800 s at flow 2.0 in each walked interval.
    on_intervals = [(idx[0] + pd.Timedelta(minutes=30), idx[0] + pd.Timedelta(minutes=90))]

    timestamps, trajectories = engine.roll_segment(
        idx,
        pd.DataFrame(index=idx),
        {},
        on_intervals,
        snapshot_sink=snapshot_sink,
        interval_begin=interval_begin,
        interval_end=interval_end,
        sample_on_zero_dt=False,
    )

    # The duplicate timestamp is skipped entirely: no sample, no snapshot,
    # no observer calls for the zero-dt pair.
    assert timestamps == [idx[0], idx[1], idx[3]]
    assert len(trajectories["a"]) == 3

    ts_walk_1, ts_walk_2 = idx[1], idx[3]
    assert [e for e in events if e[0] == "begin"] == [
        ("begin", ts_walk_1, 3600.0),
        ("begin", ts_walk_2, 3600.0),
    ]
    assert [e for e in events if e[0] == "end"] == [
        ("end", ts_walk_1, 3600.0, 2.0 * 1800.0),
        ("end", ts_walk_2, 3600.0, 2.0 * 1800.0),
    ]
    # Ordering per walked interval: begin -> walks -> end -> snapshot; the
    # initial snapshot at idx[0] precedes everything.
    assert [e[0] for e in events] == ["snap", "begin", "end", "snap", "begin", "end", "snap"]


def test_module_flux_functions_are_pde_functions():
    """The keyed flux helpers live in ``core.pde`` (shared with the sim's flux
    rates); the rollout module keeps them as import-aliases of the SAME function
    objects, so a copy-instead-of-move would fail here."""
    assert _rollout._segment_flux_dicts is _pde.segment_flux_dicts
    assert _rollout._rain_flux is _pde.rain_flux
