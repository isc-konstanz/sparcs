# -*- coding: utf-8 -*-
"""sparcs.tests.test_predictor_walk_unification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Step-9 walk-loop unification pins (issue 37): ``_integrate_horizon`` is a
thin observer-wrapper over ``RolloutEngine.roll_segment``, so the candidate
selection roll and the published re-solve run the SAME stepping code.

Two pin families:

* Cross-loop parity (slow, real Gmsh mesh + FiPy): the same IC, forecast
  index, and watering schedule rolled through ``_roll_segment`` and
  ``_integrate_horizon`` on twin cores yield identical timestamps and
  matching trajectories, with diagnostics length-aligned to the timestamps
  (NaN IC row, finite interval rows).
* Zero-dt parameterization (fast, stub PDE): ``sample_on_zero_dt`` keeps
  the two historical ``elapsed_s <= 0`` behaviors selectable (append-and-
  sample for the candidate rolls, skip for the published re-solve), and the
  interval observers fire once per WALKED interval only -- ``begin`` before
  the first walk, ``end`` after sampling and before ``snapshot_sink``,
  never for a zero-dt interval.
"""

import pytest

import numpy as np
import pandas as pd

soil_predictor = pytest.importorskip("sparcs.components.agriculture.simulation.soil_predictor")

from types import SimpleNamespace  # noqa: E402

from lories.components.weather import Weather  # noqa: E402
from sparcs.components.agriculture.simulation._predictor_rollout import RolloutEngine  # noqa: E402
from sparcs.components.agriculture.simulation._soil import ClipDiagnostics  # noqa: E402

# Extreme flow so the watering interval visibly moves the sampled Se.
_EXTREME_FLOW = 2000.0e-3 / 3600.0 * 0.5


@pytest.mark.slow
def test_integrate_horizon_matches_roll_segment(pde_core_factory, strip_probe_factory, bare_pde_predictor):
    horizon_start = pd.Timestamp("2026-07-03 08:00", tz="Europe/Berlin")
    idx = pd.DatetimeIndex(
        [horizon_start + pd.Timedelta(minutes=m) for m in (0, 10, 20, 30)],
        name="timestamp",
    )
    et_data = pd.DataFrame(index=idx)
    # Rain on one interval so _rain_flux is non-trivial on both paths.
    et_data[Weather.PRECIPITATION] = [0.0, 0.0, 1.2, 0.0]
    seg_et: dict[str, pd.DataFrame] = {}
    # One watering interval crossing a forecast boundary, so split_interval
    # produces mixed (off, on) sub-segments on both sides.
    on_intervals = [(idx[0] + pd.Timedelta(minutes=5), idx[0] + pd.Timedelta(minutes=15))]

    core_roll = pde_core_factory("walk_unification_roll")
    ic_rel_sat = core_roll.snapshot()
    predictor_roll = bare_pde_predictor(core_roll, [strip_probe_factory(core_roll)], _EXTREME_FLOW)
    core_roll.set_state(ic_rel_sat)
    roll_timestamps, roll_traj = predictor_roll._roll_segment(idx, et_data, seg_et, on_intervals)

    core_int = pde_core_factory("walk_unification_integrate")
    predictor_int = bare_pde_predictor(core_int, [strip_probe_factory(core_int)], _EXTREME_FLOW)
    predictor_int._save_state = False  # _plot_config is the class-level None
    core_int.set_state(ic_rel_sat)
    int_timestamps, int_traj, snapshots, diagnostics = predictor_int._integrate_horizon(
        et_data, seg_et, flow_schedule=on_intervals
    )

    assert list(roll_timestamps) == list(idx)
    assert list(int_timestamps) == list(idx)
    np.testing.assert_allclose(int_traj["strip"], roll_traj["strip"], atol=1e-9)

    # Neither plot nor state capture configured: no snapshots.
    assert snapshots == {}

    # Diagnostics stay length-aligned with the timestamps: NaN IC row, one
    # finite row per walked interval (_publish_results' length guard relies
    # on exactly this alignment).
    assert diagnostics
    for values in diagnostics.values():
        assert len(values) == len(int_timestamps)
        assert np.isnan(values[0])
        assert np.all(np.isfinite(values[1:]))


def _stub_engine(calls: list, flow_m3s: float = 2.0):
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
    t0 = pd.Timestamp("2026-07-03 08:00", tz="Europe/Berlin")
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
    kinds = [e[0] for e in events]
    assert kinds == ["snap", "begin", "end", "snap", "begin", "end", "snap"]
