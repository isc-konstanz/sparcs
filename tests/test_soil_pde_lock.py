# -*- coding: utf-8 -*-
"""sparcs.tests.test_soil_pde_lock
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Q11: a dedicated ``_pde_lock`` on ``SoilSimulation`` must serialize
``advance()``'s PDE-mutating span (the walk through ``_save_state``) against
``apply_state_blob``, so a state restore landing mid-``advance()`` cannot
interleave with the tick's own PDE writes -- it must block until the tick's
critical section has completed, then apply.

``object.__new__`` instances (``_pde_lock`` defaults to ``None`` at class
level, per the restore-guard tests alongside this file) must keep running
unlocked, exactly as before this lock existed.
"""

import threading
import types

import pytest

import pandas as pd

soil = pytest.importorskip("sparcs.components.agriculture.simulation.soil")
SoilSimulation = soil.SoilSimulation

_soil_core = pytest.importorskip("sparcs.components.agriculture.simulation._soil")
WalkResult = _soil_core.WalkResult


class _FakePDE:
    """Stand-in for the FiPy solver: records the restore call, no real math."""

    def __init__(self, events: list) -> None:
        self._events = events
        self.restored_with = None

    def total_water(self) -> float:
        return 0.0

    def surface_water(self) -> float:
        return 0.0

    def save_state_blob(self) -> bytes:
        return b"state-after-walk"

    def load_state_blob(self, raw: bytes) -> None:
        self.restored_with = raw
        self._events.append("restore:applied")


def _stubbed_sim(pde, events, enter_walk, release_walk):
    """A ``SoilSimulation`` with every ``advance()`` helper besides the PDE
    walk stubbed out, so the only meaningful work inside the locked critical
    section is the (fake, event-gated) walk -- the point where a racing
    restore would otherwise corrupt solver state."""
    sim = object.__new__(SoilSimulation)
    sim._name = "test_soil_pde_lock"
    sim._pde = pde
    sim._pde_lock = threading.Lock()
    sim._simulating = False
    sim._last_simulated_at = pd.Timestamp("2026-07-03 00:00", tz="UTC")
    sim._plot_config = None
    sim._anchor_cfg = types.SimpleNamespace(enabled=False)
    sim._anchor_sensors = []
    sim._discover_sensor_probes_enabled = False
    sim._sensor_probes_ready = False

    sim._compute_flux_rates = lambda et_data, seg_et, elapsed_s: "rates-stub"
    sim._record_diagnostics = lambda *a, **kw: {}

    def fake_walk(*, rates, window_s, clip_total, sim_t0, plot_interval, cancel=None):
        events.append("advance:enter_walk")
        enter_walk.set()
        # Blocks "mid-solve" until the test releases it, holding the lock the
        # whole time -- it is called from inside advance()'s locked span.
        release_walk.wait(timeout=2.0)
        events.append("advance:exit_walk")
        return WalkResult()

    sim._walk = fake_walk

    def fake_save_state(timestamp):
        events.append(("advance:save_state", timestamp))
        sim._last_simulated_at = timestamp

    sim._save_state = fake_save_state
    return sim


def test_restore_blocks_until_advance_critical_section_completes():
    """A restore racing a slow advance() must not interleave: it blocks until
    advance()'s critical section (walk through _save_state) has finished, and
    only then applies -- so the restore's timestamp, not the advance's, ends
    up as the final ``_last_simulated_at``.

    Without the lock, apply_state_blob has nothing to block on and runs
    in-line with the still-mid-walk advance(), so the advance's own (earlier)
    _save_state clobbers the restore afterwards -- the exact corruption Q11
    closes. This assertion is what makes the test fail pre-change.
    """
    events: list = []
    pde = _FakePDE(events)
    enter_walk = threading.Event()
    release_walk = threading.Event()
    restore_finished = threading.Event()
    sim = _stubbed_sim(pde, events, enter_walk, release_walk)

    now = pd.Timestamp("2026-07-03 01:00", tz="UTC")
    # Newer than `now` so apply_state_blob's own stale-timestamp guard never
    # no-ops this call in either the locked or unlocked path.
    restore_ts = pd.Timestamp("2026-07-03 01:30", tz="UTC")

    advance_thread = threading.Thread(target=sim.advance, args=(pd.DataFrame(), now, {}), name="advance")
    advance_thread.start()

    assert enter_walk.wait(timeout=2.0), "advance() never reached its critical section"

    def do_restore():
        events.append("restore:calling")
        sim.apply_state_blob(b"racing-restore", restore_ts)
        events.append("restore:returned")
        restore_finished.set()

    restore_thread = threading.Thread(target=do_restore, name="restore")
    restore_thread.start()

    # Bounded window for the (buggy, unlocked) restore to race ahead of the
    # still-blocked advance(). Generous relative to the handful of Python
    # statements apply_state_blob executes when unlocked, but it is a
    # ceiling, not the correctness mechanism: with the lock in place,
    # apply_state_blob blocks on acquire() regardless of this timeout, since
    # nothing releases it until release_walk.set() runs below.
    restore_finished.wait(timeout=0.05)
    release_walk.set()

    advance_thread.join(timeout=2.0)
    restore_thread.join(timeout=2.0)
    assert not advance_thread.is_alive(), "advance() did not finish"
    assert not restore_thread.is_alive(), "apply_state_blob() did not finish"

    # The restore must win: it is only allowed to run after advance()'s own
    # _save_state, so it is the last write to _last_simulated_at.
    assert sim._last_simulated_at == restore_ts, events
    assert pde.restored_with == b"racing-restore"

    # No interleaving: the restore's PDE mutation is recorded strictly after
    # advance()'s critical section (walk + _save_state) completes.
    save_state_index = next(i for i, e in enumerate(events) if isinstance(e, tuple) and e[0] == "advance:save_state")
    restore_applied_index = events.index("restore:applied")
    assert restore_applied_index > save_state_index, events


def test_restore_superseded_while_blocked_is_dropped():
    """A restore that passes the cheap pre-lock guard but is overtaken by the
    advance() it blocked on (whose _save_state ratchets _last_simulated_at past
    the blob's timestamp) must be dropped by the re-check inside the lock --
    never applied as a silent rewind (the guard-TOCTOU the Q11 review flagged)."""
    events: list = []
    pde = _FakePDE(events)
    enter_walk = threading.Event()
    release_walk = threading.Event()
    restore_finished = threading.Event()
    sim = _stubbed_sim(pde, events, enter_walk, release_walk)

    now = pd.Timestamp("2026-07-03 01:00", tz="UTC")
    # Newer than the initial _last_simulated_at (00:00) so the pre-lock guard
    # passes, but OLDER than the in-flight advance's own `now` -- by the time
    # the restore unblocks, _save_state has ratcheted past it.
    restore_ts = pd.Timestamp("2026-07-03 00:30", tz="UTC")

    advance_thread = threading.Thread(target=sim.advance, args=(pd.DataFrame(), now, {}), name="advance")
    advance_thread.start()
    assert enter_walk.wait(timeout=2.0), "advance() never reached its critical section"

    def do_restore():
        sim.apply_state_blob(b"stale-restore", restore_ts)
        restore_finished.set()

    restore_thread = threading.Thread(target=do_restore, name="restore")
    restore_thread.start()

    restore_finished.wait(timeout=0.05)
    release_walk.set()

    advance_thread.join(timeout=2.0)
    restore_thread.join(timeout=2.0)
    assert not advance_thread.is_alive() and not restore_thread.is_alive()

    assert pde.restored_with is None, events  # never applied
    assert sim._last_simulated_at == now  # the advance's ratchet stands
