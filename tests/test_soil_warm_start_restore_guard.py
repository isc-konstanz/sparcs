# -*- coding: utf-8 -*-
"""sparcs.tests.test_soil_warm_start_restore_guard
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once a read-side connector is wired for warm start, the restore
listener fires on every ``SIMULATION_STATE`` ``.set()`` -- including each
tick's own ``_save_state()`` self-notification -- and a stale connector read
arriving after ticking has started would rewind ``_last_simulated_at``
backwards (a huge elapsed window on the next ``advance()``). ``apply_state_blob``
must restrict the restore to the initial read: ignore any blob whose timestamp
is not strictly after ``_last_simulated_at``.

Importing ``soil`` pulls the full lories + soil (FiPy/Gmsh) stack;
``importorskip`` keeps this out of environments that lack it (the full check
runs on the box). ``apply_state_blob`` only touches ``self._pde``/
``self._last_simulated_at``, so a bare ``object.__new__`` instance with a stub
``_pde`` exercises it without a Component/PDE bootstrap.
"""

import types

import pytest

import pandas as pd

soil = pytest.importorskip("sparcs.components.agriculture.simulation.soil")
SoilSimulation = soil.SoilSimulation


def _bare_sim(last_simulated_at):
    sim = object.__new__(SoilSimulation)
    sim._name = "test_soil_simulation"
    sim._last_simulated_at = last_simulated_at
    applied: list[bytes] = []
    sim._pde = types.SimpleNamespace(load_state_blob=lambda raw: applied.append(raw))
    return sim, applied


def test_apply_state_blob_ignores_self_notification_at_same_timestamp():
    """The tick's own ``_save_state()`` sets ``_last_simulated_at = now`` before
    the listener re-fires with that same (blob, timestamp) -- must not re-apply."""
    t1 = pd.Timestamp("2026-07-03 01:00", tz="UTC")
    sim, applied = _bare_sim(last_simulated_at=t1)

    sim.apply_state_blob(b"self-notify", t1)

    assert applied == []
    assert sim._last_simulated_at == t1


def test_apply_state_blob_ignores_stale_blob_older_than_last_simulated():
    t1 = pd.Timestamp("2026-07-03 01:00", tz="UTC")
    t0 = pd.Timestamp("2026-07-03 00:00", tz="UTC")
    sim, applied = _bare_sim(last_simulated_at=t1)

    sim.apply_state_blob(b"stale", t0)

    assert applied == []
    assert sim._last_simulated_at == t1  # must not rewind


def test_apply_state_blob_across_two_advances_restores_at_most_once():
    """Models: initial restore, tick 1 advance (-> ``_save_state`` sets t1, then
    its own ``.set()`` self-notifies the restore listener), tick 2 advance
    (-> t2, self-notification again). The blob must be applied exactly once
    (the initial read); ``_last_simulated_at`` must never move backwards."""
    t0 = pd.Timestamp("2026-07-03 00:00", tz="UTC")
    t1 = pd.Timestamp("2026-07-03 01:00", tz="UTC")
    t2 = pd.Timestamp("2026-07-03 02:00", tz="UTC")
    sim, applied = _bare_sim(last_simulated_at=None)

    # Warm start: initial read.
    sim.apply_state_blob(b"initial", t0)
    assert applied == [b"initial"]

    # advance() #1: _save_state(t1) sets _last_simulated_at = t1 first, then its
    # own .set() self-notifies the restore listener with the same (blob, t1).
    sim._last_simulated_at = t1
    sim.apply_state_blob(b"self-notify-1", t1)

    # advance() #2: same pattern at t2.
    sim._last_simulated_at = t2
    sim.apply_state_blob(b"self-notify-2", t2)

    assert applied == [b"initial"]
    assert sim._last_simulated_at == t2
