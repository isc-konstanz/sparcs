# -*- coding: utf-8 -*-
"""sparcs.tests.test_soil_advance_shutdown_cancel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

B8: FieldSimulation's shutdown interrupt threads into SoilPDECore.walk_window's
existing ``cancel`` param (via SoilSimulation.advance -> _walk) so a grinding
walk exits promptly on shutdown. A cancelled walk must persist nothing: no
diagnostics recorded, no state save, no anchor application, and the frontier
(``_last_simulated_at``) must not ratchet. ``cancel=None`` (the default, used
by simulate_loop's offline/soil_tuning callers) must behave exactly as before.

``advance()`` only touches the attributes/methods stubbed below, so a bare
``object.__new__`` instance with a stub ``_pde`` exercises it without a
Component/PDE bootstrap (same pattern as the restore-guard tests).
"""

import types

import pytest

import pandas as pd

soil = pytest.importorskip("sparcs.components.agriculture.simulation.soil")
SoilSimulation = soil.SoilSimulation

_soil_core = pytest.importorskip("sparcs.components.agriculture.simulation._soil")
WalkResult = _soil_core.WalkResult


def _bare_sim(walk_window):
    """A bare SoilSimulation with just what ``advance()`` touches before/at the
    walk stubbed out, plus recording stand-ins for everything a cancelled walk
    must not reach (render/anchor/diagnostics/save)."""
    sim = object.__new__(SoilSimulation)
    sim._name = "test_soil_simulation"
    sim._simulating = False
    sim._discover_sensor_probes_enabled = False
    sim._strip_flux_warned = False
    sim._total_drip_line_length_m = 1.0
    sim._plot_config = None
    sim._anchor_cfg = types.SimpleNamespace(enabled=False)
    sim._anchor_sensors = []
    sim._pde = types.SimpleNamespace(
        walk_window=walk_window,
        total_water=lambda: 0.0,
        surface_water=lambda: 0.0,
    )
    calls: dict = {"render": [], "anchor": [], "diagnostics": [], "save_state": []}

    def _render_progress_if_due(now):
        calls["render"].append(now)

    def _apply_anchor(now, water_after_walk):
        calls["anchor"].append(now)

    def _record_diagnostics(rates, now, delta_storage, elapsed_s, clip, skipped_s):
        calls["diagnostics"].append(now)
        return {"x": 1.0}

    def _save_state(now):
        calls["save_state"].append(now)

    sim._render_progress_if_due = _render_progress_if_due
    sim._apply_anchor = _apply_anchor
    sim._record_diagnostics = _record_diagnostics
    sim._save_state = _save_state
    return sim, calls


def test_cancelled_walk_persists_nothing_and_frontier_holds():
    """A stubbed walk_window that reports cancellation: advance() must thread
    the cancel callable through, return {}, and skip every persistence side
    effect -- the frontier must not move."""
    t0 = pd.Timestamp("2026-07-16 10:00", tz="UTC")
    now = pd.Timestamp("2026-07-16 11:00", tz="UTC")
    received: dict = {}

    def _walk_window_stub(**kwargs):
        received["cancel"] = kwargs.get("cancel")
        return WalkResult(ok=False, cancelled=True, reason="cancelled")

    sim, calls = _bare_sim(_walk_window_stub)
    sim._last_simulated_at = t0

    def my_cancel():
        return True  # equivalent to _tick_interrupt.is_set once shutdown is signalled

    result = sim.advance(pd.DataFrame(index=[now]), now, {}, cancel=my_cancel)

    assert result == {}
    assert received["cancel"] is my_cancel
    assert calls == {"render": [], "anchor": [], "diagnostics": [], "save_state": []}
    assert sim._last_simulated_at == t0  # frontier holds


def test_uncancelled_walk_with_cancel_none_persists_as_before():
    """cancel=None (the default; simulate_loop's offline/soil_tuning callers
    never pass one) must behave exactly like today: walk_window sees
    cancel=None and every persistence side effect still runs."""
    t0 = pd.Timestamp("2026-07-16 10:00", tz="UTC")
    now = pd.Timestamp("2026-07-16 11:00", tz="UTC")
    received: dict = {}

    def _walk_window_stub(**kwargs):
        received["cancel"] = kwargs.get("cancel")
        return WalkResult()

    sim, calls = _bare_sim(_walk_window_stub)
    sim._last_simulated_at = t0

    result = sim.advance(pd.DataFrame(index=[now]), now, {})

    assert received["cancel"] is None
    assert calls["render"] == [now]
    assert calls["anchor"] == []  # anchoring off (_anchor_cfg.enabled=False)
    assert calls["save_state"] == [now]
    assert result == {"x": 1.0}
