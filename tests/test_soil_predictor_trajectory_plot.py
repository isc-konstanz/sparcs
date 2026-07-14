# -*- coding: utf-8 -*-
"""sparcs.tests.test_soil_predictor_trajectory_plot
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tests for the debug trajectory field-dump's control logic:
``SoilPredictor._write_trajectory_fields`` -- gating on ``save_candidate_field_plots``,
one soil-field PNG per forecast step per candidate, per-run subdir, chosen-candidate
naming, and degrade-on-failure (a debug plot must never abort the forecast). The
PDE walk (``_roll_segment``) and the matplotlib render (``_render_snapshot_png``)
are stubbed here: real pixels and a real solve only run on the box -- ``savefig``
crashes under this dev env's numpy/matplotlib build -- so the PNG output is
box-verified, like the DB round-trip. What is verified here is that the sink fires
once per recorded step, for every candidate, writing correctly-named files.
"""

import os
import types

import pytest

import numpy as np
import pandas as pd

# Force headless Agg before matplotlib loads (dev-box GUI backend is unusable).
os.environ.setdefault("MPLBACKEND", "Agg")

soil_predictor = pytest.importorskip("sparcs.components.agriculture.simulation.soil_predictor")
SoilPredictor = soil_predictor.SoilPredictor

_TZ = "Europe/Berlin"
_FAKE_PNG = b"\x89PNG\r\n\x1a\nFAKE"


def _td(minutes: int) -> pd.Timedelta:
    return pd.Timedelta(minutes=minutes)


def _forecast_index(periods: int = 3) -> pd.DatetimeIndex:
    return pd.date_range("2026-07-03 02:00", periods=periods, freq="1h", tz=_TZ)


def _make_predictor(tmp_path, enabled=True, ladder=None):
    p = object.__new__(SoilPredictor)
    p._name = "test_predictor"
    p._save_candidate_field_plots = enabled
    p._trajectory_plot_dir = str(tmp_path)
    p._ladder = ladder if ladder is not None else [(_td(0),), (_td(30),)]
    p._windows = ["w0"]  # opaque here: _build_flow_schedule is stubbed
    p._flow_m3s = 1.0
    p._pde = types.SimpleNamespace(
        set_state=lambda arr, **_k: None,
        snapshot=lambda: np.zeros(3),
    )
    return p


def _stub_walk(p, calls):
    """Stub _roll_segment to drive the sink once per forecast timestamp (no real PDE),
    and _build_flow_schedule/_render_snapshot_png so nothing heavy runs. ``calls``
    records set_state / flow-schedule invocations for assertions."""

    def _set_state(arr, **_k):
        calls.setdefault("set_state", 0)
        calls["set_state"] += 1

    p._pde.set_state = _set_state

    def _flow_schedule(windows, durations, flow, start, end):
        calls.setdefault("flow_schedule", []).append(tuple(durations))
        return []

    p._build_flow_schedule = _flow_schedule

    def _render(rel_sat, sim_t, *, title="x"):
        calls.setdefault("titles", []).append(title)
        return _FAKE_PNG

    p._render_snapshot_png = _render

    def _roll(idx, et_data, seg_et, on_intervals, snapshot_sink=None):
        if snapshot_sink is not None:
            for ts in idx:
                snapshot_sink(ts)
        return list(idx), {}

    p._roll_segment = _roll


def test_writes_one_png_per_step_per_candidate(tmp_path):
    idx = _forecast_index(periods=3)
    et_data = pd.DataFrame({"x": range(len(idx))}, index=idx)
    ladder = [(_td(0),), (_td(30),), (_td(60),)]
    p = _make_predictor(tmp_path, enabled=True, ladder=ladder)
    calls = {}
    _stub_walk(p, calls)

    run_ts = pd.Timestamp("2026-07-03 01:30:00", tz=_TZ)
    p._write_trajectory_fields(np.zeros(3), et_data, {}, idx[0], idx[-1], (_td(30),), run_ts)

    run_dir = tmp_path / "trajectory_20260703T013000"
    pngs = sorted(run_dir.glob("*.png"))
    # 3 candidates x 3 forecast steps.
    assert len(pngs) == len(ladder) * len(idx)
    assert all(png.read_bytes() == _FAKE_PNG for png in pngs)
    # One re-roll per candidate.
    assert calls["set_state"] == len(ladder)
    assert calls["flow_schedule"] == [(_td(0),), (_td(30),), (_td(60),)]

    names = [png.name for png in pngs]
    # Chosen candidate (30 min) files carry the CHOSEN_ prefix; the others don't.
    chosen = [n for n in names if n.startswith("CHOSEN_")]
    assert len(chosen) == len(idx)
    assert all("30min" in n for n in chosen)
    # Each forecast timestamp appears once per candidate.
    assert sum("20260703T020000" in n for n in names) == len(ladder)
    # Non-chosen candidates are not prefixed and carry their own slug.
    assert any(n.startswith("0min_") for n in names)
    assert any(n.startswith("60min_") for n in names)


def test_skips_when_disabled(tmp_path):
    idx = _forecast_index()
    et_data = pd.DataFrame({"x": range(len(idx))}, index=idx)
    p = _make_predictor(tmp_path, enabled=False)
    calls = {}
    _stub_walk(p, calls)

    p._write_trajectory_fields(
        np.zeros(3), et_data, {}, idx[0], idx[-1], (_td(0),), pd.Timestamp("2026-07-03 01:30", tz=_TZ)
    )

    assert list(tmp_path.glob("**/*.png")) == []
    assert calls == {}  # nothing re-rolled


def test_render_failure_skips_candidate_but_continues(tmp_path):
    """A render blow-up on one candidate must not propagate and must not stop the
    remaining candidates from being written."""
    idx = _forecast_index(periods=2)
    et_data = pd.DataFrame({"x": range(len(idx))}, index=idx)
    ladder = [(_td(0),), (_td(30),)]
    p = _make_predictor(tmp_path, enabled=True, ladder=ladder)
    calls = {}
    _stub_walk(p, calls)

    def _render_boom_for_first(rel_sat, sim_t, *, title="x"):
        if title.startswith("Candidate 0min"):
            raise RuntimeError("render blew up")
        return _FAKE_PNG

    p._render_snapshot_png = _render_boom_for_first

    p._write_trajectory_fields(
        np.zeros(3), et_data, {}, idx[0], idx[-1], (_td(30),), pd.Timestamp("2026-07-03 01:30", tz=_TZ)
    )  # must not raise

    run_dir = tmp_path / "trajectory_20260703T013000"
    names = [png.name for png in run_dir.glob("*.png")]
    # Only the surviving candidate's steps land.
    assert names and all(n.startswith("CHOSEN_30min") for n in names)
    assert len(names) == len(idx)


def test_roll_failure_swallowed(tmp_path):
    """A re-roll exception must not propagate (must not abort the forecast)."""
    idx = _forecast_index()
    et_data = pd.DataFrame({"x": range(len(idx))}, index=idx)
    p = _make_predictor(tmp_path, enabled=True, ladder=[(_td(0),)])
    calls = {}
    _stub_walk(p, calls)

    def _roll_boom(*_a, **_k):
        raise RuntimeError("solve blew up")

    p._roll_segment = _roll_boom

    p._write_trajectory_fields(
        np.zeros(3), et_data, {}, idx[0], idx[-1], (_td(0),), pd.Timestamp("2026-07-03 01:30", tz=_TZ)
    )  # must not raise

    run_dir = tmp_path / "trajectory_20260703T013000"
    assert list(run_dir.glob("*.png")) == []
