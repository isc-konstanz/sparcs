# -*- coding: utf-8 -*-
"""sparcs.tests.test_render_session
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RenderSession pins for the plot consolidation: the lazy fig/ax/norm triple
is initialised once and reused across renders, and
``SoilPredictor._render_snapshot_png`` resolves the site-local timezone
itself -- rooted at ``getattr(self, "context", None)`` because six-plus
tests drive the real ``_publish_results`` on bare context-less predictors
-- so persisted ``agri_field_forecast_image`` titles carry the same clock
as the sim's progress images (and ``tz=None`` safely when no context).
"""

import types

import pytest

soil_predictor = pytest.importorskip("sparcs.components.agriculture.simulation.soil_predictor")

from sparcs.components.agriculture.simulation import plot_render  # noqa: E402

SoilPredictor = soil_predictor.SoilPredictor


def test_render_session_lazy_inits_once_and_reuses_triple(monkeypatch):
    calls = []

    def _fake_init(width_m, height_m):
        calls.append((width_m, height_m))
        return "fig", "ax", "norm"

    seen = []

    def _fake_render(fig, ax, norm, mesh, rel_sat, sim_t, *, title="Relative saturation", tz=None):
        seen.append((fig, ax, norm, mesh, tz))
        return b"png"

    monkeypatch.setattr(plot_render, "init_rel_sat_figure", _fake_init)
    monkeypatch.setattr(plot_render, "render_rel_sat_png", _fake_render)

    session = plot_render.RenderSession(3.0, 1.5)
    out1 = session.render("mesh", "values", "t1")
    out2 = session.render("mesh", "values", "t2")

    assert out1 == out2 == b"png"
    assert calls == [(3.0, 1.5)], "init_rel_sat_figure must run exactly once (lazy, reused)"
    assert seen[0][:3] == ("fig", "ax", "norm")
    assert seen[1][:3] == ("fig", "ax", "norm")


def _bare_predictor():
    p = object.__new__(SoilPredictor)
    p._name = "test_render_session"
    p._mesh_config = types.SimpleNamespace(width=3.0, height=1.5)
    p._pde = types.SimpleNamespace(mesh="mesh")
    return p


def _capture_tz(monkeypatch):
    captured = {}
    monkeypatch.setattr(plot_render, "init_rel_sat_figure", lambda w, h: ("fig", "ax", "norm"))

    def _fake_render(fig, ax, norm, mesh, rel_sat, sim_t, *, title="", tz=None):
        captured["tz"] = tz
        return b"png"

    monkeypatch.setattr(plot_render, "render_rel_sat_png", _fake_render)
    return captured


def test_snapshot_png_passes_site_local_tz_when_context_has_one(monkeypatch):
    captured = _capture_tz(monkeypatch)
    p = _bare_predictor()
    p._Registrator__context = types.SimpleNamespace(location=types.SimpleNamespace(timezone="Europe/Berlin"))
    p._render_snapshot_png("rel_sat", "t")
    assert captured["tz"] == "Europe/Berlin"


def test_snapshot_png_tz_none_on_bare_contextless_predictor(monkeypatch):
    captured = _capture_tz(monkeypatch)
    p = _bare_predictor()
    p._render_snapshot_png("rel_sat", "t")
    assert captured["tz"] is None
