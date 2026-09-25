# -*- coding: utf-8 -*-
"""sparcs.tests.test_fieldsim_plots
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Progress-image renderers of the ``fieldsim`` skeleton as pure functions
from data to PNG bytes: no FiPy, no Gmsh, no lories channel/component state.
"""

import datetime as dt

import numpy as np
import pandas as pd
from sparcs.components.agriculture.fieldsim.core.config import PlotConfig
from sparcs.components.agriculture.fieldsim.core.plots import (
    ShadingEnvelope,
    render_due,
    render_rel_sat_png,
    render_shading_png,
    shading_envelope,
)

UTC = dt.timezone.utc

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class _FakeMesh:
    def __init__(self, cell_centers: np.ndarray) -> None:
        self.cellCenters = cell_centers


class _PVSetupStub:
    def __init__(self, *, distance: float, height: float, width: float) -> None:
        self.distance = distance
        self.height = height
        self.width = width


class _TrackerStub:
    def __init__(self, *, max_angle: float) -> None:
        self.max_angle = max_angle


def _rng() -> np.random.Generator:
    return np.random.default_rng(0)


def test_render_rel_sat_png_returns_png_bytes_deterministic_length():
    rng = _rng()
    width_m, height_m = 4.0, 3.0
    x = rng.uniform(0.0, width_m, size=200)
    y = rng.uniform(-height_m, 0.0, size=200)
    mesh = _FakeMesh(np.vstack([x, y]))
    se = rng.uniform(0.0, 1.0, size=200)
    sim_t = pd.Timestamp("2026-06-01 12:00", tz=UTC)

    png1 = render_rel_sat_png(mesh, se, sim_t, width_m=width_m, height_m=height_m)
    png2 = render_rel_sat_png(mesh, se, sim_t, width_m=width_m, height_m=height_m)

    assert png1.startswith(_PNG_SIGNATURE)
    assert png2.startswith(_PNG_SIGNATURE)
    assert len(png1) == len(png2)


def test_render_shading_png_night_empty_pv_rows_returns_png_bytes():
    envelope = ShadingEnvelope(x_half=5.25, y_min=-5.5, y_max=1.0)
    ground = [
        ((-100.0, 0.0), (-2.0, 0.0), {"qinc": 0.0}),
        ((-2.0, 0.0), (2.0, 0.0), {"qinc": 0.0}),
        ((2.0, 0.0), (100.0, 0.0), {"qinc": 0.0}),
    ]
    sun_state = (95.0, 180.0, 100.0)  # below _ZENITH_DAYTIME_LIMIT threshold -> night

    png = render_shading_png(pd.Timestamp("2026-06-01 02:00", tz=UTC), ground, [], sun_state, envelope)

    assert png.startswith(_PNG_SIGNATURE)


def test_render_shading_png_daytime_with_pv_rows_returns_png_bytes():
    envelope = ShadingEnvelope(x_half=5.25, y_min=-5.5, y_max=4.77)
    ground = [
        ((-100.0, 0.0), (-1.0, 0.0), {"qinc": 200.0}),
        ((-1.0, 0.0), (1.0, 0.0), {"qinc": 850.0}),
        ((1.0, 0.0), (100.0, 0.0), {"qinc": 200.0}),
    ]
    pv_rows = [
        ((-1.0, 0.0), (-1.0, 3.77), {"qinc_front": 900.0, "qinc_back": 50.0}),
        ((1.0, 0.0), (1.0, 3.77), {"qinc_front": 900.0, "qinc_back": 50.0}),
    ]
    sun_state = (30.0, 170.0, 100.0)

    png = render_shading_png(
        pd.Timestamp("2026-06-01 12:00", tz=UTC),
        ground,
        pv_rows,
        sun_state,
        envelope,
        title="Ground shading",
    )

    assert png.startswith(_PNG_SIGNATURE)


def test_shading_envelope_free_field_matches_bay_width_formula():
    bay_width = 3.5
    mesh_height = 5.0

    envelope = shading_envelope(
        mode="free_field",
        pv_setups=[],
        tracker=None,
        surface_tilt=0.0,
        bay_width=bay_width,
        mesh_height=mesh_height,
    )

    assert envelope.x_half == bay_width * 1.5
    assert envelope.y_max == 1.0
    assert envelope.y_min == -mesh_height - 0.5


def test_shading_envelope_as_is_matches_pv_setup_formula():
    setup = _PVSetupStub(distance=7.0, height=3.77, width=1.134)
    surface_tilt = 20.0
    mesh_height = 5.0

    envelope = shading_envelope(
        mode="as_is",
        pv_setups=[setup],
        tracker=None,
        surface_tilt=surface_tilt,
        bay_width=3.5,
        mesh_height=mesh_height,
    )

    tilt_rad = np.radians(abs(surface_tilt))
    expected_y_max = setup.height + (setup.width / 2.0) * np.sin(tilt_rad) + 1.0
    assert envelope.x_half == setup.distance * 1.5
    assert envelope.y_max == expected_y_max
    assert envelope.y_min == -mesh_height - 0.5


def test_shading_envelope_trackable_uses_tracker_max_angle():
    setup = _PVSetupStub(distance=7.0, height=3.77, width=1.134)
    tracker = _TrackerStub(max_angle=60.0)
    mesh_height = 4.0

    envelope = shading_envelope(
        mode="trackable",
        pv_setups=[setup],
        tracker=tracker,
        surface_tilt=0.0,
        bay_width=3.5,
        mesh_height=mesh_height,
    )

    tilt_rad = np.radians(abs(tracker.max_angle))
    expected_y_max = setup.height + (setup.width / 2.0) * np.sin(tilt_rad) + 1.0
    assert envelope.x_half == setup.distance * 1.5
    assert envelope.y_max == expected_y_max
    assert envelope.y_min == -mesh_height - 0.5


def test_render_due_true_when_never_rendered_and_enabled():
    config = PlotConfig.from_dict({"interval": "1h"})
    assert render_due(None, pd.Timestamp("2026-06-01 12:00", tz=UTC), config) is True


def test_render_due_false_before_interval_elapsed_or_disabled():
    config = PlotConfig.from_dict({"interval": "1h"})
    last = pd.Timestamp("2026-06-01 12:00", tz=UTC)
    assert render_due(last, last + pd.Timedelta(minutes=30), config) is False

    # A [plot] block with `enabled = false` is never configured (lories reserves
    # the key as the table's own switch); the chain then holds plots=None.
    config = None
    assert render_due(None, last, config) is False
