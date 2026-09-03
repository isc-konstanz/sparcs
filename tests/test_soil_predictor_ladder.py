# -*- coding: utf-8 -*-
"""sparcs.tests.test_soil_predictor_ladder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pure tests for issue 04's fill-order ladder (candidate set): ``SoilPredictor._build_ladder``
and the ``combo_cap`` fail-fast check at ``configure()``.

Importing ``soil_predictor`` pulls the full lories + soil (FiPy/Gmsh) stack via ``soil.py``;
``importorskip`` keeps this out of environments that lack it (the full check runs on the
box). ``_build_ladder`` is a ``@staticmethod``, called directly off the class with no
``Component``/PDE instantiation needed.
"""

import pytest

import pandas as pd

soil_predictor = pytest.importorskip("sparcs.components.agriculture.simulation.soil_predictor")
SoilPredictor = soil_predictor.SoilPredictor


def _td(minutes: int) -> pd.Timedelta:
    return pd.Timedelta(minutes=minutes)


# --- fill_order ladder generation --------------------------------------------


def test_fill_order_two_windows_sweep_then_mesh_longest():
    """Two windows, durations [0,30,60] each: sweep window 0 with window 1 off,
    then mesh window 0's max with window 1's non-zero sweep. Order matters: the
    ladder is generated fill-earlier-first, so this asserts the exact sequence."""
    window_durations = [
        [_td(0), _td(30), _td(60)],
        [_td(0), _td(30), _td(60)],
    ]

    ladder = SoilPredictor._build_ladder(window_durations, grid_mode="fill_order")

    assert ladder == [
        (_td(0), _td(0)),
        (_td(30), _td(0)),
        (_td(60), _td(0)),
        (_td(60), _td(30)),
        (_td(60), _td(60)),
    ]


def test_fill_order_total_water_strictly_increasing():
    window_durations = [
        [_td(0), _td(30), _td(60)],
        [_td(0), _td(30), _td(60)],
    ]
    ladder = SoilPredictor._build_ladder(window_durations, grid_mode="fill_order")

    totals = [sum((d for d in candidate), pd.Timedelta(0)) for candidate in ladder]
    assert totals == sorted(totals)
    assert len(set(totals)) == len(totals)


def test_fill_order_drops_back_loaded_candidate():
    """(0min morning, 60min evening) is never generated -- front-load dominance."""
    window_durations = [
        [_td(0), _td(30), _td(60)],
        [_td(0), _td(30), _td(60)],
    ]
    ladder = SoilPredictor._build_ladder(window_durations, grid_mode="fill_order")

    assert (_td(0), _td(60)) not in ladder
    assert (_td(0), _td(30)) not in ladder


def test_fill_order_three_windows():
    window_durations = [
        [_td(0), _td(30)],
        [_td(0), _td(20)],
        [_td(0), _td(10), _td(40)],
    ]

    ladder = SoilPredictor._build_ladder(window_durations, grid_mode="fill_order")

    # window 0 contributes ALL durations: (0,0,0), (30,0,0)
    # window 1 contributes non-zero only, meshed onto max0=30: (30,20,0)
    # window 2 contributes non-zero only, meshed onto max0=30,max1=20: (30,20,10), (30,20,40)
    assert ladder == [
        (_td(0), _td(0), _td(0)),
        (_td(30), _td(0), _td(0)),
        (_td(30), _td(20), _td(0)),
        (_td(30), _td(20), _td(10)),
        (_td(30), _td(20), _td(40)),
    ]

    totals = [sum((d for d in candidate), pd.Timedelta(0)) for candidate in ladder]
    assert totals == sorted(totals)
    assert len(set(totals)) == len(totals)

    count = len(window_durations[0]) + sum(len(d) - 1 for d in window_durations[1:])
    assert len(ladder) == count


def test_full_grid_mode_is_cartesian_product():
    window_durations = [
        [_td(0), _td(30), _td(60)],
        [_td(0), _td(30)],
    ]

    ladder = SoilPredictor._build_ladder(window_durations, grid_mode="full")

    assert len(ladder) == 3 * 2
    assert set(ladder) == {(d0, d1) for d0 in window_durations[0] for d1 in window_durations[1]}
    # Full mode keeps every combination, including the back-loaded one dropped by fill_order.
    assert (_td(0), _td(30)) in ladder


def test_unknown_grid_mode_raises():
    with pytest.raises(ValueError):
        SoilPredictor._build_ladder([[_td(0), _td(30)]], grid_mode="bogus")


# --- combo_cap fail-fast ------------------------------------------------------


def test_combo_cap_exceeded_raises():
    """A ladder longer than combo_cap must raise -- the exact check configure()
    performs (_check_combo_cap) against _build_ladder's static count."""
    window_durations = [
        [_td(m) for m in range(0, 10 * 30, 30)],  # 10 candidates on window 0 alone
        [_td(0), _td(30)],
    ]
    ladder = SoilPredictor._build_ladder(window_durations, grid_mode="fill_order")
    combo_cap = 5

    assert len(ladder) > combo_cap
    with pytest.raises(ValueError):
        SoilPredictor._check_combo_cap(ladder, combo_cap)


def test_combo_cap_not_exceeded_does_not_raise():
    window_durations = [
        [_td(0), _td(30), _td(60)],
        [_td(0), _td(30), _td(60)],
    ]
    ladder = SoilPredictor._build_ladder(window_durations, grid_mode="fill_order")
    combo_cap = 16

    assert len(ladder) <= combo_cap
    SoilPredictor._check_combo_cap(ladder, combo_cap)  # must not raise
