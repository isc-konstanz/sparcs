# -*- coding: utf-8 -*-
"""
tests.test_fieldsim_candidates_grid
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``build_candidate_grid`` (fill-order ladder and full product) and the
``combo_cap`` fail-fast check.
"""

import pytest
from conftest import td

import pandas as pd
from sparcs.components.agriculture.simulation.core.candidates import build_candidate_grid, check_candidate_cap

# --- fill_order ladder generation --------------------------------------------


def test_fill_order_two_windows_sweep_then_mesh_longest():
    """Two windows, durations [0,30,60] each: sweep window 0 with window 1 off,
    then mesh window 0's max with window 1's non-zero sweep. Order matters: the
    ladder is generated fill-earlier-first, so this asserts the exact sequence."""
    window_durations = [
        [td(0), td(30), td(60)],
        [td(0), td(30), td(60)],
    ]

    ladder = build_candidate_grid(window_durations, grid_mode="fill_order")

    assert ladder == [
        (td(0), td(0)),
        (td(30), td(0)),
        (td(60), td(0)),
        (td(60), td(30)),
        (td(60), td(60)),
    ]


def test_fill_order_total_water_strictly_increasing():
    window_durations = [
        [td(0), td(30), td(60)],
        [td(0), td(30), td(60)],
    ]
    ladder = build_candidate_grid(window_durations, grid_mode="fill_order")

    totals = [sum((d for d in candidate), pd.Timedelta(0)) for candidate in ladder]
    assert totals == sorted(totals)
    assert len(set(totals)) == len(totals)


def test_fill_order_drops_back_loaded_candidate():
    """(0min morning, 60min evening) is never generated -- front-load dominance."""
    window_durations = [
        [td(0), td(30), td(60)],
        [td(0), td(30), td(60)],
    ]
    ladder = build_candidate_grid(window_durations, grid_mode="fill_order")

    assert (td(0), td(60)) not in ladder
    assert (td(0), td(30)) not in ladder


def test_fill_order_three_windows():
    window_durations = [
        [td(0), td(30)],
        [td(0), td(20)],
        [td(0), td(10), td(40)],
    ]

    ladder = build_candidate_grid(window_durations, grid_mode="fill_order")

    # window 0 contributes ALL durations: (0,0,0), (30,0,0)
    # window 1 contributes non-zero only, meshed onto max0=30: (30,20,0)
    # window 2 contributes non-zero only, meshed onto max0=30,max1=20: (30,20,10), (30,20,40)
    assert ladder == [
        (td(0), td(0), td(0)),
        (td(30), td(0), td(0)),
        (td(30), td(20), td(0)),
        (td(30), td(20), td(10)),
        (td(30), td(20), td(40)),
    ]

    totals = [sum((d for d in candidate), pd.Timedelta(0)) for candidate in ladder]
    assert totals == sorted(totals)
    assert len(set(totals)) == len(totals)

    count = len(window_durations[0]) + sum(len(d) - 1 for d in window_durations[1:])
    assert len(ladder) == count


def test_full_grid_mode_is_cartesian_product():
    window_durations = [
        [td(0), td(30), td(60)],
        [td(0), td(30)],
    ]

    ladder = build_candidate_grid(window_durations, grid_mode="full")

    assert len(ladder) == 3 * 2
    assert set(ladder) == {(d0, d1) for d0 in window_durations[0] for d1 in window_durations[1]}
    # Full mode keeps every combination, including the back-loaded one dropped by fill_order.
    assert (td(0), td(30)) in ladder


def test_unknown_grid_mode_raises():
    with pytest.raises(ValueError):
        build_candidate_grid([[td(0), td(30)]], grid_mode="bogus")


# --- combo_cap fail-fast ------------------------------------------------------


def test_combo_cap_exceeded_raises():
    window_durations = [
        [td(m) for m in range(0, 10 * 30, 30)],  # 10 candidates on window 0 alone
        [td(0), td(30)],
    ]
    ladder = build_candidate_grid(window_durations, grid_mode="fill_order")
    combo_cap = 5

    assert len(ladder) > combo_cap
    with pytest.raises(ValueError):
        check_candidate_cap(ladder, combo_cap)


def test_combo_cap_not_exceeded_does_not_raise():
    window_durations = [
        [td(0), td(30), td(60)],
        [td(0), td(30), td(60)],
    ]
    ladder = build_candidate_grid(window_durations, grid_mode="fill_order")

    check_candidate_cap(ladder, 16)  # must not raise
