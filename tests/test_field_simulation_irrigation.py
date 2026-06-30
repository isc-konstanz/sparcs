# -*- coding: utf-8 -*-
"""Regression: the live irrigation callback must read NULL flow as "not watering".

``_irrigation_callback`` fires on every flow update (``how="any"``). When the
logger writes a NULL flow row, ``data.iloc[-1, 0]`` is ``NaN``; ``float(NaN)``
does not raise, so the old code stored ``NaN`` in ``_irrigation_flow_lpm`` and it
flowed straight into the soil PDE source (``soil.py``: ``flow_m3s = lpm/60_000``),
poisoning the simulation. This is the live analog of the bench asof-latch fix
(commit 20431c7). NULL must read as 0.

Importing ``FieldSimulation`` pulls the full lories + soil (FiPy) stack;
``importorskip`` keeps this out of environments that lack it (the full check runs
on the box). The callback only touches ``self`` and ``data``, so a bare instance
built with ``object.__new__`` exercises it without the Component machinery.
"""

import pytest

import numpy as np
import pandas as pd

base = pytest.importorskip("sparcs.components.agriculture.simulation.base")
FieldSimulation = base.FieldSimulation


def _callback_result(value) -> float:
    sim = object.__new__(FieldSimulation)
    sim._irrigation_flow_lpm = 999.0  # a latched prior burst, to prove it clears
    df = pd.DataFrame({"flow": [value]}, index=pd.to_datetime(["2026-05-01 10:00"]))
    FieldSimulation._irrigation_callback(sim, df)
    return sim._irrigation_flow_lpm


def test_null_flow_reads_as_zero():
    """A NULL/NaN flow row -> 0 L/min, not NaN latched into the PDE source."""
    assert _callback_result(np.nan) == 0.0


def test_nonzero_flow_passes_through():
    """A real flow reading is stored unchanged."""
    assert _callback_result(50.0) == pytest.approx(50.0)


def test_explicit_zero_stops_watering():
    """An explicit 0 flow clears a previously latched burst."""
    assert _callback_result(0.0) == 0.0
