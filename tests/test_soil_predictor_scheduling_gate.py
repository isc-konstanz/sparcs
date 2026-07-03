# -*- coding: utf-8 -*-
"""sparcs.tests.test_soil_predictor_scheduling_gate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Unit tests for issue 02: the ``interval``/``offset`` scheduling gate that makes the
predictor run once per day at a configured site-local time, and the ``cooldown``
property that feeds the per-listener backpressure floor in ``base.py``.

Importing ``soil_predictor`` pulls the full lories + soil (FiPy/Gmsh) stack via
``soil.py``; ``importorskip`` keeps this out of environments that lack it (the
full check runs on the box). ``_current_boundary`` and ``cooldown`` are pure/
config-only seams, exercised without any PDE/Component instantiation.
"""

import pytest

import pandas as pd

soil_predictor = pytest.importorskip("sparcs.components.agriculture.simulation.soil_predictor")
SoilPredictor = soil_predictor.SoilPredictor


# --- _current_boundary: the pure scheduling seam -----------------------------


def test_boundary_is_fixed_daily_time_past_local_midnight():
    """interval=1440 (daily), offset=60 -> boundary at 01:00 local, every day."""
    tz = "Europe/Berlin"
    now = pd.Timestamp("2026-07-03 10:00", tz=tz)

    boundary = SoilPredictor._current_boundary(now, tz, interval_min=1440, offset_min=60)

    assert boundary == pd.Timestamp("2026-07-03 01:00", tz=tz)


def test_boundary_before_offset_falls_back_to_previous_day():
    """now before today's 01:00 boundary -> most-recent boundary is yesterday's."""
    tz = "Europe/Berlin"
    now = pd.Timestamp("2026-07-03 00:30", tz=tz)

    boundary = SoilPredictor._current_boundary(now, tz, interval_min=1440, offset_min=60)

    assert boundary == pd.Timestamp("2026-07-02 01:00", tz=tz)


def test_boundary_constant_within_a_day_steps_once_per_interval():
    """Over a multi-day sequence, the boundary is constant within a day and advances
    by exactly one `interval` at each day's crossing -- never more, never less."""
    tz = "Europe/Berlin"
    interval_min = 1440
    offset_min = 60

    same_day_ticks = [
        pd.Timestamp("2026-07-03 01:00", tz=tz),
        pd.Timestamp("2026-07-03 06:00", tz=tz),
        pd.Timestamp("2026-07-03 12:00", tz=tz),
        pd.Timestamp("2026-07-03 23:59", tz=tz),
    ]
    boundaries = [SoilPredictor._current_boundary(t, tz, interval_min, offset_min) for t in same_day_ticks]
    assert len(set(boundaries)) == 1
    assert boundaries[0] == pd.Timestamp("2026-07-03 01:00", tz=tz)

    next_day_boundary = SoilPredictor._current_boundary(
        pd.Timestamp("2026-07-04 01:00", tz=tz), tz, interval_min, offset_min
    )
    assert next_day_boundary - boundaries[0] == pd.Timedelta(minutes=interval_min)


def test_boundary_site_local_same_instant_different_timezones():
    """The same UTC instant gates to different local boundaries in different
    site timezones (Europe/Berlin vs America/Los_Angeles), because the boundary
    is computed in site-local wall-clock time, not UTC."""
    instant_utc = pd.Timestamp("2026-07-03 09:00", tz="UTC")
    interval_min = 1440
    offset_min = 60

    berlin_boundary = SoilPredictor._current_boundary(instant_utc, "Europe/Berlin", interval_min, offset_min)
    la_boundary = SoilPredictor._current_boundary(instant_utc, "America/Los_Angeles", interval_min, offset_min)

    # Same absolute instant, different local boundary dates once converted back to UTC.
    assert berlin_boundary.tz_convert("UTC") != la_boundary.tz_convert("UTC")
    # Berlin (UTC+2 in July): 09:00 UTC = 11:00 local -> today's 01:00 local boundary.
    assert berlin_boundary == pd.Timestamp("2026-07-03 01:00", tz="Europe/Berlin")
    # Los Angeles (UTC-7 in July): 09:00 UTC = 02:00 local -> today's 01:00 local boundary.
    assert la_boundary == pd.Timestamp("2026-07-03 01:00", tz="America/Los_Angeles")


def test_boundary_custom_interval_and_offset():
    """Non-daily interval/offset combination steps at the configured cadence."""
    tz = "Europe/Berlin"
    interval_min = 60  # hourly
    offset_min = 15

    boundary = SoilPredictor._current_boundary(pd.Timestamp("2026-07-03 10:20", tz=tz), tz, interval_min, offset_min)
    assert boundary == pd.Timestamp("2026-07-03 10:15", tz=tz)

    boundary_before_offset = SoilPredictor._current_boundary(
        pd.Timestamp("2026-07-03 10:10", tz=tz), tz, interval_min, offset_min
    )
    assert boundary_before_offset == pd.Timestamp("2026-07-03 09:15", tz=tz)


# --- predict() fires once per boundary ---------------------------------------


class _FakeLocation:
    def __init__(self, timezone):
        self.timezone = timezone


class _FakeContext:
    def __init__(self, timezone):
        self.location = _FakeLocation(timezone)


def _make_gate_only_predictor(tz: str, interval_min: int, offset_min: int):
    """Build a bare ``SoilPredictor`` instance with only the attributes the
    scheduling-gate seam of ``predict()`` touches before the heavy roll-out
    (the forecast fetch), so this stays PDE-free.

    ``name`` and ``context`` are read-only properties (backed by ``_name`` and
    the name-mangled ``_Registrator__context``), so those backing attributes
    are set directly -- the same pattern ``test_soil_watering_normalization.py``
    uses for a bare ``SoilSimulation``.
    """
    predictor = object.__new__(SoilPredictor)
    predictor._name = "test_soil_predictor"
    predictor._Registrator__context = _FakeContext(tz)
    predictor._interval_min = interval_min
    predictor._offset_min = offset_min
    predictor._last_boundary_run = None
    predictor._last_predicted_key = None
    # Referenced by predict()'s no-forecast log message when a test returns None.
    predictor._horizon = pd.Timedelta("24h")

    # A present (non-empty) forecast lets predict() reach the boundary claim (which
    # sits just after the forecast guard); the fake context has no ``soil_simulation``,
    # so predict() then returns at the soil guard -- after the mark, before any PDE.
    # Tests exercising the missing-forecast path override this to return None.
    predictor._fetch_forecast = lambda now: pd.DataFrame({"_": [0.0]}, index=pd.DatetimeIndex([now]))
    return predictor


def test_predict_gate_fires_once_per_boundary_and_noops_between():
    tz = "Europe/Berlin"
    predictor = _make_gate_only_predictor(tz, interval_min=1440, offset_min=60)

    day1_boundary = pd.Timestamp("2026-07-03 01:00", tz=tz)
    ticks_day1 = [
        pd.Timestamp("2026-07-03 01:00", tz=tz),
        pd.Timestamp("2026-07-03 06:00", tz=tz),
        pd.Timestamp("2026-07-03 23:00", tz=tz),
    ]
    for now in ticks_day1:
        predictor.predict(now, forecast_creation=now)
        # The gate runs (updates _last_boundary_run) once for the first tick that
        # crosses day1_boundary; every later same-day tick no-ops before updating it.
        assert predictor._last_boundary_run == day1_boundary

    # A tick on day 2 crosses a new boundary -> gate fires again exactly once.
    day2_tick = pd.Timestamp("2026-07-04 02:00", tz=tz)
    predictor.predict(day2_tick, forecast_creation=day2_tick)
    day2_boundary = pd.Timestamp("2026-07-04 01:00", tz=tz)
    assert predictor._last_boundary_run == day2_boundary
    assert day2_boundary != day1_boundary


def test_predict_gate_noops_between_boundaries_tracked_via_fetch_calls():
    """Assert the gate short-circuits before touching `_fetch_forecast` on
    same-boundary ticks (proving the no-op is a true early return, not just an
    idempotent recompute)."""
    tz = "Europe/Berlin"
    predictor = _make_gate_only_predictor(tz, interval_min=1440, offset_min=60)

    calls = []

    def _tracking_fetch(now):
        calls.append(now)
        return pd.DataFrame({"_": [0.0]}, index=pd.DatetimeIndex([now]))

    predictor._fetch_forecast = _tracking_fetch

    first = pd.Timestamp("2026-07-03 01:00", tz=tz)
    second_same_day = pd.Timestamp("2026-07-03 20:00", tz=tz)

    predictor.predict(first, forecast_creation=first)
    assert len(calls) == 1  # gate passed, forecast fetched, boundary claimed

    predictor.predict(second_same_day, forecast_creation=second_same_day)
    assert len(calls) == 1  # same boundary -> gate no-op, fetch never called

    third_next_day = pd.Timestamp("2026-07-04 01:30", tz=tz)
    predictor.predict(third_next_day, forecast_creation=third_next_day)
    assert len(calls) == 2  # new boundary -> gate passed again


def test_predict_gate_missing_forecast_does_not_consume_the_boundary():
    """A transiently missing forecast at the boundary tick must NOT claim the
    boundary: the next tick (forecast now present) retries the same boundary
    instead of the whole day being silently skipped."""
    tz = "Europe/Berlin"
    predictor = _make_gate_only_predictor(tz, interval_min=1440, offset_min=60)

    calls = []
    forecast_box = {"value": None}  # first call: missing; later: present

    def _fetch(now):
        calls.append(now)
        return forecast_box["value"]

    predictor._fetch_forecast = _fetch

    boundary = pd.Timestamp("2026-07-03 01:00", tz=tz)

    # Boundary tick, forecast missing -> gate passes, fetch attempted, but the
    # boundary is NOT claimed (mark sits after the forecast guard).
    predictor.predict(pd.Timestamp("2026-07-03 01:00", tz=tz), forecast_creation=boundary)
    assert len(calls) == 1
    assert predictor._last_boundary_run is None

    # Later same-day tick, forecast now present -> the same boundary is retried
    # and claimed this time.
    forecast_box["value"] = pd.DataFrame({"_": [0.0]}, index=pd.DatetimeIndex([boundary]))
    predictor.predict(pd.Timestamp("2026-07-03 02:00", tz=tz), forecast_creation=boundary)
    assert len(calls) == 2
    assert predictor._last_boundary_run == boundary


def test_predict_gate_site_local_two_timezones_same_instant_gate_differently():
    """Two `predict()` calls at the same UTC instant but different site timezones
    can gate differently: an instant that is a fresh boundary in one timezone can
    still be within the same boundary as a prior call in another timezone."""
    instant_utc = pd.Timestamp("2026-07-03 09:00", tz="UTC")

    berlin = _make_gate_only_predictor("Europe/Berlin", interval_min=1440, offset_min=60)
    la = _make_gate_only_predictor("America/Los_Angeles", interval_min=1440, offset_min=60)

    berlin.predict(instant_utc, forecast_creation=instant_utc)
    la.predict(instant_utc, forecast_creation=instant_utc)

    assert berlin._last_boundary_run == pd.Timestamp("2026-07-03 01:00", tz="Europe/Berlin")
    assert la._last_boundary_run == pd.Timestamp("2026-07-03 01:00", tz="America/Los_Angeles")
    # Different absolute instants despite being "the same 01:00 local boundary" label.
    assert berlin._last_boundary_run.tz_convert("UTC") != la._last_boundary_run.tz_convert("UTC")


# --- cooldown property --------------------------------------------------------


def test_cooldown_default_is_sixty_minutes():
    predictor = object.__new__(SoilPredictor)
    predictor._cooldown_min = soil_predictor._DEFAULT_COOLDOWN_MIN

    assert predictor.cooldown == pd.Timedelta(minutes=60)


def test_cooldown_reflects_configured_value():
    predictor = object.__new__(SoilPredictor)
    predictor._cooldown_min = 15

    assert predictor.cooldown == pd.Timedelta(minutes=15)
