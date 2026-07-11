# -*- coding: utf-8 -*-
"""sparcs.tests.test_soil_intake_delay
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pure unit tests for the ``intake_delay`` feature on ``FieldSimulation``: the
frame-clipping helper that holds the whole chain behind wall-clock so it only
advances over replicated data, and the ``[field_simulation] intake_delay``
config parse.

Importing ``base`` pulls the full lories + FiPy/Gmsh stack; ``importorskip``
keeps this out of environments that lack it (the full check runs on the box).
The targets are a ``@staticmethod`` and a config parse, so no ``Component`` /
PDE instantiation or solver run is needed -- the wall-clock read in
``_replication_cutoff`` is deliberately kept out of the assertions.
"""

import types
from unittest.mock import Mock

import pytest

import pandas as pd

_base = pytest.importorskip("sparcs.components.agriculture.simulation.base")
FieldSimulation = _base.FieldSimulation

from lories import Configurations  # noqa: E402


def _weather_frame() -> pd.DataFrame:
    # tz-aware UTC index, matching what lories channels stamp (Channel.set).
    index = pd.date_range("2026-07-07 10:00", periods=5, freq="15min", tz="UTC")
    return pd.DataFrame({"ghi": [100.0, 200.0, 300.0, 400.0, 500.0]}, index=index)


def _configs(tmp_path, **values) -> Configurations:
    return Configurations.load("test.conf", conf_dir=str(tmp_path), require=False, **values)


# --- _clip_to_cutoff --------------------------------------------------------


def test_clip_to_cutoff_none_returns_frame_unchanged():
    """intake_delay == 0 -> cutoff None -> the exact same frame object, untouched."""
    frame = _weather_frame()
    clipped = FieldSimulation._clip_to_cutoff(frame, None)
    assert clipped is frame


def test_clip_to_cutoff_interior_drops_later_rows():
    frame = _weather_frame()
    cutoff = pd.Timestamp("2026-07-07 10:30", tz="UTC")  # the third of five rows
    clipped = FieldSimulation._clip_to_cutoff(frame, cutoff)
    assert len(clipped) == 3
    assert clipped.index[-1] == cutoff  # now stays the latest row at or before the cutoff


def test_clip_to_cutoff_keeps_row_exactly_at_cutoff():
    """The keep is inclusive, so a cutoff landing on the last row keeps all rows."""
    frame = _weather_frame()
    cutoff = frame.index[-1]
    clipped = FieldSimulation._clip_to_cutoff(frame, cutoff)
    assert len(clipped) == len(frame)
    assert clipped.index[-1] == cutoff


def test_clip_to_cutoff_before_frame_returns_empty():
    """A cutoff before the whole frame empties it, so the callback no-ops the tick."""
    frame = _weather_frame()
    cutoff = pd.Timestamp("2026-07-07 09:00", tz="UTC")  # before the first row
    clipped = FieldSimulation._clip_to_cutoff(frame, cutoff)
    assert clipped.empty


# --- config parse -----------------------------------------------------------


def test_intake_delay_absent_parses_to_zero(tmp_path):
    configs = _configs(tmp_path)
    assert FieldSimulation._parse_intake_delay(configs) == pd.Timedelta(0)


def test_intake_delay_parses_duration_string(tmp_path):
    configs = _configs(tmp_path, intake_delay="30min")
    assert FieldSimulation._parse_intake_delay(configs) == pd.Timedelta(minutes=30)


def test_intake_delay_default_attr_is_zero_on_field():
    """The class default is the off-state, so a field whose config omits the key
    runs the byte-for-byte pre-feature path."""
    assert FieldSimulation._intake_delay == pd.Timedelta(0)


def test_backfill_max_absent_parses_to_one_day(tmp_path):
    configs = _configs(tmp_path)
    assert FieldSimulation._parse_backfill_max(configs) == pd.Timedelta(days=1)


def test_backfill_max_parses_duration_string(tmp_path):
    configs = _configs(tmp_path, backfill_max="6h")
    assert FieldSimulation._parse_backfill_max(configs) == pd.Timedelta(hours=6)


# --- _ranged_window ----------------------------------------------------------


def test_ranged_window_none_last_returns_bounded_warmup_window():
    """First tick (no cursor yet): a bounded warm-up window ending at the frontier."""
    cutoff = pd.Timestamp("2026-07-07 12:00", tz="UTC")
    backfill_max = pd.Timedelta(hours=6)
    window = FieldSimulation._ranged_window(None, cutoff, backfill_max)
    assert window == (cutoff - backfill_max, cutoff)


def test_ranged_window_last_within_cap_returns_up_to_cutoff():
    """Backlog smaller than the cap drains in a single window reaching the cutoff."""
    cutoff = pd.Timestamp("2026-07-07 12:00", tz="UTC")
    last = pd.Timestamp("2026-07-07 10:00", tz="UTC")  # 2h behind, cap is 6h
    backfill_max = pd.Timedelta(hours=6)
    window = FieldSimulation._ranged_window(last, cutoff, backfill_max)
    assert window == (last, cutoff)


def test_ranged_window_last_far_behind_caps_window_below_cutoff():
    """A backlog bigger than the cap is drained in bounded chunks, not in one shot."""
    cutoff = pd.Timestamp("2026-07-08 12:00", tz="UTC")
    last = pd.Timestamp("2026-07-07 00:00", tz="UTC")  # 36h behind, cap is 6h
    backfill_max = pd.Timedelta(hours=6)
    window = FieldSimulation._ranged_window(last, cutoff, backfill_max)
    assert window == (last, last + backfill_max)
    assert window[1] < cutoff


def test_ranged_window_last_equal_cutoff_returns_none():
    cutoff = pd.Timestamp("2026-07-07 12:00", tz="UTC")
    window = FieldSimulation._ranged_window(cutoff, cutoff, pd.Timedelta(hours=6))
    assert window is None


def test_ranged_window_last_after_cutoff_returns_none():
    cutoff = pd.Timestamp("2026-07-07 12:00", tz="UTC")
    last = cutoff + pd.Timedelta(minutes=1)
    window = FieldSimulation._ranged_window(last, cutoff, pd.Timedelta(hours=6))
    assert window is None


# --- _weather_callback dispatch (intake_delay > 0) ---------------------------


def test_weather_callback_delay_positive_delegates_to_advance_ranged():
    """delay > 0 -> trigger-only: delegate to _advance_ranged, never touch the
    snapshot _weather_channels frame."""
    weather_channels = Mock()
    advance_ranged = Mock()
    stub = types.SimpleNamespace(
        _weather_inputs_valid=lambda: True,
        _intake_delay=pd.Timedelta("1h"),
        _weather_channels=weather_channels,
        _advance_ranged=advance_ranged,
    )
    FieldSimulation._weather_callback(stub, _weather_frame())
    advance_ranged.assert_called_once_with()
    weather_channels.to_frame.assert_not_called()


def test_weather_callback_delay_zero_does_not_delegate():
    """Regression: delay == 0 keeps taking the live snapshot path unchanged."""
    weather_channels = Mock()
    weather_channels.to_frame.return_value = pd.DataFrame()
    advance_ranged = Mock()
    stub = types.SimpleNamespace(
        _weather_inputs_valid=lambda: True,
        _intake_delay=pd.Timedelta(0),
        _weather_channels=weather_channels,
        _advance_ranged=advance_ranged,
    )
    FieldSimulation._weather_callback(stub, _weather_frame())
    advance_ranged.assert_not_called()
    weather_channels.to_frame.assert_called_once_with(unique=True)


# --- _advance_ranged ----------------------------------------------------------


def _advance_stub(*, last, cutoff, backfill_max, read_result=None, read_side_effect=None, weather_channels=None):
    weather = types.SimpleNamespace(data=Mock())
    if read_side_effect is not None:
        weather.data.read.side_effect = read_side_effect
    else:
        weather.data.read.return_value = read_result
    soil_simulation = Mock()
    soil_simulation._last_simulated_at = last
    if weather_channels is None:
        weather_channels = Mock(name="weather_channels")
    return types.SimpleNamespace(
        name="field",
        weather=weather,
        soil_simulation=soil_simulation,
        _backfill_max=backfill_max,
        _replication_cutoff=lambda: cutoff,
        _ranged_window=FieldSimulation._ranged_window,
        _weather_channels=weather_channels,
        _run_chain=Mock(),
    )


def test_advance_ranged_happy_path_steps_soil_via_simulate_loop():
    """Multi-row windows must step the PDE per row (simulate_loop), not one lumped advance()."""
    cutoff = pd.Timestamp("2026-07-07 12:00", tz="UTC")
    last = pd.Timestamp("2026-07-07 10:00", tz="UTC")
    backfill_max = pd.Timedelta(hours=6)
    index = pd.date_range("2026-07-07 10:15", periods=3, freq="15min", tz="UTC")
    frame = pd.DataFrame({"ghi": [1.0, 2.0, 3.0]}, index=index)
    stub = _advance_stub(last=last, cutoff=cutoff, backfill_max=backfill_max, read_result=frame)
    et_data = pd.DataFrame({"et": [1.0, 2.0, 3.0]}, index=index)
    seg_et = {"seg": pd.DataFrame()}
    stub._run_chain.return_value = (et_data, seg_et)

    FieldSimulation._advance_ranged(stub)

    expected_window = FieldSimulation._ranged_window(last, cutoff, backfill_max)
    stub.weather.data.read.assert_called_once_with(
        start=expected_window[0], end=expected_window[1], channels=stub._weather_channels
    )
    called_frame = stub._run_chain.call_args.args[0]
    pd.testing.assert_frame_equal(called_frame, frame)
    stub.soil_simulation.simulate_loop.assert_called_once_with(et_data, seg_et)
    stub.soil_simulation.advance.assert_not_called()


def test_advance_ranged_drops_row_exactly_at_last():
    """DB range reads are start-inclusive; the boundary row (already simulated) is dropped."""
    cutoff = pd.Timestamp("2026-07-07 12:00", tz="UTC")
    last = pd.Timestamp("2026-07-07 10:00", tz="UTC")
    backfill_max = pd.Timedelta(hours=6)
    index = pd.date_range("2026-07-07 10:00", periods=3, freq="15min", tz="UTC")  # first row == last
    frame = pd.DataFrame({"ghi": [1.0, 2.0, 3.0]}, index=index)
    stub = _advance_stub(last=last, cutoff=cutoff, backfill_max=backfill_max, read_result=frame)
    et_data = pd.DataFrame({"et": [2.0, 3.0]}, index=index[1:])
    stub._run_chain.return_value = (et_data, {})

    FieldSimulation._advance_ranged(stub)

    called_frame = stub._run_chain.call_args.args[0]
    assert list(called_frame.index) == list(index[1:])


def test_advance_ranged_empty_read_skips_chain():
    cutoff = pd.Timestamp("2026-07-07 12:00", tz="UTC")
    last = pd.Timestamp("2026-07-07 10:00", tz="UTC")
    backfill_max = pd.Timedelta(hours=6)
    frame = pd.DataFrame({"ghi": []}, index=pd.DatetimeIndex([], tz="UTC"))
    stub = _advance_stub(last=last, cutoff=cutoff, backfill_max=backfill_max, read_result=frame)

    FieldSimulation._advance_ranged(stub)

    stub._run_chain.assert_not_called()
    stub.soil_simulation.simulate_loop.assert_not_called()


def test_advance_ranged_all_rows_at_or_before_last_skips_chain():
    cutoff = pd.Timestamp("2026-07-07 12:00", tz="UTC")
    last = pd.Timestamp("2026-07-07 10:00", tz="UTC")
    backfill_max = pd.Timedelta(hours=6)
    index = pd.date_range("2026-07-07 09:00", periods=2, freq="15min", tz="UTC")  # both <= last
    frame = pd.DataFrame({"ghi": [1.0, 2.0]}, index=index)
    stub = _advance_stub(last=last, cutoff=cutoff, backfill_max=backfill_max, read_result=frame)

    FieldSimulation._advance_ranged(stub)

    stub._run_chain.assert_not_called()
    stub.soil_simulation.simulate_loop.assert_not_called()


def test_advance_ranged_read_failure_does_not_raise():
    """A tick must never raise; a failed ranged read logs and no-ops."""
    cutoff = pd.Timestamp("2026-07-07 12:00", tz="UTC")
    last = pd.Timestamp("2026-07-07 10:00", tz="UTC")
    backfill_max = pd.Timedelta(hours=6)
    stub = _advance_stub(last=last, cutoff=cutoff, backfill_max=backfill_max, read_side_effect=RuntimeError("boom"))

    FieldSimulation._advance_ranged(stub)

    stub.soil_simulation.simulate_loop.assert_not_called()


def test_advance_ranged_tz_naive_read_frame_localized_to_utc():
    """SQL TIMESTAMP/DATETIME extract can return a tz-naive index; it must be localized to UTC
    before the boundary drop against the tz-aware ``last`` cursor, without raising."""
    cutoff = pd.Timestamp("2026-07-07 12:00", tz="UTC")
    last = pd.Timestamp("2026-07-07 10:00", tz="UTC")
    backfill_max = pd.Timedelta(hours=6)
    naive_index = pd.date_range("2026-07-07 10:15", periods=3, freq="15min")  # tz-naive
    frame = pd.DataFrame({"ghi": [1.0, 2.0, 3.0]}, index=naive_index)
    stub = _advance_stub(last=last, cutoff=cutoff, backfill_max=backfill_max, read_result=frame)
    aware_index = naive_index.tz_localize("UTC")
    et_data = pd.DataFrame({"et": [1.0, 2.0, 3.0]}, index=aware_index)
    stub._run_chain.return_value = (et_data, {})

    FieldSimulation._advance_ranged(stub)  # must not raise

    called_frame = stub._run_chain.call_args.args[0]
    assert called_frame.index.tz is not None
    assert len(called_frame) == 3  # all rows strictly after last, none dropped by the boundary filter
    stub.soil_simulation.simulate_loop.assert_called_once()


def test_advance_ranged_tz_naive_last_normalized_before_window():
    """A tz-naive cursor (e.g. handed back by a warm-start restore) is localized to UTC before
    ``_ranged_window`` -- comparing it against the tz-aware cutoff must not raise."""
    cutoff = pd.Timestamp("2026-07-07 12:00", tz="UTC")
    last_naive = pd.Timestamp("2026-07-07 10:00")  # tz-naive
    backfill_max = pd.Timedelta(hours=6)
    index = pd.date_range("2026-07-07 10:15", periods=2, freq="15min", tz="UTC")
    frame = pd.DataFrame({"ghi": [1.0, 2.0]}, index=index)
    stub = _advance_stub(last=last_naive, cutoff=cutoff, backfill_max=backfill_max, read_result=frame)
    et_data = pd.DataFrame({"et": [1.0, 2.0]}, index=index)
    stub._run_chain.return_value = (et_data, {})

    FieldSimulation._advance_ranged(stub)  # must not raise

    expected_last = last_naive.tz_localize("UTC")
    expected_window = FieldSimulation._ranged_window(expected_last, cutoff, backfill_max)
    stub.weather.data.read.assert_called_once_with(
        start=expected_window[0], end=expected_window[1], channels=stub._weather_channels
    )
    stub.soil_simulation.simulate_loop.assert_called_once()


def test_advance_ranged_chain_evaluation_failure_does_not_raise():
    """Backfilled rows are gap/NaN-prone; a chain-evaluation error (e.g. Evapotranspiration's
    ValueError on NaN required columns) must not propagate out of the tick."""
    cutoff = pd.Timestamp("2026-07-07 12:00", tz="UTC")
    last = pd.Timestamp("2026-07-07 10:00", tz="UTC")
    backfill_max = pd.Timedelta(hours=6)
    index = pd.date_range("2026-07-07 10:15", periods=2, freq="15min", tz="UTC")
    frame = pd.DataFrame({"ghi": [1.0, 2.0]}, index=index)
    stub = _advance_stub(last=last, cutoff=cutoff, backfill_max=backfill_max, read_result=frame)
    stub._run_chain.side_effect = ValueError("NaN in required column")

    FieldSimulation._advance_ranged(stub)  # must not raise

    stub.soil_simulation.simulate_loop.assert_not_called()
