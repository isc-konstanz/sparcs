# -*- coding: utf-8 -*-
"""sparcs.tests.test_soil_predictor_irrigation_table
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Unit tests for the ``agri_field_forecast_irrigation`` edge-row table: the chosen
candidate's watering schedule, persisted as state-transition rows
(``SoilPredictor._build_irrigation_frame``) and channel-registered
(``SoilPredictor._register_irrigation_channels``) the same never-``.set()`` /
logger-gated way the header/detail tables are (``_register_header_channels``,
``_build_header_frame``).

Importing ``soil_predictor`` pulls the full lories + soil (FiPy/Gmsh) stack via
``soil.py``; ``importorskip`` keeps this out of environments that lack it (the
full check runs on the box).
"""

import datetime

import pytest

import pandas as pd

soil_predictor = pytest.importorskip("sparcs.components.agriculture.simulation.soil_predictor")
SoilPredictor = soil_predictor.SoilPredictor
WateringWindow = soil_predictor.WateringWindow


def _td(minutes: int) -> pd.Timedelta:
    return pd.Timedelta(minutes=minutes)


def _make_bare_predictor() -> SoilPredictor:
    """A bare ``SoilPredictor`` instance carrying only the attributes
    ``_build_irrigation_frame`` touches (``_windows``, ``_flow_m3s``, plus
    ``_build_flow_schedule`` it re-calls internally) -- the same
    ``object.__new__`` pattern ``test_soil_predictor_trajectory_table.py`` uses
    to avoid a full Component/PDE bootstrap.
    """
    predictor = object.__new__(SoilPredictor)
    predictor._name = "test_soil_predictor_irrigation"
    predictor._windows = [
        WateringWindow(start=datetime.time(8, 0)),
        WateringWindow(start=datetime.time(18, 0)),
    ]
    predictor._flow_m3s = 1.0e-5
    return predictor


# --- _build_irrigation_frame (pure) ------------------------------------------


def test_irrigation_frame_two_windows_yields_four_edge_rows():
    """Morning 08:00/30min + evening 18:00/1h -> two on/off interval pairs ->
    exactly 4 edge rows, correct timestamps/values, the SAME run timestamp on
    every row."""
    predictor = _make_bare_predictor()
    candidate = (_td(30), _td(60))
    horizon_start = pd.Timestamp("2026-07-03 01:00", tz="Europe/Berlin")
    horizon_end = horizon_start + pd.Timedelta(hours=24)
    run_ts = pd.Timestamp("2026-07-03 01:00", tz="Europe/Berlin")

    frame = predictor._build_irrigation_frame(candidate, horizon_start, horizon_end, run_ts)

    assert len(frame) == 4
    expected_index = [
        pd.Timestamp("2026-07-03 08:00", tz="Europe/Berlin"),
        pd.Timestamp("2026-07-03 08:30", tz="Europe/Berlin"),
        pd.Timestamp("2026-07-03 18:00", tz="Europe/Berlin"),
        pd.Timestamp("2026-07-03 19:00", tz="Europe/Berlin"),
    ]
    assert list(frame.index) == expected_index
    assert list(frame["irrigation_state"]) == [True, False, True, False]
    assert (frame["irrigation_timestamp_creation"] == run_ts).all()


def test_irrigation_frame_zero_duration_candidate_yields_zero_rows():
    """A do-nothing (all-0min) candidate has no state transition to record --
    zero rows, not a marker row (settled: edges are transitions; the
    do-nothing run is visible via its header row instead)."""
    predictor = _make_bare_predictor()
    candidate = (_td(0), _td(0))
    horizon_start = pd.Timestamp("2026-07-03 01:00", tz="Europe/Berlin")
    horizon_end = horizon_start + pd.Timedelta(hours=24)
    run_ts = pd.Timestamp("2026-07-03 01:00", tz="Europe/Berlin")

    frame = predictor._build_irrigation_frame(candidate, horizon_start, horizon_end, run_ts)

    assert frame.empty
    assert "irrigation_state" in frame.columns
    assert "irrigation_timestamp_creation" in frame.columns


def test_irrigation_frame_horizon_clamped_off_edge_emits_closing_false():
    """A window whose configured duration would run past the horizon must
    still close (never dangle True): ``_build_flow_schedule`` clamps the off
    edge to ``horizon_end``, and that clamped value is what gets emitted as
    the closing ``False`` row here."""
    predictor = _make_bare_predictor()
    predictor._windows = [WateringWindow(start=datetime.time(23, 0))]
    horizon_start = pd.Timestamp("2026-07-03 07:00", tz="Europe/Berlin")
    horizon_end = horizon_start + pd.Timedelta(hours=24)  # 2026-07-04 07:00
    candidate = (_td(10 * 60),)  # 10h, would end 2026-07-04 09:00 -- past horizon_end
    run_ts = horizon_start

    frame = predictor._build_irrigation_frame(candidate, horizon_start, horizon_end, run_ts)

    assert len(frame) == 2
    assert list(frame.index) == [pd.Timestamp("2026-07-03 23:00", tz="Europe/Berlin"), horizon_end]
    assert list(frame["irrigation_state"]) == [True, False]


def test_irrigation_frame_empty_schedule_returns_empty_frame_with_columns():
    predictor = _make_bare_predictor()
    predictor._windows = []
    horizon_start = pd.Timestamp("2026-07-03 01:00", tz="Europe/Berlin")
    horizon_end = horizon_start + pd.Timedelta(hours=24)

    frame = predictor._build_irrigation_frame((), horizon_start, horizon_end, horizon_start)

    assert frame.empty
    assert "irrigation_state" in frame.columns


def test_irrigation_frame_touching_windows_merge_into_one_interval():
    """Window A's clamped off_ts lands exactly on window B's on_ts -- nothing
    at configure() forbids this, and irrigation stays on through the joint,
    so it must merge into ONE interval (2 edge rows), not emit a spurious
    (False, True) pair on the identical timestamp."""
    predictor = _make_bare_predictor()  # windows = [08:00, 18:00]
    candidate = (_td(600), _td(30))  # window A: 08:00 + 10h == window B's 18:00 on_ts
    horizon_start = pd.Timestamp("2026-07-03 07:00", tz="Europe/Berlin")
    horizon_end = horizon_start + pd.Timedelta(hours=24)
    run_ts = horizon_start

    frame = predictor._build_irrigation_frame(candidate, horizon_start, horizon_end, run_ts)

    assert len(frame) == 2
    assert not frame.index.duplicated().any()
    assert list(frame.index) == [
        pd.Timestamp("2026-07-03 08:00", tz="Europe/Berlin"),
        pd.Timestamp("2026-07-03 18:30", tz="Europe/Berlin"),
    ]
    assert list(frame["irrigation_state"]) == [True, False]


def test_irrigation_frame_overlapping_windows_merge_into_one_interval():
    """Window A's interval extends past window B's on_ts (a true overlap, not
    just a touch) -- same merge behavior as the touching case."""
    predictor = _make_bare_predictor()  # windows = [08:00, 18:00]
    candidate = (_td(11 * 60), _td(30))  # window A: 08:00 + 11h = 19:00, past window B's 18:00 on_ts
    horizon_start = pd.Timestamp("2026-07-03 07:00", tz="Europe/Berlin")
    horizon_end = horizon_start + pd.Timedelta(hours=24)
    run_ts = horizon_start

    frame = predictor._build_irrigation_frame(candidate, horizon_start, horizon_end, run_ts)

    assert len(frame) == 2
    assert not frame.index.duplicated().any()
    assert list(frame.index) == [
        pd.Timestamp("2026-07-03 08:00", tz="Europe/Berlin"),
        pd.Timestamp("2026-07-03 19:00", tz="Europe/Berlin"),
    ]
    assert list(frame["irrigation_state"]) == [True, False]


def test_irrigation_frame_short_horizon_drops_degenerate_window_but_keeps_others():
    """predict() only guarantees >= 2 forecast rows, so horizon_end can land
    at or before a later window's resolved on_ts: off_ts = min(on_ts +
    duration, horizon_end) then gives on_ts >= off_ts. That window must
    contribute zero rows -- but an earlier, still-valid window's edges must
    survive untouched."""
    predictor = _make_bare_predictor()  # windows = [08:00, 18:00]
    candidate = (_td(30), _td(30))
    horizon_start = pd.Timestamp("2026-07-03 07:00", tz="Europe/Berlin")
    horizon_end = pd.Timestamp("2026-07-03 12:00", tz="Europe/Berlin")  # cuts off window B (18:00) entirely
    run_ts = horizon_start

    frame = predictor._build_irrigation_frame(candidate, horizon_start, horizon_end, run_ts)

    assert len(frame) == 2
    assert list(frame.index) == [
        pd.Timestamp("2026-07-03 08:00", tz="Europe/Berlin"),
        pd.Timestamp("2026-07-03 08:30", tz="Europe/Berlin"),
    ]
    assert list(frame["irrigation_state"]) == [True, False]


# --- _register_irrigation_channels: exact data.add() kwargs ------------------


class _RecordingData:
    """Stand-in for ``self.data``: captures every ``add(...)`` call's kwargs."""

    def __init__(self):
        self.added: list[tuple] = []

    def add(self, key, **kwargs) -> None:
        self.added.append((key, kwargs))


def _bare_predictor_for_registration(monkeypatch, **extra) -> tuple:
    predictor = object.__new__(SoilPredictor)
    predictor._name = "test_soil_predictor_irrigation"
    for key, value in extra.items():
        setattr(predictor, key, value)
    fake = _RecordingData()
    monkeypatch.setattr(SoilPredictor, "data", property(lambda self: fake))
    return predictor, fake


def test_register_irrigation_channels_binds_irrigation_table(monkeypatch):
    """Both channels route to the configured logger connector's
    agri_field_forecast_irrigation table with logger.enabled=True (direct-write path);
    irrigation_state is a plain data column (not primary), and
    timestamp_creation is the primary/non-nullable PK partner with a
    'timestamp_creation' column override (its channel key differs from the
    DB column name, mirroring the detail table's twins)."""
    predictor, fake = _bare_predictor_for_registration(monkeypatch, _logger_id="mariadb")

    predictor._register_irrigation_channels()

    by_id = dict(fake.added)
    assert SoilPredictor._IRRIGATION_STATE_KEY in by_id
    assert SoilPredictor._IRRIGATION_TIMESTAMP_CREATION_KEY in by_id

    for channel_id, kwargs in fake.added:
        assert kwargs["logger"]["table"] == SoilPredictor._IRRIGATION_TABLE_NAME
        assert kwargs["logger"]["connector"] == "mariadb"
        assert kwargs["logger"]["enabled"] is True

    state_kwargs = by_id[SoilPredictor._IRRIGATION_STATE_KEY]
    assert state_kwargs["type"] is bool
    assert state_kwargs["logger"].get("primary") is not True
    assert "column" not in state_kwargs["logger"]

    creation_kwargs = by_id[SoilPredictor._IRRIGATION_TIMESTAMP_CREATION_KEY]
    assert creation_kwargs["logger"]["column"] == "timestamp_creation"
    assert creation_kwargs["logger"]["primary"] is True
    assert creation_kwargs["logger"]["nullable"] is False


# --- _write_irrigation_table: degrade-on-missing-connector --------------------


def test_write_irrigation_table_skips_when_logger_not_configured():
    predictor = _make_bare_predictor()
    predictor._logger_id = None

    frame = pd.DataFrame({"irrigation_state": [True]}, index=pd.DatetimeIndex([pd.Timestamp("2026-07-03", tz="UTC")]))
    # Must not raise, and must not attempt any connector resolution.
    predictor._write_irrigation_table(frame)


def test_write_irrigation_table_empty_frame_is_a_noop():
    """An empty irrigation frame (e.g. a do-nothing run) must be a no-op
    BEFORE any connector resolution is attempted -- deliberately no
    connectors patch here, so this would error (not silently pass) if the
    empty-frame guard were removed."""
    predictor = _make_bare_predictor()
    predictor._logger_id = "db"

    predictor._write_irrigation_table(pd.DataFrame())


# --- Real write path: build -> write -> rename -> connector, no stray .set() -


class _FakeSetChannel:
    """Records every ``.set()`` call; mirrors the ``Channel`` surface the
    predictor touches (``.set(timestamp, value)``, ``.id``, ``.logger``)."""

    def __init__(self, channel_id: str):
        self.id = channel_id
        self.calls: list[tuple[pd.Timestamp, object]] = []
        self.timestamp = pd.NaT  # matches a never-.set() lories Channel

    def set(self, timestamp, value) -> None:
        self.calls.append((timestamp, value))
        self.timestamp = timestamp

    @property
    def logger(self):
        # No pre-bound registrator -- forces _resolve_logger_connector's
        # id-based fallback (getattr(self.connectors, logger_id)).
        class _NullLogger:
            @staticmethod
            def _get_registrator():
                return None

        return _NullLogger()


class _FakeDataAccess:
    """Minimal ``self.data`` stand-in: a dict of key -> _FakeSetChannel, with
    ``__getitem__`` matching ``DataAccess``'s access pattern."""

    def __init__(self, keys: list[str]):
        self._channels = {key: _FakeSetChannel(f"test_soil_predictor_irrigation.{key}") for key in keys}

    def __getitem__(self, key: str) -> _FakeSetChannel:
        return self._channels[key]


def _patch_data(monkeypatch, fake_data) -> None:
    """``Component.data``/``Component.connectors`` are read-only class
    properties (no setter), so a bare ``object.__new__`` instance cannot get
    a plain instance attribute assigned over them -- patch the CLASS property
    for the duration of the test instead (``monkeypatch`` auto-restores)."""
    monkeypatch.setattr(SoilPredictor, "data", property(lambda self: fake_data))


def _patch_connectors(monkeypatch, fake_connectors) -> None:
    monkeypatch.setattr(SoilPredictor, "connectors", property(lambda self: fake_connectors))


def test_write_irrigation_table_renames_to_channel_ids_and_never_calls_set(monkeypatch):
    """Exercises the REAL write path end to end (``_build_irrigation_frame`` ->
    ``_write_irrigation_table`` -> the publisher's lazy ``_ids_for`` id map ->
    rename -> ``connector.write``) against a recording fake connector: the
    frame the connector receives must be keyed by the RESOLVED channel ids,
    not the bare in-frame keys, and neither irrigation channel is ever
    ``.set()``. A typo'd key in the ``_ids_for`` registration would surface
    here as a KeyError or a column that never got renamed."""
    predictor = _make_bare_predictor()
    predictor._logger_id = "db"
    keys = [SoilPredictor._IRRIGATION_STATE_KEY, SoilPredictor._IRRIGATION_TIMESTAMP_CREATION_KEY]
    fake_data = _FakeDataAccess(keys)
    _patch_data(monkeypatch, fake_data)

    written = []

    class _Connector:
        def write(self, frame):
            written.append(frame)

    class _Connectors:
        db = _Connector()

    _patch_connectors(monkeypatch, _Connectors())

    candidate = (_td(30), _td(60))
    horizon_start = pd.Timestamp("2026-07-03 01:00", tz="Europe/Berlin")
    horizon_end = horizon_start + pd.Timedelta(hours=24)
    run_ts = horizon_start

    frame = predictor._build_irrigation_frame(candidate, horizon_start, horizon_end, run_ts)
    predictor._write_irrigation_table(frame)

    assert len(written) == 1, "the connector must have received the write"
    written_frame = written[0]
    expected_ids = {fake_data[key].id for key in keys}
    assert set(written_frame.columns) == expected_ids

    for key in keys:
        assert fake_data[key].calls == [], f"'{key}' must never be .set()"
        assert pd.isna(fake_data[key].timestamp), f"'{key}' timestamp must stay NaT"
