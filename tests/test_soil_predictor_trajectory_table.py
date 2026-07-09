# -*- coding: utf-8 -*-
"""sparcs.tests.test_soil_predictor_trajectory_table
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Unit tests for the forecast header/detail tables: the recommendation
stage is collapsed (no ``soil_predictor_recommendation`` table, no
``_publish_recommendation``), and the old wide-column all-candidate
trajectory table is replaced by a header/detail pair --
``SoilPredictor._build_header_frame`` (one ``agri_field_forecast`` row per
candidate per run) and ``SoilPredictor._build_detail_frame`` (per-probe LONG
``agri_soil_forecast`` rows for every candidate) -- both pure, exercised
against a bare instance with a fake ``self.data``/``self.connectors`` so no
``Component``/PDE bootstrap is needed. ``_write_header_table``/
``_write_detail_table``'s connector-resolution/degrade-on-missing-connector
behavior is exercised the same way the old ``_write_trajectory_table`` was.

Importing ``soil_predictor`` pulls the full lories + soil (FiPy/Gmsh) stack via
``soil.py``; ``importorskip`` keeps this out of environments that lack it (the
full check runs on the box).

The duplicate-timestamp composite-PK round-trip against a real MariaDB/MySQL
logger (PRD Prerequisite 2 / this issue's spike) is BOX-PENDING: there is no
local DB server available and the lories SQL connector rejects sqlite, so it
cannot be faked against an in-memory backend without misrepresenting the
upsert behavior under test. See ``test_soil_predictor_trajectory_roundtrip.py``
for the skipped placeholder.
"""

import datetime

import pytest

import numpy as np
import pandas as pd

soil_predictor = pytest.importorskip("sparcs.components.agriculture.simulation.soil_predictor")
SoilPredictor = soil_predictor.SoilPredictor
WateringWindow = soil_predictor.WateringWindow


def _td(minutes: int) -> pd.Timedelta:
    return pd.Timedelta(minutes=minutes)


def _make_bare_predictor(max_windows: int = 4, n_windows: int = 2) -> SoilPredictor:
    """A bare ``SoilPredictor`` instance carrying only the attributes
    ``_build_header_frame``/``_build_detail_frame`` touch -- the same
    ``object.__new__`` pattern ``test_soil_predictor_scheduling_gate.py`` and
    ``test_soil_predictor_ladder_rollout.py`` use to avoid a full Component/PDE
    bootstrap.
    """
    predictor = object.__new__(SoilPredictor)
    predictor._name = "test_soil_predictor_trajectory"
    predictor._max_windows = max_windows
    predictor._windows = [
        WateringWindow(start=datetime.time(8, 0)),
        WateringWindow(start=datetime.time(18, 0)),
    ][:n_windows]
    predictor._header_window_min_keys = [f"w{i}_min" for i in range(max_windows)]
    predictor._header_window_start_keys = [f"w{i}_start" for i in range(max_windows)]
    predictor._traj_channel_keys = {"root_20": "traj_root_20", "root_40": "traj_root_40"}
    predictor._detail_creation_keys = {
        "root_20": "traj_root_20_timestamp_creation",
        "root_40": "traj_root_40_timestamp_creation",
    }
    predictor._detail_forecast_id_keys = {
        "root_20": "traj_root_20_forecast_id",
        "root_40": "traj_root_40_forecast_id",
    }
    predictor._logger_id = "db"
    return predictor


# --- _build_header_frame (pure) ------------------------------------------


def _synthetic_ladder():
    """A 2-window fill-order-style ladder: 3 candidates."""
    return [(_td(0), _td(0)), (_td(30), _td(0)), (_td(60), _td(0))]


_RUN_TS = pd.Timestamp("2026-07-03 01:00", tz="Europe/Berlin")
_WEATHER_TS = pd.Timestamp("2026-07-03 00:00", tz="UTC")


def test_header_frame_has_expected_columns():
    predictor = _make_bare_predictor(max_windows=4, n_windows=2)
    ladder = _synthetic_ladder()
    chosen = ladder[1]

    frame = predictor._build_header_frame(
        ladder, chosen, pd.Timestamp("2026-07-03 01:00", tz="Europe/Berlin"), pd.Timestamp("2026-07-03 00:00", tz="UTC")
    )

    expected_columns = {
        "forecast_id",
        "w0_min",
        "w1_min",
        "w2_min",
        "w3_min",
        "w0_start",
        "w1_start",
        "w2_start",
        "w3_start",
        "is_recommended",
        "total_min",
        "weather_creation",
    }
    assert set(frame.columns) == expected_columns


def test_header_frame_row_count_is_one_per_candidate():
    predictor = _make_bare_predictor(max_windows=4, n_windows=2)
    ladder = _synthetic_ladder()
    chosen = ladder[0]

    frame = predictor._build_header_frame(
        ladder, chosen, pd.Timestamp("2026-07-03 01:00", tz="Europe/Berlin"), pd.Timestamp("2026-07-03 00:00", tz="UTC")
    )

    assert len(frame) == len(ladder)


def test_header_frame_indexed_at_run_timestamp_for_every_row():
    """Every candidate's header row is indexed at the single RUN timestamp --
    forecast_id, not a second timestamp channel, is the header's PK partner."""
    predictor = _make_bare_predictor(max_windows=4, n_windows=2)
    ladder = _synthetic_ladder()
    run_ts = pd.Timestamp("2026-07-03 01:00", tz="Europe/Berlin")

    frame = predictor._build_header_frame(ladder, ladder[0], run_ts, pd.Timestamp("2026-07-03 00:00", tz="UTC"))

    assert (frame.index == run_ts).all()


def test_header_frame_unconfigured_windows_are_null_not_sentinel():
    """Ladder has 2 active windows but max_windows=4 -- w2_min/w3_min/w2_start/
    w3_start must be NULL (None/NaN) for every row; no -1.0 fill sentinel."""
    predictor = _make_bare_predictor(max_windows=4, n_windows=2)
    ladder = _synthetic_ladder()

    frame = predictor._build_header_frame(ladder, ladder[0], _RUN_TS, _WEATHER_TS)

    assert frame["w2_min"].isna().all()
    assert frame["w3_min"].isna().all()
    assert frame["w2_start"].isna().all()
    assert frame["w3_start"].isna().all()
    # The two configured windows must NOT be null, and must not be -1 either.
    assert not frame["w0_min"].isna().any()
    assert not frame["w1_min"].isna().any()
    assert not (frame["w0_min"] == -1.0).any()


def test_header_frame_carries_window_start_clock_times():
    """w{i}_start persists the configured window's clock time (previously it
    lived only in the config, never in the DB)."""
    predictor = _make_bare_predictor(max_windows=4, n_windows=2)
    ladder = _synthetic_ladder()

    frame = predictor._build_header_frame(ladder, ladder[0], _RUN_TS, _WEATHER_TS)

    assert (frame["w0_start"] == "08:00").all()
    assert (frame["w1_start"] == "18:00").all()


def test_header_frame_is_recommended_true_exactly_once_per_run():
    predictor = _make_bare_predictor(max_windows=4, n_windows=2)
    ladder = _synthetic_ladder()
    chosen = ladder[1]

    frame = predictor._build_header_frame(ladder, chosen, _RUN_TS, _WEATHER_TS)

    assert frame["is_recommended"].sum() == 1
    recommended_row = frame[frame["is_recommended"]]
    assert recommended_row["w0_min"].iloc[0] == 30.0
    assert recommended_row["w1_min"].iloc[0] == 0.0


def test_header_frame_weather_creation_is_constant_across_rows():
    """weather_creation carries the weather forecast issue time this run used --
    distinct from the run timestamp (the index)."""
    predictor = _make_bare_predictor(max_windows=4, n_windows=2)
    ladder = _synthetic_ladder()
    weather_creation = pd.Timestamp("2026-07-02 23:00", tz="UTC")

    frame = predictor._build_header_frame(
        ladder, ladder[0], pd.Timestamp("2026-07-03 01:00", tz="Europe/Berlin"), weather_creation
    )

    assert (frame["weather_creation"] == weather_creation).all()


def test_header_frame_forecast_id_is_deterministic_ladder_position():
    """forecast_id is the candidate's position in the ladder -- stable across
    repeated calls with the SAME ladder (as it must be for the header/detail
    pair to agree)."""
    predictor = _make_bare_predictor(max_windows=4, n_windows=2)
    ladder = _synthetic_ladder()

    run_2 = pd.Timestamp("2026-07-04 01:00", tz="Europe/Berlin")
    weather_2 = pd.Timestamp("2026-07-04 00:00", tz="UTC")

    frame_1 = predictor._build_header_frame(ladder, ladder[0], _RUN_TS, _WEATHER_TS)
    frame_2 = predictor._build_header_frame(ladder, ladder[2], run_2, weather_2)

    ids_by_candidate_1 = dict(zip(frame_1["w0_min"], frame_1["forecast_id"]))
    ids_by_candidate_2 = dict(zip(frame_2["w0_min"], frame_2["forecast_id"]))
    assert ids_by_candidate_1 == ids_by_candidate_2
    assert sorted(frame_1["forecast_id"]) == [0, 1, 2]


def test_header_frame_empty_ladder_returns_empty_frame_with_columns():
    predictor = _make_bare_predictor(max_windows=4, n_windows=2)

    frame = predictor._build_header_frame(
        [], (), pd.Timestamp("2026-07-03 01:00", tz="Europe/Berlin"), pd.Timestamp("2026-07-03 00:00", tz="UTC")
    )

    assert frame.empty
    assert "w0_min" in frame.columns


# --- _build_detail_frame (pure, per-probe LONG shape) ------------------------


def _synthetic_ladder_trajectories():
    """3 candidates x 3 timestamps x 2 probes."""
    timestamps = [pd.Timestamp("2026-07-03 08:00", tz="Europe/Berlin") + pd.Timedelta(hours=h) for h in range(3)]
    ladder = _synthetic_ladder()
    trajectories = {
        ladder[0]: (timestamps, {"root_20": [0.9, 0.85, 0.8], "root_40": [0.9, 0.9, 0.89]}),
        ladder[1]: (timestamps, {"root_20": [0.9, 0.92, 0.91], "root_40": [0.9, 0.9, 0.9]}),
        ladder[2]: (timestamps, {"root_20": [0.9, 0.95, 0.95], "root_40": [0.9, 0.91, 0.91]}),
    }
    return ladder, trajectories, timestamps


def test_detail_frame_has_expected_columns():
    """Each probe gets its OWN timestamp_creation/forecast_id twins (per-probe
    PK partners) -- a single shared pair could not carry N probes' different
    soil_id surrogate attributes."""
    predictor = _make_bare_predictor(max_windows=4, n_windows=2)
    ladder, trajectories, _ = _synthetic_ladder_trajectories()

    frame = predictor._build_detail_frame(ladder, trajectories, pd.Timestamp("2026-07-03 01:00", tz="Europe/Berlin"))

    expected_columns = {
        "traj_root_20",
        "traj_root_20_timestamp_creation",
        "traj_root_20_forecast_id",
        "traj_root_40",
        "traj_root_40_timestamp_creation",
        "traj_root_40_forecast_id",
    }
    assert set(frame.columns) == expected_columns


def test_detail_frame_row_count_is_candidates_times_timestamps_times_probes():
    """The frame is LONG -- len(ladder) * len(timestamps) * len(probes) rows,
    one per candidate x timestamp x probe -- not wide with one row per
    candidate+timestamp and probes packed as columns."""
    predictor = _make_bare_predictor(max_windows=4, n_windows=2)
    ladder, trajectories, timestamps = _synthetic_ladder_trajectories()

    frame = predictor._build_detail_frame(ladder, trajectories, pd.Timestamp("2026-07-03 01:00", tz="Europe/Berlin"))

    assert len(frame) == len(ladder) * len(timestamps) * 2


def test_detail_frame_is_long_one_probes_full_triplet_populated_per_row():
    """Each row populates exactly ONE probe's full column TRIPLET (tension +
    its own timestamp_creation + its own forecast_id); the other probe's
    triplet is NaN on that row. This is what lets the direct-write path's
    per-attribute-set grouping (table.py) split rows back out per probe by its
    own surrogate soil_id/field_id -- and why `dropna(how="all")` on a probe's
    OWN group never drops a row that belongs to it (all three of its columns
    are populated together, never partially)."""
    predictor = _make_bare_predictor(max_windows=4, n_windows=2)
    ladder, trajectories, _ = _synthetic_ladder_trajectories()

    frame = predictor._build_detail_frame(ladder, trajectories, pd.Timestamp("2026-07-03 01:00", tz="Europe/Berlin"))

    root_20_cols = ["traj_root_20", "traj_root_20_timestamp_creation", "traj_root_20_forecast_id"]
    root_40_cols = ["traj_root_40", "traj_root_40_timestamp_creation", "traj_root_40_forecast_id"]

    root_20_present = frame[root_20_cols].notna().all(axis="columns")
    root_20_absent = frame[root_20_cols].isna().all(axis="columns")
    root_40_present = frame[root_40_cols].notna().all(axis="columns")
    root_40_absent = frame[root_40_cols].isna().all(axis="columns")

    # Every row is EITHER fully root_20 OR fully root_40 -- never mixed, never partial.
    assert (root_20_present | root_20_absent).all()
    assert (root_40_present | root_40_absent).all()
    assert not (root_20_present & root_40_present).any()
    assert not (root_20_absent & root_40_absent).any()


def test_detail_frame_all_candidates_present():
    predictor = _make_bare_predictor(max_windows=4, n_windows=2)
    ladder, trajectories, _ = _synthetic_ladder_trajectories()

    frame = predictor._build_detail_frame(ladder, trajectories, pd.Timestamp("2026-07-03 01:00", tz="Europe/Berlin"))

    for forecast_id_col in ("traj_root_20_forecast_id", "traj_root_40_forecast_id"):
        present = frame[frame[forecast_id_col].notna()]
        assert set(present[forecast_id_col]) == {0, 1, 2}
        for forecast_id in (0, 1, 2):
            assert (present[forecast_id_col] == forecast_id).sum() == 3  # 3 timestamps


def test_header_and_detail_frames_agree_on_forecast_id_per_candidate():
    """Header and detail rows built for one run must label the SAME candidate
    with the SAME forecast_id -- the join key readers use to pull the
    recommended candidate's trajectory out of ``agri_soil_forecast``. Both
    frames derive it from one ``_forecast_ids(ladder)`` enumeration; this pins
    that they stay in step."""
    predictor = _make_bare_predictor(max_windows=4, n_windows=2)
    ladder, trajectories, _ = _synthetic_ladder_trajectories()

    header = predictor._build_header_frame(ladder, ladder[1], _RUN_TS, _WEATHER_TS)
    detail = predictor._build_detail_frame(ladder, trajectories, _RUN_TS)

    for position, candidate in enumerate(ladder):
        header_row = header[header["forecast_id"] == position]
        assert len(header_row) == 1
        assert header_row["w0_min"].iloc[0] == candidate[0].total_seconds() / 60.0

        detail_rows = detail[detail["traj_root_20_forecast_id"] == position].sort_index()
        _, probe_values = trajectories[candidate]
        assert list(detail_rows["traj_root_20"]) == probe_values["root_20"]


def test_detail_frame_timestamp_creation_is_run_time_not_weather_issue_time():
    """Every probe's timestamp_creation twin must be the RUN time (not the
    weather issue time), so two runs sharing one weather issue do not
    collide."""
    predictor = _make_bare_predictor(max_windows=4, n_windows=2)
    ladder, trajectories, _ = _synthetic_ladder_trajectories()
    run_ts = pd.Timestamp("2026-07-03 01:00", tz="Europe/Berlin")

    frame = predictor._build_detail_frame(ladder, trajectories, run_ts)

    for creation_col in ("traj_root_20_timestamp_creation", "traj_root_40_timestamp_creation"):
        present = frame[creation_col].dropna()
        assert (present == run_ts).all()


def test_detail_frame_two_runs_same_weather_issue_get_distinct_run_timestamps():
    """The no-upsert-overwrite invariant at the frame level: two runs that
    shared the SAME weather issue time must still produce DISTINCT
    timestamp_creation values on every probe's twin (distinct PK), so neither
    run's rows collide with the other's at the connector."""
    predictor = _make_bare_predictor(max_windows=4, n_windows=2)
    ladder, trajectories, _ = _synthetic_ladder_trajectories()
    run_1 = pd.Timestamp("2026-07-03 01:00", tz="Europe/Berlin")
    run_2 = pd.Timestamp("2026-07-04 01:00", tz="Europe/Berlin")

    frame_1 = predictor._build_detail_frame(ladder, trajectories, run_1)
    frame_2 = predictor._build_detail_frame(ladder, trajectories, run_2)

    for creation_col in ("traj_root_20_timestamp_creation", "traj_root_40_timestamp_creation"):
        assert (frame_1[creation_col].dropna() == run_1).all()
        assert (frame_2[creation_col].dropna() == run_2).all()
    assert run_1 != run_2


def test_detail_frame_carries_per_probe_values_as_is():
    """The frame stores each candidate's per-probe trajectory values as-is -- a
    pure pass-through. In production these are already signed matric potential
    (negative hPa) from the retention model's psi_from_se; the frame does not
    re-convert."""
    predictor = _make_bare_predictor(max_windows=4, n_windows=2)
    ladder, trajectories, _ = _synthetic_ladder_trajectories()

    frame = predictor._build_detail_frame(ladder, trajectories, pd.Timestamp("2026-07-03 01:00", tz="Europe/Berlin"))

    zero_candidate_id = 0  # ladder[0] == (_td(0), _td(0))
    root_20_rows = frame[(frame["traj_root_20_forecast_id"] == zero_candidate_id) & frame["traj_root_20"].notna()]
    np.testing.assert_allclose(sorted(root_20_rows["traj_root_20"]), sorted([0.9, 0.85, 0.8]))


def test_detail_frame_empty_ladder_trajectories_returns_empty_frame_with_columns():
    predictor = _make_bare_predictor(max_windows=4, n_windows=2)

    frame = predictor._build_detail_frame([], {}, pd.Timestamp("2026-07-03 01:00", tz="Europe/Berlin"))

    assert frame.empty
    assert "traj_root_20" in frame.columns


# --- No-double-write: the auto flush must never emit header/detail rows -----


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
        self._channels = {key: _FakeSetChannel(f"test_soil_predictor_trajectory.{key}") for key in keys}

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


def _header_keys(predictor) -> list[str]:
    return [
        predictor._HEADER_FORECAST_ID_KEY,
        *predictor._header_window_min_keys,
        *predictor._header_window_start_keys,
        predictor._HEADER_IS_RECOMMENDED_KEY,
        predictor._HEADER_TOTAL_MIN_KEY,
        predictor._HEADER_WEATHER_CREATION_KEY,
    ]


def _detail_keys(predictor) -> list[str]:
    return [
        *predictor._traj_channel_keys.values(),
        *predictor._detail_creation_keys.values(),
        *predictor._detail_forecast_id_keys.values(),
    ]


def test_header_and_detail_writes_never_call_set(monkeypatch):
    """The invariant the PRD calls out: a stray `.set()` on a header/detail
    channel must never happen -- these are schema-declaring, direct-write-only
    channels. Exercises the REAL write path (_write_header_table /
    _write_detail_table against a real built frame + a recording fake
    connector), not just an untouched initial state, so a future `.set()` call
    added inside either write method would fail this test."""
    predictor = _make_bare_predictor(max_windows=4, n_windows=2)
    predictor._logger_id = "db"
    all_keys = _header_keys(predictor) + _detail_keys(predictor)
    fake_data = _FakeDataAccess(all_keys)
    _patch_data(monkeypatch, fake_data)

    written = []

    class _Connector:
        def write(self, frame):
            written.append(frame)

    class _Connectors:
        db = _Connector()

    _patch_connectors(monkeypatch, _Connectors())

    ladder = _synthetic_ladder()
    run_ts = pd.Timestamp("2026-07-03 01:00", tz="Europe/Berlin")
    header_frame = predictor._build_header_frame(ladder, ladder[0], run_ts, pd.Timestamp("2026-07-03 00:00", tz="UTC"))
    predictor._write_header_table(header_frame)

    ladder2, trajectories, _ = _synthetic_ladder_trajectories()
    detail_frame = predictor._build_detail_frame(ladder2, trajectories, run_ts)
    predictor._write_detail_table(detail_frame)

    assert len(written) == 2, "both writes must have reached the connector"
    for key in all_keys:
        assert fake_data[key].calls == [], f"'{key}' must never be .set()"
        assert pd.isna(fake_data[key].timestamp), f"'{key}' timestamp must stay NaT"


# --- _write_header_table / _write_detail_table: degrade-on-missing-connector -


def test_write_header_table_skips_when_logger_not_configured():
    predictor = _make_bare_predictor(max_windows=4, n_windows=2)
    predictor._logger_id = None

    frame = pd.DataFrame({"forecast_id": [0]}, index=pd.DatetimeIndex([pd.Timestamp("2026-07-03", tz="UTC")]))
    # Must not raise, and must not attempt any connector resolution.
    predictor._write_header_table(frame)


def test_write_detail_table_skips_when_connector_missing(monkeypatch, caplog):
    predictor = _make_bare_predictor(max_windows=4, n_windows=2)
    predictor._logger_id = "db"

    class _FakeConnectors:
        def __getitem__(self, item):
            raise KeyError(item)

    # getattr(..., "db") -> AttributeError (no such attribute); __getitem__ -> KeyError.
    _patch_connectors(monkeypatch, _FakeConnectors())

    frame = pd.DataFrame({"traj_root_20": [0.9]}, index=pd.DatetimeIndex([pd.Timestamp("2026-07-03", tz="UTC")]))
    with caplog.at_level("WARNING"):
        predictor._write_detail_table(frame)

    assert any("not found" in message for message in caplog.messages)


def test_write_header_table_empty_frame_is_a_noop(monkeypatch):
    """An empty header frame must be a no-op BEFORE any connector resolution is
    attempted -- deliberately no connectors patch here, so this would error
    (not silently pass) if the empty-frame guard were removed."""
    predictor = _make_bare_predictor(max_windows=4, n_windows=2)
    predictor._logger_id = "db"

    predictor._write_header_table(pd.DataFrame())


def test_write_detail_table_uses_channel_resolved_connector(monkeypatch):
    """A nested predictor references a ROOT-level ``[connectors.<id>]`` connector:
    ``self.connectors``' component-scoped lookup cannot resolve the bare id, but
    the header's forecast_id channel already bound the connector at registration
    (walking the component path) -- ``_logger_connector_from_channel`` anchors
    there for BOTH tables (connector resolution is table-agnostic). This test
    fails if that anchor channel is missing/unresolvable: the box failure mode
    ('logger connector not found; skipping the detail table direct write')."""
    predictor = _make_bare_predictor(max_windows=4, n_windows=2)
    predictor._logger_id = "mariadb"

    written = {}

    class _Connector:
        def write(self, frame):
            written["frame"] = frame

    class _Logger:
        def _get_registrator(self):
            return _Connector()

    class _Channel:
        def __init__(self, channel_id):
            self.id = channel_id
            self.logger = _Logger()

    class _Data:
        def __init__(self, keys):
            self._channels = {key: _Channel(f"test_soil_predictor_trajectory.{key}") for key in keys}

        def __getitem__(self, key):
            return self._channels[key]

    _patch_data(monkeypatch, _Data([predictor._HEADER_FORECAST_ID_KEY, *_detail_keys(predictor)]))

    class _NoConnectors:
        """Component-scoped registry that cannot resolve the root connector."""

        def __getitem__(self, item):
            raise KeyError(item)

    _patch_connectors(monkeypatch, _NoConnectors())

    frame = pd.DataFrame(
        {"traj_root_20": [0.9]},
        index=pd.DatetimeIndex([pd.Timestamp("2026-07-03", tz="UTC")], name="timestamp"),
    )
    predictor._write_detail_table(frame)

    assert "frame" in written, "must write via the connector the detail channel resolved"
