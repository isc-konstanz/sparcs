# -*- coding: utf-8 -*-
"""sparcs.tests.test_soil_predictor_trajectory_table
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Unit tests for issue 06's recommendation channels + all-candidate trajectory
table + direct connector write: ``SoilPredictor._build_trajectory_frame``
(pure), ``_publish_recommendation`` and the no-double-write invariant (both
exercised against a bare instance with a fake ``self.data``/``self.connectors``
so no ``Component``/PDE bootstrap is needed), and ``_write_trajectory_table``'s
connector-resolution/degrade-on-missing-connector behavior.

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

import pytest

import numpy as np
import pandas as pd

soil_predictor = pytest.importorskip("sparcs.components.agriculture.simulation.soil_predictor")
SoilPredictor = soil_predictor.SoilPredictor
_UNUSED_WINDOW_SENTINEL = soil_predictor._UNUSED_WINDOW_SENTINEL


def _td(minutes: int) -> pd.Timedelta:
    return pd.Timedelta(minutes=minutes)


def _make_bare_predictor(max_windows: int = 4, n_windows: int = 2) -> SoilPredictor:
    """A bare ``SoilPredictor`` instance carrying only the attributes
    ``_build_trajectory_frame``/``_publish_recommendation`` touch -- the same
    ``object.__new__`` pattern ``test_soil_predictor_scheduling_gate.py`` and
    ``test_soil_predictor_ladder_rollout.py`` use to avoid a full Component/PDE
    bootstrap.
    """
    predictor = object.__new__(SoilPredictor)
    predictor._name = "test_soil_predictor_trajectory"
    predictor._max_windows = max_windows
    predictor._traj_window_keys = [f"w{i}_min" for i in range(max_windows)]
    predictor._traj_channel_keys = {"root_20": "traj_root_20", "root_40": "traj_root_40"}
    predictor._recommend_window_keys = [f"recommend_w{i}_min" for i in range(max_windows)]
    predictor._logger_id = "db"
    return predictor


# --- _build_trajectory_frame (pure) ------------------------------------------


def _synthetic_ladder_trajectories():
    """3 candidates x 3 timestamps x 2 probes; a 2-window ladder."""
    timestamps = [pd.Timestamp("2026-07-03 08:00", tz="Europe/Berlin") + pd.Timedelta(hours=h) for h in range(3)]
    ladder = {
        (_td(0), _td(0)): (timestamps, {"root_20": [0.9, 0.85, 0.8], "root_40": [0.9, 0.9, 0.89]}),
        (_td(30), _td(0)): (timestamps, {"root_20": [0.9, 0.92, 0.91], "root_40": [0.9, 0.9, 0.9]}),
        (_td(60), _td(0)): (timestamps, {"root_20": [0.9, 0.95, 0.95], "root_40": [0.9, 0.91, 0.91]}),
    }
    return ladder, timestamps


def test_trajectory_frame_has_expected_columns():
    predictor = _make_bare_predictor(max_windows=4, n_windows=2)
    ladder, _ = _synthetic_ladder_trajectories()
    chosen = (_td(30), _td(0))

    frame = predictor._build_trajectory_frame(ladder, chosen, pd.Timestamp("2026-07-03 01:00", tz="Europe/Berlin"))

    expected_columns = {
        predictor._TRAJ_TIMESTAMP_CREATION_KEY,
        "w0_min",
        "w1_min",
        "w2_min",
        "w3_min",
        predictor._TRAJ_IS_RECOMMENDED_KEY,
        "traj_root_20",
        "traj_root_40",
    }
    assert set(frame.columns) == expected_columns


def test_trajectory_frame_row_count_is_candidates_times_timestamps():
    predictor = _make_bare_predictor(max_windows=4, n_windows=2)
    ladder, timestamps = _synthetic_ladder_trajectories()
    chosen = (_td(30), _td(0))

    frame = predictor._build_trajectory_frame(ladder, chosen, pd.Timestamp("2026-07-03 01:00", tz="Europe/Berlin"))

    assert len(frame) == len(ladder) * len(timestamps)


def test_trajectory_frame_duplicates_timestamps_across_candidates():
    predictor = _make_bare_predictor(max_windows=4, n_windows=2)
    ladder, timestamps = _synthetic_ladder_trajectories()
    chosen = (_td(30), _td(0))

    frame = predictor._build_trajectory_frame(ladder, chosen, pd.Timestamp("2026-07-03 01:00", tz="Europe/Berlin"))

    # Every forecast timestamp appears once per candidate -- duplicated across
    # the ladder, not collapsed.
    for ts in timestamps:
        assert (frame.index == ts).sum() == len(ladder)


def test_trajectory_frame_unused_windows_get_sentinel_fill():
    """Ladder has 2 active windows but max_windows=4 -- w2_min/w3_min must be
    the -1 sentinel for every row, keeping the PK column set fixed-arity."""
    predictor = _make_bare_predictor(max_windows=4, n_windows=2)
    ladder, _ = _synthetic_ladder_trajectories()
    chosen = (_td(30), _td(0))

    frame = predictor._build_trajectory_frame(ladder, chosen, pd.Timestamp("2026-07-03 01:00", tz="Europe/Berlin"))

    assert (frame["w2_min"] == _UNUSED_WINDOW_SENTINEL).all()
    assert (frame["w3_min"] == _UNUSED_WINDOW_SENTINEL).all()
    # The two configured windows must NOT carry the sentinel.
    assert not (frame["w0_min"] == _UNUSED_WINDOW_SENTINEL).any()
    assert not (frame["w1_min"] == _UNUSED_WINDOW_SENTINEL).any()


def test_trajectory_frame_is_recommended_true_only_for_chosen():
    predictor = _make_bare_predictor(max_windows=4, n_windows=2)
    ladder, _ = _synthetic_ladder_trajectories()
    chosen = (_td(30), _td(0))

    frame = predictor._build_trajectory_frame(ladder, chosen, pd.Timestamp("2026-07-03 01:00", tz="Europe/Berlin"))

    chosen_minutes = [30.0, 0.0, _UNUSED_WINDOW_SENTINEL, _UNUSED_WINDOW_SENTINEL]
    chosen_rows = (
        (frame["w0_min"] == chosen_minutes[0])
        & (frame["w1_min"] == chosen_minutes[1])
        & (frame["w2_min"] == chosen_minutes[2])
        & (frame["w3_min"] == chosen_minutes[3])
    )
    assert frame.loc[chosen_rows, "is_recommended"].all()
    assert not frame.loc[~chosen_rows, "is_recommended"].any()
    # Exactly one candidate's rows must be marked recommended.
    assert frame.loc[chosen_rows, "is_recommended"].sum() == 3  # 3 timestamps


def test_trajectory_frame_carries_per_probe_values():
    """The frame stores each candidate's per-probe trajectory values as-is. In
    production these are already water tension (hPa), converted upstream at the
    roll->publish boundary; the frame itself is a pure pass-through."""
    predictor = _make_bare_predictor(max_windows=4, n_windows=2)
    ladder, timestamps = _synthetic_ladder_trajectories()
    chosen = (_td(0), _td(0))

    frame = predictor._build_trajectory_frame(ladder, chosen, pd.Timestamp("2026-07-03 01:00", tz="Europe/Berlin"))

    zero_rows = frame[(frame["w0_min"] == 0.0) & (frame["w1_min"] == 0.0)]
    np.testing.assert_allclose(sorted(zero_rows["traj_root_20"]), sorted([0.9, 0.85, 0.8]))
    np.testing.assert_allclose(sorted(zero_rows["traj_root_40"]), sorted([0.9, 0.9, 0.89]))


def test_trajectory_frame_empty_ladder_returns_empty_frame_with_columns():
    predictor = _make_bare_predictor(max_windows=4, n_windows=2)

    frame = predictor._build_trajectory_frame({}, (), pd.Timestamp("2026-07-03 01:00", tz="Europe/Berlin"))

    assert frame.empty
    assert "traj_root_20" in frame.columns


# --- No-double-write: the auto flush must never emit trajectory rows --------


class _FakeSetChannel:
    """Records every ``.set()`` call; mirrors the ``Channel`` surface the
    predictor touches (``.set(timestamp, value)``, ``.id``)."""

    def __init__(self, channel_id: str):
        self.id = channel_id
        self.calls: list[tuple[pd.Timestamp, object]] = []
        self.timestamp = pd.NaT  # matches a never-.set() lories Channel

    def set(self, timestamp: pd.Timestamp, value) -> None:
        self.calls.append((timestamp, value))
        self.timestamp = timestamp


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


def test_publish_recommendation_never_touches_trajectory_channel_keys(monkeypatch):
    """The invariant the PRD calls out: a stray `.set()` on a trajectory
    channel must never happen. Assert `_publish_recommendation` only calls
    `.set()` on recommendation channel keys, never on any `_traj_*`/`w{i}_min`
    trajectory key."""
    predictor = _make_bare_predictor(max_windows=4, n_windows=2)
    recommend_keys = [
        *predictor._recommend_window_keys,
        predictor._RECOMMEND_TOTAL_KEY,
        predictor._RECOMMEND_STATUS_KEY,
    ]
    trajectory_keys = [
        predictor._TRAJ_TIMESTAMP_CREATION_KEY,
        *predictor._traj_window_keys,
        predictor._TRAJ_IS_RECOMMENDED_KEY,
        *predictor._traj_channel_keys.values(),
    ]
    _patch_data(monkeypatch, _FakeDataAccess(recommend_keys + trajectory_keys))

    run_ts = pd.Timestamp("2026-07-03 01:00", tz="Europe/Berlin")
    predictor._publish_recommendation((_td(30), _td(0)), "ok", run_ts, run_ts)

    for key in trajectory_keys:
        assert predictor.data[key].calls == [], f"trajectory channel '{key}' must never be .set()"
        assert pd.isna(predictor.data[key].timestamp), f"trajectory channel '{key}' timestamp must stay NaT"

    for key in recommend_keys:
        assert len(predictor.data[key].calls) == 1, f"recommendation channel '{key}' must be set exactly once"


def test_publish_recommendation_sets_expected_window_minutes_total_and_status(monkeypatch):
    predictor = _make_bare_predictor(max_windows=4, n_windows=2)
    recommend_keys = [
        *predictor._recommend_window_keys,
        predictor._RECOMMEND_TOTAL_KEY,
        predictor._RECOMMEND_STATUS_KEY,
    ]
    _patch_data(monkeypatch, _FakeDataAccess(recommend_keys))

    run_ts = pd.Timestamp("2026-07-03 01:00", tz="Europe/Berlin")
    chosen = (_td(45), _td(15))
    predictor._publish_recommendation(chosen, "ok", run_ts, run_ts)

    assert predictor.data["recommend_w0_min"].calls == [(run_ts, 45.0)]
    assert predictor.data["recommend_w1_min"].calls == [(run_ts, 15.0)]
    # Unconfigured windows (index >= len(chosen)) get 0.0, NOT the -1 sentinel.
    assert predictor.data["recommend_w2_min"].calls == [(run_ts, 0.0)]
    assert predictor.data["recommend_w3_min"].calls == [(run_ts, 0.0)]
    assert predictor.data[predictor._RECOMMEND_TOTAL_KEY].calls == [(run_ts, 60.0)]
    assert predictor.data[predictor._RECOMMEND_STATUS_KEY].calls == [(run_ts, "ok")]


def test_publish_recommendation_status_values_pass_through_unmodified(monkeypatch):
    predictor = _make_bare_predictor(max_windows=4, n_windows=2)
    recommend_keys = [
        *predictor._recommend_window_keys,
        predictor._RECOMMEND_TOTAL_KEY,
        predictor._RECOMMEND_STATUS_KEY,
    ]

    for status in ("ok", "none_needed", "infeasible"):
        _patch_data(monkeypatch, _FakeDataAccess(recommend_keys))
        run_ts = pd.Timestamp("2026-07-03 01:00", tz="Europe/Berlin")
        predictor._publish_recommendation((_td(0), _td(0)), status, run_ts, run_ts)
        assert predictor.data[predictor._RECOMMEND_STATUS_KEY].calls == [(run_ts, status)]


# --- _write_trajectory_table: degrade-on-missing-connector -------------------


def test_write_trajectory_table_skips_when_logger_not_configured(caplog):
    predictor = _make_bare_predictor(max_windows=4, n_windows=2)
    predictor._logger_id = None

    frame = pd.DataFrame({"traj_root_20": [0.9]}, index=pd.DatetimeIndex([pd.Timestamp("2026-07-03", tz="UTC")]))
    # Must not raise, and must not attempt any connector resolution.
    predictor._write_trajectory_table(frame)


def test_write_trajectory_table_skips_when_connector_missing(monkeypatch, caplog):
    predictor = _make_bare_predictor(max_windows=4, n_windows=2)
    predictor._logger_id = "db"

    class _FakeConnectors:
        def __getitem__(self, item):
            raise KeyError(item)

    # getattr(..., "db") -> AttributeError (no such attribute); __getitem__ -> KeyError.
    _patch_connectors(monkeypatch, _FakeConnectors())

    frame = pd.DataFrame({"traj_root_20": [0.9]}, index=pd.DatetimeIndex([pd.Timestamp("2026-07-03", tz="UTC")]))
    with caplog.at_level("WARNING"):
        predictor._write_trajectory_table(frame)

    assert any("not found" in message for message in caplog.messages)


def test_write_trajectory_table_empty_frame_is_a_noop(monkeypatch):
    """An empty trajectory frame must be a no-op BEFORE any connector
    resolution is attempted -- deliberately no connectors patch here, so this
    would error (not silently pass) if the empty-frame guard were removed."""
    predictor = _make_bare_predictor(max_windows=4, n_windows=2)
    predictor._logger_id = "db"

    predictor._write_trajectory_table(pd.DataFrame())


def test_write_trajectory_table_uses_channel_resolved_connector(monkeypatch):
    """A nested predictor references a ROOT-level ``[connectors.<id>]`` connector:
    ``self.connectors``' component-scoped lookup cannot resolve the bare id, but
    the trajectory channels already bound the connector at registration (walking
    the component path). ``_write_trajectory_table`` must write via the
    channel-resolved connector -- the box failure mode ('logger connector not
    found; skipping the trajectory-table direct write')."""
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

    keys = [
        predictor._TRAJ_TIMESTAMP_CREATION_KEY,
        predictor._TRAJ_IS_RECOMMENDED_KEY,
        *predictor._traj_window_keys,
        *predictor._traj_channel_keys.values(),
    ]
    _patch_data(monkeypatch, _Data(keys))

    class _NoConnectors:
        """Component-scoped registry that cannot resolve the root connector."""

        def __getitem__(self, item):
            raise KeyError(item)

    _patch_connectors(monkeypatch, _NoConnectors())

    frame = pd.DataFrame(
        {"traj_root_20": [0.9]},
        index=pd.DatetimeIndex([pd.Timestamp("2026-07-03", tz="UTC")], name="timestamp"),
    )
    predictor._write_trajectory_table(frame)

    assert "frame" in written, "must write via the connector the trajectory channel resolved"
