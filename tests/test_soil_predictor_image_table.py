# -*- coding: utf-8 -*-
"""sparcs.tests.test_soil_predictor_image_table
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Unit tests for the ``agri_field_forecast_image`` table: the RECOMMENDED
candidate's soil-saturation field snapshots persisted as PNG bytes
(``SoilPredictor._build_image_frame``), channel-registered
(``_register_image_channels``) and direct-written (``_write_image_table``) the
same never-``.set()`` / logger-gated way ``agri_field_forecast_irrigation`` is, plus the
``_publish_results`` -> reuse-the-rendered-bytes handshake that feeds it.

Importing ``soil_predictor`` pulls the full lories + soil (FiPy/Gmsh) stack via
``soil.py``; ``importorskip`` keeps this out of environments that lack it.
"""

import pytest

import numpy as np
import pandas as pd

soil_predictor = pytest.importorskip("sparcs.components.agriculture.simulation.soil_predictor")
SoilPredictor = soil_predictor.SoilPredictor

_TZ = "Europe/Berlin"
_PNG_A = b"\x89PNG\r\n\x1a\nA"
_PNG_B = b"\x89PNG\r\n\x1a\nB"


def _save_index(periods: int = 2) -> pd.DatetimeIndex:
    return pd.date_range("2026-07-03 02:00", periods=periods, freq="6h", tz=_TZ, name="timestamp")


# --- _build_image_frame (pure) -----------------------------------------------


def test_image_frame_one_row_per_snapshot_stamped_with_run_time():
    predictor = object.__new__(SoilPredictor)
    predictor._name = "test_predictor"
    save_index = _save_index(2)
    run_ts = pd.Timestamp("2026-07-03 01:00", tz=_TZ)

    frame = predictor._build_image_frame(save_index, [_PNG_A, _PNG_B], run_ts)

    assert list(frame.index) == list(save_index)
    assert list(frame[SoilPredictor._IMAGE_KEY]) == [_PNG_A, _PNG_B]
    assert (frame[SoilPredictor._IMAGE_TIMESTAMP_CREATION_KEY] == run_ts).all()
    assert list(frame.columns) == [SoilPredictor._IMAGE_KEY, SoilPredictor._IMAGE_TIMESTAMP_CREATION_KEY]


def test_image_frame_empty_returns_empty_frame_with_columns():
    predictor = object.__new__(SoilPredictor)
    predictor._name = "test_predictor"
    frame = predictor._build_image_frame(_save_index(0), [], pd.Timestamp("2026-07-03 01:00", tz=_TZ))
    assert frame.empty
    assert SoilPredictor._IMAGE_KEY in frame.columns
    assert SoilPredictor._IMAGE_TIMESTAMP_CREATION_KEY in frame.columns


# --- _register_image_channels: exact data.add() kwargs -----------------------


class _RecordingData:
    def __init__(self):
        self.added: list[tuple] = []

    def add(self, key, **kwargs) -> None:
        self.added.append((key, kwargs))


def test_register_image_channels_binds_image_table(monkeypatch):
    predictor = object.__new__(SoilPredictor)
    predictor._name = "test_predictor"
    predictor._logger_id = "mariadb"
    fake = _RecordingData()
    monkeypatch.setattr(SoilPredictor, "data", property(lambda self: fake))

    predictor._register_image_channels()

    by_id = dict(fake.added)
    assert SoilPredictor._IMAGE_KEY in by_id
    assert SoilPredictor._IMAGE_TIMESTAMP_CREATION_KEY in by_id

    for _key, kwargs in fake.added:
        assert kwargs["logger"]["table"] == SoilPredictor._IMAGE_TABLE_NAME
        assert kwargs["logger"]["connector"] == "mariadb"
        assert kwargs["logger"]["enabled"] is True

    image_kwargs = by_id[SoilPredictor._IMAGE_KEY]
    assert image_kwargs["type"] is bytes
    assert image_kwargs["logger"]["column"] == "image"
    assert image_kwargs["logger"].get("primary") is not True

    twin_kwargs = by_id[SoilPredictor._IMAGE_TIMESTAMP_CREATION_KEY]
    assert twin_kwargs["logger"]["column"] == "timestamp_creation"
    assert twin_kwargs["logger"]["primary"] is True
    assert twin_kwargs["logger"]["nullable"] is False


# --- _write_image_table: degrade-on-missing-connector / empty -----------------


def test_write_image_table_skips_when_logger_not_configured():
    predictor = object.__new__(SoilPredictor)
    predictor._name = "test_predictor"
    predictor._logger_id = None
    frame = pd.DataFrame(
        {SoilPredictor._IMAGE_KEY: [_PNG_A]},
        index=pd.DatetimeIndex([pd.Timestamp("2026-07-03", tz="UTC")]),
    )
    predictor._write_image_table(frame)  # must not raise / resolve any connector


def test_write_image_table_empty_frame_is_a_noop():
    predictor = object.__new__(SoilPredictor)
    predictor._name = "test_predictor"
    predictor._logger_id = "db"
    predictor._write_image_table(pd.DataFrame())  # empty guard fires before connector resolution


# --- real write path: build -> write -> rename -> connector, no stray .set() --


class _FakeSetChannel:
    def __init__(self, channel_id: str):
        self.id = channel_id
        self.calls: list[tuple] = []
        self.timestamp = pd.NaT

    def set(self, timestamp, value) -> None:
        self.calls.append((timestamp, value))
        self.timestamp = timestamp

    @property
    def logger(self):
        class _NullLogger:
            @staticmethod
            def _get_registrator():
                return None

        return _NullLogger()


class _FakeDataAccess:
    def __init__(self, keys):
        self._channels = {key: _FakeSetChannel(f"test_predictor.{key}") for key in keys}

    def __getitem__(self, key):
        return self._channels[key]


def test_write_image_table_renames_to_channel_ids_and_never_calls_set(monkeypatch):
    predictor = object.__new__(SoilPredictor)
    predictor._name = "test_predictor"
    predictor._logger_id = "db"
    keys = [SoilPredictor._IMAGE_KEY, SoilPredictor._IMAGE_TIMESTAMP_CREATION_KEY]
    fake_data = _FakeDataAccess(keys)
    monkeypatch.setattr(SoilPredictor, "data", property(lambda self: fake_data))

    written = []

    class _Connector:
        def write(self, frame):
            written.append(frame)

    class _Connectors:
        db = _Connector()

    monkeypatch.setattr(SoilPredictor, "connectors", property(lambda self: _Connectors()))

    save_index = _save_index(2)
    frame = predictor._build_image_frame(save_index, [_PNG_A, _PNG_B], pd.Timestamp("2026-07-03 01:00", tz=_TZ))
    predictor._write_image_table(frame)

    assert len(written) == 1
    assert set(written[0].columns) == {fake_data[key].id for key in keys}
    for key in keys:
        assert fake_data[key].calls == [], f"'{key}' must never be .set()"
        assert pd.isna(fake_data[key].timestamp)


# --- _publish_results returns the rendered recommended plot -------------------


def test_publish_results_returns_rendered_plot_for_reuse(monkeypatch):
    """save_plot on -> _publish_results renders once, sets predict_plot, and
    RETURNS (save_index, png bytes) so predict() can persist without re-rendering."""
    predictor = object.__new__(SoilPredictor)
    predictor._name = "test_predictor"
    predictor._channel_keys = {}
    predictor._save_state = False
    predictor._save_plot = True

    fake_data = _FakeDataAccess(
        [SoilPredictor._TIMESTAMP_CREATION_KEY, SoilPredictor._PLOT_CHANNEL_KEY]
        + [c.key for c in soil_predictor._DIAGNOSTIC_CONSTANTS]
    )
    monkeypatch.setattr(SoilPredictor, "data", property(lambda self: fake_data))

    rendered = []
    monkeypatch.setattr(
        SoilPredictor,
        "_render_snapshot_png",
        lambda self, arr, t, **_k: rendered.append(t) or (_PNG_A if len(rendered) == 1 else _PNG_B),
    )

    index = pd.date_range("2026-07-03 02:00", periods=2, freq="6h", tz=_TZ)
    snapshots = {index[0]: np.zeros(3), index[1]: np.zeros(3)}
    diagnostics = {c.key: [float("nan"), float("nan")] for c in soil_predictor._DIAGNOSTIC_CONSTANTS}

    result = predictor._publish_results([], [], list(index), snapshots, diagnostics, index[0])

    assert result is not None
    save_index, plot_values = result
    assert list(save_index) == list(index)
    assert plot_values == [_PNG_A, _PNG_B]
    # And the in-memory predict_plot channel still got the same series (reuse, not bypass).
    assert fake_data[SoilPredictor._PLOT_CHANNEL_KEY].calls


def test_publish_results_returns_none_when_save_plot_off(monkeypatch):
    predictor = object.__new__(SoilPredictor)
    predictor._name = "test_predictor"
    predictor._channel_keys = {}
    predictor._save_state = False
    predictor._save_plot = False

    fake_data = _FakeDataAccess(
        [SoilPredictor._TIMESTAMP_CREATION_KEY] + [c.key for c in soil_predictor._DIAGNOSTIC_CONSTANTS]
    )
    monkeypatch.setattr(SoilPredictor, "data", property(lambda self: fake_data))

    index = pd.date_range("2026-07-03 02:00", periods=2, freq="6h", tz=_TZ)
    snapshots = {index[0]: np.zeros(3)}
    diagnostics = {c.key: [float("nan"), float("nan")] for c in soil_predictor._DIAGNOSTIC_CONSTANTS}

    assert predictor._publish_results([], [], list(index), snapshots, diagnostics, index[0]) is None


def test_publish_results_returns_none_when_render_fails(monkeypatch):
    """A mid-render failure resets plot_values and breaks -> _publish_results
    returns None (never a partial/mismatched tuple) and predict_plot is not set."""
    predictor = object.__new__(SoilPredictor)
    predictor._name = "test_predictor"
    predictor._channel_keys = {}
    predictor._save_state = False
    predictor._save_plot = True

    fake_data = _FakeDataAccess(
        [SoilPredictor._TIMESTAMP_CREATION_KEY, SoilPredictor._PLOT_CHANNEL_KEY]
        + [c.key for c in soil_predictor._DIAGNOSTIC_CONSTANTS]
    )
    monkeypatch.setattr(SoilPredictor, "data", property(lambda self: fake_data))

    def _boom(self, arr, t, **_k):
        raise RuntimeError("render failed")

    monkeypatch.setattr(SoilPredictor, "_render_snapshot_png", _boom)

    index = pd.date_range("2026-07-03 02:00", periods=2, freq="6h", tz=_TZ)
    snapshots = {index[0]: np.zeros(3), index[1]: np.zeros(3)}
    diagnostics = {c.key: [float("nan"), float("nan")] for c in soil_predictor._DIAGNOSTIC_CONSTANTS}

    assert predictor._publish_results([], [], list(index), snapshots, diagnostics, index[0]) is None
    assert fake_data[SoilPredictor._PLOT_CHANNEL_KEY].calls == []
