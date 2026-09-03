# -*- coding: utf-8 -*-
"""sparcs.tests.test_soil_predictor_write_failures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Issue 24 (W2.6): a failed direct table write used to be log-and-return with no
counter and no row count -- frames rebuild only at the next boundary, so one
failed write silently loses that table's rows for the whole run.
``_write_direct_frame`` now bumps a monotonic per-table failure counter ONLY in
the ``connector.write`` except (never on the skip paths, which stay pinned by
the trajectory/irrigation/image table tests), adds the row count to the ERROR,
and surfaces the count via in-memory channels registered next to the
diagnostics ones (logger disabled -- Dash Data accordion, per Q6).
"""

from types import SimpleNamespace

import pytest

import pandas as pd

soil_predictor = pytest.importorskip("sparcs.components.agriculture.simulation.soil_predictor")
SoilPredictor = soil_predictor.SoilPredictor


class _RecordingChannel:
    def __init__(self, channel_id: str):
        self.id = channel_id
        self.calls: list[tuple] = []

    def set(self, timestamp, value) -> None:
        self.calls.append((timestamp, value))


class _RecordingData:
    def __init__(self):
        self.channels: dict = {}
        self.added: list[tuple] = []

    def add(self, key, **kwargs):
        self.added.append((key, kwargs))

    def __getitem__(self, key: str) -> _RecordingChannel:
        return self.channels.setdefault(key, _RecordingChannel(key))


class _RaisingData:
    """Every channel access raises -- the counter bump must swallow this."""

    def add(self, key, **kwargs):
        raise KeyError(key)

    def __getitem__(self, key: str):
        raise KeyError(key)


class _RaisingConnector:
    def write(self, frame):
        raise RuntimeError("db down")


class _OkConnector:
    def __init__(self):
        self.written: list = []

    def write(self, frame):
        self.written.append(frame)


def _bare_predictor(monkeypatch, data, connector) -> SoilPredictor:
    p = object.__new__(SoilPredictor)
    p._name = "test_predictor"
    p._logger_id = "db"
    p._logger_connector_from_channel = lambda: None  # force the id-based fallback
    monkeypatch.setattr(SoilPredictor, "data", property(lambda self: data))
    monkeypatch.setattr(SoilPredictor, "connectors", property(lambda self: SimpleNamespace(db=connector)))
    return p


def _frame(rows: int = 2) -> pd.DataFrame:
    index = pd.date_range("2026-07-03 01:00", periods=rows, freq="15min", tz="UTC")
    return pd.DataFrame({"x": [0.0] * rows}, index=index)


def test_write_failure_bumps_only_that_tables_counter(monkeypatch, caplog):
    data = _RecordingData()
    p = _bare_predictor(monkeypatch, data, _RaisingConnector())

    with caplog.at_level("ERROR"):
        p._write_direct_frame(_frame(), lambda: {}, "header table")
        p._write_direct_frame(_frame(), lambda: {}, "header table")

    assert [v for _, v in data.channels["header_write_failures"].calls] == [1.0, 2.0]
    for other in ("detail_write_failures", "irrigation_write_failures", "image_write_failures"):
        assert other not in data.channels


def test_write_failure_error_names_table_and_row_count(monkeypatch, caplog):
    p = _bare_predictor(monkeypatch, _RecordingData(), _RaisingConnector())

    with caplog.at_level("ERROR"):
        p._write_direct_frame(_frame(rows=3), lambda: {}, "irrigation table")

    errors = [r for r in caplog.records if "direct write" in r.getMessage()]
    assert len(errors) == 1
    message = errors[0].getMessage()
    assert "irrigation table" in message
    assert "3 rows" in message


def test_successful_write_and_skip_paths_never_bump(monkeypatch):
    data = _RecordingData()
    connector = _OkConnector()
    p = _bare_predictor(monkeypatch, data, connector)

    p._write_direct_frame(_frame(), lambda: {}, "header table")  # success
    p._write_direct_frame(pd.DataFrame(), lambda: {}, "header table")  # empty: skip
    p._logger_id = None
    p._write_direct_frame(_frame(), lambda: {}, "header table")  # unconfigured: skip

    assert len(connector.written) == 1
    assert "header_write_failures" not in data.channels
    assert p._write_failures in (None, {})


def test_counter_bump_survives_a_raising_channel_access(monkeypatch, caplog):
    """Bare/fake data whose channel access raises (the trajectory-table test
    family's KeyError fixture): the failure handler's job is to log, not crash."""
    p = _bare_predictor(monkeypatch, _RaisingData(), _RaisingConnector())

    with caplog.at_level("ERROR"):
        p._write_direct_frame(_frame(), lambda: {}, "detail table")  # must not raise

    assert p._write_failures == {"detail table": 1}


def test_register_write_failure_channels_registers_four_in_memory(monkeypatch):
    data = _RecordingData()
    p = object.__new__(SoilPredictor)
    p._name = "test_predictor"
    monkeypatch.setattr(SoilPredictor, "data", property(lambda self: data))

    p._register_write_failure_channels()

    keys = [key for key, _ in data.added]
    assert sorted(keys) == [
        "detail_write_failures",
        "header_write_failures",
        "image_write_failures",
        "irrigation_write_failures",
    ]
    for _, kwargs in data.added:
        assert kwargs["logger"] == {"enabled": False}
        assert kwargs["aggregate"] == "last"
