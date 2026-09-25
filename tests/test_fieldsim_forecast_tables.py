# -*- coding: utf-8 -*-
"""sparcs.tests.test_fieldsim_forecast_tables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``ForecastTablePublisher``: the four persisted forecast tables' channel
registrations, the probe-identity resolution they carry, the image frame
builder, and the shared direct-write path with its skip/degrade branches and
write-failure counters.

The publisher is a per-call view over a ``SoilPredictor``; the tests drive it
through a bare ``object.__new__`` instance whose ``data``/``connectors`` class
properties are monkeypatched, so no Component bootstrap is needed.
"""

import types

import pytest

import pandas as pd
from lories.core import ConfigurationError
from sparcs.components.agriculture.fieldsim import components
from sparcs.components.agriculture.fieldsim.components import SoilPredictor

_TZ = "Europe/Berlin"
_PNG_A = b"\x89PNG\r\n\x1a\nA"
_PNG_B = b"\x89PNG\r\n\x1a\nB"
_RUN_TS = pd.Timestamp("2026-07-03 01:00", tz=_TZ)


class _RecordingAdds:
    """Stand-in for ``self.data``: captures every ``add(...)`` call's kwargs."""

    def __init__(self):
        self.added: list[tuple] = []

    def add(self, key, **kwargs) -> None:
        self.added.append((key, kwargs))


def _bare(monkeypatch=None, data=None, connectors=None, **extra) -> SoilPredictor:
    predictor = object.__new__(SoilPredictor)
    predictor._name = "test_predictor"
    for key, value in extra.items():
        setattr(predictor, key, value)
    if monkeypatch is not None and data is not None:
        monkeypatch.setattr(SoilPredictor, "data", property(lambda self: data))
    if monkeypatch is not None and connectors is not None:
        monkeypatch.setattr(SoilPredictor, "connectors", property(lambda self: connectors))
    return predictor


def _registering(monkeypatch, **extra) -> tuple:
    data = _RecordingAdds()
    return _bare(monkeypatch, data=data, **extra), data


def _probes(*channel_ids: str) -> list:
    return [types.SimpleNamespace(channel_id=cid, name=f"Probe {cid}") for cid in channel_ids]


# --- Channel registration ----------------------------------------------------


def test_register_header_channels_binds_header_table(monkeypatch):
    """Every header channel routes to the configured logger connector's
    agri_field_forecast table with logger.enabled=True (direct-write path)."""
    predictor, data = _registering(monkeypatch, _logger_id="mariadb", _max_windows=4)

    window_min_keys, window_start_keys = predictor.tables().register_header_channels()

    assert window_min_keys == ["w0_min", "w1_min", "w2_min", "w3_min"]
    assert window_start_keys == ["w0_start", "w1_start", "w2_start", "w3_start"]
    added_ids = [channel_id for channel_id, _ in data.added]
    assert SoilPredictor._HEADER_FORECAST_ID_KEY in added_ids
    assert SoilPredictor._HEADER_IS_RECOMMENDED_KEY in added_ids
    assert SoilPredictor._HEADER_TOTAL_MIN_KEY in added_ids
    assert SoilPredictor._HEADER_WEATHER_CREATION_KEY in added_ids
    for _channel_id, kwargs in data.added:
        assert kwargs["logger"]["table"] == SoilPredictor._HEADER_TABLE_NAME
        assert kwargs["logger"]["connector"] == "mariadb"
        assert kwargs["logger"]["enabled"] is True

    # forecast_id is the header's PK partner; window min/start are plain data
    # columns (NOT primary -- nullable, no -1 sentinel).
    by_id = dict(data.added)
    assert by_id[SoilPredictor._HEADER_FORECAST_ID_KEY]["logger"]["primary"] is True
    assert by_id[SoilPredictor._HEADER_FORECAST_ID_KEY]["logger"]["nullable"] is False
    assert by_id["w0_min"]["logger"].get("primary") is not True


def test_register_detail_channels_binds_detail_table_with_shared_water_tension_column(monkeypatch):
    """The detail table's probe channels share ONE 'water_tension' DB column
    (per-probe distinction via soil_id); each probe's OWN timestamp_creation/
    forecast_id TWINS are its PK partners, because a single shared pair cannot
    carry N different probes' soil_ids at once."""
    predictor, data = _registering(monkeypatch, _logger_id="mariadb")
    identities = {"root_20": {"soil_id": 20, "field_id": 2}, "root_40": {"soil_id": 40, "field_id": 2}}

    tension_keys, creation_keys, forecast_id_keys = predictor.tables().register_detail_channels(
        _probes("root_20", "root_40"), identities
    )

    assert tension_keys == {"root_20": "traj_root_20", "root_40": "traj_root_40"}
    assert creation_keys == {
        "root_20": "traj_root_20_timestamp_creation",
        "root_40": "traj_root_40_timestamp_creation",
    }
    assert forecast_id_keys == {"root_20": "traj_root_20_forecast_id", "root_40": "traj_root_40_forecast_id"}

    by_id = dict(data.added)
    assert by_id["traj_root_20"]["logger"]["column"] == "water_tension"
    assert by_id["traj_root_40"]["logger"]["column"] == "water_tension"
    assert by_id["traj_root_20"]["logger"]["table"] == SoilPredictor._DETAIL_TABLE_NAME
    assert not by_id["traj_root_20"]["logger"].get("primary")

    for probe_key in ("root_20", "root_40"):
        creation_kwargs = by_id[f"traj_{probe_key}_timestamp_creation"]
        assert creation_kwargs["logger"]["column"] == "timestamp_creation"
        assert creation_kwargs["logger"]["primary"] is True
        assert creation_kwargs["logger"]["nullable"] is False

        forecast_id_kwargs = by_id[f"traj_{probe_key}_forecast_id"]
        assert forecast_id_kwargs["logger"]["column"] == "forecast_id"
        assert forecast_id_kwargs["logger"]["primary"] is True
        assert forecast_id_kwargs["logger"]["nullable"] is False


def test_register_detail_channels_every_channel_carries_matching_soil_id_and_field_id(monkeypatch):
    """The SQL connector's per-attribute-set write grouping raises for any
    resource on a keyed table missing a declared surrogate attribute, so every
    one of a probe's THREE channels must carry the IDENTICAL soil_id/field_id
    pair, and different probes must carry DIFFERENT soil_ids (same field_id)."""
    predictor, data = _registering(monkeypatch, _logger_id="mariadb")
    identities = {"root_20": {"soil_id": 20, "field_id": 2}, "root_40": {"soil_id": 40, "field_id": 2}}

    predictor.tables().register_detail_channels(_probes("root_20", "root_40"), identities)

    by_id = dict(data.added)
    for probe_key, expected in identities.items():
        for channel_id in (
            f"traj_{probe_key}",
            f"traj_{probe_key}_timestamp_creation",
            f"traj_{probe_key}_forecast_id",
        ):
            assert by_id[channel_id]["soil_id"] == expected["soil_id"], channel_id
            assert by_id[channel_id]["field_id"] == expected["field_id"], channel_id
    assert by_id["traj_root_20"]["soil_id"] != by_id["traj_root_40"]["soil_id"]
    assert by_id["traj_root_20"]["field_id"] == by_id["traj_root_40"]["field_id"]


def test_register_irrigation_channels_binds_irrigation_table(monkeypatch):
    """Both channels route to the irrigation table with logger.enabled=True;
    irrigation_state is a plain data column (not primary, no column key at all),
    and timestamp_creation is the primary/non-nullable PK partner."""
    predictor, data = _registering(monkeypatch, _logger_id="mariadb")

    predictor.tables().register_irrigation_channels()

    by_id = dict(data.added)
    assert SoilPredictor._IRRIGATION_STATE_KEY in by_id
    assert SoilPredictor._IRRIGATION_TIMESTAMP_CREATION_KEY in by_id
    for _channel_id, kwargs in data.added:
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


def test_register_image_channels_binds_image_table(monkeypatch):
    predictor, data = _registering(monkeypatch, _logger_id="mariadb")

    predictor.tables().register_image_channels()

    by_id = dict(data.added)
    assert SoilPredictor._IMAGE_KEY in by_id
    assert SoilPredictor._IMAGE_TIMESTAMP_CREATION_KEY in by_id
    for _channel_id, kwargs in data.added:
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


# --- resolve_probe_identities ------------------------------------------------


class _FakeLeafConfig:
    def __init__(self, values: dict):
        self._values = values

    def get(self, key, default=None):
        return self._values.get(key, default)


class _FakeChannelsConfig:
    """``[soil_simulation.data.channels]`` stand-in: ``.get("field_id")`` for the
    component-wide default, ``.get_member(<probe_key>)`` for the per-probe
    ``soil_id`` block."""

    def __init__(self, field_id=None, per_probe_soil_ids: dict = None):
        self._field_id = field_id
        self._per_probe = per_probe_soil_ids or {}

    def get(self, key, default=None):
        assert key == "field_id"
        return self._field_id if self._field_id is not None else default

    def get_member(self, key, defaults=None):
        soil_id = self._per_probe.get(key)
        return _FakeLeafConfig({"soil_id": soil_id} if soil_id is not None else {})


class _FakeDataConfig:
    def __init__(self, channels_cfg):
        self._channels_cfg = channels_cfg

    def get_member(self, key, defaults=None):
        assert key == "channels"
        return self._channels_cfg


class _FakeSoilBlock:
    def __init__(self, channels_cfg):
        self._data_cfg = _FakeDataConfig(channels_cfg)

    def get_member(self, key, defaults=None):
        assert key == "data"
        return self._data_cfg


def test_resolve_probe_identities_reads_soil_id_and_field_id():
    predictor = _bare()
    soil_block = _FakeSoilBlock(_FakeChannelsConfig(field_id=2, per_probe_soil_ids={"root_20": 20, "root_40": 40}))

    identities = predictor.tables().resolve_probe_identities(soil_block, _probes("root_20", "root_40"))

    assert identities == {"root_20": {"field_id": 2, "soil_id": 20}, "root_40": {"field_id": 2, "soil_id": 40}}


def test_resolve_probe_identities_missing_soil_id_only_warns(caplog):
    """A probe with no configured soil_id only warns and simply gets no soil_id
    kwarg -- its channels then fail loudly at connector connect time instead."""
    predictor = _bare()
    soil_block = _FakeSoilBlock(_FakeChannelsConfig(field_id=2, per_probe_soil_ids={}))

    with caplog.at_level("WARNING"):
        identities = predictor.tables().resolve_probe_identities(soil_block, _probes("root_20"))

    assert identities == {"root_20": {"field_id": 2}}
    assert any("soil_id" in message for message in caplog.messages)


def test_resolve_probe_identities_duplicate_soil_id_raises():
    """Two probes on one soil_id would upsert onto the same forecast row key
    and clobber each other."""
    predictor = _bare()
    soil_block = _FakeSoilBlock(_FakeChannelsConfig(per_probe_soil_ids={"soil_30cm": 3, "soil_60cm": 3}))

    with pytest.raises(ConfigurationError):
        predictor.tables().resolve_probe_identities(soil_block, _probes("soil_30cm", "soil_60cm"))


def test_resolve_probe_identities_unique_soil_ids_do_not_warn(caplog):
    predictor = _bare()
    soil_block = _FakeSoilBlock(_FakeChannelsConfig(per_probe_soil_ids={"soil_30cm": 3, "soil_60cm": 6}))

    with caplog.at_level("WARNING"):
        identities = predictor.tables().resolve_probe_identities(soil_block, _probes("soil_30cm", "soil_60cm"))

    assert identities == {"soil_30cm": {"soil_id": 3}, "soil_60cm": {"soil_id": 6}}
    assert caplog.messages == []


# --- build_image_frame -------------------------------------------------------


def _save_index(periods: int = 2) -> pd.DatetimeIndex:
    return pd.date_range("2026-07-03 02:00", periods=periods, freq="6h", tz=_TZ, name="timestamp")


def test_image_frame_one_row_per_snapshot_stamped_with_run_time():
    frame = _bare().tables().build_image_frame(_save_index(2), [_PNG_A, _PNG_B], _RUN_TS)

    assert list(frame.index) == list(_save_index(2))
    assert list(frame[SoilPredictor._IMAGE_KEY]) == [_PNG_A, _PNG_B]
    assert (frame[SoilPredictor._IMAGE_TIMESTAMP_CREATION_KEY] == _RUN_TS).all()
    assert list(frame.columns) == [SoilPredictor._IMAGE_KEY, SoilPredictor._IMAGE_TIMESTAMP_CREATION_KEY]


def test_image_frame_empty_returns_empty_frame_with_columns():
    frame = _bare().tables().build_image_frame(_save_index(0), [], _RUN_TS)

    assert frame.empty
    assert SoilPredictor._IMAGE_KEY in frame.columns
    assert SoilPredictor._IMAGE_TIMESTAMP_CREATION_KEY in frame.columns


# --- Direct-write path -------------------------------------------------------


class _FakeSetChannel:
    """Records every ``.set()`` call; mirrors the ``Channel`` surface the
    publisher touches (``.set(timestamp, value)``, ``.id``, ``.logger``)."""

    def __init__(self, channel_id: str):
        self.id = channel_id
        self.calls: list[tuple] = []
        self.timestamp = pd.NaT  # matches a never-.set() lories Channel

    def set(self, timestamp, value) -> None:
        self.calls.append((timestamp, value))
        self.timestamp = timestamp

    @property
    def logger(self):
        # No pre-bound registrator -- forces the id-based fallback ladder.
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


class _RecordingConnector:
    def __init__(self):
        self.written: list = []

    def write(self, frame):
        self.written.append(frame)


def _connectors(connector):
    return types.SimpleNamespace(db=connector)


def _header_keys(predictor) -> list:
    return [
        predictor._HEADER_FORECAST_ID_KEY,
        *predictor._header_window_min_keys,
        *predictor._header_window_start_keys,
        predictor._HEADER_IS_RECOMMENDED_KEY,
        predictor._HEADER_TOTAL_MIN_KEY,
        predictor._HEADER_WEATHER_CREATION_KEY,
    ]


def _detail_keys(predictor) -> list:
    return [
        *predictor._traj_channel_keys.values(),
        *predictor._detail_creation_keys.values(),
        *predictor._detail_forecast_id_keys.values(),
    ]


def _frame(column: str, value, rows: int = 1) -> pd.DataFrame:
    index = pd.date_range("2026-07-03 01:00", periods=rows, freq="15min", tz="UTC", name="timestamp")
    return pd.DataFrame({column: [value] * rows}, index=index)


def test_write_header_table_skips_when_logger_not_configured():
    predictor = _bare(_logger_id=None)

    # Must not raise, and must not attempt any connector resolution.
    predictor.tables().write_header_table(_frame("forecast_id", 0))


def test_write_header_table_empty_frame_is_a_noop():
    """The empty guard fires BEFORE any connector resolution -- deliberately no
    connectors patch here, so this would error if the guard were removed."""
    predictor = _bare(_logger_id="db")

    predictor.tables().write_header_table(pd.DataFrame())


def test_write_image_table_skips_when_logger_not_configured():
    predictor = _bare(_logger_id=None)

    predictor.tables().write_image_table(_frame(SoilPredictor._IMAGE_KEY, _PNG_A))


def test_write_detail_table_skips_when_connector_missing(monkeypatch, caplog):
    class _NoConnectors:
        def __getitem__(self, item):
            raise KeyError(item)

    predictor = _bare(monkeypatch, connectors=_NoConnectors(), _logger_id="db", _traj_channel_keys={})
    predictor._logger_connector_from_channel = lambda: None

    with caplog.at_level("WARNING"):
        predictor.tables().write_detail_table(_frame("traj_root_20", 0.9))

    assert any("not found" in message for message in caplog.messages)


def test_write_detail_table_uses_channel_resolved_connector(monkeypatch):
    """A nested predictor references a ROOT-level connector: the component-scoped
    id lookup cannot resolve the bare id, but the header's forecast_id channel
    already bound it at registration, and that anchor serves BOTH tables."""
    connector = _RecordingConnector()

    class _Logger:
        def _get_registrator(self):
            return connector

    class _Channel:
        def __init__(self, channel_id):
            self.id = channel_id
            self.logger = _Logger()

    class _Data:
        def __init__(self, keys):
            self._channels = {key: _Channel(f"test_predictor.{key}") for key in keys}

        def __getitem__(self, key):
            return self._channels[key]

    class _NoConnectors:
        def __getitem__(self, item):
            raise KeyError(item)

    predictor = _bare(
        monkeypatch,
        connectors=_NoConnectors(),
        _logger_id="mariadb",
        _traj_channel_keys={"root_20": "traj_root_20"},
        _detail_creation_keys={},
        _detail_forecast_id_keys={},
    )
    monkeypatch.setattr(
        SoilPredictor, "data", property(lambda self: _Data([SoilPredictor._HEADER_FORECAST_ID_KEY, "traj_root_20"]))
    )

    predictor.tables().write_detail_table(_frame("traj_root_20", 0.9))

    assert connector.written, "must write via the connector the header channel resolved"


def test_header_and_detail_writes_rename_to_ids_and_never_call_set(monkeypatch):
    """Exercises the REAL write path (build -> write -> lazy id map -> rename ->
    connector.write): the connector receives columns keyed by RESOLVED channel
    ids, and no schema-declaring channel is ever ``.set()``."""
    predictor = _bare(
        _logger_id="db",
        _header_window_min_keys=["w0_min"],
        _header_window_start_keys=["w0_start"],
        _traj_channel_keys={"root_20": "traj_root_20"},
        _detail_creation_keys={"root_20": "traj_root_20_timestamp_creation"},
        _detail_forecast_id_keys={"root_20": "traj_root_20_forecast_id"},
    )
    all_keys = _header_keys(predictor) + _detail_keys(predictor)
    data = _FakeDataAccess(all_keys)
    connector = _RecordingConnector()
    monkeypatch.setattr(SoilPredictor, "data", property(lambda self: data))
    monkeypatch.setattr(SoilPredictor, "connectors", property(lambda self: _connectors(connector)))

    predictor.tables().write_header_table(_frame(SoilPredictor._HEADER_FORECAST_ID_KEY, 0))
    predictor.tables().write_detail_table(_frame("traj_root_20", 0.9))

    assert len(connector.written) == 2
    assert set(connector.written[0].columns) == {data[SoilPredictor._HEADER_FORECAST_ID_KEY].id}
    assert set(connector.written[1].columns) == {data["traj_root_20"].id}
    for key in all_keys:
        assert data[key].calls == [], f"'{key}' must never be .set()"
        assert pd.isna(data[key].timestamp), f"'{key}' timestamp must stay NaT"


@pytest.mark.parametrize(
    ("table", "keys"),
    [
        ("irrigation", [SoilPredictor._IRRIGATION_STATE_KEY, SoilPredictor._IRRIGATION_TIMESTAMP_CREATION_KEY]),
        ("image", [SoilPredictor._IMAGE_KEY, SoilPredictor._IMAGE_TIMESTAMP_CREATION_KEY]),
    ],
)
def test_single_table_write_renames_to_channel_ids_and_never_calls_set(monkeypatch, table, keys):
    predictor = _bare(_logger_id="db")
    data = _FakeDataAccess(keys)
    connector = _RecordingConnector()
    monkeypatch.setattr(SoilPredictor, "data", property(lambda self: data))
    monkeypatch.setattr(SoilPredictor, "connectors", property(lambda self: _connectors(connector)))

    index = pd.DatetimeIndex([pd.Timestamp("2026-07-03 08:00", tz=_TZ)], name="timestamp")
    frame = pd.DataFrame({keys[0]: [True], keys[1]: [_RUN_TS]}, index=index)
    getattr(predictor.tables(), f"write_{table}_table")(frame)

    assert len(connector.written) == 1
    assert set(connector.written[0].columns) == {data[key].id for key in keys}
    for key in keys:
        assert data[key].calls == [], f"'{key}' must never be .set()"
        assert pd.isna(data[key].timestamp)


# --- Write-failure counters --------------------------------------------------


class _CountingChannel:
    def __init__(self):
        self.calls: list = []

    def set(self, timestamp, value) -> None:
        self.calls.append((timestamp, value))


class _CountingData(dict):
    def __missing__(self, key):
        channel = _CountingChannel()
        self[key] = channel
        return channel


class _RaisingData:
    """Every channel access raises -- the counter bump must swallow this."""

    def __getitem__(self, key):
        raise KeyError(key)


class _RaisingConnector:
    def write(self, frame):
        raise RuntimeError("db down")


def _failing(monkeypatch, data, connector=None) -> SoilPredictor:
    predictor = _bare(monkeypatch, connectors=_connectors(connector or _RaisingConnector()), _logger_id="db")
    predictor._write_failures = None
    predictor._logger_connector_from_channel = lambda: None  # force the id-based fallback
    monkeypatch.setattr(SoilPredictor, "data", property(lambda self: data))
    return predictor


def test_write_failure_bumps_only_that_tables_counter(monkeypatch, caplog):
    data = _CountingData()
    predictor = _failing(monkeypatch, data)

    with caplog.at_level("ERROR"):
        predictor.tables().write_direct_frame(_frame("x", 0.0, rows=2), lambda: {}, "header table")
        predictor.tables().write_direct_frame(_frame("x", 0.0, rows=2), lambda: {}, "header table")

    header = components._WRITE_FAILURE_CHANNELS["header table"]
    assert [v for _, v in data[header].calls] == [1.0, 2.0]
    for label in ("detail table", "irrigation table", "image table"):
        assert components._WRITE_FAILURE_CHANNELS[label] not in data


def test_write_failure_error_names_table_and_row_count(monkeypatch, caplog):
    predictor = _failing(monkeypatch, _CountingData())

    with caplog.at_level("ERROR"):
        predictor.tables().write_direct_frame(_frame("x", 0.0, rows=3), lambda: {}, "irrigation table")

    errors = [r for r in caplog.records if "direct write" in r.getMessage()]
    assert len(errors) == 1
    assert "irrigation table" in errors[0].getMessage()
    assert "3 rows" in errors[0].getMessage()


def test_successful_write_and_skip_paths_never_bump(monkeypatch):
    data = _CountingData()
    connector = _RecordingConnector()
    predictor = _failing(monkeypatch, data, connector=connector)

    predictor.tables().write_direct_frame(_frame("x", 0.0, rows=2), lambda: {}, "header table")  # success
    predictor.tables().write_direct_frame(pd.DataFrame(), lambda: {}, "header table")  # empty: skip
    predictor._logger_id = None
    predictor.tables().write_direct_frame(_frame("x", 0.0, rows=2), lambda: {}, "header table")  # unconfigured

    assert len(connector.written) == 1
    assert components._WRITE_FAILURE_CHANNELS["header table"] not in data
    assert predictor._write_failures in (None, {})


def test_counter_bump_survives_a_raising_channel_access(monkeypatch, caplog):
    """The failure handler's job is to log, not crash, when the counter channel
    is unavailable."""
    predictor = _failing(monkeypatch, _RaisingData())

    with caplog.at_level("ERROR"):
        predictor.tables().write_direct_frame(_frame("x", 0.0, rows=2), lambda: {}, "detail table")  # must not raise

    assert predictor._write_failures == {"detail table": 1}
