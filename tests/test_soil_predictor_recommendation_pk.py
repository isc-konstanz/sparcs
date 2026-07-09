# -*- coding: utf-8 -*-
"""sparcs.tests.test_soil_predictor_recommendation_pk
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Regression tests for the collapsed recommendation stage: there is no
separate recommendation table/channels anymore -- ``SoilPredictor`` no longer
defines ``_publish_recommendation``, ``_RECOMMEND_TABLE_NAME``, or any
``recommend_*`` channel keys (the chosen candidate is the header's
``is_recommended`` row instead, see ``test_soil_predictor_trajectory_table.py``).

This file also pins the exact ``data.add()`` kwargs the small
``_register_*`` channel-registration helpers (``configure()``'s extracted,
unit-testable building blocks -- mirrors ``soil.py``'s
``_register_state_channel``/``_register_progress_image_channel``) pass: every
in-memory-only channel this issue collapses (``timestamp_creation``,
``predict_<probe>``, ``predict_state``, ``predict_plot``, the water-balance
diagnostics) gets ``logger.enabled = False``, so none of them can ever
auto-create the old vestigial ``soil_predictor`` table again.
"""

import types

import pytest

soil_predictor = pytest.importorskip("sparcs.components.agriculture.simulation.soil_predictor")
SoilPredictor = soil_predictor.SoilPredictor


# --- Collapse: the recommendation table/channels are gone --------------------


def test_publish_recommendation_method_removed():
    assert not hasattr(SoilPredictor, "_publish_recommendation")


def test_recommend_table_name_constant_removed():
    assert not hasattr(SoilPredictor, "_RECOMMEND_TABLE_NAME")


def test_recommend_total_key_constant_removed():
    assert not hasattr(SoilPredictor, "_RECOMMEND_TOTAL_KEY")


def test_unused_window_sentinel_removed():
    """Unconfigured windows persist as NULL header columns; a fill sentinel
    must not creep back."""
    assert not hasattr(soil_predictor, "_UNUSED_WINDOW_SENTINEL")


# --- Registration helpers: exact data.add() kwargs ---------------------------


class _RecordingData:
    """Stand-in for ``self.data``: captures every ``add(...)`` call's kwargs."""

    def __init__(self):
        self.added: list[tuple] = []

    def add(self, key, **kwargs) -> None:
        # `key` mirrors the real signature (lories/data/access.py:
        # add(self, key, **configs)) so positional and keyword call
        # styles both record correctly.
        self.added.append((key, kwargs))


def _bare_predictor(monkeypatch, **extra):
    predictor = object.__new__(SoilPredictor)
    predictor._name = "test_predictor"
    for key, value in extra.items():
        setattr(predictor, key, value)
    fake = _RecordingData()
    monkeypatch.setattr(SoilPredictor, "data", property(lambda self: fake))
    return predictor, fake


def test_register_timestamp_creation_channel_disables_logger(monkeypatch):
    """The main run-time channel stays in-memory (Dash) but must not log: a
    lone enabled channel here would auto-create a vestigial group-named table
    of PK-only rows once the config's table default drops."""
    predictor, fake = _bare_predictor(monkeypatch)

    predictor._register_timestamp_creation_channel()

    channel_id, kwargs = fake.added[0]
    assert channel_id == SoilPredictor._TIMESTAMP_CREATION_KEY
    assert kwargs["logger"]["enabled"] is False
    assert kwargs["logger"]["primary"] is True
    assert kwargs["logger"]["nullable"] is False


def test_register_predict_channels_disables_logger(monkeypatch):
    """predict_<probe> stays in-memory only: the chosen candidate's forecast
    persists via the header's is_recommended and its agri_soil_forecast
    rows."""
    predictor, fake = _bare_predictor(monkeypatch)
    probe = types.SimpleNamespace(channel_id="root_20", name="Root 20cm")

    channel_keys = predictor._register_predict_channels([probe])

    assert channel_keys == {"root_20": "predict_root_20"}
    channel_id, kwargs = fake.added[0]
    assert channel_id == "predict_root_20"
    assert kwargs["logger"]["enabled"] is False


def test_register_state_channel_disables_logger(monkeypatch):
    predictor, fake = _bare_predictor(monkeypatch)

    predictor._register_state_channel()

    channel_id, kwargs = fake.added[0]
    assert channel_id == SoilPredictor._STATE_CHANNEL_KEY
    assert kwargs["logger"]["enabled"] is False


def test_register_plot_channel_disables_logger(monkeypatch):
    predictor, fake = _bare_predictor(monkeypatch)

    predictor._register_plot_channel()

    channel_id, kwargs = fake.added[0]
    assert channel_id == SoilPredictor._PLOT_CHANNEL_KEY
    assert kwargs["logger"]["enabled"] is False


def test_register_diagnostic_channels_disables_logger(monkeypatch):
    """The water-balance diagnostics mirror stays in-memory only (no longer
    persisted)."""
    predictor, fake = _bare_predictor(monkeypatch)

    predictor._register_diagnostic_channels()

    assert len(fake.added) == len(soil_predictor._DIAGNOSTIC_CONSTANTS)
    for _channel_id, kwargs in fake.added:
        assert kwargs["logger"]["enabled"] is False


def test_register_header_channels_binds_header_table(monkeypatch):
    """Every header channel routes to the configured logger connector's
    agri_field_forecast table with logger.enabled=True (direct-write path)."""
    predictor, fake = _bare_predictor(monkeypatch, _logger_id="mariadb", _max_windows=4)

    window_min_keys, window_start_keys = predictor._register_header_channels()

    assert window_min_keys == ["w0_min", "w1_min", "w2_min", "w3_min"]
    assert window_start_keys == ["w0_start", "w1_start", "w2_start", "w3_start"]
    added_ids = [channel_id for channel_id, _ in fake.added]
    assert SoilPredictor._HEADER_FORECAST_ID_KEY in added_ids
    assert SoilPredictor._HEADER_IS_RECOMMENDED_KEY in added_ids
    assert SoilPredictor._HEADER_TOTAL_MIN_KEY in added_ids
    assert SoilPredictor._HEADER_WEATHER_CREATION_KEY in added_ids
    for channel_id, kwargs in fake.added:
        assert kwargs["logger"]["table"] == SoilPredictor._HEADER_TABLE_NAME
        assert kwargs["logger"]["connector"] == "mariadb"
        assert kwargs["logger"]["enabled"] is True

    # forecast_id is the header's PK partner; window min/start are plain data
    # columns (NOT primary -- nullable, no -1 sentinel).
    forecast_id_kwargs = dict(fake.added)[SoilPredictor._HEADER_FORECAST_ID_KEY]
    assert forecast_id_kwargs["logger"]["primary"] is True
    assert forecast_id_kwargs["logger"]["nullable"] is False
    w0_min_kwargs = dict(fake.added)["w0_min"]
    assert w0_min_kwargs["logger"].get("primary") is not True


def test_register_detail_channels_binds_detail_table_with_shared_water_tension_column(monkeypatch):
    """The detail table's probe channels share ONE 'water_tension' DB column
    (per-probe distinction via soil_id) and there is no w{i}_min PK column --
    each probe's OWN timestamp_creation/forecast_id TWINS are its PK partners,
    because a single shared pair cannot carry N different probes' soil_ids at
    once."""
    predictor, fake = _bare_predictor(monkeypatch, _logger_id="mariadb")
    probes = [
        types.SimpleNamespace(channel_id="root_20", name="Root 20cm"),
        types.SimpleNamespace(channel_id="root_40", name="Root 40cm"),
    ]
    probe_identities = {
        "root_20": {"soil_id": 20, "field_id": 2},
        "root_40": {"soil_id": 40, "field_id": 2},
    }

    tension_keys, creation_keys, forecast_id_keys = predictor._register_detail_channels(probes, probe_identities)

    assert tension_keys == {"root_20": "traj_root_20", "root_40": "traj_root_40"}
    assert creation_keys == {
        "root_20": "traj_root_20_timestamp_creation",
        "root_40": "traj_root_40_timestamp_creation",
    }
    assert forecast_id_keys == {
        "root_20": "traj_root_20_forecast_id",
        "root_40": "traj_root_40_forecast_id",
    }

    by_id = dict(fake.added)
    assert by_id["traj_root_20"]["logger"]["column"] == "water_tension"
    assert by_id["traj_root_40"]["logger"]["column"] == "water_tension"
    assert by_id["traj_root_20"]["logger"]["table"] == SoilPredictor._DETAIL_TABLE_NAME
    assert "primary" not in by_id["traj_root_20"]["logger"] or not by_id["traj_root_20"]["logger"]["primary"]

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
    """The regression this correction fixes: the SQL connector's per-
    attribute-set write grouping (``table.py``'s ``_groupby``) raises
    ``ResourceError`` for any resource on a keyed table missing a declared
    surrogate attribute -- a single shared timestamp_creation/forecast_id pair
    could never carry N different probes' soil_ids at once. Every one of a
    probe's THREE channels (tension, timestamp_creation twin, forecast_id
    twin) must carry the IDENTICAL soil_id/field_id pair as that probe, and
    different probes must carry DIFFERENT soil_ids (same field_id)."""
    predictor, fake = _bare_predictor(monkeypatch, _logger_id="mariadb")
    probes = [
        types.SimpleNamespace(channel_id="root_20", name="Root 20cm"),
        types.SimpleNamespace(channel_id="root_40", name="Root 40cm"),
    ]
    probe_identities = {
        "root_20": {"soil_id": 20, "field_id": 2},
        "root_40": {"soil_id": 40, "field_id": 2},
    }

    predictor._register_detail_channels(probes, probe_identities)

    by_id = dict(fake.added)
    for probe_key, expected in probe_identities.items():
        triplet = [f"traj_{probe_key}", f"traj_{probe_key}_timestamp_creation", f"traj_{probe_key}_forecast_id"]
        for channel_id in triplet:
            assert by_id[channel_id]["soil_id"] == expected["soil_id"], channel_id
            assert by_id[channel_id]["field_id"] == expected["field_id"], channel_id
    # Different probes must NOT share a soil_id -- it is what distinguishes their rows.
    assert by_id["traj_root_20"]["soil_id"] != by_id["traj_root_40"]["soil_id"]
    # Same field (both probes belong to the same predictor/field).
    assert by_id["traj_root_20"]["field_id"] == by_id["traj_root_40"]["field_id"]


# --- _resolve_probe_identities: reuses SoilSimulation's own probe identity ---


class _FakeLeafConfig:
    def __init__(self, values: dict):
        self._values = values

    def get(self, key, default=None):
        return self._values.get(key, default)


class _FakeChannelsConfig:
    """``[soil_simulation.data.channels]`` stand-in: ``.get("field_id")`` for
    the component-wide default, ``.get_member(<probe_key>)`` for the per-probe
    ``soil_id`` block -- the exact two reads ``_resolve_probe_identities``
    performs (mirrors ``soil.py``'s ``_validate_probe_soil_ids`` fakes in
    ``test_soil_probe_tension.py``)."""

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


def test_resolve_probe_identities_reads_soil_id_and_field_id(monkeypatch):
    predictor, _ = _bare_predictor(monkeypatch)
    soil_block = _FakeSoilBlock(_FakeChannelsConfig(field_id=2, per_probe_soil_ids={"root_20": 20, "root_40": 40}))
    probes = [
        types.SimpleNamespace(channel_id="root_20", name="Root 20cm"),
        types.SimpleNamespace(channel_id="root_40", name="Root 40cm"),
    ]

    identities = predictor._resolve_probe_identities(soil_block, probes)

    assert identities == {
        "root_20": {"field_id": 2, "soil_id": 20},
        "root_40": {"field_id": 2, "soil_id": 40},
    }


def test_resolve_probe_identities_missing_soil_id_only_warns(monkeypatch, caplog):
    """Mirrors soil.py's _validate_probe_soil_ids: a probe with no configured
    soil_id only warns (fixtures gain soil_ids in a later issue) and simply
    gets no soil_id kwarg -- its channels then fail loudly at connector
    connect time instead (the existing, accepted failure mode)."""
    predictor, _ = _bare_predictor(monkeypatch)
    soil_block = _FakeSoilBlock(_FakeChannelsConfig(field_id=2, per_probe_soil_ids={}))
    probe = types.SimpleNamespace(channel_id="root_20", name="Root 20cm")

    with caplog.at_level("WARNING"):
        identities = predictor._resolve_probe_identities(soil_block, [probe])

    assert identities == {"root_20": {"field_id": 2}}
    assert any("soil_id" in message for message in caplog.messages)
