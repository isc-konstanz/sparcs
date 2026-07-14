# -*- coding: utf-8 -*-
"""sparcs.tests.test_soil_warm_start_read_config
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The warm-start read wiring for ``simulation_state`` only works when
``table``/``column`` are TOP-LEVEL channel keys next to a bare
``connector = "<id>"`` string. A nested ``[<channel>.connector]`` block stores
them on the channel's connector wrapper, which the SQL read path never
consults: ``SqlDatabase.read()`` resolves the table via the channel's OWN
config (``c.get("table", default=c.group)``) and the plain read path applies
no flatten step (only the logger path has ``from_logger()``). Under the
nested shape the read silently queries the channel-group table, the raised
``DatabaseError`` is caught and logged as a warning, and the restore listener
never receives data — while ``has_connector()`` still passes, so no
configure-time warning fires either.

Two guards here: the reference fixture keeps the working shape, and the
framework resolution semantics that shape relies on stay true.
"""

from pathlib import Path

import pytest
import tomllib

from lories._core._connector import _Connector
from lories._core._converter import _Converter
from lories._core._tasks import _TaskContext
from lories.data.channels.channel import Channel

_KOB_SOIL_SIMULATION_CONF = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "test_kob_soil_predictor"
    / "conf"
    / "agri_pv.d"
    / "field_2.d"
    / "field_simulation.d"
    / "soil_simulation.conf"
)


def _load_conf(path: Path) -> dict:
    # lories' TOML loader rewrites leading ``;`` comments to ``#`` before
    # parsing; mirror that so the fixture parses with stock tomllib.
    text = "\n".join(
        "#" + line[1:] if line.lstrip().startswith(";") else line for line in path.read_text().splitlines()
    )
    return tomllib.loads(text)


def test_kob_fixture_simulation_state_read_keys_are_top_level():
    if not _KOB_SOIL_SIMULATION_CONF.is_file():
        pytest.skip("test_kob_soil_predictor fixture not present")
    conf = _load_conf(_KOB_SOIL_SIMULATION_CONF)

    channel = conf["data"]["channels"]["simulation_state"]
    assert channel.get("connector") == "mariadb"
    assert channel.get("table") == "agri_field_simulation_soil_state"
    assert channel.get("column") == "state"
    # The read-side table/column must not live (only) inside a nested
    # connector sub-block — the SQL read path cannot see them there.
    assert not isinstance(channel.get("connector"), dict)
    # Write side unchanged: the logger block keeps its own table.
    assert channel["logger"]["table"] == "agri_field_simulation_soil_state"


# --- framework semantics the fixture shape relies on --------------------------
#
# Bare-instance scaffold: a minimal _TaskContext/_Connector/_Converter set so a
# real Channel can be built without a full application bootstrap (same
# object.__new__ approach the other bare-instance tests use).


class _FakeConnector(_Connector):
    pass


_FakeConnector.__abstractmethods__ = frozenset()


def _make_connector(connector_id: str):
    connector = object.__new__(_FakeConnector)
    connector._id = connector_id
    connector._key = connector_id.split(".")[-1]
    connector._name = connector_id
    connector.is_enabled = lambda: True
    connector.is_configured = lambda: True
    return connector


class _FakeConverter(_Converter):
    pass


_FakeConverter.__abstractmethods__ = frozenset()


def _make_converter():
    converter = object.__new__(_FakeConverter)
    converter._id = "bytes"
    converter._key = "bytes"
    converter._name = "bytes"
    return converter


class _FakeConnectors:
    def __init__(self, connectors):
        self._connectors = {c.id: c for c in connectors}

    def get(self, connector_id, default=None):
        return self._connectors.get(connector_id, default)

    def keys(self):
        return self._connectors.keys()

    def __contains__(self, connector_id):
        return connector_id in self._connectors


class _FakeConverters:
    def get_by_dtype(self, dtype):
        return _make_converter()

    def get(self, converter_id, default=None):
        return _make_converter()

    def keys(self):
        return []


class _FakeContext(_TaskContext):
    @property
    def connectors(self):
        return self.__dict__["_conns"]

    @property
    def processors(self):
        return None

    @property
    def converters(self):
        return self.__dict__["_converters"]

    def has_logged(self, *args, **kwargs):
        return False

    def read_logged(self, *args, **kwargs):
        return None

    def read(self, *args, **kwargs):
        return None

    def write(self, *args, **kwargs):
        return None

    def is_enabled(self):
        return True

    def is_configured(self):
        return True

    def is_active(self):
        return True


_FakeContext.__abstractmethods__ = frozenset()


def _build_channel(connector, **top):
    context = _FakeContext.__new__(_FakeContext)
    context.__dict__["_conns"] = _FakeConnectors([_make_connector("mariadb")])
    context.__dict__["_converters"] = _FakeConverters()
    return Channel(
        context=context,
        id="soil_simulation.simulation_state",
        key="simulation_state",
        type=bytes,
        connector=connector,
        logger={"enabled": True, "column": "state", "table": "agri_field_simulation_soil_state"},
        **top,
    )


def test_top_level_table_and_column_resolve_on_the_raw_read_channel():
    channel = _build_channel(
        connector="mariadb",
        table="agri_field_simulation_soil_state",
        column="state",
    )

    assert channel.get("table") == "agri_field_simulation_soil_state"
    assert channel.get("column") == "state"
    assert channel.has_connector("mariadb")
    # Write side is untouched by the top-level read keys: the logger flatten
    # still resolves the same table/column.
    logged = channel.from_logger()
    assert logged.get("table") == "agri_field_simulation_soil_state"
    assert logged.get("column") == "state"


def test_nested_connector_block_keys_are_invisible_to_the_read_path():
    """The trap the fixture comment warns about: table/column inside a nested
    connector block never reach the channel's own config, so the SQL read
    would fall back to the channel group — while has_connector() still
    passes, hiding the misconfiguration from the configure-time warning."""
    channel = _build_channel(
        connector={"connector": "mariadb", "table": "agri_field_simulation_soil_state", "column": "state"},
    )

    assert channel.get("table") is None
    assert channel.get("column") is None
    assert channel.has_connector("mariadb")
