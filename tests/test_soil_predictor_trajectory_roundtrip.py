# -*- coding: utf-8 -*-
"""sparcs.tests.test_soil_predictor_trajectory_roundtrip
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

BOX-PENDING: the PRD Prerequisite 2 / issue 06 direct-write spike -- a
duplicate-timestamp composite-PK round-trip through the real lories SQL
connector (``lories.connectors.sql.database.SqlDatabase``) against MariaDB or
MySQL.

Two same-timestamp rows that differ only in ``w0_min`` must survive
``connector.write(frame)`` -> ``connector.read(resources)`` as two DISTINCT
rows keyed on the full composite PK ``(timestamp, timestamp_creation,
w0_min, ...)``, not collapsed -- this is the exact behavior
``_write_trajectory_table`` depends on (``Table.write`` upserts on the full
composite PK via ``ON DUPLICATE KEY UPDATE``; two rows sharing ``timestamp``
but differing in ``w0_min`` are different PK tuples, hence different rows).

There is NO local MariaDB/MySQL server available in this environment (see
project memory: no local DB server), and the lories SQL connector's own
``configure()`` explicitly rejects ``sqlite`` (``ConfigurationError:
Unsupported database type``), so the upsert-on-duplicate-composite-PK
behavior under test cannot be faked against an in-memory backend without
misrepresenting what is actually being proven. This test is gated on real
connection parameters via environment variables and SKIPS (does not fail)
when they are absent or the connection cannot be established.

A real ``SqlDatabase`` connector cannot be constructed standalone: its base
``Registrator.__init__`` asserts a real ``RegistratorContext``
(``lories/core/register/registrator.py``), which only a full ``Application``
provides via the framework's own config-driven load path -- there is no
lighter, still-real construction available (confirmed against
``lories/application/main.py``/``settings.py``; the same friction the
ai-inference prototype ran into, see ``.scratch/_archive/ai-inference-
prototype/PRD.md``). This test therefore bootstraps a minimal headless
project (``settings.conf`` + ``system.conf`` declaring one ``[connectors.sql]``
block and no bound channels) via ``lories.load(...)``, mirroring that
prototype's verified recipe, then fetches the configured (not yet connected --
a channel-less connector never auto-connects) connector and connects it
manually against the test's own ad-hoc ``Resources``.

Run on the box against a real MariaDB/MySQL instance to close this out, with:

    SPARCS_TEST_SQL_HOST, SPARCS_TEST_SQL_PORT, SPARCS_TEST_SQL_USER,
    SPARCS_TEST_SQL_PASSWORD, SPARCS_TEST_SQL_DATABASE
    (SPARCS_TEST_SQL_DIALECT, default "mariadb")
"""

import os

import pytest

import pandas as pd

pytestmark = pytest.mark.slow

lories = pytest.importorskip("lories")
# NOTE: lories.typing.Resource/Resources are TypeVars, not constructable classes;
# the concrete dataclasses live in lories.core. Importing the TypeVars here is what
# made this box-pending test error at _build_resources() before it could connect.
_resource_mod = pytest.importorskip("lories.core.resource")
_resources_mod = pytest.importorskip("lories.core.resources")

Resource = _resource_mod.Resource
Resources = _resources_mod.Resources

_ENV_HOST = "SPARCS_TEST_SQL_HOST"
_ENV_PORT = "SPARCS_TEST_SQL_PORT"
_ENV_USER = "SPARCS_TEST_SQL_USER"
_ENV_PASSWORD = "SPARCS_TEST_SQL_PASSWORD"
_ENV_DATABASE = "SPARCS_TEST_SQL_DATABASE"
_ENV_DIALECT = "SPARCS_TEST_SQL_DIALECT"

_REQUIRED_ENV = (_ENV_HOST, _ENV_PORT, _ENV_USER, _ENV_PASSWORD, _ENV_DATABASE)

TABLE_NAME = "soil_predictor_trajectory_roundtrip_test"
TIMESTAMP_CREATION_ID = "test_roundtrip.traj_timestamp_creation"
W0_ID = "test_roundtrip.w0_min"
SE_ID = "test_roundtrip.traj_root_20"

_SETTINGS_CONF = """
name = "sparcs_traj_roundtrip_test"
action = "run"

[interface]
enabled = false
"""

_SYSTEM_CONF_TEMPLATE = """
key = "traj_roundtrip_test"
name = "Trajectory Roundtrip Test"

[connectors.sql]
type = "sql"
enabled = true
dialect = "{dialect}"
host = "{host}"
port = {port}
user = "{user}"
password = "{password}"
database = "{database}"
"""


def _missing_env() -> list:
    return [key for key in _REQUIRED_ENV if not os.environ.get(key)]


def _build_project(tmp_path) -> None:
    conf_dir = tmp_path / "conf"
    conf_dir.mkdir()
    (conf_dir / "settings.conf").write_text(_SETTINGS_CONF)
    (conf_dir / "system.conf").write_text(
        _SYSTEM_CONF_TEMPLATE.format(
            dialect=os.environ.get(_ENV_DIALECT, "mariadb"),
            host=os.environ[_ENV_HOST],
            port=int(os.environ[_ENV_PORT]),
            user=os.environ[_ENV_USER],
            password=os.environ[_ENV_PASSWORD],
            database=os.environ[_ENV_DATABASE],
        )
    )


def _build_resources() -> "Resources":
    timestamp_creation = Resource(
        id=TIMESTAMP_CREATION_ID,
        key="traj_timestamp_creation",
        name="Timestamp Creation",
        type=pd.Timestamp,
        table=TABLE_NAME,
        primary=True,
        nullable=False,
    )
    w0_min = Resource(
        id=W0_ID,
        key="w0_min",
        name="Window 0 duration",
        type=float,
        table=TABLE_NAME,
        primary=True,
        nullable=False,
    )
    se = Resource(
        id=SE_ID,
        key="traj_root_20",
        name="Trajectory root_20",
        type=float,
        table=TABLE_NAME,
    )
    return Resources([timestamp_creation, w0_min, se])


def _build_two_combo_frame() -> pd.DataFrame:
    """Two rows sharing the SAME `timestamp` index value, differing only in
    `w0_min` -- the exact duplicate-timestamp composite-PK scenario the
    trajectory table depends on."""
    ts = pd.Timestamp("2026-07-03 08:00", tz="UTC")
    creation = pd.Timestamp("2026-07-03 01:00", tz="UTC")
    index = pd.DatetimeIndex([ts, ts], name="timestamp")
    return pd.DataFrame(
        {
            TIMESTAMP_CREATION_ID: [creation, creation],
            W0_ID: [0.0, 30.0],
            SE_ID: [0.8, 0.9],
        },
        index=index,
    )


@pytest.fixture
def sql_connector(tmp_path, monkeypatch):
    missing = _missing_env()
    if missing:
        pytest.skip(
            f"No local MariaDB/MySQL reachable: missing env var(s) {missing}. "
            "This is the PRD Prerequisite 2 direct-write spike -- run on the box "
            "against a real MariaDB/MySQL instance."
        )

    _build_project(tmp_path)
    monkeypatch.chdir(tmp_path)

    try:
        app = lories.load("sparcs_traj_roundtrip_test")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Unable to load a headless lories project: {e}")

    connector = None
    for candidate in app.connectors.values():
        if candidate.key == "sql":
            connector = candidate
            break
    if connector is None:
        pytest.skip("connectors.sql not found on the loaded headless project.")

    resources = _build_resources()
    try:
        connector.connect(resources)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Unable to connect to the SQL server: {e}")

    yield connector, resources

    try:
        connector.disconnect()
    except Exception:  # noqa: BLE001
        pass


def test_duplicate_timestamp_distinct_w0_min_survive_as_distinct_rows(sql_connector):
    connector, resources = sql_connector
    frame = _build_two_combo_frame()

    connector.write(frame)

    # The trajectory table cannot be read back through connector.read(): its whole
    # point is multiple rows per timestamp (one per candidate), but lories' read
    # path rejects a non-unique DatetimeIndex (validate_index in
    # lories/data/validation.py raises "Invalid series with non unique index"), and
    # an unbounded read additionally applies .limit(1). Offline analysis therefore
    # reads this table with direct SQL -- which is what proves the composite-PK
    # persistence here: both rows physically survive as DISTINCT PK tuples.
    from sqlalchemy import text

    rows = connector.connection.execute(
        text(f"SELECT w0_min, traj_root_20 FROM {TABLE_NAME} ORDER BY w0_min")
    ).fetchall()

    assert len(rows) == 2, (
        "two rows sharing `timestamp` but differing in `w0_min` must survive as "
        "two DISTINCT rows keyed on the full composite PK, not collapsed to one"
    )
    assert sorted(float(r[0]) for r in rows) == [0.0, 30.0]
    assert sorted(float(r[1]) for r in rows) == [0.8, 0.9]  # each row keeps its own Se
