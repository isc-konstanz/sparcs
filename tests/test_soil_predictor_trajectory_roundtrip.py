# -*- coding: utf-8 -*-
"""sparcs.tests.test_soil_predictor_trajectory_roundtrip
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two same-timestamp rows differing only in ``forecast_id`` survive a real SQL write as two distinct composite-PK rows.
Skips unless SPARCS_TEST_SQL_HOST/PORT/USER/PASSWORD/DATABASE reach MariaDB or MySQL (the connector rejects sqlite).
"""

import os

import pytest

import pandas as pd

pytestmark = pytest.mark.slow

lories = pytest.importorskip("lories")
# lories.typing.Resource/Resources are TypeVars, not constructable classes; the dataclasses are in lories.core.
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

TABLE_NAME = "agri_soil_forecast_roundtrip_test"
TIMESTAMP_CREATION_ID = "test_roundtrip.traj_timestamp_creation"
FORECAST_ID_ID = "test_roundtrip.traj_forecast_id"
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
    forecast_id = Resource(
        id=FORECAST_ID_ID,
        key="forecast_id",
        name="Forecast candidate id",
        type=int,
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
    return Resources([timestamp_creation, forecast_id, se])


def _build_two_combo_frame() -> pd.DataFrame:
    """Two rows sharing the same `timestamp`, differing only in `forecast_id` (two candidates, same future step)."""
    ts = pd.Timestamp("2026-07-03 08:00", tz="UTC")
    creation = pd.Timestamp("2026-07-03 01:00", tz="UTC")
    index = pd.DatetimeIndex([ts, ts], name="timestamp")
    return pd.DataFrame(
        {
            TIMESTAMP_CREATION_ID: [creation, creation],
            FORECAST_ID_ID: [0, 1],
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


def test_duplicate_timestamp_distinct_forecast_id_survive_as_distinct_rows(sql_connector):
    connector, resources = sql_connector
    frame = _build_two_combo_frame()

    connector.write(frame)

    # connector.read() rejects a non-unique DatetimeIndex, so the rows are read back with direct SQL.
    from sqlalchemy import text

    with connector.engine.connect() as connection:
        rows = connection.execute(
            text(f"SELECT forecast_id, traj_root_20 FROM {TABLE_NAME} ORDER BY forecast_id")
        ).fetchall()

    assert len(rows) == 2, (
        "two rows sharing `timestamp` but differing in `forecast_id` must survive as "
        "two DISTINCT rows keyed on the full composite PK, not collapsed to one"
    )
    assert sorted(int(r[0]) for r in rows) == [0, 1]
    assert sorted(float(r[1]) for r in rows) == [0.8, 0.9]  # each row keeps its own Se
