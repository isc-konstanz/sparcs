# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.simulation.forecast_tables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Forecast-table persistence for ``SoilPredictor``: channel registrations, frame builders and the direct-write path.
All state stays on the predictor and is read at use time; importing ``components`` at runtime would cycle.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable

import pandas as pd
from lories.core import ConfigurationError, ConfigurationUnavailableError
from lories.typing import Configurations

from .core.pde import ProbeSpec

if TYPE_CHECKING:
    from typing import Iterable, Optional

    from .components import SoilPredictor

logger = logging.getLogger(__name__)


def forecast_ids(ladder: list[tuple[pd.Timedelta, ...]]) -> dict[tuple[pd.Timedelta, ...], int]:
    """A candidate's ``forecast_id`` is its position in ``ladder``, stable from run to run."""
    return {candidate: forecast_id for forecast_id, candidate in enumerate(ladder)}


def _merge_irrigation_intervals(
    intervals: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Drop intervals with ``on_ts >= off_ts``, then merge touching or overlapping ones.
    Separate edges at a shared timestamp would write an ambiguous ``(False, True)`` pair on one primary key."""
    valid = sorted((on_ts, off_ts) for on_ts, off_ts in intervals if on_ts < off_ts)
    if not valid:
        return []

    merged: list[list[pd.Timestamp]] = [list(valid[0])]
    for on_ts, off_ts in valid[1:]:
        if on_ts <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], off_ts)
        else:
            merged.append([on_ts, off_ts])
    return [(on_ts, off_ts) for on_ts, off_ts in merged]


class ForecastTablePublisher:
    """Per-call view over one ``SoilPredictor`` for the four persisted forecast tables.
    Sibling calls go through the predictor's ``_x`` delegates so stubs patched onto the predictor still intercept."""

    def __init__(self, predictor: "SoilPredictor") -> None:
        self._predictor = predictor

    # --- Channel registration (data.add() kwargs, unit-testable) --------------

    def _add_forecast_channel(
        self,
        key: str,
        *,
        table: str,
        type: type,
        name: str,
        unit: "Optional[str]" = None,
        column: "Optional[str]" = None,
        primary: bool = False,
        identity: "Optional[dict[str, Any]]" = None,
    ) -> None:
        """Declare one persisted-table channel on the ``logger`` connector; no ``column`` key when ``column`` is None.
        ``identity`` (soil_id/field_id) goes in as top-level ``data.add`` kwargs."""
        p = self._predictor
        logger_cfg: dict[str, Any] = {"connector": p._logger_id, "table": table}
        if column is not None:
            logger_cfg["column"] = column
        if primary:
            logger_cfg["primary"] = True
            logger_cfg["nullable"] = False
        logger_cfg["enabled"] = True
        kwargs: dict[str, Any] = {"type": type, "name": name}
        if unit is not None:
            kwargs["unit"] = unit
        p.data.add(key, aggregate="last", logger=logger_cfg, **kwargs, **(identity or {}))

    def _add_creation_twin(
        self,
        key: str,
        table: str,
        name: str,
        identity: "Optional[dict[str, Any]]" = None,
    ) -> None:
        """Declare the per-run ``timestamp_creation`` primary-key channel a table pairs with its rows."""
        self._add_forecast_channel(
            key,
            table=table,
            type=pd.Timestamp,
            name=name,
            column="timestamp_creation",
            primary=True,
            identity=identity,
        )

    def register_header_channels(self) -> tuple[list[str], list[str]]:
        """`agri_field_forecast`: one row per candidate per run; returns the window min and start keys in order.
        These channels are never `.set()`: the automatic log flush skips a channel whose timestamp is NaT."""
        p = self._predictor
        table = p._HEADER_TABLE_NAME
        self._add_forecast_channel(
            p._HEADER_FORECAST_ID_KEY, table=table, type=int, name="Forecast candidate id", primary=True
        )

        window_min_keys = []
        for i in range(p._max_windows):
            key = f"w{i}_min"
            window_min_keys.append(key)
            self._add_forecast_channel(key, table=table, type=float, name=f"Window {i} duration", unit="min")

        window_start_keys = []
        for i in range(p._max_windows):
            key = f"w{i}_start"
            window_start_keys.append(key)
            self._add_forecast_channel(key, table=table, type=str, name=f"Window {i} start")

        self._add_forecast_channel(
            p._HEADER_IS_RECOMMENDED_KEY, table=table, type=bool, name="Is recommended candidate"
        )
        self._add_forecast_channel(
            p._HEADER_TOTAL_MIN_KEY, table=table, type=float, name="Total watering duration", unit="min"
        )
        self._add_forecast_channel(
            p._HEADER_WEATHER_CREATION_KEY, table=table, type=pd.Timestamp, name="Weather forecast issue time"
        )
        return window_min_keys, window_start_keys

    def resolve_probe_identities(
        self,
        soil_block: Configurations,
        probes: list[ProbeSpec],
    ) -> dict[str, dict[str, Any]]:
        """Per-probe soil_id/field_id kwargs from ``[soil_simulation.data.channels]``.
        A probe without soil_id is logged; two probes sharing a soil_id raise ConfigurationError."""
        p = self._predictor
        channels_cfg = soil_block.get_member("data", defaults={}).get_member("channels", defaults={})
        field_id = channels_cfg.get("field_id", default=None)

        identities: dict[str, dict[str, Any]] = {}
        seen: dict[Any, str] = {}
        for probe in probes:
            identity: dict[str, Any] = {}
            if field_id is not None:
                identity["field_id"] = field_id
            soil_id = channels_cfg.get_member(probe.channel_id, defaults={}).get("soil_id", default=None)
            if soil_id is None:
                logger.warning(
                    "%s: probe '%s' has no soil_id configured on "
                    "[soil_simulation.data.channels.%s]; its agri_soil_forecast "
                    "rows cannot be attributed to a probe.",
                    p.name,
                    probe.channel_id,
                    probe.channel_id,
                )
            else:
                if soil_id in seen:
                    raise ConfigurationError(
                        f"{p.name}: duplicate soil_id {soil_id!r} on probes '{seen[soil_id]}' and '{probe.channel_id}'"
                    )
                seen[soil_id] = probe.channel_id
                identity["soil_id"] = soil_id
            identities[probe.channel_id] = identity
        return identities

    def register_detail_channels(
        self,
        probes: list[ProbeSpec],
        probe_identities: dict[str, dict[str, Any]],
    ) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
        """`agri_soil_forecast`: every candidate's per-probe tension rows, with per-probe creation and id twins.
        The SQL connector groups writes by attribute set, so a probe's three channels share one soil_id/field_id."""
        p = self._predictor
        table = p._DETAIL_TABLE_NAME
        tension_keys: dict[str, str] = {}
        creation_keys: dict[str, str] = {}
        forecast_id_keys: dict[str, str] = {}
        for probe in probes:
            identity = probe_identities.get(probe.channel_id, {})
            key = f"traj_{probe.channel_id}"
            tension_keys[probe.channel_id] = key
            self._add_forecast_channel(
                key,
                table=table,
                type=float,
                name=f"Trajectory {probe.name}",
                unit="hPa",
                column="water_tension",
                identity=identity,
            )

            creation_key = f"{key}{p._DETAIL_TIMESTAMP_CREATION_SUFFIX}"
            creation_keys[probe.channel_id] = creation_key
            self._add_creation_twin(creation_key, table, f"Trajectory {probe.name} run timestamp", identity=identity)

            forecast_id_key = f"{key}{p._DETAIL_FORECAST_ID_SUFFIX}"
            forecast_id_keys[probe.channel_id] = forecast_id_key
            self._add_forecast_channel(
                forecast_id_key,
                table=table,
                type=int,
                name=f"Trajectory {probe.name} candidate id",
                column="forecast_id",
                primary=True,
                identity=identity,
            )
        return tension_keys, creation_keys, forecast_id_keys

    def register_irrigation_channels(self) -> None:
        """`agri_field_forecast_irrigation`: the chosen candidate's watering schedule as state-transition edge rows."""
        p = self._predictor
        table = p._IRRIGATION_TABLE_NAME
        self._add_forecast_channel(p._IRRIGATION_STATE_KEY, table=table, type=bool, name="Irrigation plan state")
        self._add_creation_twin(p._IRRIGATION_TIMESTAMP_CREATION_KEY, table, "Irrigation plan run timestamp")

    def register_image_channels(self) -> None:
        """`agri_field_forecast_image`: the recommended candidate's field-plot PNGs.
        Separate from the in-memory `predict_plot` channel, which stays `.set()` for Dash."""
        p = self._predictor
        table = p._IMAGE_TABLE_NAME
        self._add_forecast_channel(
            p._IMAGE_KEY, table=table, type=bytes, name="Predicted soil field image", unit="png", column=p._IMAGE_COLUMN
        )
        self._add_creation_twin(p._IMAGE_TIMESTAMP_CREATION_KEY, table, "Predicted image run timestamp")

    # --- Frame builders (pure, unit-testable) ---------------------------------

    def build_image_frame(
        self,
        save_index: pd.DatetimeIndex,
        plot_values: list[bytes],
        run_timestamp: pd.Timestamp,
    ) -> pd.DataFrame:
        """One PNG row per snapshot at its future timestamp, each stamped with ``run_timestamp``.
        Columns are bare channel keys, not channel ids."""
        p = self._predictor
        columns = [p._IMAGE_KEY, p._IMAGE_TIMESTAMP_CREATION_KEY]
        rows: list[dict[str, Any]] = []
        index: list[pd.Timestamp] = []
        for ts, png in zip(save_index, plot_values):
            rows.append({p._IMAGE_KEY: png, p._IMAGE_TIMESTAMP_CREATION_KEY: run_timestamp})
            index.append(ts)
        if not rows:
            return pd.DataFrame(columns=columns)
        frame = pd.DataFrame.from_records(rows, index=pd.DatetimeIndex(index, name="timestamp"))
        return frame.loc[:, columns]

    # --- Direct-write path ----------------------------------------------------

    def write_direct_frame(
        self,
        frame: pd.DataFrame,
        id_by_key_fn: Callable[[], dict[str, str]],
        table_label: str,
    ) -> None:
        """Rename key columns to channel ids and write once; never raises, a failure is logged and counted.
        ``id_by_key_fn`` runs only after the connector resolved with a write(), so a skip never touches ``data``."""
        p = self._predictor
        if p._logger_id is None:
            return
        if frame.empty:
            logger.debug("%s: %s frame empty; skipping direct write.", p.name, table_label)
            return

        connector = p._resolve_logger_connector(p._logger_id)
        if connector is None:
            logger.warning(
                "%s: logger connector '%s' not found; skipping the %s direct write.",
                p.name,
                p._logger_id,
                table_label,
            )
            return
        if not hasattr(connector, "write"):
            logger.warning(
                "%s: logger connector '%s' (%s) has no write(); skipping the %s direct write.",
                p.name,
                p._logger_id,
                type(connector).__name__,
                table_label,
            )
            return

        write_frame = frame.rename(columns=id_by_key_fn())
        try:
            connector.write(write_frame)
        except Exception:  # noqa: BLE001
            logger.exception(
                "%s: direct write of the %s (%d rows) to logger '%s' failed.",
                p.name,
                table_label,
                len(write_frame),
                p._logger_id,
            )
            p._bump_write_failure(table_label)
            return
        logger.info(
            "%s: %s written: %d rows to logger '%s'.",
            p.name,
            table_label,
            len(write_frame),
            p._logger_id,
        )

    def _ids_for(self, keys: "Iterable[str]") -> dict[str, str]:
        """Key to full channel id for ``keys``, one ``data`` lookup per key."""
        p = self._predictor
        return {key: p.data[key].id for key in keys}

    def write_header_table(self, frame: pd.DataFrame) -> None:
        """Direct-write the ``agri_field_forecast`` header frame."""
        p = self._predictor
        p._write_direct_frame(
            frame,
            lambda: self._ids_for(
                [
                    p._HEADER_FORECAST_ID_KEY,
                    *p._header_window_min_keys,
                    *p._header_window_start_keys,
                    p._HEADER_IS_RECOMMENDED_KEY,
                    p._HEADER_TOTAL_MIN_KEY,
                    p._HEADER_WEATHER_CREATION_KEY,
                ]
            ),
            "header table",
        )

    def write_detail_table(self, frame: pd.DataFrame) -> None:
        """Direct-write the ``agri_soil_forecast`` detail frame."""
        p = self._predictor
        p._write_direct_frame(
            frame,
            lambda: self._ids_for(
                [
                    *p._traj_channel_keys.values(),
                    *p._detail_creation_keys.values(),
                    *p._detail_forecast_id_keys.values(),
                ]
            ),
            "detail table",
        )

    def write_irrigation_table(self, frame: pd.DataFrame) -> None:
        """Direct-write the ``agri_field_forecast_irrigation`` edge-row frame."""
        p = self._predictor
        p._write_direct_frame(
            frame,
            lambda: self._ids_for([p._IRRIGATION_STATE_KEY, p._IRRIGATION_TIMESTAMP_CREATION_KEY]),
            "irrigation table",
        )

    def write_image_table(self, frame: pd.DataFrame) -> None:
        """Direct-write the ``agri_field_forecast_image`` frame."""
        p = self._predictor
        p._write_direct_frame(
            frame,
            lambda: self._ids_for([p._IMAGE_KEY, p._IMAGE_TIMESTAMP_CREATION_KEY]),
            "image table",
        )

    # --- Connector resolution ------------------------------------------------

    def validate_logger_connector(self) -> None:
        """Refuse to start when ``logger`` names a connector that can never write."""
        p = self._predictor
        if p._logger_id is None:
            return
        connector = self.resolve_logger_connector(p._logger_id)
        if connector is None:
            raise ConfigurationUnavailableError(
                f"{p.name}: [soil_predictor] logger = '{p._logger_id}' resolves to no connector; "
                "every forecast-table write would be skipped. Point it at a declared "
                "[connectors.<id>] or remove the key to disable the direct writes."
            )
        if not callable(getattr(connector, "write", None)):
            raise ConfigurationUnavailableError(
                f"{p.name}: [soil_predictor] logger = '{p._logger_id}' resolves to "
                f"{type(connector).__name__}, which has no write(); forecast tables need a writing connector."
            )

    def resolve_logger_connector(self, logger_id: str) -> "Optional[Any]":
        """Resolve the direct-write connector; a bare id is tried under each prefix of the predictor's path first.
        Innermost prefix wins, so a root-level ``[connectors.<id>]`` also resolves for a nested predictor."""
        p = self._predictor
        context = p.connectors.context
        connector_id = logger_id
        if "." not in connector_id:
            for i in reversed(range(1, len(p.path) + 1)):
                candidate = ".".join([*p.path[:i], logger_id])
                if candidate in context.keys():
                    connector_id = candidate
                    break
        connector = context.get(connector_id, None)
        if connector is not None:
            return connector
        try:
            connector = getattr(p.connectors, logger_id)
        except AttributeError:
            connector = None
        if connector is not None:
            return connector
        try:
            return p.connectors[logger_id]
        except (KeyError, TypeError):
            return None
