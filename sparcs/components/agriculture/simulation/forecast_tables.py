# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.simulation.forecast_tables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Forecast-table persistence for ``SoilPredictor``: the four
persisted tables' channel registrations, frame builders, and the shared
direct-write path. ``ForecastTablePublisher`` is a PER-CALL view over the
predictor instance -- it reads ``predictor.data`` / ``predictor.connectors``
/ ``predictor._logger_id`` etc. live through the predictor at use time,
never copying at configure time: the forecast-table tests monkeypatch
``SoilPredictor.data``/``.connectors`` as class properties, stub
collaborators as instance attributes, and assert the write-failure
counters on the predictor. Every cross-method dispatch below therefore
goes back THROUGH the predictor's bound ``_x`` delegate names (so
instance-attr overrides keep intercepting), and all mutable state
(counters, key lists) stays predictor-resident. Table names and channel
keys stay as ``SoilPredictor`` class constants (test-pinned); this module
reads them off the instance. Nothing here imports ``components`` at
runtime (would cycle).
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
    """Deterministic candidate-id enumeration for one run: a candidate's
    ``forecast_id`` is its position in ``ladder``. ``ladder`` (``self._ladder``,
    built once at ``configure()``) has a stable order for a given deployment
    regardless of grid_mode (``fill_order`` or ``full``) or which candidates a
    parallel roll-out happens to finish first, and that order does not change
    run over run -- so the same candidate gets the same id every run. Shared by
    ``planner.build_header_frame``/``planner.build_detail_frame`` so header and
    detail rows agree on every candidate's id.
    """
    return {candidate: forecast_id for forecast_id, candidate in enumerate(ladder)}


def _merge_irrigation_intervals(
    intervals: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Drop degenerate/inverted intervals (``on_ts >= off_ts`` -- a window
    whose resolved start lands at or after a horizon too short to reach
    it, or whose clamped ``off_ts`` collapses back onto ``on_ts``), then
    sort the rest by ``on_ts`` and merge any pair that touches or
    overlaps (``next_on_ts <= current_off_ts``) into one interval.
    Nothing upstream forbids two configured windows from abutting or
    overlapping once resolved onto the horizon, and irrigation staying on
    continuously across such a joint has no state transition to record
    there -- emitting independent edges per window would instead place a
    ``(False, True)`` pair on the identical timestamp, an ambiguous PK
    write the connector's upsert would resolve nondeterministically.
    """
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
    """Per-call view over one ``SoilPredictor`` for the four persisted
    forecast tables. Holds ONLY the predictor reference; every attribute
    read (``data``, ``connectors``, ``_logger_id``, key lists, class
    constants) goes through the predictor live, and every call to a moved
    sibling goes through the predictor's bound delegate (``p._x``), never
    publisher-internal dispatch -- instance-attr stubs and class-property
    monkeypatches on the predictor must keep intercepting exactly as they
    did when these were plain methods.
    """

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
        """Declare one persisted-table channel: bound to the predictor's
        ``logger`` connector id, ``aggregate="last"`` always, and the logger
        dict built BY OMISSION -- no ``column`` key when ``column`` is None
        (hard absence pin: the irrigation state channel must carry no column
        key at all), ``primary=True``/``nullable=False`` emitted only for PK
        partners. ``identity`` (soil_id/field_id) passes through as TOP-LEVEL
        ``data.add`` kwargs, identical across a probe's channels; a probe with
        a partial identity set stays warn-not-raise upstream
        (``resolve_probe_identities``)."""
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
        """Declare a ``timestamp_creation`` PK twin: the per-run timestamp
        value channel every persisted table pairs with its rows (shared
        ``timestamp_creation`` DB column, ``primary``/``nullable=False``)."""
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
        """`agri_field_forecast`: one row per candidate per run. Bound to the
        configured `logger` connector; these channels are NEVER `.set()` by the
        predictor -- the automatic flush (Channels.to_frame(unique=True)) skips
        any channel whose timestamp is NaT, so leaving them un-set is what keeps
        the auto path silent for them (see the module docstring). Returns
        (window_min_keys, window_start_keys), w0 ... w{max_windows-1}, in order.
        Only called when `predictor._logger_id is not None` (see configure())."""
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
        """soil_id/field_id kwargs per probe, for the detail table's per-probe
        channel triplet (`_register_detail_channels`). Read from the SAME
        config the live ``SoilSimulation``'s own ``agri_soil_simulation`` probe
        channel resolves -- ``[soil_simulation.data.channels.<probe_key>].soil_id``
        (per probe) and ``[soil_simulation.data.channels].field_id``
        (component-wide) -- reused code-side (mirrors ``soil.py``'s
        ``_validate_probe_soil_ids``), NOT re-declared under the predictor's own
        config: every channel sharing one probe's rows must carry an IDENTICAL
        surrogate pair, because the SQL connector's per-attribute-set write
        grouping (``table.py``) raises ``ResourceError`` if any resource in a
        keyed-table write is missing a declared surrogate attribute.

        Two probes sharing one ``soil_id`` raise: their forecast rows would
        upsert onto the same key and clobber each other.

        A probe with no configured ``soil_id`` is warned (not raised, matching
        ``soil.py``) and gets no ``soil_id`` kwarg at all -- the detail write
        then fails at the next ``connector.write()`` for this table (caught by
        that table's best-effort try in predict(), logged and counted, not
        raised to the caller), and because the write grouping raises on the
        FIRST resource missing the attribute, ALL probes' rows for that tick
        are dropped along with the misconfigured probe's. A residual
        misconfiguration signal only -- the reference configs carry soil_ids
        since the config overhaul.
        """
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
        """`agri_soil_forecast`: ALL candidates' per-probe tension rows. Same
        never-`.set()` / logger-gated contract as `_register_header_channels`.

        Each probe gets THREE channels: `traj_<probe>` (shared "water_tension"
        DB column, like today) and its OWN `timestamp_creation`/`forecast_id`
        TWINS (`traj_<probe>_timestamp_creation` / `traj_<probe>_forecast_id`,
        sharing the "timestamp_creation"/"forecast_id" DB columns across probes
        -- exactly the same shared-column pattern as `water_tension`) instead of
        one shared pair for every probe: the connector's per-attribute-set
        write grouping requires every channel on this table to carry the SAME
        surrogate attributes as the group it belongs to, and a single shared
        pair cannot carry N different probes' soil_ids at once. `probe_identities`
        (`_resolve_probe_identities`) supplies the soil_id/field_id kwargs,
        applied IDENTICALLY across a probe's three channels (declarations are
        otherwise identical across probes too -- the schema's duplicate-column
        guard is dead code, first-wins, so consistency here is load-bearing).

        Returns (tension_keys, creation_keys, forecast_id_keys), each
        probe.channel_id -> that probe's channel key. Only called when
        `predictor._logger_id is not None` (see configure())."""
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
        """`agri_field_forecast_irrigation`: the chosen candidate's watering schedule as
        state-transition edge rows. Same never-`.set()` / logger-gated contract
        as `_register_header_channels`; only called when `predictor._logger_id is
        not None` (see configure()). Only one field per predictor component, so
        a single shared `timestamp_creation` value channel is enough -- no
        per-probe twins like the detail table needs."""
        p = self._predictor
        table = p._IRRIGATION_TABLE_NAME
        self._add_forecast_channel(p._IRRIGATION_STATE_KEY, table=table, type=bool, name="Irrigation plan state")
        self._add_creation_twin(p._IRRIGATION_TIMESTAMP_CREATION_KEY, table, "Irrigation plan run timestamp")

    def register_image_channels(self) -> None:
        """`agri_field_forecast_image`: the recommended candidate's field-plot PNGs.
        Same never-`.set()` / logger-gated / single-`timestamp_creation`-twin
        contract as `_register_irrigation_channels`; these two channels are DISTINCT
        from the in-memory `predict_plot` channel (which stays `.set()` for Dash),
        so the auto-log path never fires for them. Only called when
        `predictor._logger_id is not None` AND plotting is enabled
        (`predictor._plot_config is not None`; see configure())."""
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
        """Build the ``agri_field_forecast_image`` frame: one PNG-bytes row per
        recommended-candidate snapshot, indexed at the snapshot's future
        timestamp, every row stamped with ``run_timestamp`` (the run time, the
        ``timestamp_creation`` PK partner). ``save_index`` and ``plot_values`` are
        the aligned pair ``_publish_results`` returned (rendered once, reused
        here). Pure and unit-testable: no ``Channel``/connector access; column
        NAMES are the bare channel keys, renamed to ids by ``_write_image_table``.
        """
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
        """Shared direct-write path for the header/detail/irrigation tables:
        rename ``frame``'s bare channel-key columns to full channel ids (what
        ``connector.write`` matches against) and write once. Warns and skips
        (never raises) if `logger` is not configured or the connector cannot be
        resolved/is not a writable connector -- a grid persistence failure must
        never abort the forecast on the main channels.

        ``id_by_key_fn`` is called ONLY once the connector has resolved and is
        writable -- it touches ``predictor.data`` (one lookup per channel), which
        a skip must never do (mirrors the old single-table write's behavior: a
        missing `logger`/connector short-circuits before any channel lookup).
        """
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
        """key -> full channel id for ``keys`` (one ``data`` lookup per key).
        Only ever called through ``_write_direct_frame``'s lazy ``id_by_key_fn``
        -- the skip paths must never touch ``predictor.data``."""
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
        """Resolve the connector for the header/detail direct-writes.

        A bare id is tried against every prefix of the predictor's id path,
        innermost first, then as given, the way a channel resolves its own
        connector, so a root-level ``[connectors.<id>]`` (the common case for a
        shared SQL logger) resolves for a nested predictor. The component-scoped
        attribute and item lookups on ``predictor.connectors`` stay as fallbacks.
        """
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
