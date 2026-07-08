# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.simulation.soil_predictor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Forecast-driven horizon predictor. Integrates the Richards-equation soil
PDE over the available weather forecast with no irrigation applied and
publishes the predicted relative saturation at every configured probe.
Optionally archives saturation-field snapshots at ``save_freq`` cadence
as npz blobs (``predict_state``) and PNGs (``predict_plot``).
"""

from __future__ import annotations

import datetime
import io
import itertools
import logging
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from lories.components.weather import Weather
from lories.typing import Configurations
from lories.util import floor_date, to_timedelta
from sparcs.components.agriculture.simulation.soil import (
    PDEConfig,
    ProbeSpec,
    SoilPDECore,
    SoilSimulation,
    resolve_probes,
)

from . import plot_render
from ._soil import ClipDiagnostics, FluxRates, MeshConfig, SoilBase, apply_surface_forcing, ensure_mesh

_DEFAULT_HORIZON: str = "24h"
_DEFAULT_SAVE_FREQ: str = "1h"

# Scheduling gate defaults -- the predictor's OWN cadence, distinct from
# WeatherForecast's interval=60/offset=0 (do not inherit those). `interval`/
# `offset` are the run-cadence config (daily at ~01:00 local); `cooldown` is
# a separate notion, the lories listener-backpressure floor (base.py's
# `data.register(..., interval=...)`), well below the run cadence so it never
# gates the fixed daily boundary.
_DEFAULT_INTERVAL_MIN: int = 1440
_DEFAULT_OFFSET_MIN: int = 60
_DEFAULT_COOLDOWN_MIN: int = 60

# Drip-flow derivation defaults; mirrors the [soil_simulation] default of a
# single already-per-metre line (see soil.py:186) when a field has no
# [soil_predictor.drip] block of its own.
_DEFAULT_NOZZLE_FLOW_LPH: float = 1.0
_DEFAULT_NOZZLE_COUNT: int = 1
_DEFAULT_DRIP_LINE_LENGTH_M: float = 1.0

# Candidate-set (ladder) defaults.
_DEFAULT_COMBO_CAP: int = 16
_DEFAULT_GRID_MODE: str = "fill_order"
_GRID_MODES = ("fill_order", "full")

# Execution-strategy default: keep the proven sequential caterpillar unless an
# operator opts into parallel independent rolls. Existing deployments are
# unchanged until they set parallel=true (see docs/adr/0005-...).
_DEFAULT_PARALLEL: bool = False

# Recommendation-scoring defaults. `threshold_hpa` is a positive hPa magnitude
# target tension (setpoint) the RMS score is measured against; an operator
# calibration input, not a physical constant, so this default is a placeholder,
# not a validated field value.
_DEFAULT_THRESHOLD_HPA: float = 300.0

# Trajectory table: fixed PK arity (unused window columns filled with the
# sentinel below) and the sentinel value itself.
_DEFAULT_MAX_WINDOWS: int = 4
_UNUSED_WINDOW_SENTINEL: float = -1.0


@dataclass(frozen=True)
class WateringWindow:
    """One configured watering window: a site-local clock time the emitters start at.

    The candidate duration for a given ladder rung is passed alongside, not stored
    here, so the same window definition is reused across every candidate.
    """

    start: datetime.time


_DIAGNOSTIC_CONSTANTS = (
    SoilSimulation.WATER_TOP_IN,
    SoilSimulation.WATER_TOP_OUT,
    SoilSimulation.WATER_BOTTOM,
    SoilSimulation.WATER_TRANSP,
    SoilSimulation.WATER_RUNOFF,
    SoilSimulation.WATER_DEMAND_UNMET,
    SoilSimulation.WATER_BALANCE_RESIDUAL,
)


class SoilPredictor(SoilBase):
    REL_SAT_NAME: str = "predictor relative saturation"

    TYPE: str = "soil_predictor"
    INCLUDES = ["pde"]

    _TIMESTAMP_CREATION_KEY: str = "timestamp_creation"
    _STATE_CHANNEL_KEY: str = "predict_state"
    _PLOT_CHANNEL_KEY: str = "predict_plot"

    # Recommendation channels (normal auto-logged path, one row per run).
    _RECOMMEND_TOTAL_KEY: str = "recommend_total_min"
    # The recommendation lives in its own table keyed by run time only (no
    # timestamp_creation composite PK), so each run appends one row and the
    # recommendation history accumulates through the normal auto-logger. Keeping it
    # out of the soil_predictor forecast table is what avoids the composite-PK
    # partner problem entirely -- see _publish_recommendation.
    _RECOMMEND_TABLE_NAME: str = "soil_predictor_recommendation"

    # Trajectory-table channels (direct connector write). Distinct-key scheme:
    # every trajectory column uses a `traj_`/`w{i}_min` prefix so it can never
    # collide with a legacy auto-logged channel key (`predict_<probe>`,
    # `timestamp_creation`, ...) -- see the module docstring / configure().
    _TRAJ_TIMESTAMP_CREATION_KEY: str = "traj_timestamp_creation"
    _TRAJ_IS_RECOMMENDED_KEY: str = "is_recommended"
    _TRAJ_TABLE_NAME: str = "soil_predictor_trajectory"

    _horizon: pd.Timedelta
    _save_freq: pd.Timedelta
    _save_state: bool
    _save_plot: bool
    _save_trajectory_plot: bool = False
    _trajectory_plot_dir: str

    # Scheduling gate: run cadence, own defaults (not WeatherForecast's).
    _interval_min: int
    _offset_min: int
    # Per-listener backpressure floor (base.py's `data.register(..., interval=...)`);
    # distinct from `_interval_min`, the run-cadence config above.
    _cooldown_min: int

    # Last boundary `predict()` actually ran for; None before the first run.
    _last_boundary_run: Optional[pd.Timestamp] = None

    _mesh_config: MeshConfig
    _ode_config: PDEConfig
    _pde: SoilPDECore

    # Fixed design flow derived from the drip layout (m³/s per out-of-plane
    # metre of row); see _derive_flow_m3s.
    _flow_m3s: float

    # Watering windows (ordered by start) and each window's parallel,
    # ascending, zero-inclusive duration list; see _build_ladder.
    _windows: list[WateringWindow]
    _window_durations: list[list[pd.Timedelta]]
    _combo_cap: int
    _grid_mode: str
    _ladder: list[tuple[pd.Timedelta, ...]]

    # Execution strategy for the ladder roll-out (orthogonal to _grid_mode, which
    # picks the candidate SET). _parallel=True rolls every candidate independently
    # across an in-component spawn ProcessPoolExecutor of _max_workers; False keeps
    # the sequential prefix-shared caterpillar (_rollout_ladder) as the efficient
    # path and correctness oracle. See docs/adr/0005-...
    _parallel: bool
    _max_workers: int

    # Recommendation scoring: the target tension (setpoint) and the probe subset
    # the RMS-to-setpoint score is evaluated over; see _score_candidate / _select.
    _threshold_hpa: float
    _decision_probes: list[str]

    _plot_fig: Any = None
    _plot_ax: Any = None
    _plot_norm: Any = None

    _channel_keys: dict[str, str]

    _probes: list[ProbeSpec]

    # Fixed PK arity for the trajectory table (see _build_trajectory_frame) and
    # the id of the SQL logger connector the direct grid write targets; None
    # when [soil_predictor].logger is not configured (the grid write is then
    # skipped with a warning -- see configure()/_write_trajectory_table).
    _max_windows: int
    _logger_id: Optional[str]

    # Recommendation channel keys, w0_min ... w{max_windows-1}_min, in window order.
    _recommend_window_keys: list[str]
    # Trajectory channel keys, w0_min ... w{max_windows-1}_min (PK columns), in window order.
    _traj_window_keys: list[str]
    # probe.channel_id -> trajectory channel key (traj_<probe>), distinct from
    # _channel_keys (the legacy predict_<probe> auto-logged keys).
    _traj_channel_keys: dict[str, str]

    # Dedup gate: skip if (now, forecast_creation) was already published.
    _last_predicted_key: Optional[tuple[pd.Timestamp, pd.Timestamp]] = None

    @staticmethod
    def _resolve_ode_config(
        configs: Configurations,
        soil_pde: PDEConfig,
        model_configs: Configurations,
    ) -> PDEConfig:
        """Build the predictor's PDE config, inheriting surface forcing from the sim.

        The predictor parses its OWN ``[pde]`` block (solver / IC / timestep), so
        any key it does not restate falls back to the ``PDEConfig`` default. That
        is intentional -- the predictor warm-starts from live soil state, so its IC
        keys stay predictor-local. But the surface-forcing blocks ``[ponding]`` and
        ``[feddes]`` are siblings of ``[pde]`` (``soil_pde`` already carries the
        live sim's, attached by the caller), and they must track the sim unless the
        predictor deliberately overrides them: a predictor left on the 5 mm
        ``watering_h_max_mm`` default while the sim ponds to 50 mm overflows its
        watering rolls ~10x sooner, reading too dry and biasing the recommendation.

        Contract: with no ``[pde]`` block the predictor inherits ``soil_pde``
        wholesale (forcing included); with its own ``[pde]`` it overrides the
        solver keys but still inherits the sim's ponding + feddes, unless it
        supplies its own ``[soil_predictor.ponding]`` / ``[soil_predictor.feddes]``
        (which then win via apply_surface_forcing).
        """
        if configs.has_member("pde"):
            ode_config = PDEConfig(configs.get_member("pde"), model_configs=model_configs)
        else:
            ode_config = soil_pde
        # Seed the sim's surface forcing, then let the predictor's own sibling
        # blocks override it. (In the no-[pde] branch ode_config IS soil_pde, so the
        # seed is a no-op and only an explicit predictor override changes anything.)
        ode_config.ponding = soil_pde.ponding
        ode_config.feddes = soil_pde.feddes
        apply_surface_forcing(ode_config, configs)
        return ode_config

    def configure(self, configs: Configurations) -> None:
        super().configure(configs)

        mesh_config = getattr(self.context, "mesh_config", None)
        if mesh_config is None:
            raise ValueError(
                f"{self.id}: parent FieldSimulation has no mesh_config; "
                "predictor needs a [soil_simulation.mesh] block to derive "
                "the soil cross-section."
            )
        self._mesh_config = mesh_config

        self._horizon = to_timedelta(configs.get("horizon", default=_DEFAULT_HORIZON))
        self._save_freq = to_timedelta(configs.get("save_freq", default=_DEFAULT_SAVE_FREQ))
        self._save_state = configs.get_bool("save_state", default=False)
        self._save_plot = configs.get_bool("save_plot", default=False)

        # Debug dump of the soil saturation field at every forecast step (hourly),
        # for every watering candidate on the ladder -- candidates x steps PNGs per
        # run, in a per-run subdir on disk. Off by default: when on it re-simulates
        # the whole ladder (slow, disk-heavy) but never blocks or aborts the forecast.
        self._save_trajectory_plot = configs.get_bool("save_trajectory_plot", default=False)
        if self._save_trajectory_plot:
            self._trajectory_plot_dir = configs.get(
                "plot.dir",
                default=str(configs.dirs.data.joinpath("soil_predictor")),
            )
            os.makedirs(self._trajectory_plot_dir, exist_ok=True)

        self._interval_min = configs.get_int("interval", default=_DEFAULT_INTERVAL_MIN)
        self._offset_min = configs.get_int("offset", default=_DEFAULT_OFFSET_MIN)
        self._cooldown_min = configs.get_int("cooldown", default=_DEFAULT_COOLDOWN_MIN)

        model_block = self.context.configs.get_member("model", defaults={}, ensure_exists=True)
        soil_block = self.context.configs.get_member(SoilSimulation.TYPE, defaults={}, ensure_exists=True)
        soil_pde = PDEConfig(
            soil_block.get_member("pde", defaults={}, ensure_exists=True),
            model_configs=model_block,
        )
        # Attach the live sim's ponding + feddes (sibling blocks of its [pde]) so
        # soil_pde carries the sim's surface forcing for the predictor to inherit.
        apply_surface_forcing(soil_pde, soil_block)

        drip_block = configs.get_member("drip", defaults={}, ensure_exists=True)
        nozzle_flow_lph = drip_block.get_float("nozzle_flow_lph", default=_DEFAULT_NOZZLE_FLOW_LPH)
        nozzle_count = drip_block.get_int("nozzle_count", default=_DEFAULT_NOZZLE_COUNT)
        total_drip_line_length_m = soil_block.get_float("total_drip_line_length_m", default=_DEFAULT_DRIP_LINE_LENGTH_M)
        self._flow_m3s = self._derive_flow_m3s(nozzle_count, nozzle_flow_lph, total_drip_line_length_m)

        self._windows = []
        self._window_durations = []
        if configs.has_member("windows"):
            for name, window_cfg in configs.get_member("windows").items():
                start = pd.Timestamp(str(window_cfg["start"])).time()
                durations = sorted(to_timedelta(d) for d in window_cfg["durations"])
                if pd.Timedelta(0) not in durations:
                    raise ValueError(
                        f"{self.id}: [soil_predictor.windows.{name}] "
                        "is missing a '0min' duration; every window's durations list must include zero."
                    )
                self._windows.append(WateringWindow(start=start))
                self._window_durations.append(durations)

        order = sorted(range(len(self._windows)), key=lambda i: self._windows[i].start)
        self._windows = [self._windows[i] for i in order]
        self._window_durations = [self._window_durations[i] for i in order]

        self._combo_cap = configs.get_int("combo_cap", default=_DEFAULT_COMBO_CAP)
        self._grid_mode = str(configs.get("grid_mode", default=_DEFAULT_GRID_MODE))
        if self._grid_mode not in _GRID_MODES:
            raise ValueError(f"{self.id}: grid_mode={self._grid_mode!r} not in {_GRID_MODES}.")

        self._ladder = self._build_ladder(self._window_durations, self._grid_mode)
        self._check_combo_cap(self._ladder, self._combo_cap, log_name=self.id)

        # Execution strategy: parallel independent rolls vs the sequential
        # caterpillar. max_workers defaults to one below the core count (leaving a
        # core for the parent/OS); only used when parallel=true.
        self._parallel = configs.get_bool("parallel", default=_DEFAULT_PARALLEL)
        default_workers = max(1, (os.cpu_count() or 2) - 1)
        self._max_workers = configs.get_int("max_workers", default=default_workers)
        if self._max_workers < 1:
            raise ValueError(f"{self.id}: max_workers={self._max_workers} must be >= 1.")

        self._max_windows = configs.get_int("max_windows", default=_DEFAULT_MAX_WINDOWS)
        if len(self._windows) > self._max_windows:
            raise ValueError(
                f"{self.id}: {len(self._windows)} [soil_predictor.windows] configured, "
                f"exceeding max_windows={self._max_windows}; max_windows is the fixed "
                "trajectory-table PK arity and needs a manual table migration to raise "
                "(see SOIL.md)."
            )

        self._logger_id = configs.get("logger", default=None)
        if self._logger_id is not None:
            self._logger_id = str(self._logger_id)
        else:
            logging.warning(
                "%s: [soil_predictor].logger not configured; the all-candidate "
                "trajectory table will not be written (the recommendation channels "
                "still log normally).",
                self.name,
            )

        self._ode_config = self._resolve_ode_config(configs, soil_pde, model_block)

        self._pde = self._build_pde()

        probes_cfg = soil_block.get_member("probes", defaults={}, ensure_exists=True)
        self._probes = resolve_probes(
            probes_cfg,
            self._pde.mesh,
            self._mesh_config,
            log_name=self.name,
        )

        self._threshold_hpa = configs.get_float("threshold_hpa", default=_DEFAULT_THRESHOLD_HPA)

        all_probe_ids = [probe.channel_id for probe in self._probes]
        decision_probes_cfg = configs.get("decision_probes", default=None)
        if not decision_probes_cfg:
            self._decision_probes = list(all_probe_ids)
            logging.warning(
                "%s: decision_probes not configured; using ALL probes (%s) for the "
                "tension decision -- surface and deep probes may distort the result.",
                self.name,
                self._decision_probes,
            )
        else:
            decision_probes = [str(p) for p in decision_probes_cfg]
            known = [p for p in decision_probes if p in all_probe_ids]
            unknown = [p for p in decision_probes if p not in all_probe_ids]
            if not known:
                # An all-unknown decision set can never yield a tension sample, so
                # every candidate would score +inf and the argmin would be an
                # arbitrary tie-break over garbage. Fail fast at configure() instead
                # of shipping a miscalibrated recommender.
                raise ValueError(
                    f"{self.id}: decision_probes {decision_probes} match none of the "
                    f"resolved probe channel-ids {all_probe_ids}; the tension decision "
                    "would have no probe to evaluate. Fix the decision_probes ids."
                )
            if unknown:
                logging.warning(
                    "%s: decision_probes %s not found among resolved probe channel-ids "
                    "%s; unknown ids are kept as configured but will never contribute "
                    "a tension sample.",
                    self.name,
                    unknown,
                    all_probe_ids,
                )
            self._decision_probes = decision_probes

        self.data.add(
            key=self._TIMESTAMP_CREATION_KEY,
            name="Predictor Creation Timestamp",
            type=pd.Timestamp,
            aggregate="last",
            logger={
                "primary": True,
                "nullable": False,
                "enabled": True,
            },
        )

        self._channel_keys = {}
        for probe in self._probes:
            key = f"predict_{probe.channel_id}"
            self._channel_keys[probe.channel_id] = key
            self.data.add(
                key,
                type=float,
                name=f"Predicted {probe.name}",
                unit="hPa",
                aggregate="last",
                logger={"enabled": True},
            )

        if self._save_state:
            self.data.add(
                self._STATE_CHANNEL_KEY,
                type=bytes,
                name="Predicted soil state",
                aggregate="last",
                logger={"enabled": True},
            )

        if self._save_plot:
            self.data.add(
                self._PLOT_CHANNEL_KEY,
                type=bytes,
                name="Predicted soil progress image",
                unit="png",
                aggregate="last",
                logger={"enabled": True},
            )

        for c in _DIAGNOSTIC_CONSTANTS:
            self.data.add(c, aggregate="last", logger={"enabled": True})

        if not self._probes:
            logging.warning(
                "%s: no probes resolved from [soil_simulation.probes]; "
                "predictor will still run but has no per-probe channels to "
                "publish.",
                self.name,
            )

        # --- Recommendation channels (normal auto-logged path) ---------------
        # One row per run in the dedicated _RECOMMEND_TABLE_NAME table, keyed by run
        # time only: recommend_w0_min ... recommend_w{max_windows-1}_min,
        # recommend_total_min. Its own table (no timestamp_creation
        # composite PK) lets the recommendation history accumulate through the normal
        # auto-logger. Distinct keys from the trajectory table's w{i}_min PK columns
        # below (different channels, different table).
        self._recommend_window_keys = []
        for i in range(self._max_windows):
            key = f"recommend_w{i}_min"
            self._recommend_window_keys.append(key)
            self.data.add(
                key,
                type=float,
                name=f"Recommended window {i} duration",
                unit="min",
                aggregate="last",
                logger={"enabled": True, "table": self._RECOMMEND_TABLE_NAME},
            )

        self.data.add(
            self._RECOMMEND_TOTAL_KEY,
            type=float,
            name="Recommended total watering duration",
            unit="min",
            aggregate="last",
            logger={"enabled": True, "table": self._RECOMMEND_TABLE_NAME},
        )

        # --- Trajectory-table channels (direct connector write) --------------
        # Bound to the configured `logger` connector, in a dedicated table
        # (_TRAJ_TABLE_NAME) distinct from the default logger's predictor table.
        # These channels are NEVER `.set()` by the predictor -- the automatic
        # flush (Channels.to_frame(unique=True)) skips any channel whose
        # timestamp is NaT, so leaving them un-set is what keeps the auto path
        # silent for them (see the module docstring / PRD "Further Notes").
        # Skipped entirely when no `logger` is configured (degrade, don't crash).
        self._traj_window_keys = []
        self._traj_channel_keys = {}
        if self._logger_id is not None:
            self.data.add(
                self._TRAJ_TIMESTAMP_CREATION_KEY,
                name="Trajectory Creation Timestamp",
                type=pd.Timestamp,
                aggregate="last",
                logger={
                    "connector": self._logger_id,
                    "table": self._TRAJ_TABLE_NAME,
                    "primary": True,
                    "nullable": False,
                    "enabled": True,
                },
            )

            for i in range(self._max_windows):
                key = f"w{i}_min"
                self._traj_window_keys.append(key)
                self.data.add(
                    key,
                    type=float,
                    name=f"Trajectory window {i} duration",
                    unit="min",
                    aggregate="last",
                    logger={
                        "connector": self._logger_id,
                        "table": self._TRAJ_TABLE_NAME,
                        "primary": True,
                        "nullable": False,
                        "enabled": True,
                    },
                )

            self.data.add(
                self._TRAJ_IS_RECOMMENDED_KEY,
                type=bool,
                name="Is recommended candidate",
                aggregate="last",
                logger={
                    "connector": self._logger_id,
                    "table": self._TRAJ_TABLE_NAME,
                    "enabled": True,
                },
            )

            for probe in self._probes:
                key = f"traj_{probe.channel_id}"
                self._traj_channel_keys[probe.channel_id] = key
                self.data.add(
                    key,
                    type=float,
                    name=f"Trajectory {probe.name}",
                    unit="hPa",
                    aggregate="last",
                    logger={
                        "connector": self._logger_id,
                        "table": self._TRAJ_TABLE_NAME,
                        "enabled": True,
                    },
                )

    @property
    def cooldown(self) -> pd.Timedelta:
        """Per-listener backpressure floor for the ``_predict_callback`` registration
        (``base.py``'s ``data.register(..., interval=...)``). Distinct from the
        ``interval``/``offset`` run-cadence config above: this is a low floor (default
        60 min) well below the daily cadence, so it never gates the fixed boundary --
        it only protects against the listener re-dispatching on every live-sim tick.
        """
        return pd.Timedelta(minutes=self._cooldown_min)

    @staticmethod
    def _current_boundary(now: pd.Timestamp, tz, interval_min: int, offset_min: int) -> pd.Timestamp:
        """Most-recent run boundary at or before ``now``, site-local. Mirrors the
        WeatherForecast pattern (lories ``forecast.py:98-101``).
        """
        boundary = floor_date(now, tz, freq=f"{interval_min}T") + pd.Timedelta(minutes=offset_min)
        if boundary > now:
            boundary -= pd.Timedelta(minutes=interval_min)
        return boundary

    # Public driver

    def predict(self, now: pd.Timestamp, forecast_creation: Optional[pd.Timestamp]) -> None:
        """One prediction tick; silently skips if no forecast or no live soil state yet.

        ``forecast_creation`` is the composite-PK partner (forecast issue time) stamped
        on every emitted row. Falls back to ``now`` when unavailable so the channel
        always emits a non-null value.
        """
        if forecast_creation is None:
            logging.debug(
                "%s: forecast_creation unavailable (upstream "
                "weather.forecast.timestamp_creation not valid yet); "
                "falling back to now=%s for the PK column.",
                self.name,
                now,
            )
            forecast_creation = now

        tz = self.context.location.timezone
        boundary = self._current_boundary(now, tz, self._interval_min, self._offset_min)
        if boundary == self._last_boundary_run:
            logging.debug(
                "%s: predict skipped (no new %d-min boundary since %s).",
                self.name,
                self._interval_min,
                self._last_boundary_run,
            )
            return

        key = (now, forecast_creation)
        if self._last_predicted_key == key:
            logging.debug(
                "%s: predict skipped (already published for now=%s, creation=%s).",
                self.name,
                now,
                forecast_creation,
            )
            return

        forecast = self._fetch_forecast(now)
        if forecast is None or forecast.empty:
            logging.info(
                "%s: predict skipped: no forecast rows in [%s, %s].",
                self.name,
                now,
                now + self._horizon,
            )
            return

        field = self.context
        soil = getattr(field, "soil_simulation", None)
        if soil is None:
            logging.info("%s: predict skipped: no soil_simulation sibling.", self.name)
            return

        if getattr(soil, "_last_simulated_at", None) is None:
            logging.debug(
                "%s: predict skipped: live solver has no state yet (cold-start still running) at %s.",
                self.name,
                now,
            )
            return

        # Claim the boundary only once every transient precondition has passed (a
        # present forecast AND live soil state), so a missing-forecast or cold-start
        # tick retries on the next tick instead of silently burning the whole day,
        # while a ready tick bounds the heavy chain replay + roll-out to one attempt
        # per boundary.
        self._last_boundary_run = boundary

        try:
            et_data, seg_et = field._run_chain(forecast, publish=False)
        except Exception:  # noqa: BLE001
            logging.exception("%s: chain replay on forecast failed; skipping tick.", self.name)
            return
        if et_data.empty or et_data.shape[0] < 2:
            logging.info(
                "%s: predict skipped: chain replay returned %d row(s), need ≥ 2.",
                self.name,
                et_data.shape[0],
            )
            return

        ic_rel_sat = soil.get_rel_sat_snapshot()
        try:
            zf_timestamps, zf_trajectories, zf_snapshots, zf_diagnostics = self._solve(ic_rel_sat, et_data, seg_et)
        except Exception:  # noqa: BLE001
            logging.exception("%s: integration failed; skipping tick.", self.name)
            return

        # Se -> tension at the roll->publish boundary: the roll stays in Se (so the
        # roll-mechanics tests hold), everything downstream is hPa. The zero-flow
        # roll is computed every tick but HELD, not published here: it is the
        # fallback that keeps a forecast on the main channels if the watering-grid
        # block below throws, and the forecast for deployments with no windows.
        zf_trajectories = self._trajectories_to_tension(zf_trajectories)

        published = False
        chosen: Optional[tuple[pd.Timedelta, ...]] = None
        ladder_traj: dict = {}
        horizon_start = et_data.index[0]
        horizon_end = et_data.index[-1]

        # Watering-grid roll-out: pick the recommended candidate and publish ITS
        # roll on the main channels (not the zero-flow roll). Only when windows are
        # configured. Isolated in its own try/except so any roll-out/select/re-solve
        # failure falls through to the zero-flow fallback publish below -- a ready
        # tick always lands one complete, self-consistent forecast.
        if self._windows:
            try:
                ladder_traj = self._rollout_dispatch(ic_rel_sat, et_data, seg_et, horizon_start, horizon_end)
                # Same Se -> tension boundary as the zero-flow roll, per candidate.
                ladder_traj = {
                    candidate: (candidate_ts, self._trajectories_to_tension(probe_series))
                    for candidate, (candidate_ts, probe_series) in ladder_traj.items()
                }
                chosen = self._select(
                    self._ladder,
                    ladder_traj,
                    self._decision_probes,
                    self._threshold_hpa,
                    self._grid_mode,
                )

                # Recover the chosen candidate's snapshots + diagnostics for the main
                # channels: the ladder roll keeps only trajectories, so re-solve the
                # single chosen candidate. When it is the all-0min rung the zero-flow
                # solve already IS that roll -- reuse it, no re-solve.
                if chosen == tuple(pd.Timedelta(0) for _ in chosen):
                    ch_timestamps, ch_trajectories = zf_timestamps, zf_trajectories
                    ch_snapshots, ch_diagnostics = zf_snapshots, zf_diagnostics
                else:
                    ch_timestamps, ch_trajectories, ch_snapshots, ch_diagnostics = self._solve_candidate(
                        ic_rel_sat, chosen, et_data, seg_et, horizon_start, horizon_end
                    )
                    ch_trajectories = self._trajectories_to_tension(ch_trajectories)

                self._publish_results(
                    ch_trajectories,
                    self._probes,
                    ch_timestamps,
                    ch_snapshots,
                    ch_diagnostics,
                    forecast_creation,
                )
                published = True
            except Exception:  # noqa: BLE001
                logging.exception(
                    "%s: watering-grid roll-out/selection/publish failed "
                    "(now=%s, creation=%s); falling back to the zero-flow forecast.",
                    self.name,
                    now,
                    forecast_creation,
                )

        if not published:
            try:
                self._publish_results(
                    zf_trajectories,
                    self._probes,
                    zf_timestamps,
                    zf_snapshots,
                    zf_diagnostics,
                    forecast_creation,
                )
            except Exception:  # noqa: BLE001
                logging.exception(
                    "%s: publishing results failed; predictor channels stay stale this tick (now=%s, creation=%s).",
                    self.name,
                    now,
                    forecast_creation,
                )
                return

        self._last_predicted_key = key
        logging.info(
            "%s: predict OK: %d probes, %d rows emitted (now=%s, creation=%s).",
            self.name,
            len(self._probes),
            len(zf_timestamps),
            now,
            forecast_creation,
        )

        # Secondary watering-grid writes -- the recommendation table, the
        # all-candidate trajectory table, and the debug field plots. Best-effort and
        # only when the grid path produced a recommendation: a failure here never
        # affects the forecast already published on the main channels above.
        if published and chosen is not None:
            try:
                self._publish_recommendation(chosen, now, forecast_creation)
                trajectory_frame = self._build_trajectory_frame(ladder_traj, chosen, forecast_creation)
                self._write_trajectory_table(trajectory_frame)
                self._write_trajectory_fields(ic_rel_sat, et_data, seg_et, horizon_start, horizon_end, chosen, now)
            except Exception:  # noqa: BLE001
                logging.exception(
                    "%s: recommendation/trajectory-write failed (now=%s, creation=%s); "
                    "the forecast published on the main channels is unaffected.",
                    self.name,
                    now,
                    forecast_creation,
                )

    # PDE backend

    def _solve(
        self,
        ic_rel_sat: np.ndarray,
        et_data: pd.DataFrame,
        seg_et: dict[str, pd.DataFrame],
    ) -> tuple[
        list[pd.Timestamp],
        dict[str, list[float]],
        dict[pd.Timestamp, np.ndarray],
        dict[str, list[float]],
    ]:
        """Load IC into the predictor PDE and integrate over the forecast horizon.

        Returns ``(timestamps, trajectories, snapshots, diagnostics)``; see
        ``_integrate_horizon`` for field descriptions.
        """
        self._pde.set_state(ic_rel_sat)
        return self._integrate_horizon(et_data, seg_et)

    def _solve_candidate(
        self,
        ic_rel_sat: np.ndarray,
        candidate: tuple[pd.Timedelta, ...],
        et_data: pd.DataFrame,
        seg_et: dict[str, pd.DataFrame],
        horizon_start: pd.Timestamp,
        horizon_end: pd.Timestamp,
    ) -> tuple[
        list[pd.Timestamp],
        dict[str, list[float]],
        dict[pd.Timestamp, np.ndarray],
        dict[str, list[float]],
    ]:
        """Full-capture solve of ONE watering candidate: load the IC and integrate
        the horizon under ``candidate``'s flow schedule, returning the same
        ``(timestamps, trajectories, snapshots, diagnostics)`` 4-tuple as ``_solve``
        so ``_publish_results`` consumes it unchanged.

        The ladder roll-out keeps only trajectories per candidate; this recovers the
        chosen candidate's snapshots and diagnostics for the main forecast channels
        (see ``predict``). ``set_state`` resets the PDE, so this is independent of
        whatever state the zero-flow solve or the ladder roll left behind.
        """
        schedule = self._build_flow_schedule(self._windows, list(candidate), self._flow_m3s, horizon_start, horizon_end)
        self._pde.set_state(ic_rel_sat)
        return self._integrate_horizon(et_data, seg_et, flow_schedule=schedule)

    # Watering schedule (pure)

    @staticmethod
    def _derive_flow_m3s(
        nozzle_count: int,
        nozzle_flow_lph: float,
        total_drip_line_length_m: float,
    ) -> float:
        """Fixed design flow from the drip layout: nozzle output x count, normalized
        per out-of-plane metre of row.

        Mirrors the live sim's inline arithmetic at ``soil.py:412``
        (``SoilSimulation._compute_flux_rates``), but the meter is DERIVED from the
        layout here instead of read from the live flow meter.
        """
        flow_lpm = nozzle_count * nozzle_flow_lph / 60.0
        return flow_lpm / (60_000.0 * total_drip_line_length_m)

    @staticmethod
    def _build_flow_schedule(
        windows: list[WateringWindow],
        durations: list[pd.Timedelta],
        flow_m3s: float,
        horizon_start: pd.Timestamp,
        horizon_end: pd.Timestamp,
    ) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        """Build one candidate's "on" intervals from its per-window durations.

        ``windows`` and ``durations`` are parallel sequences (one chosen duration per
        window, this candidate's rung). Each window's ``start`` clock time is resolved
        onto ``horizon_start``'s date (site-local, tz-aware); a zero duration
        contributes no interval. ``off_ts`` clamps to ``horizon_end``. The derived
        ``flow_m3s`` is not stored per interval -- callers apply it uniformly during
        every returned interval and zero elsewhere.
        """
        intervals: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        for window, duration in zip(windows, durations):
            if duration <= pd.Timedelta(0):
                continue
            on_ts = SoilPredictor._resolve_window_start(window.start, horizon_start)
            off_ts = min(on_ts + duration, horizon_end)
            intervals.append((on_ts, off_ts))
        return intervals

    @staticmethod
    def _split_interval(
        on_intervals: list[tuple[pd.Timestamp, pd.Timestamp]],
        ts_prev: pd.Timestamp,
        ts_next: pd.Timestamp,
        flow_m3s: float,
    ) -> list[tuple[float, float]]:
        """Split ``[ts_prev, ts_next]`` at every on/off edge that falls strictly inside it.

        Returns ``[(sub_window_s, flow_m3s), ...]``, contiguous, summing to
        ``(ts_next - ts_prev).total_seconds()``; flow is ``flow_m3s`` where the
        sub-window lies inside an on-interval, else ``0.0``. Empty ``on_intervals``
        (the all-``0min`` schedule) returns a single segment covering the whole
        interval at zero flow, so the zero-flow roll integrates identically whether
        it runs through this split path or a bare ``walk_window``.
        """
        elapsed_s = (ts_next - ts_prev).total_seconds()
        if not on_intervals:
            return [(elapsed_s, 0.0)]

        edges: set[float] = {0.0, elapsed_s}
        for on_ts, off_ts in on_intervals:
            on_offset = (on_ts - ts_prev).total_seconds()
            off_offset = (off_ts - ts_prev).total_seconds()
            if 0.0 < on_offset < elapsed_s:
                edges.add(on_offset)
            if 0.0 < off_offset < elapsed_s:
                edges.add(off_offset)
        sorted_edges = sorted(edges)

        segments: list[tuple[float, float]] = []
        for edge_prev, edge_next in zip(sorted_edges[:-1], sorted_edges[1:]):
            width = edge_next - edge_prev
            if width <= 0.0:
                continue
            mid_offset = (edge_prev + edge_next) / 2.0
            mid_ts = ts_prev + pd.Timedelta(seconds=mid_offset)
            active = any(on_ts <= mid_ts < off_ts for on_ts, off_ts in on_intervals)
            segments.append((width, flow_m3s if active else 0.0))
        return segments

    # Fill-order ladder (candidate set)

    @staticmethod
    def _build_ladder(
        window_durations: list[list[pd.Timedelta]],
        grid_mode: str,
    ) -> list[tuple[pd.Timedelta, ...]]:
        """Build the candidate set: the fill-order ladder (default) or the full grid.

        ``window_durations`` is one ascending, zero-inclusive duration list per window,
        windows already ordered by ``start``. Each candidate is a tuple of one duration
        per window.

        ``fill_order`` (front-load dominance; see the PRD): window 0 contributes ALL of
        its durations (``(d0, 0, ..., 0)``, including the all-zero candidate); each later
        window i contributes only its NON-ZERO durations, meshed onto the max of every
        earlier window (``(max0, ..., max_{i-1}, d_i, 0, ..., 0)``). This excludes the
        duplicate ``(..., max_{i-1}, 0, ...)`` candidate that window i-1 already
        contributed and drops every back-loaded candidate. Count =
        ``|D0| + sum_{i>=1}(|D_i| - 1)``; total-water is strictly increasing.

        ``full``: the Cartesian product of every window's duration list.
        """
        if not window_durations:
            return [()]

        if grid_mode == "full":
            return list(itertools.product(*window_durations))

        if grid_mode != "fill_order":
            raise ValueError(f"Unknown grid_mode {grid_mode!r}; expected 'fill_order' or 'full'.")

        n = len(window_durations)
        maxima = [max(durations) for durations in window_durations]
        ladder: list[tuple[pd.Timedelta, ...]] = []

        for d0 in window_durations[0]:
            ladder.append((d0,) + (pd.Timedelta(0),) * (n - 1))

        for i in range(1, n):
            for d_i in window_durations[i]:
                if d_i <= pd.Timedelta(0):
                    continue
                candidate = tuple(maxima[:i]) + (d_i,) + (pd.Timedelta(0),) * (n - i - 1)
                ladder.append(candidate)

        return ladder

    @staticmethod
    def _check_combo_cap(
        ladder: list[tuple[pd.Timedelta, ...]],
        combo_cap: int,
        log_name: str = "",
    ) -> None:
        """Fail fast at ``configure()`` if the (static) ladder length exceeds ``combo_cap``,
        instead of silently skipping candidates at runtime.
        """
        if len(ladder) > combo_cap:
            raise ValueError(
                f"{log_name}: ladder has {len(ladder)} candidates, exceeding "
                f"combo_cap={combo_cap}; reduce the per-window durations lists, "
                "raise combo_cap, or drop windows."
            )

    # Prefix-shared roll-out (the caterpillar)

    def _roll_segment(
        self,
        idx: pd.DatetimeIndex,
        et_data: pd.DataFrame,
        seg_et: dict[str, pd.DataFrame],
        on_intervals: list[tuple[pd.Timestamp, pd.Timestamp]],
        snapshot_sink: Optional[Callable[[pd.Timestamp], None]] = None,
    ) -> tuple[list[pd.Timestamp], dict[str, list[float]]]:
        """Walk the PDE across ``idx`` (>=1 forecast timestamps; the live PDE state is
        already the state at ``idx[0]``), applying ``on_intervals`` inside each
        ``(t, t+dt)`` sub-interval via ``_split_interval``. Returns per-forecast-
        timestamp Se at every probe, including ``idx[0]`` (sampled as-is, no walk).

        Shared by the prefix roll and every per-window sweep in ``_rollout_ladder``,
        and by ``_rollout_independent``'s single full-horizon roll.

        ``snapshot_sink``, when given, is called with each recorded forecast timestamp
        right after that state is reached -- the live ``self._pde`` is the field at
        that timestamp, so the sink can read ``self._pde.snapshot()``. Only the debug
        field-plot re-roll passes it; the forecast/recommendation paths leave it
        ``None`` (no per-step cost).
        """
        timestamps: list[pd.Timestamp] = [idx[0]]
        trajectories: dict[str, list[float]] = {p.channel_id: [self._pde.sample(p)] for p in self._probes}
        if snapshot_sink is not None:
            snapshot_sink(idx[0])

        for ts_prev, ts_next in zip(idx[:-1], idx[1:]):
            elapsed_s = (ts_next - ts_prev).total_seconds()
            if elapsed_s <= 0:
                timestamps.append(ts_next)
                for p in self._probes:
                    trajectories[p.channel_id].append(self._pde.sample(p))
                if snapshot_sink is not None:
                    snapshot_sink(ts_next)
                continue

            seg_evap, seg_transp = self._segment_flux_dicts(seg_et, ts_next)
            rain_flux = self._rain_flux(et_data, ts_next, elapsed_s)
            sub_segments = self._split_interval(on_intervals, ts_prev, ts_next, self._flow_m3s)
            for sub_window_s, sub_flow_m3s in sub_segments:
                if sub_window_s <= 0.0:
                    continue
                sub_rates = FluxRates(
                    seg_evap=seg_evap,
                    seg_transp=seg_transp,
                    flow_m3s=sub_flow_m3s,
                    rain_flux=rain_flux,
                )
                self._pde.walk_window(
                    rates=sub_rates,
                    window_s=sub_window_s,
                    accept_at_dt_min=True,
                    log_name=self.name,
                )

            timestamps.append(ts_next)
            for p in self._probes:
                trajectories[p.channel_id].append(self._pde.sample(p))
            if snapshot_sink is not None:
                snapshot_sink(ts_next)

        return timestamps, trajectories

    @staticmethod
    def _extend_trajectory(
        base_timestamps: list[pd.Timestamp],
        base_trajectories: dict[str, list[float]],
        tail_timestamps: list[pd.Timestamp],
        tail_trajectories: dict[str, list[float]],
    ) -> tuple[list[pd.Timestamp], dict[str, list[float]]]:
        """Concatenate ``base`` (up to and including a window start) with ``tail``
        (from that same window start to its segment end), dropping the tail's
        duplicated leading timestamp.
        """
        timestamps = list(base_timestamps) + list(tail_timestamps[1:])
        trajectories = {
            channel_id: list(base_trajectories[channel_id]) + list(tail_trajectories[channel_id][1:])
            for channel_id in base_trajectories
        }
        return timestamps, trajectories

    def _rollout_ladder(
        self,
        ic_rel_sat: np.ndarray,
        ladder: list[tuple[pd.Timedelta, ...]],
        et_data: pd.DataFrame,
        seg_et: dict[str, pd.DataFrame],
        flow_m3s: float,
        horizon_start: pd.Timestamp,
        horizon_end: pd.Timestamp,
    ) -> dict[tuple[pd.Timedelta, ...], tuple[list[pd.Timestamp], dict[str, list[float]]]]:
        """Caterpillar roll-out (``self._grid_mode == "fill_order"``): integrate the
        shared prefix once, then sweep each window's ladder-contributed durations
        from a save of the max-prefix branch, saving/restoring branch state with
        ``save_state_blob``/``load_state_blob`` (never ``snapshot``/``set_state`` --
        the latter drop the ``surface_h`` ponds that watering fills; see the module
        docstring / PRD).

        The fill-order chain's prefix-sharing only applies to the ``fill_order``
        candidate set: for ``self._grid_mode == "full"`` (the full Cartesian
        product, not a single chain) every candidate is rolled independently via
        ``_rollout_independent`` instead, with no prefix sharing.

        ``ladder`` is ``_build_ladder``'s output; ``self._windows`` (ordered by
        ``start``) supplies the window clock times. Window starts are assumed to fall
        exactly on a forecast timestamp in ``et_data.index`` (the common on-the-hour
        case); if a window start does not land on a forecast timestamp, the nearest
        forecast timestamp at or before it is used as the segment boundary instead.

        Returns ``{candidate: (timestamps, {probe_id: [Se, ...]})}`` for every rung.
        """
        windows = self._windows
        idx = et_data.index
        results: dict[tuple[pd.Timedelta, ...], tuple[list[pd.Timestamp], dict[str, list[float]]]] = {}

        if self._grid_mode == "full" or not windows:
            # No caterpillar prefix-sharing for the full Cartesian product (or the
            # no-windows degenerate case): roll every candidate independently.
            for candidate in ladder:
                results[candidate] = self._rollout_independent(
                    ic_rel_sat, candidate, et_data, seg_et, flow_m3s, horizon_start, horizon_end
                )
            return results

        self._pde.set_state(ic_rel_sat)

        maxima = [max(durations) for durations in self._window_durations]
        window_starts = [self._resolve_window_start(w.start, horizon_start) for w in windows]

        def _floor_idx(ts: pd.Timestamp) -> pd.Timestamp:
            eligible = idx[idx <= ts]
            return eligible[-1] if len(eligible) > 0 else idx[0]

        segment_bounds = [_floor_idx(ts) for ts in window_starts] + [horizon_end if horizon_end in idx else idx[-1]]

        # The caterpillar's prefix-sharing is only valid when the floored segment
        # bounds are STRICTLY increasing: each window's floored start must fall
        # strictly after the previous one, and the horizon end strictly after the
        # last window. That can fail two ways -- two window starts flooring to the
        # same forecast timestamp (a window pair inside one forecast interval), or a
        # window resolving out of temporal order (e.g. a near-midnight window rolled
        # to the next day landing after a later-clock-time window). In either case
        # the segment-based save/restore would silently drop or misattribute a
        # window's water, so fall back to correct (unshared) independent rolls.
        if not all(segment_bounds[k] < segment_bounds[k + 1] for k in range(len(segment_bounds) - 1)):
            logging.debug(
                "%s: caterpillar segment bounds not strictly increasing (%s); "
                "falling back to independent per-candidate rolls.",
                self.name,
                segment_bounds,
            )
            for candidate in ladder:
                results[candidate] = self._rollout_independent(
                    ic_rel_sat, candidate, et_data, seg_et, flow_m3s, horizon_start, horizon_end
                )
            return results

        prefix_idx = idx[idx <= segment_bounds[0]]
        prefix_timestamps, prefix_trajectories = self._roll_segment(prefix_idx, et_data, seg_et, [])
        prev_blob = self._pde.save_state_blob()

        for i, window in enumerate(windows):
            seg_start = segment_bounds[i]
            seg_end = segment_bounds[i + 1]
            seg_idx = idx[(idx >= seg_start) & (idx <= seg_end)]

            durations = self._window_durations[i]
            sweep = durations if i == 0 else [d for d in durations if d > pd.Timedelta(0)]
            max_duration = maxima[i]

            for d_i in sweep:
                self._pde.load_state_blob(prev_blob)
                on_intervals = self._build_flow_schedule([window], [d_i], flow_m3s, seg_start, horizon_end)
                tail_timestamps, tail_trajectories = self._roll_segment(
                    idx[idx >= seg_start], et_data, seg_et, on_intervals
                )
                full_timestamps, full_trajectories = self._extend_trajectory(
                    prefix_timestamps, prefix_trajectories, tail_timestamps, tail_trajectories
                )
                # Positions before i carry EACH earlier window's OWN max (maxima[j]),
                # not window i's max -- the state already reflects every earlier
                # window at its own max (via the max-branch save below), so the key
                # must label that accurately to match _build_ladder's candidates.
                candidate = tuple(
                    maxima[j] if j < i else (d_i if j == i else pd.Timedelta(0)) for j in range(len(windows))
                )
                results[candidate] = (full_timestamps, full_trajectories)

                if d_i == max_duration and i + 1 < len(windows):
                    self._pde.load_state_blob(prev_blob)
                    on_intervals_seg = self._build_flow_schedule([window], [d_i], flow_m3s, seg_start, seg_end)
                    seg_timestamps, seg_trajectories = self._roll_segment(seg_idx, et_data, seg_et, on_intervals_seg)
                    prefix_timestamps, prefix_trajectories = self._extend_trajectory(
                        prefix_timestamps, prefix_trajectories, seg_timestamps, seg_trajectories
                    )
                    prev_blob = self._pde.save_state_blob()

        return results

    def _rollout_independent(
        self,
        ic_rel_sat: np.ndarray,
        candidate: tuple[pd.Timedelta, ...],
        et_data: pd.DataFrame,
        seg_et: dict[str, pd.DataFrame],
        flow_m3s: float,
        horizon_start: pd.Timestamp,
        horizon_end: pd.Timestamp,
    ) -> tuple[list[pd.Timestamp], dict[str, list[float]]]:
        """Reference roll-out for one candidate: reset to the IC and integrate the
        whole horizon in a single pass with no prefix sharing. Ground truth that
        ``_rollout_ladder``'s per-candidate trajectory must match.
        """
        self._pde.set_state(ic_rel_sat)
        on_intervals = self._build_flow_schedule(self._windows, list(candidate), flow_m3s, horizon_start, horizon_end)
        idx = et_data.index
        return self._roll_segment(idx, et_data, seg_et, on_intervals)

    def _rollout_dispatch(
        self,
        ic_rel_sat: np.ndarray,
        et_data: pd.DataFrame,
        seg_et: dict[str, pd.DataFrame],
        horizon_start: pd.Timestamp,
        horizon_end: pd.Timestamp,
    ) -> dict[tuple[pd.Timedelta, ...], tuple[list[pd.Timestamp], dict[str, list[float]]]]:
        """Roll every ladder candidate; return ``{candidate: (timestamps, trajectories)}``.

        Dispatches on ``self._parallel``: independent parallel rolls
        (``_rollout_parallel``) vs the sequential prefix-shared caterpillar
        (``_rollout_ladder``). The two are equal within solver tolerance -- parallel
        is a pure wall-time win, not a change to what is stored (``docs/adr/0005-...``).
        On any parallel-execution failure the run degrades to the caterpillar and
        logs it; a parallelism failure must never abort the daily forecast.
        """
        if self._parallel:
            try:
                return self._rollout_parallel(ic_rel_sat, et_data, seg_et, horizon_start, horizon_end)
            except Exception:  # noqa: BLE001
                logging.exception(
                    "%s: parallel roll-out failed; falling back to the sequential caterpillar for this run.",
                    self.name,
                )
        return self._rollout_ladder(
            ic_rel_sat, self._ladder, et_data, seg_et, self._flow_m3s, horizon_start, horizon_end
        )

    def _rollout_parallel(
        self,
        ic_rel_sat: np.ndarray,
        et_data: pd.DataFrame,
        seg_et: dict[str, pd.DataFrame],
        horizon_start: pd.Timestamp,
        horizon_end: pd.Timestamp,
    ) -> dict[tuple[pd.Timedelta, ...], tuple[list[pd.Timestamp], dict[str, list[float]]]]:
        """Roll every ladder candidate as an independent parallel roll across an
        in-component spawn ``ProcessPoolExecutor``. Each worker rebuilds the PDE once
        from the pickled ``MeshConfig`` + ``PDEConfig`` (spawn-safe) and rolls its
        assigned candidates via ``_rollout_independent``; the parent gathers the
        ``{candidate: (timestamps, trajectories)}`` map and does every downstream
        step (select, frame, write) serially. Same candidate set and same stored
        trajectories as the caterpillar within solver tolerance (``docs/adr/0005-...``).

        The pool is created and torn down per call -- daily cadence makes the setup
        cost negligible. Raises on pool/worker failure so ``_rollout_dispatch`` can
        degrade to the caterpillar.
        """
        ladder = self._ladder
        # No point spawning more workers than candidates; always at least one.
        n_workers = max(1, min(self._max_workers, len(ladder)))
        ctx = multiprocessing.get_context("spawn")
        # The chain-replay frames carry lories Constant column labels (e.g.
        # Weather.PRECIPITATION). Constant is a str subclass whose __new__ takes
        # (type, key, ...), which clashes with how pickle reconstructs a str
        # subclass: unpickling in a spawned worker calls Constant(<value>) with
        # key=None and raises "Constant '...' is None", killing the worker. Coerce
        # every column label to a plain str for transport; a Constant equals its
        # key str, so the worker's Constant-keyed lookups (_rain_flux etc.) still
        # match. The caterpillar path keeps the original frames (no pickling).
        et_data = _stringify_columns(et_data)
        seg_et = {name: _stringify_columns(frame) for name, frame in seg_et.items()}
        initargs = (
            self._mesh_config,
            self._ode_config,
            self.REL_SAT_NAME,
            self.name,
            self._probes,
            self._windows,
            self._flow_m3s,
            self._grid_mode,
            ic_rel_sat,
            et_data,
            seg_et,
            horizon_start,
            horizon_end,
        )
        results: dict[tuple[pd.Timedelta, ...], tuple[list[pd.Timestamp], dict[str, list[float]]]] = {}
        with ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=ctx,
            initializer=_worker_init,
            initargs=initargs,
        ) as pool:
            futures = [pool.submit(_worker_roll, candidate) for candidate in ladder]
            for fut in as_completed(futures):
                candidate, result = fut.result()
                results[candidate] = result
        logging.debug(
            "%s: parallel roll-out complete: %d candidates across %d workers.",
            self.name,
            len(results),
            n_workers,
        )
        return results

    @staticmethod
    def _resolve_window_start(start: datetime.time, horizon_start: pd.Timestamp) -> pd.Timestamp:
        """Resolve a window's clock time onto ``horizon_start``'s date, rolling
        forward a **calendar** day if that time already elapsed before
        ``horizon_start``. The roll-forward re-resolves the wall-clock fields on the
        next calendar day rather than adding a fixed ``Timedelta(days=1)``, so the
        result stays at the intended local clock time across a DST transition (a
        fixed 24h add would land an hour off on the spring-forward / fall-back night).
        The single canonical resolver; ``_build_flow_schedule`` calls it too.
        """
        on_ts = horizon_start.replace(
            hour=start.hour,
            minute=start.minute,
            second=start.second,
            microsecond=start.microsecond,
        )
        if on_ts < horizon_start:
            on_ts = (horizon_start + pd.Timedelta(days=1)).replace(
                hour=start.hour,
                minute=start.minute,
                second=start.second,
                microsecond=start.microsecond,
            )
        return on_ts

    def _trajectories_to_tension(self, trajectories: dict[str, list[float]]) -> dict[str, list[float]]:
        """Convert a roll's per-probe Se trajectories to water tension (hPa).

        The PDE roll (``_roll_segment`` / ``_integrate_horizon``) works in the solver's
        native relative saturation Se, and the roll-mechanics tests compare it against
        ``SoilPDECore.sample`` (also Se). This is the single boundary where the roll
        output crosses into the tension-native decision + publish layer; ``psi_from_se``
        returns the signed matric potential (negative hPa; drier soil -> more
        negative). It is published unchanged; the scorer compares its magnitude to
        ``threshold_hpa`` (see ``_score_candidate``).
        """
        model = self._pde.soil_model
        return {
            channel_id: [float(v) for v in model.psi_from_se(np.asarray(values, dtype=float))]
            for channel_id, values in trajectories.items()
        }

    # Tension conversion, candidate scoring, and ladder selection (pure)

    @staticmethod
    def _score_candidate(
        trajectory: tuple[list[pd.Timestamp], dict[str, list[float]]],
        decision_probes: list[str],
        threshold_hpa: float,
    ) -> float:
        """RMS distance of a candidate's water tension from the setpoint
        ``threshold_hpa``, over the whole horizon, pooled across the decision
        probes. Lower is better; ``_select`` takes the argmin.

        The trajectory values are water tension (hPa), converted from the solver's
        native Se at the roll->publish boundary in ``predict()`` (see
        ``_trajectories_to_tension``). ``threshold_hpa`` is read here as a TARGET
        tension (setpoint), not a ceiling: tension above OR below it adds to the
        score, so the recommended candidate is the one that tracks the setpoint
        most closely.

        Probes not present in ``decision_probes`` are ignored. Returns ``+inf`` if
        ``decision_probes`` selects no probe present in the trajectory, so a
        misconfigured probe subset scores as WORST (fail safe) and can never be the
        argmin. ``configure()`` additionally hard-fails when the configured
        ``decision_probes`` resolve to zero known ids.

        This is the single scoring seam: swap the formula here -- for example to a
        one-sided ceiling ``max(0, tension - threshold)`` -- without touching the
        selector or the publish path.
        """
        _timestamps, probe_series = trajectory
        deviations: list[np.ndarray] = []
        for channel_id in decision_probes:
            tension_values = probe_series.get(channel_id)
            if not tension_values:
                continue
            # Trajectories are signed matric potential (negative hPa); compare their
            # suction MAGNITUDE against the positive ``threshold_hpa`` setpoint, so
            # the setpoint stays a plain positive dryness target.
            deviations.append(np.abs(np.asarray(tension_values, dtype=float)) - threshold_hpa)
        if not deviations:
            return float("inf")
        stacked = np.concatenate(deviations)
        return float(np.sqrt(np.mean(np.square(stacked))))

    @classmethod
    def _select(
        cls,
        ladder: list[tuple[pd.Timedelta, ...]],
        trajectories: dict[tuple[pd.Timedelta, ...], tuple[list[pd.Timestamp], dict[str, list[float]]]],
        decision_probes: list[str],
        threshold_hpa: float,
        grid_mode: str,
    ) -> tuple[pd.Timedelta, ...]:
        """Select the recommended candidate: the rung whose water-tension
        trajectory tracks the ``threshold_hpa`` setpoint most closely, scored by
        ``_score_candidate`` (RMS-to-setpoint, lower is better).

        Both grid modes reduce to the same rule -- score every candidate and take
        the argmin, breaking ties by least total watering (``_total_minutes``) for a
        deterministic pick. There is no feasibility test and no status: the ceiling
        and the monotone-feasibility walk are gone.

        ``fill_order`` is an APPROXIMATE search: it scores only the front-loaded
        ladder subset, not the full Cartesian grid, so the argmin is
        best-on-the-ladder, not a proven global optimum (the RMS-to-setpoint score
        is not monotone in total water, so the true optimum can be interior). Use
        ``grid_mode = "full"`` when the recommendation must be exact.
        """
        if not ladder:
            raise ValueError("_select requires a non-empty ladder.")
        if grid_mode not in ("fill_order", "full"):
            raise ValueError(f"Unknown grid_mode {grid_mode!r}; expected 'fill_order' or 'full'.")

        scores = {c: cls._score_candidate(trajectories[c], decision_probes, threshold_hpa) for c in ladder}
        return min(ladder, key=lambda c: (scores[c], cls._total_minutes(c)))

    @staticmethod
    def _total_minutes(candidate: tuple[pd.Timedelta, ...]) -> float:
        """Total watering minutes across a candidate's per-window durations."""
        return sum((d.total_seconds() / 60.0 for d in candidate), 0.0)

    # Forecast retrieval

    def _fetch_forecast(self, now: pd.Timestamp) -> Optional[pd.DataFrame]:
        """Read the in-memory forecast cache via ``data.to_frame()`` and slice to ``[now, now+horizon]``."""
        weather = getattr(self.context, "weather", None)
        forecast_sub = getattr(weather, "forecast", None)
        if forecast_sub is None or not forecast_sub.is_enabled():
            return None
        try:
            df = forecast_sub.data.to_frame(unique=False)
        except Exception as e:  # noqa: BLE001
            logging.warning("%s: forecast read failed: %s", self.name, e)
            return None
        if df.empty:
            return None
        # Align tz: forecast index is location-tz-aware; ``now`` matches.
        end = now + self._horizon
        sliced = df.loc[(df.index >= now) & (df.index <= end)]
        return sliced if not sliced.empty else None

    # PDE integration over horizon

    def _integrate_horizon(
        self,
        et_data: pd.DataFrame,
        seg_et: dict[str, pd.DataFrame],
        flow_schedule: Optional[list[tuple[pd.Timestamp, pd.Timestamp]]] = None,
    ) -> tuple[
        list[pd.Timestamp],
        dict[str, list[float]],
        dict[pd.Timestamp, np.ndarray],
        dict[str, list[float]],
    ]:
        """Step the predictor PDE through every (t, t+Δt) forecast interval.

        ``flow_schedule`` is a candidate's "on" intervals (see ``_build_flow_schedule``);
        ``None``/empty applies no irrigation, reproducing today's zero-flow roll exactly
        (each forecast interval is a single ``walk_window`` call at ``flow_m3s=0.0``).
        When non-empty, each forecast interval is split at every on/off edge that falls
        strictly inside it (``_split_interval``) and walked once per sub-segment at that
        sub-segment's flow, so a sub-hourly duration integrates the exact water volume.
        Trajectories/diagnostics/snapshots are still sampled once per forecast interval,
        at ``ts_next`` -- the split is for integration accuracy, not extra sample points.

        Returns ``(timestamps, trajectories, snapshots, diagnostics)``.
        ``diagnostics`` values are kg/(m²·h) flux densities; NaN at the IC row (no interval yet).
        """
        on_intervals = flow_schedule or []
        idx = et_data.index
        timestamps: list[pd.Timestamp] = []
        trajectories: dict[str, list[float]] = {p.channel_id: [] for p in self._probes}
        snapshots: dict[pd.Timestamp, np.ndarray] = {}
        diagnostics: dict[str, list[float]] = {c.key: [] for c in _DIAGNOSTIC_CONSTANTS}
        last_save_ts: Optional[pd.Timestamp] = None
        capture_snapshots = self._save_state or self._save_plot

        def _maybe_snapshot(ts: pd.Timestamp) -> None:
            nonlocal last_save_ts
            if not capture_snapshots:
                return
            if last_save_ts is not None and (ts - last_save_ts) < self._save_freq:
                return
            snapshots[ts] = self._pde.snapshot()
            last_save_ts = ts

        if len(idx) > 0:
            timestamps.append(idx[0])
            for p in self._probes:
                trajectories[p.channel_id].append(self._pde.sample(p))
            for key in diagnostics:
                diagnostics[key].append(float("nan"))
            _maybe_snapshot(idx[0])

        for ts_prev, ts_next in zip(idx[:-1], idx[1:]):
            elapsed_s = (ts_next - ts_prev).total_seconds()
            if elapsed_s <= 0:
                continue

            seg_evap, seg_transp = self._segment_flux_dicts(seg_et, ts_next)
            rain_flux = self._rain_flux(et_data, ts_next, elapsed_s)
            storage_before = self._total_water() + self._pde.surface_water()

            sub_segments = self._split_interval(on_intervals, ts_prev, ts_next, self._flow_m3s)
            clip_total = ClipDiagnostics()
            irrigated_mass = 0.0
            for sub_window_s, sub_flow_m3s in sub_segments:
                if sub_window_s <= 0.0:
                    continue
                sub_rates = FluxRates(
                    seg_evap=seg_evap,
                    seg_transp=seg_transp,
                    flow_m3s=sub_flow_m3s,
                    rain_flux=rain_flux,
                )
                walk = self._pde.walk_window(
                    rates=sub_rates,
                    window_s=sub_window_s,
                    accept_at_dt_min=True,
                    log_name=self.name,
                )
                clip_total.add(walk.clip)
                irrigated_mass += sub_flow_m3s * sub_window_s

            timestamps.append(ts_next)
            for p in self._probes:
                trajectories[p.channel_id].append(self._pde.sample(p))
            delta_storage = self._total_water() + self._pde.surface_water() - storage_before
            # Time-weighted average flow over the interval: rates.flow_m3s * elapsed_s
            # reproduces the same irrigated mass that _compute_diagnostics' mass-balance
            # math expects, whether the interval was split or not.
            interval_flow_m3s = irrigated_mass / elapsed_s if elapsed_s > 0 else 0.0
            interval_rates = FluxRates(
                seg_evap=seg_evap,
                seg_transp=seg_transp,
                flow_m3s=interval_flow_m3s,
                rain_flux=rain_flux,
            )
            interval_diag = self._compute_diagnostics(
                interval_rates,
                delta_storage,
                elapsed_s,
                clip_total,
            )
            for key in diagnostics:
                diagnostics[key].append(interval_diag.get(key, float("nan")))
            _maybe_snapshot(ts_next)

        if capture_snapshots and timestamps:
            final_ts = timestamps[-1]
            if final_ts not in snapshots:
                snapshots[final_ts] = self._pde.snapshot()

        return timestamps, trajectories, snapshots, diagnostics

    @staticmethod
    def _segment_flux_dicts(
        seg_et: dict[str, pd.DataFrame],
        ts: pd.Timestamp,
    ) -> tuple[dict[str, float], dict[str, float]]:
        seg_evap: dict[str, float] = {}
        seg_transp: dict[str, float] = {}
        for name, frame in seg_et.items():
            if ts not in frame.index:
                continue
            evap = max(0.0, float(frame.loc[ts, "evap"]))
            transp = max(0.0, float(frame.loc[ts, "transp"]))
            if evap > 0.0:
                seg_evap[name] = evap
            if transp > 0.0:
                seg_transp[name] = transp
        return seg_evap, seg_transp

    @staticmethod
    def _rain_flux(et_data: pd.DataFrame, ts: pd.Timestamp, elapsed_s: float) -> float:
        col = Weather.PRECIPITATION
        if elapsed_s <= 0 or col not in et_data.columns or ts not in et_data.index:
            return 0.0
        precip = et_data.loc[ts, col]
        if pd.isna(precip) or precip <= 0:
            return 0.0
        return float(precip) / elapsed_s  # mm/s == kg/(m²·s)

    # Channel publishing

    def _publish_results(
        self,
        trajectories: dict[str, list[float]],
        probes: list[ProbeSpec],
        timestamps: list[pd.Timestamp],
        snapshots: dict[pd.Timestamp, np.ndarray],
        diagnostics: dict[str, list[float]],
        forecast_creation: pd.Timestamp,
    ) -> None:
        if not timestamps:
            return

        index = pd.DatetimeIndex(timestamps, name="timestamp")

        creation_series = pd.Series(forecast_creation, index=index)
        self.data[self._TIMESTAMP_CREATION_KEY].set(index[0], creation_series)

        for probe in probes:
            key = self._channel_keys.get(probe.channel_id)
            if key is None:
                continue
            traj = trajectories.get(probe.channel_id, [])
            if len(traj) != len(index):
                continue
            self.data[key].set(index[0], pd.Series(traj, index=index, dtype=float))

        for constant in _DIAGNOSTIC_CONSTANTS:
            values = diagnostics.get(constant.key, [])
            if len(values) != len(index):
                continue
            self.data[constant.key].set(
                index[0],
                pd.Series(values, index=index, dtype=float),
            )

        if not snapshots:
            return

        save_index = pd.DatetimeIndex(sorted(snapshots), name="timestamp")

        if self._save_state:
            state_values = [self._encode_state(snapshots[t]) for t in save_index]
            self.data[self._STATE_CHANNEL_KEY].set(
                save_index[0],
                pd.Series(state_values, index=save_index, dtype=object),
            )

        if self._save_plot:
            plot_values: list[bytes] = []
            for t in save_index:
                try:
                    plot_values.append(self._render_snapshot_png(snapshots[t], t))
                except Exception:  # noqa: BLE001
                    logging.exception(
                        "%s: predict_plot render failed at %s; skipping remaining plot snapshots this tick.",
                        self.name,
                        t,
                    )
                    plot_values = []
                    break
            if plot_values:
                self.data[self._PLOT_CHANNEL_KEY].set(
                    save_index[0],
                    pd.Series(plot_values, index=save_index, dtype=object),
                )

    # Recommendation + trajectory-table publishing (the watering-grid outputs)

    def _publish_recommendation(
        self,
        chosen: tuple[pd.Timedelta, ...],
        run_timestamp: pd.Timestamp,
        forecast_creation: pd.Timestamp,
    ) -> None:
        """Publish the chosen candidate on the recommendation channels (normal
        auto-logged path): one row per run at ``run_timestamp`` in the dedicated
        ``_RECOMMEND_TABLE_NAME`` table. That table is keyed by run time only (no
        ``timestamp_creation`` composite PK), so each run appends a row and the
        recommendation history accumulates through the auto-logger -- no PK partner
        to keep in sync, unlike the forecast table.

        Per-window minutes (unused windows -- i.e. ``chosen`` shorter than
        ``_max_windows``, which cannot happen given the configure()-time length
        check, but the fill below is defensive -- get ``0.0``, not the trajectory
        table's ``-1`` sentinel; a "no window configured" reading of 0 minutes is
        unambiguous on the advisory channel, whereas the sentinel is only meaningful
        alongside the trajectory table's fixed-arity PK columns) and the total
        watering minutes. A ``recommend_total_min`` of ``0`` is the "do nothing"
        recommendation; there is no separate status channel.

        ``forecast_creation`` is accepted for call-site symmetry with the trajectory
        write but is not persisted here: the recommendation row is keyed by run time,
        not by the forecast issue.
        """
        total_minutes = self._total_minutes(chosen)

        for i, key in enumerate(self._recommend_window_keys):
            minutes = chosen[i].total_seconds() / 60.0 if i < len(chosen) else 0.0
            self.data[key].set(run_timestamp, minutes)

        self.data[self._RECOMMEND_TOTAL_KEY].set(run_timestamp, total_minutes)
        logging.debug(
            "%s: recommendation published: total=%.1fmin (now=%s).",
            self.name,
            total_minutes,
            run_timestamp,
        )

    def _build_trajectory_frame(
        self,
        ladder_trajectories: dict[tuple[pd.Timedelta, ...], tuple[list[pd.Timestamp], dict[str, list[float]]]],
        chosen: tuple[pd.Timedelta, ...],
        forecast_creation: pd.Timestamp,
    ) -> pd.DataFrame:
        """Build the all-candidate trajectory frame: index = forecast timestamps
        (duplicated across candidates); columns = the trajectory channel KEYS
        (``_TRAJ_TIMESTAMP_CREATION_KEY``, ``w0_min ... w{max_windows-1}_min``,
        ``is_recommended``, ``traj_<probe>`` for every probe present in
        ``ladder_trajectories``).

        Every candidate contributes one row per forecast timestamp: its
        per-window minutes (``_UNUSED_WINDOW_SENTINEL`` for any window index
        beyond ``len(candidate)``, so the PK column set is stable regardless of
        how many windows a deployment configures), ``is_recommended`` (True only
        for ``chosen``'s rows), and its per-probe signed matric potential
        (negative hPa, from the retention model's ``psi_from_se``).

        Pure and unit-testable: no ``Channel``/connector access here. Column
        NAMES are the bare channel keys, not full channel ids --
        ``_write_trajectory_table`` renames them to ids (``connector.write``
        matches on ``resource.id in data.columns``) right before the write.
        """
        rows: list[dict[str, Any]] = []
        index: list[pd.Timestamp] = []

        for candidate, (timestamps, probe_series) in ladder_trajectories.items():
            is_recommended = candidate == chosen
            window_minutes = [
                candidate[i].total_seconds() / 60.0 if i < len(candidate) else _UNUSED_WINDOW_SENTINEL
                for i in range(self._max_windows)
            ]
            for t_idx, ts in enumerate(timestamps):
                row: dict[str, Any] = {self._TRAJ_TIMESTAMP_CREATION_KEY: forecast_creation}
                for key, minutes in zip(self._traj_window_keys, window_minutes):
                    row[key] = minutes
                row[self._TRAJ_IS_RECOMMENDED_KEY] = is_recommended
                for probe_id, key in self._traj_channel_keys.items():
                    values = probe_series.get(probe_id)
                    row[key] = values[t_idx] if values is not None and t_idx < len(values) else np.nan
                rows.append(row)
                index.append(ts)

        columns = [self._TRAJ_TIMESTAMP_CREATION_KEY, *self._traj_window_keys, self._TRAJ_IS_RECOMMENDED_KEY]
        columns.extend(self._traj_channel_keys.values())
        if not rows:
            return pd.DataFrame(columns=columns)

        frame = pd.DataFrame.from_records(rows, index=pd.DatetimeIndex(index, name="timestamp"))
        return frame.loc[:, columns]

    def _write_trajectory_table(self, frame: pd.DataFrame) -> None:
        """Direct-write the all-candidate trajectory frame to the configured
        `logger` connector, once. Renames ``frame``'s bare channel-key columns
        to full channel ids (what ``connector.write`` matches against) first.
        Warns and skips (never raises) if `logger` is not configured or the
        connector cannot be resolved/is not a writable connector -- a grid
        persistence failure must never abort the forecast on the main channels.
        """
        if self._logger_id is None:
            return
        if frame.empty:
            logging.debug("%s: trajectory frame empty; skipping direct write.", self.name)
            return

        connector = self._resolve_logger_connector(self._logger_id)
        if connector is None:
            logging.warning(
                "%s: logger connector '%s' not found; skipping the trajectory-table direct write.",
                self.name,
                self._logger_id,
            )
            return
        if not hasattr(connector, "write"):
            logging.warning(
                "%s: logger connector '%s' (%s) has no write(); skipping the trajectory-table direct write.",
                self.name,
                self._logger_id,
                type(connector).__name__,
            )
            return

        id_by_key = {
            self._TRAJ_TIMESTAMP_CREATION_KEY: self.data[self._TRAJ_TIMESTAMP_CREATION_KEY].id,
            self._TRAJ_IS_RECOMMENDED_KEY: self.data[self._TRAJ_IS_RECOMMENDED_KEY].id,
        }
        for key in self._traj_window_keys:
            id_by_key[key] = self.data[key].id
        for key in self._traj_channel_keys.values():
            id_by_key[key] = self.data[key].id

        write_frame = frame.rename(columns=id_by_key)
        try:
            connector.write(write_frame)
        except Exception:  # noqa: BLE001
            logging.exception(
                "%s: direct write of the trajectory table to logger '%s' failed.",
                self.name,
                self._logger_id,
            )
            return
        logging.info(
            "%s: trajectory table written: %d rows to logger '%s'.",
            self.name,
            len(write_frame),
            self._logger_id,
        )

    def _resolve_logger_connector(self, logger_id: str) -> Optional[Any]:
        """Resolve the connector for the trajectory direct-write.

        Prefer the connector the trajectory channels already bound at
        registration: ``ChannelConnector`` walks the component path to reach a
        root-level ``[connectors.<id>]`` connector (the common case for a shared
        SQL logger). ``self.connectors``' bare-key/id lookup is **component
        scoped** -- ``RegistratorAccess.__getattr__`` only sees this component's
        own connector map and ``__getitem__`` prefixes a dot-less id with this
        component's id -- so a root-level connector is unreachable from a nested
        predictor that way. Reusing the channel's resolution is what makes
        ``logger = "<bare id>"`` work for a deeply nested predictor; the id-based
        lookups stay as fallbacks (a full dotted ``logger`` value, or before the
        channels are bound).
        """
        connector = self._logger_connector_from_channel()
        if connector is not None:
            return connector
        try:
            connector = getattr(self.connectors, logger_id)
        except AttributeError:
            connector = None
        if connector is not None:
            return connector
        try:
            return self.connectors[logger_id]
        except (KeyError, TypeError):
            return None

    def _logger_connector_from_channel(self) -> Optional[Any]:
        """The connector a trajectory channel already resolved against the root
        context (``ChannelConnector``'s component-path walk), or ``None`` if the
        channels are not registered / not bound yet. Defensive: a resolution
        failure must degrade to the id-based fallback, never raise.
        """
        try:
            return self.data[self._TRAJ_TIMESTAMP_CREATION_KEY].logger._get_registrator()
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _candidate_slug(candidate: tuple[pd.Timedelta, ...]) -> str:
        """Filesystem-safe per-window-minutes slug for a ladder candidate, e.g.
        ``(30 min, 0 min)`` -> ``30-0min``. Used in debug field-plot filenames."""
        return "-".join(f"{c.total_seconds() / 60:g}" for c in candidate) + "min"

    def _write_trajectory_fields(
        self,
        ic_rel_sat: np.ndarray,
        et_data: pd.DataFrame,
        seg_et: dict[str, pd.DataFrame],
        horizon_start: pd.Timestamp,
        horizon_end: pd.Timestamp,
        chosen: tuple[pd.Timedelta, ...],
        run_timestamp: pd.Timestamp,
    ) -> None:
        """Debug dump: save the soil saturation field as a PNG at every forecast
        timestamp (hourly), for every watering candidate on the ladder, into a
        per-run subdirectory of the on-disk plot dir. Gated by `save_trajectory_plot`.

        Each candidate is re-rolled independently from the initial condition (the
        ground-truth path ``_rollout_independent`` uses), with a snapshot sink that
        renders and writes each hourly field inline -- so this never accumulates the
        full mesh across steps, and never touches the parallel/caterpillar forecast
        roll-out above. Warns and skips (never raises), per-candidate and overall: a
        debug-plot failure must never abort the forecast/recommendation already
        published. Off by default; when on it re-simulates the whole ladder and
        writes candidates x forecast-steps PNGs -- slow and disk-heavy, debug only.
        """
        if not self._save_trajectory_plot:
            return

        run_dir = os.path.join(self._trajectory_plot_dir, f"trajectory_{run_timestamp:%Y%m%dT%H%M%S}")
        try:
            os.makedirs(run_dir, exist_ok=True)
        except OSError:
            logging.exception("%s: creating the trajectory field-plot dir '%s' failed.", self.name, run_dir)
            return

        idx = et_data.index
        written = 0
        for candidate in self._ladder:
            slug = self._candidate_slug(candidate)
            is_chosen = candidate == chosen
            prefix = f"{'CHOSEN_' if is_chosen else ''}{slug}"
            label = slug + (" (recommended)" if is_chosen else "")

            def _sink(ts: pd.Timestamp, *, _prefix: str = prefix, _label: str = label) -> None:
                nonlocal written
                png = self._render_snapshot_png(self._pde.snapshot(), ts, title=f"Candidate {_label}")
                path = os.path.join(run_dir, f"{_prefix}_{ts:%Y%m%dT%H%M%S}.png")
                with open(path, "wb") as handle:
                    handle.write(png)
                written += 1

            try:
                self._pde.set_state(ic_rel_sat)
                on_intervals = self._build_flow_schedule(
                    self._windows, list(candidate), self._flow_m3s, horizon_start, horizon_end
                )
                self._roll_segment(idx, et_data, seg_et, on_intervals, snapshot_sink=_sink)
            except Exception:  # noqa: BLE001
                logging.exception("%s: field-plot re-roll for candidate %s failed; skipping it.", self.name, slug)
                continue

        logging.debug(
            "%s: trajectory field plots written: %d PNGs across %d candidates in %s.",
            self.name,
            written,
            len(self._ladder),
            run_dir,
        )

    @staticmethod
    def _encode_state(rel_sat: np.ndarray) -> bytes:
        buf = io.BytesIO()
        np.savez(buf, rel_sat=rel_sat)
        return buf.getvalue()

    def _render_snapshot_png(
        self, rel_sat: np.ndarray, sim_t: pd.Timestamp, *, title: str = "Predicted relative saturation"
    ) -> bytes:
        """Render a saturation snapshot to PNG bytes; fig/ax/norm are lazily initialised and reused."""
        if self._plot_fig is None:
            self._plot_fig, self._plot_ax, self._plot_norm = plot_render.init_rel_sat_figure(
                self._mesh_config.width,
                self._mesh_config.height,
            )
        return plot_render.render_rel_sat_png(
            self._plot_fig,
            self._plot_ax,
            self._plot_norm,
            self._pde.mesh,
            rel_sat,
            sim_t,
            title=title,
        )


def _stringify_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Copy ``frame`` with every column label coerced to a plain ``str``.

    lories ``Constant`` column labels (str subclasses, e.g. ``Weather.PRECIPITATION``)
    do not survive pickling across a spawn worker boundary: ``Constant.__new__``
    takes ``(type, key, ...)``, so pickle's str-subclass reconstruction passes the
    value as ``type`` with ``key=None`` and the constructor raises. A Constant
    compares equal to its key str, so flattening the labels leaves Constant-keyed
    access (``_rain_flux``, ``_segment_flux_dicts``) unchanged downstream.
    """
    return frame.rename(columns=str)


# --- Parallel-executor worker (module-level, spawn-picklable) ----------------
# The ProcessPoolExecutor initializer/task functions must be importable by name
# for the spawn start method, so they live at module scope, not as methods or
# closures. Each worker rebuilds one SoilPDECore in _worker_init and reuses it
# across every candidate that worker handles; the shared per-run inputs are
# stashed in _WORKER so each task payload is just the candidate tuple. See
# docs/adr/0005-parallel-independent-rolls-over-caterpillar.md.
_WORKER: dict[str, Any] = {}


def _worker_init(
    mesh_config: MeshConfig,
    ode_config: PDEConfig,
    rel_sat_name: str,
    name: str,
    probes: list[ProbeSpec],
    windows: list[WateringWindow],
    flow_m3s: float,
    grid_mode: str,
    ic_rel_sat: np.ndarray,
    et_data: pd.DataFrame,
    seg_et: dict[str, pd.DataFrame],
    horizon_start: pd.Timestamp,
    horizon_end: pd.Timestamp,
) -> None:
    """ProcessPoolExecutor initializer (runs once per worker process): pin one
    core, rebuild the PDE from config (spawn-safe -- no fork-inherited state), and
    stash the shared per-run inputs as worker globals.
    """
    # Pin one core per worker BEFORE building/solving the PDE: OMP_NUM_THREADS=1
    # avoids OpenMP oversubscription across the pool, and KMP_DUPLICATE_LIB_OK=TRUE
    # avoids the duplicate-runtime abort (OMP #15) that otherwise crashes FiPy.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    ensure_mesh(mesh_config)
    predictor = object.__new__(SoilPredictor)
    predictor._name = name
    predictor._pde = SoilPDECore(mesh_config, ode_config, rel_sat_name=rel_sat_name)
    predictor._probes = probes
    predictor._windows = windows
    predictor._flow_m3s = flow_m3s
    predictor._grid_mode = grid_mode

    _WORKER.clear()
    _WORKER["predictor"] = predictor
    _WORKER["ic_rel_sat"] = ic_rel_sat
    _WORKER["et_data"] = et_data
    _WORKER["seg_et"] = seg_et
    _WORKER["flow_m3s"] = flow_m3s
    _WORKER["horizon_start"] = horizon_start
    _WORKER["horizon_end"] = horizon_end


def _worker_roll(
    candidate: tuple[pd.Timedelta, ...],
) -> tuple[tuple[pd.Timedelta, ...], tuple[list[pd.Timestamp], dict[str, list[float]]]]:
    """ProcessPoolExecutor task: roll one candidate on this worker's rebuilt PDE
    via the reference ``_rollout_independent``. The payload is only the candidate
    tuple; every other input comes from the worker globals set by ``_worker_init``.
    Returns ``(candidate, (timestamps, trajectories))`` so the parent can key the
    gathered map without tracking submission order.
    """
    predictor = _WORKER["predictor"]
    result = predictor._rollout_independent(
        _WORKER["ic_rel_sat"],
        candidate,
        _WORKER["et_data"],
        _WORKER["seg_et"],
        _WORKER["flow_m3s"],
        _WORKER["horizon_start"],
        _WORKER["horizon_end"],
    )
    return candidate, result
