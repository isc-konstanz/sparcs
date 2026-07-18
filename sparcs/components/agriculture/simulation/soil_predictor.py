# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.simulation.soil_predictor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Forecast-driven horizon predictor. Rolls the Richards-equation soil PDE
over the available weather forecast for every watering candidate on the
configured ladder, publishes the recommended candidate's forecast as signed
water tension (hPa) at every configured probe, and persists every
candidate's forecast as a two-table pair: one ``agri_field_forecast`` header
row per candidate per run (durations, window starts, ``is_recommended``,
``weather_creation``) and ALL candidates' per-probe tension trajectories in
``agri_soil_forecast``. There is no separate recommendation table -- the
chosen candidate is the header row with ``is_recommended = True``. The
chosen candidate's watering schedule is additionally persisted as
state-transition edge rows in ``agri_field_forecast_irrigation`` (one row per planned
on/off change, minute-exact). The in-memory ``predict_<probe>`` channels (and
the debug ``predict_state`` / ``predict_plot`` blobs) stay available for
Dash/debugging but are no longer logged.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, NamedTuple, Optional

import numpy as np
import pandas as pd
from lories.core import ConfigurationUnavailableError
from lories.typing import Configurations
from lories.util import to_timedelta
from sparcs.components.agriculture.simulation.soil import SoilSimulation

from . import _predictor_candidates, _predictor_rollout, _predictor_tables, plot_render, plot_style
from ._predictor_candidates import WateringWindow
from ._schedule import parse_tick_schedule
from ._soil import (
    _DEFAULT_NOZZLE_COUNT,
    _DEFAULT_NOZZLE_FLOW_LPH,
    SOIL_PREDICTOR_ALLOWED_KEYS,
    ClipDiagnostics,
    DripConfig,
    FluxRates,
    MeshConfig,
    PDEConfig,
    ProbeSpec,
    SoilBase,
    SoilPDECore,
    # Re-export: test_soil_predictor_ponding_inheritance.py reads
    # soil_predictor.apply_surface_forcing off this module's namespace.
    apply_surface_forcing,  # noqa: F401
    resolve_probes,
    warn_unknown_keys,
)

logger = logging.getLogger(__name__)

_DEFAULT_HORIZON: str = "24h"
# Default snapshot cadence, shared by the [plot] interval (chosen-candidate field
# images) and the [state] interval (predict_state blobs); each overridable.
_DEFAULT_SNAPSHOT_INTERVAL: str = "1h"

# Scheduling gate defaults -- the predictor's OWN cadence, distinct from
# the field-simulation tick's interval=60/offset=0 (do not inherit those).
# `interval`/`offset` are the run-cadence config (daily at ~01:00 local);
# the tick calls predict() after every advance and this gate decides whether
# a roll-out actually runs.
_DEFAULT_INTERVAL_MIN: int = 1440
_DEFAULT_OFFSET_MIN: int = 60

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

# Forecast header table: fixed PK arity for the w{i}_min/w{i}_start column
# set. Windows beyond what a deployment configures are left NULL (no sentinel).
_DEFAULT_MAX_WINDOWS: int = 4

# MeshConfig's eight __init__-set attributes (_soil.py): its generated
# @dataclass __eq__ is vacuous (a hand-written __init__ with zero field
# annotations means dataclasses.fields() is empty), so any two MeshConfig
# instances compare equal via `==`. SoilPredictor._borrow_probes must never
# use `==`; this is the attribute-wise fallback for when identity doesn't hold.
_MESH_CONFIG_ATTRS: tuple[str, ...] = (
    "filename",
    "dl",
    "width",
    "height",
    "plant_width",
    "plant_height",
    "watering_width",
    "dx",
)


def _mesh_configs_equivalent(a: MeshConfig, b: MeshConfig) -> bool:
    """Attribute-wise equality over MeshConfig's eight __init__-set fields
    (see ``_MESH_CONFIG_ATTRS``) -- NEVER MeshConfig's own ``==`` (vacuous)."""
    return all(getattr(a, attr, object()) == getattr(b, attr, object()) for attr in _MESH_CONFIG_ATTRS)


_DIAGNOSTIC_CONSTANTS = (
    SoilSimulation.WATER_TOP_IN,
    SoilSimulation.WATER_TOP_OUT,
    SoilSimulation.WATER_BOTTOM,
    SoilSimulation.WATER_TRANSP,
    SoilSimulation.WATER_RUNOFF,
    SoilSimulation.WATER_DEMAND_UNMET,
    SoilSimulation.WATER_BALANCE_RESIDUAL,
)

# In-memory failure-counter channel key per _write_direct_frame table_label
# (issue 24/W2.6). Deliberately NOT part of _DIAGNOSTIC_CONSTANTS -- its exact
# membership is test-pinned.
_WRITE_FAILURE_KEYS = {
    "header table": "header_write_failures",
    "detail table": "detail_write_failures",
    "irrigation table": "irrigation_write_failures",
    "image table": "image_write_failures",
}


class GridResult(NamedTuple):
    """One successful watering-grid pass (see ``SoilPredictor._run_grid``):
    the chosen candidate, every candidate's tension trajectories, and the
    recommended candidate's rendered field plots (``None`` when plotting is
    off). ``predict()`` treats a non-``None`` GridResult as published."""

    chosen: tuple[pd.Timedelta, ...]
    ladder_traj: dict[tuple[pd.Timedelta, ...], tuple[list[pd.Timestamp], dict[str, list[float]]]]
    recommended_plot: Optional[tuple[pd.DatetimeIndex, list[bytes]]]


class SoilPredictor(SoilBase):
    REL_SAT_NAME: str = "predictor relative saturation"

    TYPE: str = "soil_predictor"
    INCLUDES = ["pde"]

    _TIMESTAMP_CREATION_KEY: str = "timestamp_creation"
    _STATE_CHANNEL_KEY: str = "predict_state"
    _PLOT_CHANNEL_KEY: str = "predict_plot"

    # --- Header table (direct connector write) --------------------------------
    # `agri_field_forecast`: one row per candidate per run, indexed at the
    # predictor's RUN time (the default "timestamp" index column carries it --
    # no separate primary timestamp_creation channel is needed here, unlike the
    # detail table below). forecast_id is the per-row PK partner that
    # distinguishes one run's candidate rows from each other.
    _HEADER_TABLE_NAME: str = "agri_field_forecast"
    _HEADER_FORECAST_ID_KEY: str = "forecast_id"
    _HEADER_IS_RECOMMENDED_KEY: str = "is_recommended"
    _HEADER_TOTAL_MIN_KEY: str = "total_min"
    _HEADER_WEATHER_CREATION_KEY: str = "weather_creation"

    # --- Detail table (direct connector write) ---------------------------------
    # `agri_soil_forecast`: ALL candidates' per-probe tension rows, indexed at
    # the forecast step's future timestamp. Every probe channel gets its OWN
    # timestamp_creation/forecast_id TWIN (traj_<probe>_timestamp_creation /
    # traj_<probe>_forecast_id, both sharing the "timestamp_creation"/
    # "forecast_id" DB columns across probes -- same pattern as traj_<probe>
    # sharing "water_tension") rather than one pair shared by every probe: the
    # SQL connector's per-attribute-set write grouping (table.py `_groupby`)
    # requires EVERY channel written to this table to carry ALL its declared
    # surrogate attributes (soil_id, field_id), and a single shared
    # timestamp_creation/forecast_id pair cannot carry N different probes'
    # soil_ids at once. `timestamp_creation` is the predictor's RUN time --
    # distinct from the header's index and from the weather issue time
    # (header's `weather_creation`) -- so every run's rows survive, never
    # upsert-collide. `forecast_id` is the same per-run candidate enumeration
    # as the header's forecast_id. Both twins are per-row value channels with
    # `logger={primary: true, nullable: false}` (exactly the w0_min pattern the
    # old trajectory table used); soil_id/field_id are surrogate attributes,
    # resolved code-side from the SAME config SoilSimulation's own probe
    # channel reads (`_resolve_probe_identities`) and applied identically to a
    # probe's tension channel AND both its twins (config-side identity stays
    # on the probe; the twins inherit it in code -- see
    # `_register_detail_channels`).
    _DETAIL_TABLE_NAME: str = "agri_soil_forecast"
    _DETAIL_TIMESTAMP_CREATION_SUFFIX: str = "_timestamp_creation"
    _DETAIL_FORECAST_ID_SUFFIX: str = "_forecast_id"

    # --- Irrigation-plan table (direct connector write) ------------------------
    # `agri_field_forecast_irrigation`: the chosen candidate's watering schedule as
    # state-transition edge rows (one row per on/off edge), indexed at the edge's
    # OWN timestamp -- unlike the header table, this index varies per row, like
    # the detail table's future timestamps. `timestamp_creation` is the
    # predictor's RUN time, the PK partner that keeps every run's edges distinct.
    # Unlike the detail table's per-probe twins, only ONE `timestamp_creation`
    # value channel is needed: this table has a single field per predictor
    # component (field_id cascades component-wide via config, not per-row), so
    # there is no per-probe soil_id to disambiguate.
    _IRRIGATION_TABLE_NAME: str = "agri_field_forecast_irrigation"
    _IRRIGATION_STATE_KEY: str = "irrigation_state"
    _IRRIGATION_TIMESTAMP_CREATION_KEY: str = "irrigation_timestamp_creation"

    # --- Recommended-candidate field-plot image table (direct connector write) --
    # `agri_field_forecast_image`: the RECOMMENDED candidate's soil-saturation
    # field snapshots as PNG bytes, one row per saved snapshot. Same field-level
    # shape as `agri_field_forecast_irrigation` (PK `timestamp` = snapshot future time,
    # `timestamp_creation` = run time, `id` <- field_id; single value column
    # `image`), and the same single shared `timestamp_creation` twin -- one field
    # per component, so no per-probe twins. Recommended candidate only. Reuses the
    # bytes already rendered for the in-memory `predict_plot` channel -- see
    # `_publish_results`/`predict`. Gated by `[plot] enabled` AND a configured logger.
    _IMAGE_TABLE_NAME: str = "agri_field_forecast_image"
    _IMAGE_KEY: str = "predict_image"
    _IMAGE_COLUMN: str = "image"
    _IMAGE_TIMESTAMP_CREATION_KEY: str = "predict_image_timestamp_creation"

    _horizon: pd.Timedelta
    # Chosen-candidate field-plot cadence lives in _plot_config (None = plotting off);
    # state-blob capture has its own [state] gate + interval (both snapshot the roll-out).
    _plot_config: Optional[plot_style.PlotConfig] = None
    # Consecutive failing render TICKS toward plot_style.PLOT_DISABLE_AFTER
    # (W2.9); one strike per failing tick, not per snapshot.
    _plot_strikes: int = 0
    _state_freq: pd.Timedelta
    _save_state: bool

    # Scheduling gate: run cadence, own defaults (not the field-sim tick's).
    _interval_min: int
    _offset_min: int

    # Last boundary `predict()` actually ran for; None before the first run.
    _last_boundary_run: Optional[pd.Timestamp] = None

    # Monotonic per-table direct-write failure counts (issue 24/W2.6), keyed by
    # table_label; None on bare object.__new__ instances (the _pde_lock idiom).
    # Surfaced via the in-memory *_write_failures channels (Dash Data accordion).
    _write_failures: Optional[dict[str, int]] = None

    # Why the last _fetch_forecast returned None (issue 23/W2.5): lories folds
    # connector errors into empty frames, so "no forecast" needs a recorded
    # cause to be diagnosable. Consumed by predict()'s skip block, which WARNs
    # once per unclaimed boundary (_forecast_warn_boundary is the latch --
    # _last_boundary_run itself deliberately stays unclaimed during an outage
    # so retries continue, and thus cannot rate-limit the warning).
    _forecast_empty_cause: Optional[str] = None
    _forecast_warn_boundary: Optional[pd.Timestamp] = None

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

    # Fixed PK arity for the header's w{i}_min/w{i}_start columns (see
    # _build_header_frame) and the id of the SQL logger connector the direct
    # header/detail writes target; None when [soil_predictor].logger is not
    # configured (both writes are then skipped with a warning -- see
    # configure()/_write_header_table/_write_detail_table).
    _max_windows: int
    _logger_id: Optional[str]

    # Header data-column keys, w0_min ... w{max_windows-1}_min / w0_start ...
    # w{max_windows-1}_start, in window order.
    _header_window_min_keys: list[str]
    _header_window_start_keys: list[str]
    # probe.channel_id -> detail-table channel key (traj_<probe>), distinct from
    # _channel_keys (the in-memory predict_<probe> keys).
    _traj_channel_keys: dict[str, str]
    # probe.channel_id -> that probe's timestamp_creation/forecast_id TWIN
    # channel keys (traj_<probe>_timestamp_creation / traj_<probe>_forecast_id)
    # -- per-probe so each carries the SAME soil_id/field_id as its tension
    # channel (see _register_detail_channels/_resolve_probe_identities).
    _detail_creation_keys: dict[str, str]
    _detail_forecast_id_keys: dict[str, str]

    # Dedup gate: skip if (now, forecast_creation) was already published.
    _last_predicted_key: Optional[tuple[pd.Timestamp, pd.Timestamp]] = None

    # Pinned static-method names, kept for the tests that call them directly
    # off the class; each delegates to the glossary-correct, module-level
    # function in _predictor_candidates.
    _build_ladder = staticmethod(_predictor_candidates.build_candidate_grid)
    _check_combo_cap = staticmethod(_predictor_candidates.check_candidate_cap)
    _score_candidate = staticmethod(_predictor_candidates.score_candidate)
    _select = staticmethod(_predictor_candidates.select_candidate)
    _total_minutes = staticmethod(_predictor_candidates.total_minutes)
    _build_flow_schedule = staticmethod(_predictor_candidates.build_flow_schedule)
    _split_interval = staticmethod(_predictor_candidates.split_interval)
    _resolve_window_start = staticmethod(_predictor_candidates.resolve_window_start)
    _derive_flow_m3s = staticmethod(_predictor_candidates.derive_flow_m3s)
    _current_boundary = staticmethod(_predictor_candidates.current_boundary)
    _resolve_ode_config = staticmethod(_predictor_candidates.resolve_ode_config)
    # Same pattern for the flux statics shared with the rollout engine, which
    # live module-level in _predictor_rollout.
    _segment_flux_dicts = staticmethod(_predictor_rollout._segment_flux_dicts)
    _rain_flux = staticmethod(_predictor_rollout._rain_flux)
    # And for the candidate-id enumeration shared by the header/detail frame
    # builders, module-level in _predictor_tables.
    _forecast_ids = staticmethod(_predictor_tables.forecast_ids)

    @staticmethod
    def _resolve_model_block(context_configs: Configurations) -> tuple[Configurations, Configurations]:
        """Resolve the live sim's ``[soil_simulation]`` block and its effective ``[model]``.

        Also called by ``FieldSimulation.configure`` (base.py) for the eager
        ``soil_pde_config`` parse, so the two resolutions are equivalent by
        construction -- renames or semantic changes here affect base.py too.

        The sim reads ``[model]`` through the ``[soil_simulation]`` cascade
        (``Component._build_defaults(includes=["model", "plot"])``, base.py), so a
        ``[soil_simulation.model]`` key-level override (e.g. a tuned ``k_s``) wins
        over the field-level ``[model]`` block. The parent's ``_build_child``
        defaults-merge normally performs that cascade before any child configures,
        but the predictor must not assume it (standalone/unit-test construction,
        ordering-defensive since it configures BEFORE the sim) -- so it resolves
        the same key-level merge itself here: soil-level keys win, keys the soil
        block does not restate fall back to the field-level block.
        """
        soil_block = context_configs.get_member(SoilSimulation.TYPE, defaults={}, ensure_exists=True)
        field_model = context_configs.get_member("model", defaults={}, ensure_exists=True)
        model_block = soil_block.get_member("model", defaults=field_model, ensure_exists=True)
        return soil_block, model_block

    def configure(self, configs: Configurations) -> None:
        super().configure(configs)
        warn_unknown_keys(configs, SOIL_PREDICTOR_ALLOWED_KEYS, "soil_predictor")

        mesh_config = getattr(self.context, "mesh_config", None)
        if mesh_config is None:
            raise ValueError(
                f"{self.id}: parent FieldSimulation has no mesh_config; "
                "predictor needs a [soil_simulation.mesh] block to derive "
                "the soil cross-section."
            )
        self._mesh_config = mesh_config

        self._horizon = to_timedelta(configs.get("horizon", default=_DEFAULT_HORIZON))

        # Chosen-candidate field images: [plot] enabled (default on) + [plot] interval.
        # None when disabled. Shares the shared subcomponent [plot] vocabulary.
        self._plot_config = plot_style.load_plot_config(configs, default_interval=_DEFAULT_SNAPSHOT_INTERVAL)

        # State-blob capture (predict_state, debug/replay): its own [state] block,
        # decoupled from [plot] -- a state blob is not a plot.
        state = configs.get_member("state", defaults={}, ensure_exists=True)
        self._save_state = state.get_bool("save", default=False)
        self._state_freq = to_timedelta(state.get("interval", default=_DEFAULT_SNAPSHOT_INTERVAL))

        self._interval_min, self._offset_min = parse_tick_schedule(
            configs,
            default_interval=_DEFAULT_INTERVAL_MIN,
            default_offset=_DEFAULT_OFFSET_MIN,
            section_name="soil_predictor",
        )

        soil_block, model_block = self._resolve_model_block(self.context.configs)
        # ONE canonical resolution site for the [pde]/[model]/forcing cascade:
        # FieldSimulation eagerly parses this at its own configure time (before
        # any child configures) and exposes it via soil_pde_config -- consume
        # that instead of re-deriving PDEConfig here, so predictor and sim can
        # never diverge on ponding/feddes (see base.py's soil_pde_config
        # docstring). No new failure mode: a predictor without a
        # [soil_simulation] context already hard-fails on the mesh_config check
        # above.
        soil_pde = getattr(self.context, "soil_pde_config", None)
        if soil_pde is None:
            raise ValueError(
                f"{self.id}: parent FieldSimulation has no soil_pde_config; "
                "predictor needs a [soil_simulation] block to resolve "
                "the sim's PDE config."
            )

        # [soil_predictor.drip] is a PER-KEY override against the sim's
        # resolved DripConfig: an unset key inherits the sim's value (key-level
        # merge, same idiom as PondingConfig/FeddesConfig's `base` parameter)
        # instead of a hardcoded placeholder. sim_drip is None only when the
        # context has no [soil_simulation] block at all -- the mesh_config
        # check above already raises before this line is reached in that case;
        # _resolve_drip_layout falls back to the shared _soil.py defaults
        # regardless, so this stays correct if that ever changes.
        sim_drip = getattr(self.context, "drip_config", None)
        drip_block = configs.get_member("drip", defaults={}, ensure_exists=True)
        nozzle_count, nozzle_flow_lph = self._resolve_drip_layout(drip_block, sim_drip)
        # total_drip_line_length_m: single parse lives on the sim (soil.py);
        # consume it via the context property instead of re-parsing
        # [soil_simulation] ourselves (mirrors mesh_config/soil_pde_config).
        total_drip_line_length_m = self.context.total_drip_line_length_m
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
                "w{i}_min/w{i}_start column-set arity on agri_field_forecast and needs "
                "a manual table migration to raise (see SOIL.md)."
            )

        self._logger_id = configs.get("logger", default=None)
        if self._logger_id is not None:
            self._logger_id = str(self._logger_id)
        else:
            logger.warning(
                "%s: [soil_predictor].logger not configured; the "
                "agri_field_forecast/agri_soil_forecast tables will not be written.",
                self.name,
            )

        self._ode_config = self._resolve_ode_config(configs, soil_pde, model_block)

        self._pde = self._build_pde()

        probes_cfg = soil_block.get_member("probes", defaults={}, ensure_exists=True)
        borrowed_probes = self._borrow_probes(self.context, self._mesh_config)
        if borrowed_probes is not None:
            self._probes = borrowed_probes
        else:
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
            logger.warning(
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
                logger.warning(
                    "%s: decision_probes %s not found among resolved probe channel-ids "
                    "%s; unknown ids are kept as configured but will never contribute "
                    "a tension sample.",
                    self.name,
                    unknown,
                    all_probe_ids,
                )
            self._decision_probes = decision_probes

        # Main auto-logged run-time channel + in-memory-only Dash/debug channels:
        # extracted into small _register_* helpers (mirrors soil.py's
        # _register_state_channel/_register_progress_image_channel) so the exact
        # data.add() kwargs -- notably every logger.enabled=False here, the collapse
        # this issue makes -- are unit-testable without a full configure() bootstrap.
        self._register_timestamp_creation_channel()
        self._channel_keys = self._register_predict_channels(self._probes)
        if self._save_state:
            self._register_state_channel()
        if self._plot_config is not None:
            self._register_plot_channel()
        self._register_diagnostic_channels()
        self._register_write_failure_channels()
        self._register_plot_strike_channel()

        if not self._probes:
            logger.warning(
                "%s: no probes resolved from [soil_simulation.probes]; "
                "predictor will still run but has no per-probe channels to "
                "publish.",
                self.name,
            )

        # Header (`agri_field_forecast`) + detail (`agri_soil_forecast`) +
        # irrigation-plan (`agri_field_forecast_irrigation`) tables (direct connector
        # write). Skipped entirely when no `logger` is configured (degrade,
        # don't crash) -- see the module docstring.
        self._header_window_min_keys = []
        self._header_window_start_keys = []
        self._traj_channel_keys = {}
        self._detail_creation_keys = {}
        self._detail_forecast_id_keys = {}
        if self._logger_id is not None:
            self._header_window_min_keys, self._header_window_start_keys = self._register_header_channels()
            probe_identities = self._resolve_probe_identities(soil_block, self._probes)
            (
                self._traj_channel_keys,
                self._detail_creation_keys,
                self._detail_forecast_id_keys,
            ) = self._register_detail_channels(self._probes, probe_identities)
            self._register_irrigation_channels()
            # The recommended candidate's field plots persist only when they are
            # also rendered (plotting enabled) -- no point declaring the table otherwise.
            if self._plot_config is not None:
                self._register_image_channels()

    def activate(self) -> None:
        super().activate()
        self._validate_logger_connector()

    def _validate_logger_connector(self) -> None:
        """Refuse to start a predictor whose ``logger =`` id can never resolve
        (issue 25/W2.7, mirroring _validate_irrigation_input): lories connects
        connectors BEFORE activating components, so a failed resolution here IS
        a wiring error -- today it would instead warn "connector not found;
        skipping" on every table write, forever. The write-time resolution
        ladder and warn-and-skip degradation are unchanged; only the
        never-resolvable case fails fast. ``logger`` unset stays a no-op (that
        shape already warns at configure)."""
        if self._logger_id is None:
            return
        connector = self._resolve_logger_connector(self._logger_id)
        if connector is None:
            raise ConfigurationUnavailableError(
                f"{self.name}: [soil_predictor] logger = '{self._logger_id}' resolves to no "
                "connector -- every forecast-table write would be skipped. Point it at a "
                "declared [connectors.<id>] (bare root-level ids resolve via the bound "
                "channels) or remove the key to disable the direct writes."
            )
        # Deliberately stricter than _write_direct_frame's hasattr check: a
        # non-callable write attr should refuse startup here, not crash-and-log
        # inside every write later.
        if not callable(getattr(connector, "write", None)):
            raise ConfigurationUnavailableError(
                f"{self.name}: [soil_predictor] logger = '{self._logger_id}' resolved to "
                f"{type(connector).__name__}, which has no write() -- the forecast tables "
                "need a writable (SQL) connector."
            )

    # --- Channel registration helpers (data.add() kwargs, unit-testable) -------

    def _register_timestamp_creation_channel(self) -> None:
        """Main run-time channel: kept for in-memory/Dash use, but no longer
        logged -- once the reference config's stale `table = "soil_predictor"`
        default is dropped, a lone enabled channel here would still auto-create a
        vestigial group-named table of PK-only rows. The run time now travels
        via the header's timestamp index and the detail rows' timestamp_creation
        value channel (`_register_detail_channels`)."""
        self.data.add(
            self._TIMESTAMP_CREATION_KEY,
            name="Predictor Creation Timestamp",
            type=pd.Timestamp,
            aggregate="last",
            logger={
                "primary": True,
                "nullable": False,
                "enabled": False,
            },
        )

    def _register_predict_channels(self, probes: list[ProbeSpec]) -> dict[str, str]:
        """In-memory only (Dash/debug): the chosen candidate's forecast now
        persists via the header's is_recommended + its agri_soil_forecast rows.
        Returns probe.channel_id -> predict_<probe> channel key."""
        channel_keys: dict[str, str] = {}
        for probe in probes:
            key = f"predict_{probe.channel_id}"
            channel_keys[probe.channel_id] = key
            self.data.add(
                key,
                type=float,
                name=f"Predicted {probe.name}",
                unit="hPa",
                aggregate="last",
                logger={"enabled": False},
            )
        return channel_keys

    def _register_state_channel(self) -> None:
        """In-memory only (Dash/debug): no longer persisted (collapsed
        recommendation stage -- see the module docstring)."""
        self.data.add(
            self._STATE_CHANNEL_KEY,
            type=bytes,
            name="Predicted soil state",
            aggregate="last",
            logger={"enabled": False},
        )

    def _register_plot_channel(self) -> None:
        """In-memory only (Dash/debug): no longer persisted."""
        self.data.add(
            self._PLOT_CHANNEL_KEY,
            type=bytes,
            name="Predicted soil progress image",
            unit="png",
            aggregate="last",
            logger={"enabled": False},
        )

    def _register_diagnostic_channels(self) -> None:
        """In-memory only (Dash/debug): the forecast water-balance diagnostics
        are no longer persisted."""
        for c in _DIAGNOSTIC_CONSTANTS:
            self.data.add(c, aggregate="last", logger={"enabled": False})

    def _register_write_failure_channels(self) -> None:
        """In-memory only (Dash/debug): monotonic per-table direct-write failure
        counters (issue 24/W2.6). Registered or they surface nowhere -- they
        appear in the component page's Data accordion, not the DB."""
        for label, key in _WRITE_FAILURE_KEYS.items():
            self.data.add(
                key,
                type=float,
                name=f"{label.title()} Write Failures",
                aggregate="last",
                logger={"enabled": False},
            )

    def _register_plot_strike_channel(self) -> None:
        """In-memory only (Dash/debug): the W2.9 render-failure strike counter.
        Separate from _register_diagnostic_channels (its count is test-pinned)."""
        self.data.add("plot_strikes", type=float, name="Plot Strikes", aggregate="last", logger={"enabled": False})

    def _bump_write_failure(self, table_label: str) -> None:
        """Count a failed direct write and surface it on the matching in-memory
        channel. Never raises -- the failure handler's job is to log, and bare
        test predictors carry fake data accesses that raise on lookup."""
        if self._write_failures is None:
            self._write_failures = {}
        count = self._write_failures.get(table_label, 0) + 1
        self._write_failures[table_label] = count
        key = _WRITE_FAILURE_KEYS.get(table_label)
        if key is None:
            return
        try:
            self.data[key].set(pd.Timestamp.now(tz="UTC"), float(count))
        except Exception:  # noqa: BLE001
            logger.debug(
                "%s: write-failure channel '%s' unavailable; count=%d kept in memory.",
                self.name,
                key,
                count,
            )

    # Persisted-table registration, frame building, and direct writes -- bodies
    # live in _predictor_tables.ForecastTablePublisher.

    def _table_publisher(self) -> _predictor_tables.ForecastTablePublisher:
        """Assemble a per-call ``ForecastTablePublisher`` view over this
        predictor. Never cached: the pin files monkeypatch ``data``/
        ``connectors`` as class properties and stub collaborators as instance
        attributes, so every read must go through the live instance at use
        time.
        """
        return _predictor_tables.ForecastTablePublisher(self)

    def _register_header_channels(self) -> tuple[list[str], list[str]]:
        return self._table_publisher().register_header_channels()

    def _resolve_probe_identities(
        self,
        soil_block: Configurations,
        probes: list[ProbeSpec],
    ) -> dict[str, dict[str, Any]]:
        return self._table_publisher().resolve_probe_identities(soil_block, probes)

    def _register_detail_channels(
        self,
        probes: list[ProbeSpec],
        probe_identities: dict[str, dict[str, Any]],
    ) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
        return self._table_publisher().register_detail_channels(probes, probe_identities)

    def _register_irrigation_channels(self) -> None:
        self._table_publisher().register_irrigation_channels()

    def _register_image_channels(self) -> None:
        self._table_publisher().register_image_channels()

    # Public driver

    def predict(self, now: pd.Timestamp, forecast_creation: Optional[pd.Timestamp]) -> None:
        """One prediction tick; silently skips if no forecast or no live soil state yet.

        ``forecast_creation`` (the weather forecast's issue time) is persisted only
        as the header's ``weather_creation`` data column -- it is no longer a PK
        partner (``now``, the predictor's own run time, is: see
        ``_build_header_frame``/``_build_detail_frame``). Falls back to ``now`` when
        unavailable so the column always gets a non-null value.

        A spine over four phase methods -- ``_gate_boundary`` (cadence + dedup,
        returns the boundary WITHOUT claiming), ``_prepare_inputs`` (forecast
        fetch, sibling/cold-start guards, the boundary claim, chain replay),
        ``_run_grid`` (the whole watering-grid block incl. its publish;
        ``None`` on any failure) and ``_persist_grid_tables`` (the four
        best-effort table writes) -- with the zero-flow solve and the fallback
        publish kept inline between them.
        """
        if forecast_creation is None:
            logger.debug(
                "%s: forecast_creation unavailable (upstream "
                "weather.forecast.timestamp_creation not valid yet); "
                "falling back to now=%s for the PK column.",
                self.name,
                now,
            )
            forecast_creation = now

        boundary = self._gate_boundary(now, forecast_creation)
        if boundary is None:
            return

        prepared = self._prepare_inputs(now, boundary)
        if prepared is None:
            return
        et_data, seg_et, ic_rel_sat = prepared

        try:
            zf_timestamps, zf_trajectories, zf_snapshots, zf_diagnostics = self._solve(ic_rel_sat, et_data, seg_et)
        except Exception:  # noqa: BLE001
            logger.exception("%s: integration failed; skipping tick.", self.name)
            return

        # Se -> tension at the roll->publish boundary: the roll stays in Se (so the
        # roll-mechanics tests hold), everything downstream is hPa. The zero-flow
        # roll is computed every tick but HELD, not published here: it is the
        # fallback that keeps a forecast on the main channels if the watering-grid
        # block throws, and the forecast for deployments with no windows.
        zf_trajectories = self._trajectories_to_tension(zf_trajectories)

        published = False
        chosen: Optional[tuple[pd.Timedelta, ...]] = None
        ladder_traj: dict = {}
        # The recommended candidate's rendered field plots (plot_index, png bytes),
        # reused for the agri_field_forecast_image write below; None unless the grid
        # path published with plotting enabled.
        recommended_plot: Optional[tuple[pd.DatetimeIndex, list[bytes]]] = None
        horizon_start = et_data.index[0]
        horizon_end = et_data.index[-1]

        grid = self._run_grid(
            ic_rel_sat,
            et_data,
            seg_et,
            (zf_timestamps, zf_trajectories, zf_snapshots, zf_diagnostics),
            now,
            forecast_creation,
        )
        if grid is not None:
            published = True
            chosen = grid.chosen
            ladder_traj = grid.ladder_traj
            recommended_plot = grid.recommended_plot

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
                logger.exception(
                    "%s: publishing results failed; predictor channels stay stale this tick (now=%s, creation=%s).",
                    self.name,
                    now,
                    forecast_creation,
                )
                return

        self._last_predicted_key = (now, forecast_creation)
        logger.info(
            "%s: predict OK: %d probes, %d rows emitted (now=%s, creation=%s).",
            self.name,
            len(self._probes),
            len(zf_timestamps),
            now,
            forecast_creation,
        )

        if published and chosen is not None:
            self._persist_grid_tables(
                chosen, ladder_traj, now, forecast_creation, horizon_start, horizon_end, recommended_plot
            )

    def _gate_boundary(self, now: pd.Timestamp, forecast_creation: pd.Timestamp) -> Optional[pd.Timestamp]:
        """Cadence + dedup gate: compute the site-local run boundary and return
        it, or ``None`` when this tick should skip (boundary already ran, or
        this exact ``(now, forecast_creation)`` pair already published). Does
        NOT claim the boundary -- ``_prepare_inputs`` claims it only once every
        transient precondition has passed, so a missing-forecast or cold-start
        tick retries on the next tick.
        """
        tz = self.context.location.timezone
        boundary = self._current_boundary(now, tz, self._interval_min, self._offset_min)
        if boundary == self._last_boundary_run:
            logger.debug(
                "%s: predict skipped (no new %d-min boundary since %s).",
                self.name,
                self._interval_min,
                self._last_boundary_run,
            )
            return None

        if self._last_predicted_key == (now, forecast_creation):
            logger.debug(
                "%s: predict skipped (already published for now=%s, creation=%s).",
                self.name,
                now,
                forecast_creation,
            )
            return None

        return boundary

    def _prepare_inputs(
        self,
        now: pd.Timestamp,
        boundary: pd.Timestamp,
    ) -> Optional[tuple[pd.DataFrame, dict[str, pd.DataFrame], np.ndarray]]:
        """Fetch the forecast, verify the live-soil preconditions, CLAIM the
        boundary, and replay the ET chain: returns ``(et_data, seg_et,
        ic_rel_sat)``, or ``None`` on any skip.
        """
        forecast = self._fetch_forecast(now)
        if forecast is None or forecast.empty:
            logger.info(
                "%s: predict skipped: no forecast rows in [%s, %s].",
                self.name,
                now,
                now + self._horizon,
            )
            cause = self._forecast_empty_cause
            if cause is not None and boundary != self._forecast_warn_boundary:
                # Once per unclaimed boundary (the INFO above fires every retry
                # tick): a forecast-enabled predictor producing nothing is an
                # anomaly worth an operator-visible signal, not routine no-data.
                logger.warning(
                    "%s: prediction is stalled at boundary %s: %s. Retrying every tick until forecast rows appear.",
                    self.name,
                    boundary,
                    cause,
                )
                self._forecast_warn_boundary = boundary
            return None

        field = self.context
        soil = getattr(field, "soil_simulation", None)
        if soil is None:
            logger.info("%s: predict skipped: no soil_simulation sibling.", self.name)
            return None

        if getattr(soil, "_last_simulated_at", None) is None:
            logger.debug(
                "%s: predict skipped: live solver has no state yet (cold-start still running) at %s.",
                self.name,
                now,
            )
            return None

        # Claim the boundary only once every transient precondition has passed (a
        # present forecast AND live soil state), so a missing-forecast or cold-start
        # tick retries on the next tick instead of silently burning the whole day,
        # while a ready tick bounds the heavy chain replay + roll-out to one attempt
        # per boundary.
        self._last_boundary_run = boundary

        try:
            et_data, seg_et = field._run_chain(forecast, publish=False)
        except Exception:  # noqa: BLE001
            logger.exception("%s: chain replay on forecast failed; skipping tick.", self.name)
            return None
        if et_data.empty or et_data.shape[0] < 2:
            logger.info(
                "%s: predict skipped: chain replay returned %d row(s), need ≥ 2.",
                self.name,
                et_data.shape[0],
            )
            return None

        return et_data, seg_et, soil.get_rel_sat_snapshot()

    def _run_grid(
        self,
        ic_rel_sat: np.ndarray,
        et_data: pd.DataFrame,
        seg_et: dict[str, pd.DataFrame],
        zf_solution: tuple[
            list[pd.Timestamp],
            dict[str, list[float]],
            dict[pd.Timestamp, tuple[Optional[np.ndarray], Optional[bytes]]],
            dict[str, list[float]],
        ],
        now: pd.Timestamp,
        forecast_creation: pd.Timestamp,
    ) -> Optional[GridResult]:
        """Watering-grid roll-out: pick the recommended candidate and publish ITS
        roll on the main channels (not the zero-flow roll). Only when windows are
        configured. The whole block sits in one try/except so any
        roll-out/select/re-solve/publish failure returns ``None`` and the
        zero-flow fallback publish in ``predict()`` still runs -- a ready tick
        always lands one complete, self-consistent forecast. ``zf_solution`` is
        the already-tension-converted zero-flow solve, reused verbatim when the
        chosen candidate is the all-0min rung (no re-solve).
        """
        if not self._windows:
            return None

        zf_timestamps, zf_trajectories, zf_snapshots, zf_diagnostics = zf_solution
        horizon_start = et_data.index[0]
        horizon_end = et_data.index[-1]
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

            recommended_plot = self._publish_results(
                ch_trajectories,
                self._probes,
                ch_timestamps,
                ch_snapshots,
                ch_diagnostics,
                forecast_creation,
            )
            return GridResult(chosen=chosen, ladder_traj=ladder_traj, recommended_plot=recommended_plot)
        except Exception:  # noqa: BLE001
            logger.exception(
                "%s: watering-grid roll-out/selection/publish failed "
                "(now=%s, creation=%s); falling back to the zero-flow forecast.",
                self.name,
                now,
                forecast_creation,
            )
            return None

    def _persist_grid_tables(
        self,
        chosen: tuple[pd.Timedelta, ...],
        ladder_traj: dict[tuple[pd.Timedelta, ...], tuple[list[pd.Timestamp], dict[str, list[float]]]],
        now: pd.Timestamp,
        forecast_creation: pd.Timestamp,
        horizon_start: pd.Timestamp,
        horizon_end: pd.Timestamp,
        recommended_plot: Optional[tuple[pd.DatetimeIndex, list[bytes]]],
    ) -> None:
        """Secondary watering-grid writes -- the agri_field_forecast header, the
        agri_soil_forecast detail rows (ALL candidates), the chosen candidate's
        agri_field_forecast_irrigation edge rows, and the recommended candidate's
        agri_field_forecast_image field plots.

        Best-effort PER TABLE (issue 24/W2.6: one failed build/write must not
        skip the later tables) and only called when the grid path produced a
        recommendation (see predict()'s gate): a failure here never affects the
        forecast already published on the main channels. ``now`` is the run
        time (every run's rows are kept, keyed by it -- see _build_header_frame
        / _build_detail_frame / _build_irrigation_frame / _build_image_frame);
        ``forecast_creation`` (the weather issue time) is persisted only as the
        header's weather_creation data column.
        """
        try:
            self._write_header_table(self._build_header_frame(self._ladder, chosen, now, forecast_creation))
        except Exception:  # noqa: BLE001
            logger.exception(
                "%s: header-table write failed (now=%s, creation=%s); "
                "the forecast published on the main channels is unaffected.",
                self.name,
                now,
                forecast_creation,
            )
        try:
            self._write_detail_table(self._build_detail_frame(self._ladder, ladder_traj, now))
        except Exception:  # noqa: BLE001
            logger.exception(
                "%s: detail-table write failed (now=%s); the forecast published on the main channels is unaffected.",
                self.name,
                now,
            )
        try:
            self._write_irrigation_table(self._build_irrigation_frame(chosen, horizon_start, horizon_end, now))
        except Exception:  # noqa: BLE001
            logger.exception(
                "%s: irrigation-table write failed (now=%s); "
                "the forecast published on the main channels is unaffected.",
                self.name,
                now,
            )
        if recommended_plot is not None:
            try:
                save_index, plot_values = recommended_plot
                self._write_image_table(self._build_image_frame(save_index, plot_values, now))
            except Exception:  # noqa: BLE001
                logger.exception(
                    "%s: image-table write failed (now=%s); the forecast published on the main channels is unaffected.",
                    self.name,
                    now,
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
        dict[pd.Timestamp, tuple[Optional[np.ndarray], Optional[bytes]]],
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
        dict[pd.Timestamp, tuple[Optional[np.ndarray], Optional[bytes]]],
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
    def _resolve_drip_layout(
        drip_block: Configurations,
        sim_drip: Optional[DripConfig],
    ) -> tuple[int, float]:
        """Per-key override of ``[soil_predictor.drip]`` against the sim's
        resolved ``DripConfig``: an unset key inherits the sim's value (same
        key-level-merge idiom as ``PondingConfig``/``FeddesConfig``'s ``base``
        parameter), rather than a hardcoded placeholder. ``sim_drip=None`` (no
        ``[soil_simulation]`` block to inherit from) falls back to the shared
        ``_soil.py`` nozzle defaults instead.
        """
        default_nozzle_count = sim_drip.nozzle_count if sim_drip is not None else _DEFAULT_NOZZLE_COUNT
        default_nozzle_flow_lph = sim_drip.nozzle_flow_lph if sim_drip is not None else _DEFAULT_NOZZLE_FLOW_LPH
        nozzle_count = drip_block.get_int("nozzle_count", default=default_nozzle_count)
        nozzle_flow_lph = drip_block.get_float("nozzle_flow_lph", default=default_nozzle_flow_lph)
        return nozzle_count, nozzle_flow_lph

    # Probe borrowing (pure)

    @staticmethod
    def _borrow_probes(context: Any, mesh_config: MeshConfig) -> Optional[list[ProbeSpec]]:
        """Adopt the sim's already-resolved probe specs instead of
        re-resolving ``[soil_simulation.probes]`` against a second mesh
        instance, when it is safe to: ``context.get_probes()`` is non-empty
        AND the two meshes are the SAME configuration (identity -- the
        common in-context case, this predictor's own ``_mesh_config`` was
        read from ``context.mesh_config`` moments earlier at configure() --
        or, failing identity, attribute-wise equality over MeshConfig's eight
        __init__-set fields; see ``_mesh_configs_equivalent``. NEVER ``==``:
        MeshConfig's generated ``__eq__`` is vacuous).

        Returns ``None`` (caller falls back to its own ``resolve_probes``)
        when ``context`` has no ``get_probes`` (a bare stub in tests),
        ``get_probes()`` returns no probes, or the mesh guard fails -- never
        raises. Callers must not defensively copy the returned ``ProbeSpec``
        elements: sharing is by design (the parallel worker path already
        reuses parent ProbeSpecs verbatim, see
        ``_predictor_rollout._worker_init``).
        """
        get_probes = getattr(context, "get_probes", None)
        if not callable(get_probes):
            return None
        sim_probes = get_probes()
        if not sim_probes:
            return None
        context_mesh = getattr(context, "mesh_config", None)
        if context_mesh is mesh_config:
            return list(sim_probes)
        if context_mesh is not None and _mesh_configs_equivalent(context_mesh, mesh_config):
            return list(sim_probes)
        return None

    # Prefix-shared roll-out (the caterpillar) -- bodies live in
    # _predictor_rollout.RolloutEngine.

    def _rollout_engine(self) -> _predictor_rollout.RolloutEngine:
        """Assemble a per-call ``RolloutEngine`` view from this instance's loose
        attributes. Never cached: the roll-mechanics tests build bare
        ``object.__new__`` predictors carrying only the attributes the invoked
        method actually reads, so absent attrs map to ``None`` fields.
        """
        return _predictor_rollout.RolloutEngine(
            pde=getattr(self, "_pde", None),
            probes=getattr(self, "_probes", None),
            windows=getattr(self, "_windows", None),
            window_durations=getattr(self, "_window_durations", None),
            flow_m3s=getattr(self, "_flow_m3s", None),
            grid_mode=getattr(self, "_grid_mode", None),
            ladder=getattr(self, "_ladder", None),
            max_workers=getattr(self, "_max_workers", None),
            name=getattr(self, "name", None),
            mesh_config=getattr(self, "_mesh_config", None),
            ode_config=getattr(self, "_ode_config", None),
            rel_sat_name=self.REL_SAT_NAME,
        )

    def _roll_segment(
        self,
        idx: pd.DatetimeIndex,
        et_data: pd.DataFrame,
        seg_et: dict[str, pd.DataFrame],
        on_intervals: list[tuple[pd.Timestamp, pd.Timestamp]],
        snapshot_sink: Optional[Callable[[pd.Timestamp], None]] = None,
    ) -> tuple[list[pd.Timestamp], dict[str, list[float]]]:
        return self._rollout_engine().roll_segment(idx, et_data, seg_et, on_intervals, snapshot_sink)

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
        return self._rollout_engine().rollout_ladder(
            ic_rel_sat, ladder, et_data, seg_et, flow_m3s, horizon_start, horizon_end
        )

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
        return self._rollout_engine().rollout_independent(
            ic_rel_sat, candidate, et_data, seg_et, flow_m3s, horizon_start, horizon_end
        )

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
                logger.exception(
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
        return self._rollout_engine().rollout_parallel(ic_rel_sat, et_data, seg_et, horizon_start, horizon_end)

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

    # Forecast retrieval

    def _fetch_forecast(self, now: pd.Timestamp) -> Optional[pd.DataFrame]:
        """Read the in-memory forecast cache via ``data.to_frame()`` and slice to
        ``[now, now+horizon]``. On a None return, ``_forecast_empty_cause`` records
        why (empty cache vs rows-outside-horizon) for predict()'s boundary-latched
        warning -- lories folds connector errors into empty frames, so the two
        shapes are otherwise indistinguishable (issue 23/W2.5)."""
        self._forecast_empty_cause = None
        weather = getattr(self.context, "weather", None)
        forecast_sub = getattr(weather, "forecast", None)
        if forecast_sub is None or not forecast_sub.is_enabled():
            return None
        try:
            df = forecast_sub.data.to_frame(unique=False)
        except Exception as e:  # noqa: BLE001
            logger.warning("%s: forecast read failed: %s", self.name, e)
            return None
        if df.empty:
            self._forecast_empty_cause = (
                "forecast cache is empty (upstream read produced no rows -- possibly a "
                "connector error folded into an empty frame)"
            )
            return None
        # Align tz: forecast index is location-tz-aware; ``now`` matches.
        end = now + self._horizon
        sliced = df.loc[(df.index >= now) & (df.index <= end)]
        if sliced.empty:
            self._forecast_empty_cause = (
                f"forecast rows exist but none cover [{now}, {end}] "
                f"(cache spans {df.index.min()}..{df.index.max()} -- upstream forecast is stale)"
            )
            return None
        return sliced

    # PDE integration over horizon

    def _integrate_horizon(
        self,
        et_data: pd.DataFrame,
        seg_et: dict[str, pd.DataFrame],
        flow_schedule: Optional[list[tuple[pd.Timestamp, pd.Timestamp]]] = None,
    ) -> tuple[
        list[pd.Timestamp],
        dict[str, list[float]],
        dict[pd.Timestamp, tuple[Optional[np.ndarray], Optional[bytes]]],
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
        ``snapshots`` values are ``(plot_array, state_blob)`` pairs -- see the
        ``snapshots`` capture rationale below.
        """
        on_intervals = flow_schedule or []
        idx = et_data.index
        timestamps: list[pd.Timestamp] = []
        trajectories: dict[str, list[float]] = {p.channel_id: [] for p in self._probes}
        snapshots: dict[pd.Timestamp, tuple[Optional[np.ndarray], Optional[bytes]]] = {}
        diagnostics: dict[str, list[float]] = {c.key: [] for c in _DIAGNOSTIC_CONSTANTS}
        # One snapshot dict captured at the UNION of the plot and state cadences; the
        # per-sink subsets are re-derived in _publish_results (_cadence_subset). Plot
        # and state have independent intervals, so a snapshot is taken whenever either
        # is due, and each sink writes only its own due timestamps. Each entry is a
        # ``(plot_array, state_blob)`` pair: the plot sink keeps the cheap
        # ``snapshot()`` array (fine for rendering -- it already excludes the
        # surface_h ponds); the state sink now captures ``save_state_blob()`` bytes
        # so the FULL solver state -- including the surface_h ponds that
        # ``snapshot()``/``set_state`` drop -- round-trips through
        # ``load_state_blob``/``apply_state_blob``. A timestamp due for only
        # one sink leaves the other half of the pair ``None``.
        plot_interval = self._plot_config.interval if self._plot_config is not None else None
        last_plot_ts: Optional[pd.Timestamp] = None
        last_state_ts: Optional[pd.Timestamp] = None
        capture_snapshots = self._save_state or plot_interval is not None

        def _maybe_snapshot(ts: pd.Timestamp) -> None:
            nonlocal last_plot_ts, last_state_ts
            if not capture_snapshots:
                return
            plot_due = plot_interval is not None and plot_style.render_due(last_plot_ts, ts, plot_interval)
            state_due = self._save_state and plot_style.render_due(last_state_ts, ts, self._state_freq)
            if not (plot_due or state_due):
                return
            snapshots[ts] = (
                self._pde.snapshot() if plot_due else None,
                self._pde.save_state_blob() if state_due else None,
            )
            if plot_due:
                last_plot_ts = ts
            if state_due:
                last_state_ts = ts

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
            # _cadence_subset force-keeps the last index entry for BOTH sinks (the
            # forecast's final frame is always persisted), regardless of whether
            # that tick was naturally plot_due/state_due above -- so make sure
            # whichever half(s) a due-enabled sink still needs are filled in here.
            final_array, final_blob = snapshots.get(final_ts, (None, None))
            if plot_interval is not None and final_array is None:
                final_array = self._pde.snapshot()
            if self._save_state and final_blob is None:
                final_blob = self._pde.save_state_blob()
            snapshots[final_ts] = (final_array, final_blob)

        return timestamps, trajectories, snapshots, diagnostics

    # Channel publishing

    def _publish_results(
        self,
        trajectories: dict[str, list[float]],
        probes: list[ProbeSpec],
        timestamps: list[pd.Timestamp],
        snapshots: dict[pd.Timestamp, tuple[Optional[np.ndarray], Optional[bytes]]],
        diagnostics: dict[str, list[float]],
        forecast_creation: pd.Timestamp,
    ) -> Optional[tuple[pd.DatetimeIndex, list[bytes]]]:
        """Publish the (recommended or zero-flow) forecast on the in-memory
        channels. Returns the rendered ``(plot_index, png_bytes)`` when plotting
        is enabled and produced snapshots, else None -- the caller persists that
        recommended-candidate render to ``agri_field_forecast_image`` without
        re-rendering (see ``predict``).

        ``snapshots`` is captured at the union of the plot and state cadences;
        ``_cadence_subset`` re-derives each sink's own timestamps so the
        ``[plot] interval`` and ``[state] interval`` stay independent. Each value
        is a ``(plot_array, state_blob)`` pair (see ``_integrate_horizon``); this
        method reads only the half its own sink needs."""
        if not timestamps:
            return None

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
            return None

        full_index = pd.DatetimeIndex(sorted(snapshots), name="timestamp")

        if self._save_state:
            state_index = self._cadence_subset(full_index, self._state_freq)
            state_values = [self._encode_state(snapshots[t][1]) for t in state_index]
            self.data[self._STATE_CHANNEL_KEY].set(
                state_index[0],
                pd.Series(state_values, index=state_index, dtype=object),
            )

        if self._plot_config is not None:
            plot_index = self._cadence_subset(full_index, self._plot_config.interval)
            plot_values: list[bytes] = []
            for t in plot_index:
                try:
                    plot_values.append(self._render_snapshot_png(snapshots[t][0], t))
                except Exception:  # noqa: BLE001
                    # One strike per failing TICK; the tick contract is pinned:
                    # no partial series, predict_plot never set, return None.
                    self._plot_strikes, disable = plot_style.count_render_failure(
                        logger, self.name, self._plot_strikes, what="predict_plot"
                    )
                    plot_style.set_strike_channel(self, "plot_strikes", self._plot_strikes)
                    if disable:
                        self._plot_config = None
                    plot_values = []
                    break
            if plot_values:
                if self._plot_strikes:
                    self._plot_strikes = 0
                    plot_style.set_strike_channel(self, "plot_strikes", 0)
                self.data[self._PLOT_CHANNEL_KEY].set(
                    plot_index[0],
                    pd.Series(plot_values, index=plot_index, dtype=object),
                )
                return plot_index, plot_values

        return None

    @staticmethod
    def _cadence_subset(index: pd.DatetimeIndex, interval: pd.Timedelta) -> pd.DatetimeIndex:
        """The subset of ``index`` (assumed sorted) at ``interval`` cadence: the
        first timestamp, each one at least ``interval`` past the previously kept,
        and always the last (so the forecast's final frame is persisted).

        Reproduces the live ``render_due`` throttle over the captured union of
        snapshot timestamps, letting the plot and state sinks apply their own
        interval to the shared snapshot dict without a second roll-out."""
        kept: list[pd.Timestamp] = []
        last: Optional[pd.Timestamp] = None
        for ts in index:
            if plot_style.render_due(last, ts, interval):
                kept.append(ts)
                last = ts
        if len(index) > 0 and index[-1] not in kept:
            kept.append(index[-1])
        return pd.DatetimeIndex(kept, name="timestamp")

    # Header + detail forecast-table publishing (the watering-grid outputs) --
    # bodies live in _predictor_tables.ForecastTablePublisher.

    def _build_header_frame(
        self,
        ladder: list[tuple[pd.Timedelta, ...]],
        chosen: tuple[pd.Timedelta, ...],
        run_timestamp: pd.Timestamp,
        weather_creation: pd.Timestamp,
    ) -> pd.DataFrame:
        return self._table_publisher().build_header_frame(ladder, chosen, run_timestamp, weather_creation)

    def _build_detail_frame(
        self,
        ladder: list[tuple[pd.Timedelta, ...]],
        ladder_trajectories: dict[tuple[pd.Timedelta, ...], tuple[list[pd.Timestamp], dict[str, list[float]]]],
        run_timestamp: pd.Timestamp,
    ) -> pd.DataFrame:
        return self._table_publisher().build_detail_frame(ladder, ladder_trajectories, run_timestamp)

    def _build_irrigation_frame(
        self,
        candidate: tuple[pd.Timedelta, ...],
        horizon_start: pd.Timestamp,
        horizon_end: pd.Timestamp,
        run_timestamp: pd.Timestamp,
    ) -> pd.DataFrame:
        return self._table_publisher().build_irrigation_frame(candidate, horizon_start, horizon_end, run_timestamp)

    def _write_direct_frame(
        self,
        frame: pd.DataFrame,
        id_by_key_fn: Callable[[], dict[str, str]],
        table_label: str,
    ) -> None:
        self._table_publisher().write_direct_frame(frame, id_by_key_fn, table_label)

    def _write_header_table(self, frame: pd.DataFrame) -> None:
        self._table_publisher().write_header_table(frame)

    def _write_detail_table(self, frame: pd.DataFrame) -> None:
        self._table_publisher().write_detail_table(frame)

    def _write_irrigation_table(self, frame: pd.DataFrame) -> None:
        self._table_publisher().write_irrigation_table(frame)

    def _build_image_frame(
        self,
        save_index: pd.DatetimeIndex,
        plot_values: list[bytes],
        run_timestamp: pd.Timestamp,
    ) -> pd.DataFrame:
        return self._table_publisher().build_image_frame(save_index, plot_values, run_timestamp)

    def _write_image_table(self, frame: pd.DataFrame) -> None:
        self._table_publisher().write_image_table(frame)

    def _resolve_logger_connector(self, logger_id: str) -> Optional[Any]:
        return self._table_publisher().resolve_logger_connector(logger_id)

    def _logger_connector_from_channel(self) -> Optional[Any]:
        """The connector a forecast-table channel already resolved against the
        root context (``ChannelConnector``'s component-path walk), or ``None``
        if the channels are not registered / not bound yet. Defensive: a
        resolution failure must degrade to the id-based fallback, never raise.

        Anchored on the HEADER's forecast_id channel: unlike the detail table's
        per-probe timestamp_creation/forecast_id twins, it is a single, always-
        present channel whenever `self._logger_id is not None` (both tables are
        registered together in `configure()`), and connector resolution is
        table-agnostic -- every channel bound to this `logger` id resolves the
        SAME connector regardless of which table it belongs to.
        """
        try:
            return self.data[self._HEADER_FORECAST_ID_KEY].logger._get_registrator()
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _encode_state(state_blob: bytes) -> bytes:
        """Pass through the ``save_state_blob`` bytes captured for the state sink.

        ``_maybe_snapshot`` (see ``_integrate_horizon``) captures
        ``self._pde.save_state_blob()`` directly for the state-due half of each
        snapshot pair, so the value reaching this method is already the fully-
        encoded npz blob ``load_state_blob``/``apply_state_blob`` expect (rel_sat +
        rel_sat_old + the surface pond state) -- there is nothing left to encode.
        Kept as a trivially-wrapping staticmethod (rather than inlined at the
        ``_publish_results`` call site) purely so tests can still intercept the
        state sink's published value by monkeypatching this name (see
        test_soil_predictor_image_table.py's
        test_publish_results_plot_and_state_use_independent_cadences).
        """
        return state_blob

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
