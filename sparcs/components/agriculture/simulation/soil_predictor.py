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

import copy
import datetime
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
from sparcs.components.agriculture.simulation.soil import SoilSimulation

from . import plot_render, plot_style
from ._soil import (
    ClipDiagnostics,
    FluxRates,
    MeshConfig,
    PDEConfig,
    ProbeSpec,
    SoilBase,
    SoilPDECore,
    apply_surface_forcing,
    design_flow_lpm,
    ensure_mesh,
    resolve_probes,
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

# Drip-flow derivation defaults; mirrors the [soil_simulation] default of a
# single already-per-metre line (SoilSimulation.configure's
# total_drip_line_length_m) when a field has no [soil_predictor.drip] block.
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

# Forecast header table: fixed PK arity for the w{i}_min/w{i}_start column
# set. Windows beyond what a deployment configures are left NULL (no sentinel).
_DEFAULT_MAX_WINDOWS: int = 4


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
    _state_freq: pd.Timedelta
    _save_state: bool

    # Scheduling gate: run cadence, own defaults (not the field-sim tick's).
    _interval_min: int
    _offset_min: int

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

        Contract: with no ``[pde]`` block and no forcing override the predictor
        inherits ``soil_pde`` wholesale, same object (``ode is soil_pde``). With
        no ``[pde]`` but its OWN ``[ponding]``/``[feddes]``, a shallow copy of
        ``soil_pde`` is built first -- ``apply_surface_forcing`` replaces
        ``.ponding``/``.feddes`` wholesale, and without the copy ``ode_config``
        would BE ``soil_pde``, silently rewriting the sim's own resolved forcing
        (HAZARD, B4 review). With its own ``[pde]`` it always gets a fresh
        ``PDEConfig``, overriding the solver keys but still inheriting the sim's
        ponding + feddes unless it supplies its own ``[soil_predictor.ponding]`` /
        ``[soil_predictor.feddes]`` (which then win via ``apply_surface_forcing``).
        """
        if configs.has_member("pde"):
            ode_config = PDEConfig(configs.get_member("pde"), model_configs=model_configs)
        elif configs.has_member("ponding") or configs.has_member("feddes"):
            # Shallow copy is sufficient: apply_surface_forcing only ever REPLACES
            # .ponding/.feddes wholesale (never mutates them in place), so a copy
            # keeps ode_config distinct from soil_pde while still sharing every
            # other scalar field.
            ode_config = copy.copy(soil_pde)
        else:
            # No [pde] and no forcing override: return the sim's object unchanged
            # so `ode is soil_pde` holds for callers that never touch forcing.
            return soil_pde
        # Seed the sim's surface forcing onto ode_config, then let the predictor's
        # own sibling blocks override it. ode_config is never soil_pde itself here
        # (own-[pde]: a fresh PDEConfig; no-[pde]-with-override: the shallow copy),
        # so the reassignment below can never touch the sim's object.
        ode_config.ponding = soil_pde.ponding
        ode_config.feddes = soil_pde.feddes
        apply_surface_forcing(ode_config, configs, ponding_base=soil_pde.ponding, feddes_base=soil_pde.feddes)
        return ode_config

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

        self._interval_min = configs.get_int("interval", default=_DEFAULT_INTERVAL_MIN)
        self._offset_min = configs.get_int("offset", default=_DEFAULT_OFFSET_MIN)

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

    def _register_header_channels(self) -> tuple[list[str], list[str]]:
        """`agri_field_forecast`: one row per candidate per run. Bound to the
        configured `logger` connector; these channels are NEVER `.set()` by the
        predictor -- the automatic flush (Channels.to_frame(unique=True)) skips
        any channel whose timestamp is NaT, so leaving them un-set is what keeps
        the auto path silent for them (see the module docstring). Returns
        (window_min_keys, window_start_keys), w0 ... w{max_windows-1}, in order.
        Only called when `self._logger_id is not None` (see configure())."""
        self.data.add(
            self._HEADER_FORECAST_ID_KEY,
            name="Forecast candidate id",
            type=int,
            aggregate="last",
            logger={
                "connector": self._logger_id,
                "table": self._HEADER_TABLE_NAME,
                "primary": True,
                "nullable": False,
                "enabled": True,
            },
        )

        window_min_keys = []
        for i in range(self._max_windows):
            key = f"w{i}_min"
            window_min_keys.append(key)
            self.data.add(
                key,
                type=float,
                name=f"Window {i} duration",
                unit="min",
                aggregate="last",
                logger={
                    "connector": self._logger_id,
                    "table": self._HEADER_TABLE_NAME,
                    "enabled": True,
                },
            )

        window_start_keys = []
        for i in range(self._max_windows):
            key = f"w{i}_start"
            window_start_keys.append(key)
            self.data.add(
                key,
                type=str,
                name=f"Window {i} start",
                aggregate="last",
                logger={
                    "connector": self._logger_id,
                    "table": self._HEADER_TABLE_NAME,
                    "enabled": True,
                },
            )

        self.data.add(
            self._HEADER_IS_RECOMMENDED_KEY,
            type=bool,
            name="Is recommended candidate",
            aggregate="last",
            logger={
                "connector": self._logger_id,
                "table": self._HEADER_TABLE_NAME,
                "enabled": True,
            },
        )

        self.data.add(
            self._HEADER_TOTAL_MIN_KEY,
            type=float,
            name="Total watering duration",
            unit="min",
            aggregate="last",
            logger={
                "connector": self._logger_id,
                "table": self._HEADER_TABLE_NAME,
                "enabled": True,
            },
        )

        self.data.add(
            self._HEADER_WEATHER_CREATION_KEY,
            name="Weather forecast issue time",
            type=pd.Timestamp,
            aggregate="last",
            logger={
                "connector": self._logger_id,
                "table": self._HEADER_TABLE_NAME,
                "enabled": True,
            },
        )
        return window_min_keys, window_start_keys

    def _resolve_probe_identities(
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

        A probe with no configured ``soil_id`` is warned (not raised, matching
        ``soil.py``) and gets no ``soil_id`` kwarg at all -- the detail write
        then fails at the next ``connector.write()`` for this table (caught by
        the surrounding best-effort try/except and logged, not raised to the
        caller), and because the write grouping raises on the FIRST resource
        missing the attribute, ALL probes' rows for that tick are dropped
        along with the misconfigured probe's. Accepted failure mode until the
        fixtures carry soil_ids.
        """
        channels_cfg = soil_block.get_member("data", defaults={}).get_member("channels", defaults={})
        field_id = channels_cfg.get("field_id", default=None)

        identities: dict[str, dict[str, Any]] = {}
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
                    self.name,
                    probe.channel_id,
                    probe.channel_id,
                )
            else:
                identity["soil_id"] = soil_id
            identities[probe.channel_id] = identity
        return identities

    def _register_detail_channels(
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
        `self._logger_id is not None` (see configure())."""
        tension_keys: dict[str, str] = {}
        creation_keys: dict[str, str] = {}
        forecast_id_keys: dict[str, str] = {}
        for probe in probes:
            identity = probe_identities.get(probe.channel_id, {})
            key = f"traj_{probe.channel_id}"
            tension_keys[probe.channel_id] = key
            self.data.add(
                key,
                type=float,
                name=f"Trajectory {probe.name}",
                unit="hPa",
                aggregate="last",
                logger={
                    "connector": self._logger_id,
                    "table": self._DETAIL_TABLE_NAME,
                    "column": "water_tension",
                    "enabled": True,
                },
                **identity,
            )

            creation_key = f"{key}{self._DETAIL_TIMESTAMP_CREATION_SUFFIX}"
            creation_keys[probe.channel_id] = creation_key
            self.data.add(
                creation_key,
                name=f"Trajectory {probe.name} run timestamp",
                type=pd.Timestamp,
                aggregate="last",
                logger={
                    "connector": self._logger_id,
                    "table": self._DETAIL_TABLE_NAME,
                    "column": "timestamp_creation",
                    "primary": True,
                    "nullable": False,
                    "enabled": True,
                },
                **identity,
            )

            forecast_id_key = f"{key}{self._DETAIL_FORECAST_ID_SUFFIX}"
            forecast_id_keys[probe.channel_id] = forecast_id_key
            self.data.add(
                forecast_id_key,
                name=f"Trajectory {probe.name} candidate id",
                type=int,
                aggregate="last",
                logger={
                    "connector": self._logger_id,
                    "table": self._DETAIL_TABLE_NAME,
                    "column": "forecast_id",
                    "primary": True,
                    "nullable": False,
                    "enabled": True,
                },
                **identity,
            )
        return tension_keys, creation_keys, forecast_id_keys

    def _register_irrigation_channels(self) -> None:
        """`agri_field_forecast_irrigation`: the chosen candidate's watering schedule as
        state-transition edge rows. Same never-`.set()` / logger-gated contract
        as `_register_header_channels`; only called when `self._logger_id is
        not None` (see configure()). Only one field per predictor component, so
        a single shared `timestamp_creation` value channel is enough -- no
        per-probe twins like the detail table needs."""
        self.data.add(
            self._IRRIGATION_STATE_KEY,
            type=bool,
            name="Irrigation plan state",
            aggregate="last",
            logger={
                "connector": self._logger_id,
                "table": self._IRRIGATION_TABLE_NAME,
                "enabled": True,
            },
        )
        self.data.add(
            self._IRRIGATION_TIMESTAMP_CREATION_KEY,
            name="Irrigation plan run timestamp",
            type=pd.Timestamp,
            aggregate="last",
            logger={
                "connector": self._logger_id,
                "table": self._IRRIGATION_TABLE_NAME,
                "column": "timestamp_creation",
                "primary": True,
                "nullable": False,
                "enabled": True,
            },
        )

    def _register_image_channels(self) -> None:
        """`agri_field_forecast_image`: the recommended candidate's field-plot PNGs.
        Same never-`.set()` / logger-gated / single-`timestamp_creation`-twin
        contract as `_register_irrigation_channels`; these two channels are DISTINCT
        from the in-memory `predict_plot` channel (which stays `.set()` for Dash),
        so the auto-log path never fires for them. Only called when
        `self._logger_id is not None` AND plotting is enabled (`self._plot_config
        is not None`; see configure())."""
        self.data.add(
            self._IMAGE_KEY,
            type=bytes,
            name="Predicted soil field image",
            unit="png",
            aggregate="last",
            logger={
                "connector": self._logger_id,
                "table": self._IMAGE_TABLE_NAME,
                "column": self._IMAGE_COLUMN,
                "enabled": True,
            },
        )
        self.data.add(
            self._IMAGE_TIMESTAMP_CREATION_KEY,
            name="Predicted image run timestamp",
            type=pd.Timestamp,
            aggregate="last",
            logger={
                "connector": self._logger_id,
                "table": self._IMAGE_TABLE_NAME,
                "column": "timestamp_creation",
                "primary": True,
                "nullable": False,
                "enabled": True,
            },
        )

    @staticmethod
    def _current_boundary(now: pd.Timestamp, tz, interval_min: int, offset_min: int) -> pd.Timestamp:
        """Most-recent run boundary at or before ``now``, site-local. Mirrors the
        interval/offset pattern of lories ``WeatherForecast`` (forecast.py).
        """
        boundary = floor_date(now, tz, freq=f"{interval_min}min") + pd.Timedelta(minutes=offset_min)
        if boundary > now:
            boundary -= pd.Timedelta(minutes=interval_min)
        return boundary

    # Public driver

    def predict(self, now: pd.Timestamp, forecast_creation: Optional[pd.Timestamp]) -> None:
        """One prediction tick; silently skips if no forecast or no live soil state yet.

        ``forecast_creation`` (the weather forecast's issue time) is persisted only
        as the header's ``weather_creation`` data column -- it is no longer a PK
        partner (``now``, the predictor's own run time, is: see
        ``_build_header_frame``/``_build_detail_frame``). Falls back to ``now`` when
        unavailable so the column always gets a non-null value.
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

        tz = self.context.location.timezone
        boundary = self._current_boundary(now, tz, self._interval_min, self._offset_min)
        if boundary == self._last_boundary_run:
            logger.debug(
                "%s: predict skipped (no new %d-min boundary since %s).",
                self.name,
                self._interval_min,
                self._last_boundary_run,
            )
            return

        key = (now, forecast_creation)
        if self._last_predicted_key == key:
            logger.debug(
                "%s: predict skipped (already published for now=%s, creation=%s).",
                self.name,
                now,
                forecast_creation,
            )
            return

        forecast = self._fetch_forecast(now)
        if forecast is None or forecast.empty:
            logger.info(
                "%s: predict skipped: no forecast rows in [%s, %s].",
                self.name,
                now,
                now + self._horizon,
            )
            return

        field = self.context
        soil = getattr(field, "soil_simulation", None)
        if soil is None:
            logger.info("%s: predict skipped: no soil_simulation sibling.", self.name)
            return

        if getattr(soil, "_last_simulated_at", None) is None:
            logger.debug(
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
            logger.exception("%s: chain replay on forecast failed; skipping tick.", self.name)
            return
        if et_data.empty or et_data.shape[0] < 2:
            logger.info(
                "%s: predict skipped: chain replay returned %d row(s), need ≥ 2.",
                self.name,
                et_data.shape[0],
            )
            return

        ic_rel_sat = soil.get_rel_sat_snapshot()
        try:
            zf_timestamps, zf_trajectories, zf_snapshots, zf_diagnostics = self._solve(ic_rel_sat, et_data, seg_et)
        except Exception:  # noqa: BLE001
            logger.exception("%s: integration failed; skipping tick.", self.name)
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
        # The recommended candidate's rendered field plots (plot_index, png bytes),
        # reused for the agri_field_forecast_image write below; None unless the grid
        # path published with plotting enabled.
        recommended_plot: Optional[tuple[pd.DatetimeIndex, list[bytes]]] = None
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

                recommended_plot = self._publish_results(
                    ch_trajectories,
                    self._probes,
                    ch_timestamps,
                    ch_snapshots,
                    ch_diagnostics,
                    forecast_creation,
                )
                published = True
            except Exception:  # noqa: BLE001
                logger.exception(
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
                logger.exception(
                    "%s: publishing results failed; predictor channels stay stale this tick (now=%s, creation=%s).",
                    self.name,
                    now,
                    forecast_creation,
                )
                return

        self._last_predicted_key = key
        logger.info(
            "%s: predict OK: %d probes, %d rows emitted (now=%s, creation=%s).",
            self.name,
            len(self._probes),
            len(zf_timestamps),
            now,
            forecast_creation,
        )

        # Secondary watering-grid writes -- the agri_field_forecast header, the
        # agri_soil_forecast detail rows (ALL candidates), the chosen candidate's
        # agri_field_forecast_irrigation edge rows, and the recommended candidate's
        # agri_field_forecast_image field plots.
        # Best-effort and only when the grid path produced a recommendation: a
        # failure here never affects the forecast already published on the main
        # channels above. `now` is the run time (every run's rows are kept, keyed
        # by it -- see _build_header_frame / _build_detail_frame /
        # _build_irrigation_frame / _build_image_frame); `forecast_creation` (the
        # weather issue time) is persisted only as the header's weather_creation
        # data column.
        if published and chosen is not None:
            try:
                header_frame = self._build_header_frame(self._ladder, chosen, now, forecast_creation)
                self._write_header_table(header_frame)
                detail_frame = self._build_detail_frame(self._ladder, ladder_traj, now)
                self._write_detail_table(detail_frame)
                irrigation_frame = self._build_irrigation_frame(chosen, horizon_start, horizon_end, now)
                self._write_irrigation_table(irrigation_frame)
                if recommended_plot is not None:
                    save_index, plot_values = recommended_plot
                    self._write_image_table(self._build_image_frame(save_index, plot_values, now))
            except Exception:  # noqa: BLE001
                logger.exception(
                    "%s: header/detail/irrigation/image-write failed (now=%s, creation=%s); "
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
    def _derive_flow_m3s(
        nozzle_count: int,
        nozzle_flow_lph: float,
        total_drip_line_length_m: float,
    ) -> float:
        """Fixed design flow from the drip layout: nozzle output x count, normalized
        per out-of-plane metre of row.

        The l/min core is the shared ``design_flow_lpm`` (also fed to the live sim
        when its physical meter is unavailable); here it is DERIVED from the layout
        instead of read from the meter and normalized to m³/s per metre of row.
        """
        return design_flow_lpm(nozzle_count, nozzle_flow_lph) / (60_000.0 * total_drip_line_length_m)

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
            logger.debug(
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

            if not sweep and i + 1 < len(windows):
                # A window whose durations are only 0min contributes no rungs, so the
                # max-branch save above never ran; still advance the shared prefix
                # across its segment, or every later window's roll would silently
                # skip the weather in [bounds[i], bounds[i+1]].
                self._pde.load_state_blob(prev_blob)
                seg_timestamps, seg_trajectories = self._roll_segment(seg_idx, et_data, seg_et, [])
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
        # Cap per-worker threading BEFORE the pool exists: spawn children inherit
        # the parent's environment, and OpenMP/OpenBLAS read these at numpy import
        # time -- which in a spawn child happens before the initializer runs, so
        # setting them only in _worker_init is too late. The parent's own numpy is
        # already initialized, so this does not throttle the live process.
        # Scope the mutation to this call only: save the prior values here and
        # restore them in the `finally` below once the pool block exits --
        # whether it returns or raises -- so a later component/pool in this same
        # process never silently inherits the pin. NOT safe for concurrent
        # rollouts (another thread's restore could race this save); fine today,
        # predict() runs on a single thread.
        prior_omp_num_threads = os.environ.get("OMP_NUM_THREADS")
        prior_kmp_duplicate_lib_ok = os.environ.get("KMP_DUPLICATE_LIB_OK")
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        try:
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
        finally:
            if prior_omp_num_threads is None:
                os.environ.pop("OMP_NUM_THREADS", None)
            else:
                os.environ["OMP_NUM_THREADS"] = prior_omp_num_threads
            if prior_kmp_duplicate_lib_ok is None:
                os.environ.pop("KMP_DUPLICATE_LIB_OK", None)
            else:
                os.environ["KMP_DUPLICATE_LIB_OK"] = prior_kmp_duplicate_lib_ok
        logger.debug(
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
            logger.warning("%s: forecast read failed: %s", self.name, e)
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
                    logger.exception(
                        "%s: predict_plot render failed at %s; skipping remaining plot snapshots this tick.",
                        self.name,
                        t,
                    )
                    plot_values = []
                    break
            if plot_values:
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

    # Header + detail forecast-table publishing (the watering-grid outputs)

    @staticmethod
    def _forecast_ids(ladder: list[tuple[pd.Timedelta, ...]]) -> dict[tuple[pd.Timedelta, ...], int]:
        """Deterministic candidate-id enumeration for one run: a candidate's
        ``forecast_id`` is its position in ``ladder``. ``ladder`` (``self._ladder``,
        built once at ``configure()``) has a stable order for a given deployment
        regardless of grid_mode (``fill_order`` or ``full``) or which candidates a
        parallel roll-out happens to finish first, and that order does not change
        run over run -- so the same candidate gets the same id every run. Shared by
        ``_build_header_frame``/``_build_detail_frame`` so header and detail rows
        agree on every candidate's id.
        """
        return {candidate: forecast_id for forecast_id, candidate in enumerate(ladder)}

    def _build_header_frame(
        self,
        ladder: list[tuple[pd.Timedelta, ...]],
        chosen: tuple[pd.Timedelta, ...],
        run_timestamp: pd.Timestamp,
        weather_creation: pd.Timestamp,
    ) -> pd.DataFrame:
        """Build the ``agri_field_forecast`` header frame: one row per candidate in
        ``ladder``, indexed at ``run_timestamp`` (repeated across every row -- the
        header's PK partner is ``forecast_id``, not a second timestamp channel).

        Per candidate: its deterministic ``forecast_id`` (``_forecast_ids``), its
        per-window minutes (``None`` for any window index beyond the number of
        CONFIGURED windows -- NULL, not a sentinel, since every candidate has the
        same fixed-arity tuple length), the configured windows' clock-time starts
        (also ``None`` past the configured count), ``is_recommended`` (True only
        for ``chosen``), ``total_min``, and ``weather_creation`` (the weather issue
        time this run used, constant across every row).

        Pure and unit-testable: no ``Channel``/connector access here. Column NAMES
        are the bare channel keys, not full channel ids -- ``_write_header_table``
        renames them to ids (``connector.write`` matches on
        ``resource.id in data.columns``) right before the write.
        """
        forecast_ids = self._forecast_ids(ladder)
        columns = [
            self._HEADER_FORECAST_ID_KEY,
            *self._header_window_min_keys,
            *self._header_window_start_keys,
            self._HEADER_IS_RECOMMENDED_KEY,
            self._HEADER_TOTAL_MIN_KEY,
            self._HEADER_WEATHER_CREATION_KEY,
        ]
        if not ladder:
            return pd.DataFrame(columns=columns)

        rows: list[dict[str, Any]] = []
        index: list[pd.Timestamp] = []

        for candidate in ladder:
            row: dict[str, Any] = {
                self._HEADER_FORECAST_ID_KEY: forecast_ids[candidate],
                self._HEADER_IS_RECOMMENDED_KEY: candidate == chosen,
                self._HEADER_TOTAL_MIN_KEY: self._total_minutes(candidate),
                self._HEADER_WEATHER_CREATION_KEY: weather_creation,
            }
            for i, key in enumerate(self._header_window_min_keys):
                row[key] = candidate[i].total_seconds() / 60.0 if i < len(candidate) else None
            for i, key in enumerate(self._header_window_start_keys):
                row[key] = self._windows[i].start.strftime("%H:%M") if i < len(self._windows) else None
            rows.append(row)
            index.append(run_timestamp)

        frame = pd.DataFrame.from_records(rows, index=pd.DatetimeIndex(index, name="timestamp"))
        return frame.loc[:, columns]

    def _build_detail_frame(
        self,
        ladder: list[tuple[pd.Timedelta, ...]],
        ladder_trajectories: dict[tuple[pd.Timedelta, ...], tuple[list[pd.Timestamp], dict[str, list[float]]]],
        run_timestamp: pd.Timestamp,
    ) -> pd.DataFrame:
        """Build the ``agri_soil_forecast`` detail frame: per-probe LONG rows (one
        row per candidate x forecast-timestamp x probe), not the old wide
        ``traj_<probe>``-columns-side-by-side shape. Every row populates ONLY
        that probe's own THREE columns (tension, its timestamp_creation twin,
        its forecast_id twin) -- every OTHER probe's three columns are absent /
        NaN on that row. The direct-write path groups rows back together per
        probe by its channels' own surrogate ``soil_id``/``field_id``
        attributes (config-side, identical across a probe's three channels --
        see ``_register_detail_channels``/``_resolve_probe_identities``); each
        probe's group then contains exactly its own three columns, so
        ``Table._validate``'s per-group ``dropna(how="all")`` keeps every row
        that has this probe's data and drops rows belonging to other probes.

        ``run_timestamp`` (the predictor's RUN time, not the weather issue
        time) and ``forecast_id`` (``_forecast_ids(ladder)``, the SAME
        enumeration ``_build_header_frame`` uses) are written into every
        probe's OWN twin columns, so their values are identical across probes
        for a given candidate x timestamp -- only the column IDENTITY differs
        per probe, not the values.

        Pure and unit-testable: no ``Channel``/connector access here. Column NAMES
        are the bare channel keys, not full channel ids -- ``_write_detail_table``
        renames them to ids right before the write.
        """
        forecast_ids = self._forecast_ids(ladder)
        columns: list[str] = []
        for probe_id in self._traj_channel_keys:
            columns.append(self._traj_channel_keys[probe_id])
            columns.append(self._detail_creation_keys[probe_id])
            columns.append(self._detail_forecast_id_keys[probe_id])

        rows: list[dict[str, Any]] = []
        index: list[pd.Timestamp] = []

        for candidate, (timestamps, probe_series) in ladder_trajectories.items():
            forecast_id = forecast_ids[candidate]
            for probe_id, tension_key in self._traj_channel_keys.items():
                values = probe_series.get(probe_id)
                if not values:
                    continue
                creation_key = self._detail_creation_keys[probe_id]
                forecast_id_key = self._detail_forecast_id_keys[probe_id]
                for t_idx, ts in enumerate(timestamps):
                    if t_idx >= len(values):
                        continue
                    rows.append(
                        {
                            tension_key: values[t_idx],
                            creation_key: run_timestamp,
                            forecast_id_key: forecast_id,
                        }
                    )
                    index.append(ts)

        if not rows:
            return pd.DataFrame(columns=columns)

        frame = pd.DataFrame.from_records(rows, index=pd.DatetimeIndex(index, name="timestamp"))
        return frame.reindex(columns=columns)

    def _build_irrigation_frame(
        self,
        candidate: tuple[pd.Timedelta, ...],
        horizon_start: pd.Timestamp,
        horizon_end: pd.Timestamp,
        run_timestamp: pd.Timestamp,
    ) -> pd.DataFrame:
        """Build the ``agri_field_forecast_irrigation`` edge-row frame for the CHOSEN
        candidate's watering schedule: one ``(on_ts, True)`` row and one
        ``(off_ts, False)`` row per MERGED on-interval, both stamped with
        ``run_timestamp``. Re-derives the schedule via ``_build_flow_schedule``
        -- built during solving but discarded there -- instead of threading it
        through, keeping this a function of the chosen candidate's durations
        and the horizon bounds alone. A zero-duration (do-nothing) candidate
        yields no intervals and therefore zero rows: a plan with no watering
        has no state transition to record.

        ``_build_flow_schedule`` clamps every interval's ``off_ts`` to
        ``horizon_end`` but does not otherwise guarantee non-degenerate,
        disjoint, ordered intervals, so ``_merge_irrigation_intervals`` sorts,
        drops any interval a short horizon collapsed to ``on_ts >= off_ts``,
        and merges any two intervals that touch or overlap before edges are
        emitted -- see that method for why. The returned off edge for a window
        whose configured duration would run past the horizon IS the closing
        edge (the clamp above), so no separate trailing-edge case is needed
        here.

        Pure and unit-testable: no ``Channel``/connector access here. Column NAMES
        are the bare channel keys, not full channel ids -- ``_write_irrigation_table``
        renames them to ids right before the write.
        """
        columns = [self._IRRIGATION_STATE_KEY, self._IRRIGATION_TIMESTAMP_CREATION_KEY]
        schedule = self._build_flow_schedule(self._windows, list(candidate), self._flow_m3s, horizon_start, horizon_end)
        intervals = self._merge_irrigation_intervals(schedule)
        if not intervals:
            return pd.DataFrame(columns=columns)

        rows: list[dict[str, Any]] = []
        index: list[pd.Timestamp] = []
        for on_ts, off_ts in intervals:
            rows.append({self._IRRIGATION_STATE_KEY: True, self._IRRIGATION_TIMESTAMP_CREATION_KEY: run_timestamp})
            index.append(on_ts)
            rows.append({self._IRRIGATION_STATE_KEY: False, self._IRRIGATION_TIMESTAMP_CREATION_KEY: run_timestamp})
            index.append(off_ts)

        frame = pd.DataFrame.from_records(rows, index=pd.DatetimeIndex(index, name="timestamp"))
        return frame.loc[:, columns]

    @staticmethod
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

    def _write_direct_frame(
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
        writable -- it touches ``self.data`` (one lookup per channel), which a
        skip must never do (mirrors the old single-table write's behavior: a
        missing `logger`/connector short-circuits before any channel lookup).
        """
        if self._logger_id is None:
            return
        if frame.empty:
            logger.debug("%s: %s frame empty; skipping direct write.", self.name, table_label)
            return

        connector = self._resolve_logger_connector(self._logger_id)
        if connector is None:
            logger.warning(
                "%s: logger connector '%s' not found; skipping the %s direct write.",
                self.name,
                self._logger_id,
                table_label,
            )
            return
        if not hasattr(connector, "write"):
            logger.warning(
                "%s: logger connector '%s' (%s) has no write(); skipping the %s direct write.",
                self.name,
                self._logger_id,
                type(connector).__name__,
                table_label,
            )
            return

        write_frame = frame.rename(columns=id_by_key_fn())
        try:
            connector.write(write_frame)
        except Exception:  # noqa: BLE001
            logger.exception(
                "%s: direct write of the %s to logger '%s' failed.",
                self.name,
                table_label,
                self._logger_id,
            )
            return
        logger.info(
            "%s: %s written: %d rows to logger '%s'.",
            self.name,
            table_label,
            len(write_frame),
            self._logger_id,
        )

    def _header_id_by_key(self) -> dict[str, str]:
        id_by_key = {
            self._HEADER_FORECAST_ID_KEY: self.data[self._HEADER_FORECAST_ID_KEY].id,
            self._HEADER_IS_RECOMMENDED_KEY: self.data[self._HEADER_IS_RECOMMENDED_KEY].id,
            self._HEADER_TOTAL_MIN_KEY: self.data[self._HEADER_TOTAL_MIN_KEY].id,
            self._HEADER_WEATHER_CREATION_KEY: self.data[self._HEADER_WEATHER_CREATION_KEY].id,
        }
        for key in self._header_window_min_keys:
            id_by_key[key] = self.data[key].id
        for key in self._header_window_start_keys:
            id_by_key[key] = self.data[key].id
        return id_by_key

    def _write_header_table(self, frame: pd.DataFrame) -> None:
        """Direct-write the ``agri_field_forecast`` header frame."""
        self._write_direct_frame(frame, self._header_id_by_key, "header table")

    def _detail_id_by_key(self) -> dict[str, str]:
        id_by_key: dict[str, str] = {}
        for key in self._traj_channel_keys.values():
            id_by_key[key] = self.data[key].id
        for key in self._detail_creation_keys.values():
            id_by_key[key] = self.data[key].id
        for key in self._detail_forecast_id_keys.values():
            id_by_key[key] = self.data[key].id
        return id_by_key

    def _write_detail_table(self, frame: pd.DataFrame) -> None:
        """Direct-write the ``agri_soil_forecast`` detail frame."""
        self._write_direct_frame(frame, self._detail_id_by_key, "detail table")

    def _irrigation_id_by_key(self) -> dict[str, str]:
        return {
            self._IRRIGATION_STATE_KEY: self.data[self._IRRIGATION_STATE_KEY].id,
            self._IRRIGATION_TIMESTAMP_CREATION_KEY: self.data[self._IRRIGATION_TIMESTAMP_CREATION_KEY].id,
        }

    def _write_irrigation_table(self, frame: pd.DataFrame) -> None:
        """Direct-write the ``agri_field_forecast_irrigation`` edge-row frame."""
        self._write_direct_frame(frame, self._irrigation_id_by_key, "irrigation table")

    def _build_image_frame(
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
        columns = [self._IMAGE_KEY, self._IMAGE_TIMESTAMP_CREATION_KEY]
        rows: list[dict[str, Any]] = []
        index: list[pd.Timestamp] = []
        for ts, png in zip(save_index, plot_values):
            rows.append({self._IMAGE_KEY: png, self._IMAGE_TIMESTAMP_CREATION_KEY: run_timestamp})
            index.append(ts)
        if not rows:
            return pd.DataFrame(columns=columns)
        frame = pd.DataFrame.from_records(rows, index=pd.DatetimeIndex(index, name="timestamp"))
        return frame.loc[:, columns]

    def _image_id_by_key(self) -> dict[str, str]:
        return {
            self._IMAGE_KEY: self.data[self._IMAGE_KEY].id,
            self._IMAGE_TIMESTAMP_CREATION_KEY: self.data[self._IMAGE_TIMESTAMP_CREATION_KEY].id,
        }

    def _write_image_table(self, frame: pd.DataFrame) -> None:
        """Direct-write the ``agri_field_forecast_image`` frame."""
        self._write_direct_frame(frame, self._image_id_by_key, "image table")

    def _resolve_logger_connector(self, logger_id: str) -> Optional[Any]:
        """Resolve the connector for the header/detail direct-writes.

        Prefer the connector the direct-write channels already bound at
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
    # Belt-and-braces: _rollout_parallel already exported these in the parent
    # before spawning (they must be in the environment before the child's numpy
    # import for OpenMP/OpenBLAS to honor them); restated here for any caller
    # that builds a pool without going through _rollout_parallel.
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
