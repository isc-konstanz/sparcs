# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.simulation.soil_predictor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Forecast-driven horizon predictor. Integrates the Richards-equation soil
PDE over the available weather forecast **with no irrigation applied** and
publishes the predicted relative saturation at every probe defined on
the live ``SoilSimulation``. The live solver is never touched: the
predictor owns its own FiPy mesh, equation, and ``CellVariable``s.

Optionally archives the full saturation field at ``save_freq`` cadence
across the horizon on bytes channels ``predict_state`` (npz blob) and
``predict_plot`` (PNG, rendered via :mod:`plot_render`).

The expensive compute step is isolated in :meth:`SoilPredictor._solve`
so a future server-attached backend (sparcs running on an edge device,
predictions offloaded to a server) can replace just that one method
with an HTTP / RPC call without touching forecast retrieval, chain
replay, or channel publishing.
"""

from __future__ import annotations

import io
import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

from lories.components.weather import Weather
from lories.typing import Configurations
from lories.util import to_timedelta

from . import plot_render
from ._soil import ClipDiagnostics, FluxRates, MeshConfig, SoilBase
# FiPy mesh, Richards-equation assembly, segment index, and the pure
# integration primitives (apply_source, solve, sample, total_water, state
# I/O) live on the shared :class:`SoilPDECore` so SoilPredictor and
# SoilSimulation never drift apart.
from sparcs.components.agriculture.simulation.soil import (
    PDEConfig,
    ProbeSpec,
    SoilPDECore,
    SoilSimulation,
    resolve_probes,
)


_DEFAULT_HORIZON: str = "24h"
_DEFAULT_SAVE_FREQ: str = "1h"

# Diagnostic flux Constants the predictor mirrors from the live solver so
# the predict-side channels carry identical names / units (and the UI can
# match them side-by-side under different component cards). Order is the
# emission order — same as :class:`SoilSimulation.configure`.
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

    # Matches the lories Weather convention so SQL connectors that
    # special-case ``timestamp_creation`` (composite PK promotion in
    # ``connectors/sql/schema.py``) work unchanged.
    _TIMESTAMP_CREATION_KEY: str = "timestamp_creation"
    _STATE_CHANNEL_KEY: str = "predict_state"
    _PLOT_CHANNEL_KEY: str = "predict_plot"

    _horizon: pd.Timedelta
    _save_freq: pd.Timedelta
    _save_state: bool
    _save_plot: bool

    _mesh_config: MeshConfig
    _ode_config: PDEConfig
    _pde: SoilPDECore

    # Lazy-init render scaffolding (only built when ``save_plot`` is on and
    # the first snapshot needs rendering).
    _plot_fig: Any = None
    _plot_ax: Any = None
    _plot_norm: Any = None

    # probe.channel_id → predict-channel id.
    _channel_keys: dict[str, str]

    _probes: list[ProbeSpec]

    # Dedup gate keyed on ``(now, forecast_creation)`` so a fresh DWD
    # forecast at the same simulated ``now`` still flows through (that's
    # the whole point of stamping rows with ``timestamp_creation``), while
    # duplicate listener re-fires for an identical pair no-op out.
    _last_predicted_key: Optional[tuple[pd.Timestamp, pd.Timestamp]] = None

    def configure(self, configs: Configurations) -> None:
        super().configure(configs)

        # Predictor and soil_simulation are siblings; lories sorts components
        # alphabetically before configuring, so soil_simulation may not yet
        # be configured when we run. Read everything we need from the
        # *parent* (FieldSimulation) and from the loaded ``[soil_simulation]``
        # config block instead of from the live SoilSimulation instance —
        # that instance only has its attributes populated *after* configure.
        # The instance is still required at predict-time for the live IC, but
        # by then the activate phase is past and it's safe.
        mesh_config = getattr(self.context, "mesh_config", None)
        if mesh_config is None:
            raise ValueError(
                f"{self.id}: parent FieldSimulation has no mesh_config — "
                "predictor needs a [soil_simulation.mesh] block to derive "
                "the soil cross-section."
            )
        self._mesh_config = mesh_config

        self._horizon = to_timedelta(configs.get("horizon", default=_DEFAULT_HORIZON))
        self._save_freq = to_timedelta(configs.get("save_freq", default=_DEFAULT_SAVE_FREQ))
        self._save_state = configs.get_bool("save_state", default=False)
        self._save_plot = configs.get_bool("save_plot", default=False)

        # Pull soil_simulation's loaded config block (file + parent overrides
        # already merged) so we can read its [pde] params (model selector +
        # hydraulic parameters) and its [probes] block for probe definitions.
        soil_block = self.context.configs.get_member(
            SoilSimulation.TYPE, defaults={}, ensure_exists=True
        )
        soil_pde = PDEConfig(soil_block.get_member("pde", defaults={}, ensure_exists=True))

        # Predictor's own [pde] block overrides the live solver's, so users
        # can pick a coarser dt to keep prediction cost in check on the edge.
        if configs.has_member("pde"):
            self._ode_config = PDEConfig(configs.get_member("pde"))
        else:
            self._ode_config = soil_pde

        # Independent FiPy state via SoilPDECore — same .msh file, different
        # in-memory mesh / variables / equation. Predictor mutates its own
        # rel_sat freely without touching the live solver. ``_build_pde``
        # (inherited from SoilBase) also ensures the .msh file exists, so
        # the predictor can configure before ``soil_simulation`` does.
        self._pde = self._build_pde()

        # Resolve probe specs from the *config block* (not the live soil
        # instance, which may not be configured yet — see comment above).
        # Same parser, same .msh file → identical cell indices and weights
        # as the live solver's probe channels, so predictions line up
        # directly with current readings.
        probes_cfg = soil_block.get_member("probes", defaults={}, ensure_exists=True)
        self._probes = resolve_probes(
            probes_cfg, self._pde.mesh, self._mesh_config, log_name=self.name,
        )

        # Composite-PK partner for every row this predictor emits. Mirrors
        # ``WeatherForecast``'s ``timestamp_creation`` (see
        # ``lories.components.weather.dwd.provider``) so the SQL schema
        # layer promotes ``(timestamp, timestamp_creation)`` to the table's
        # primary key and multiple forecasts addressing the same target
        # timestamp coexist as distinct rows. Registered before the
        # per-probe loop so it's present even when there are no probes.
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

        # Channel layout: one ``predict_<probe_id>`` float channel per probe,
        # holding a scalar predicted rel_sat per row. ``predict()`` emits N
        # rows (one per forecast step) in a single ``set`` call per channel;
        # the target timestamps live on the row index.
        self._channel_keys = {}
        for probe in self._probes:
            key = f"predict_{probe.channel_id}"
            self._channel_keys[probe.channel_id] = key
            self.data.add(
                key,
                type=float,
                name=f"Predicted {probe.name}",
                unit="-",
                aggregate="last",
                logger={"enabled": True},
            )

        if self._save_state:
            # Saturation-field snapshots sampled at ``save_freq`` across the
            # horizon (plus the final forecast row). Same set-with-Series
            # idiom as the probe channels — each row is one binary blob.
            self.data.add(
                self._STATE_CHANNEL_KEY,
                type=bytes,
                name="Predicted soil state",
                aggregate="last",
                logger={"enabled": True},
            )

        if self._save_plot:
            # PNG of the saturation field at every save tick, rendered with
            # the same colorbar / styling as the live solver's progress.png
            # via the shared :mod:`plot_render` helpers.
            self.data.add(
                self._PLOT_CHANNEL_KEY,
                type=bytes,
                name="Predicted soil progress image",
                unit="png",
                aggregate="last",
                logger={"enabled": True},
            )

        # Per-forecast-interval flux-density diagnostics, mirroring the live
        # solver's 7 channels. Each forecast row contributes one value;
        # ``_publish_results`` emits the full horizon as a Series.
        for c in _DIAGNOSTIC_CONSTANTS:
            self.data.add(c, aggregate="last", logger={"enabled": True})

        if not self._probes:
            logging.warning(
                "%s: no probes resolved from [soil_simulation.probes] — "
                "predictor will still run but has no per-probe channels to "
                "publish.",
                self.name,
            )

    # =========================================================================
    # Public driver
    # =========================================================================

    def predict(self, now: pd.Timestamp, forecast_creation: Optional[pd.Timestamp]) -> None:
        """One prediction tick. Silently skips when no forecast is available
        or the live soil solver hasn't produced a state yet — predictions
        are best-effort and must not break the live callback chain.

        ``forecast_creation`` is the DWD forecast's ``timestamp_creation``
        (read by ``FieldSimulation._forecast_callback``) and becomes the
        composite-PK partner stamped onto every emitted row. When the
        upstream hasn't populated it yet (cold start, non-DWD weather
        source, network hiccup), we fall back to ``now`` so the predictor
        still emits instead of going dark — the PK invariant (a non-null
        Timestamp on every row) is preserved; the only thing lost is the
        join-back trace to the exact upstream forecast issue time. A
        debug log records each fallback so the cause is visible if the
        situation turns out to be persistent rather than transient.
        """
        if forecast_creation is None:
            logging.debug(
                "%s: forecast_creation unavailable (upstream "
                "weather.forecast.timestamp_creation not valid yet); "
                "falling back to now=%s for the PK column.",
                self.name, now,
            )
            forecast_creation = now

        # Dedup on the (target, creation) pair. A genuine new forecast at
        # the same ``now`` must flow through (this is the entire reason
        # for the timestamp_creation column); only an exact re-fire of the
        # same listener event no-ops out.
        key = (now, forecast_creation)
        if self._last_predicted_key == key:
            # Duplicate listener re-fires (forecast trigger spins on
            # future-dated channel.timestamp) hit this every time, so DEBUG.
            logging.debug(
                "%s: predict skipped — already published for (now=%s, creation=%s).",
                self.name, now, forecast_creation,
            )
            return

        forecast = self._fetch_forecast(now)
        if forecast is None or forecast.empty:
            logging.info(
                "%s: predict skipped — no forecast rows in [%s, %s].",
                self.name, now, now + self._horizon,
            )
            return

        field = self.context
        soil = getattr(field, "soil_simulation", None)
        if soil is None:
            logging.info("%s: predict skipped — no soil_simulation sibling.", self.name)
            return

        # Skip until the live solver has actually produced a state — predicting
        # off the static IC during the cold-start spin-up wastes compute and
        # publishes trajectories that don't match the field. SoilSimulation
        # sets ``_last_simulated_at`` from ``_save_state`` after each successful
        # advance, so this gate clears exactly when the first real callback
        # has committed a saturation field. DEBUG because cold-start refires
        # this gate many times before the first soil advance lands.
        if getattr(soil, "_last_simulated_at", None) is None:
            logging.debug(
                "%s: predict skipped — live solver has no state yet "
                "(cold-start still running) at %s.", self.name, now,
            )
            return

        try:
            et_data, seg_et = field._run_chain(forecast, publish=False)
        except Exception:  # noqa: BLE001
            logging.exception(
                "%s: chain replay on forecast failed; skipping tick.", self.name
            )
            return
        if et_data.empty or et_data.shape[0] < 2:
            logging.info(
                "%s: predict skipped — chain replay returned %d row(s), need ≥ 2.",
                self.name, et_data.shape[0],
            )
            return

        ic_rel_sat = soil.get_rel_sat_snapshot()
        try:
            timestamps, trajectories, snapshots, diagnostics = self._solve(
                ic_rel_sat, et_data, seg_et
            )
        except Exception:  # noqa: BLE001
            logging.exception(
                "%s: integration failed; skipping tick.", self.name
            )
            return

        # Wrap _publish_results: any exception here propagates up to
        # lories' Listener.__call__, whose ``finally: return self`` (see
        # listener.py:85) SWALLOWS the exception completely — no
        # traceback, no "Failed notifying listener" warning. Without
        # this try/except a converter / set_frame failure is invisible.
        try:
            self._publish_results(
                trajectories, self._probes, timestamps, snapshots,
                diagnostics, forecast_creation,
            )
        except Exception:  # noqa: BLE001
            logging.exception(
                "%s: publishing results failed; predictor channels stay "
                "stale this tick (now=%s, creation=%s).",
                self.name, now, forecast_creation,
            )
            return
        self._last_predicted_key = key
        logging.info(
            "%s: predict OK — %d probes, %d rows emitted "
            "(now=%s, creation=%s).",
            self.name, len(self._probes), len(timestamps),
            now, forecast_creation,
        )

    # =========================================================================
    # Backend split-point — replace this method with a remote call to offload
    # =========================================================================

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
        """Local PDE backend.

        A future server-attached deployment can swap this method for an
        RPC: the inputs (initial saturation field + ET-augmented forecast +
        per-segment ET decomposition) and the outputs (probe + diagnostic
        trajectories + optional state snapshots) are already a
        self-contained, pickle-friendly payload.

        Returns ``(timestamps, trajectories, snapshots, diagnostics)``.
        ``timestamps`` is the time axis from ``_integrate_horizon`` (the
        forecast row index); ``trajectories[probe_id]`` is the matching
        list of probe rel_sat values; ``snapshots`` maps save-tick
        timestamps to raw saturation-field arrays (empty when neither
        ``save_state`` nor ``save_plot`` is on); ``diagnostics[key]`` is
        the per-interval kg/(m²·h) flux density (one entry per forecast
        interval, ``NaN`` at the IC row where there is no interval to
        diagnose yet).
        """
        self._pde.set_state(ic_rel_sat)
        return self._integrate_horizon(et_data, seg_et)

    # =========================================================================
    # Forecast retrieval
    # =========================================================================

    def _fetch_forecast(self, now: pd.Timestamp) -> Optional[pd.DataFrame]:
        """Read the in-memory forecast cache and slice to ``[now, now+horizon]``.

        We deliberately bypass ``forecast.get()`` because it has a latent
        lories bug: the schedule-floor branch builds a pandas freq string
        with ``f"{self.interval}T"`` and ``self.interval`` is the
        unresolved Parameter descriptor (no ``__get__``, never converted
        to int), which raises ``ValueError: Invalid frequency format`` the
        moment the in-memory cache doesn't fully cover the range. Reading
        ``forecast.data.to_frame()`` directly skips that branch — we lose
        the logger gap-fill, but the forecast sub-component refreshes on
        its own schedule so the cache covers anything we need at predict
        time.
        """
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

    # =========================================================================
    # PDE integration over horizon
    # =========================================================================

    def _integrate_horizon(
        self,
        et_data: pd.DataFrame,
        seg_et: dict[str, pd.DataFrame],
    ) -> tuple[
        list[pd.Timestamp],
        dict[str, list[float]],
        dict[pd.Timestamp, np.ndarray],
        dict[str, list[float]],
    ]:
        """Step the predictor PDE through every (t, t+Δt) interval covered
        by the forecast. Per-segment ET / rain values are read from the
        forecast row at each step; no irrigation source is applied.

        Returns ``(timestamps, trajectories, snapshots, diagnostics)``.
        Diagnostics carry one entry per timestamp — ``NaN`` at the IC row
        (no interval to diagnose yet), then per-interval kg/(m²·h) flux
        densities computed from the same mass-balance math the live
        solver runs via :meth:`SoilBase._compute_diagnostics`.
        """
        dt_max = self._ode_config.dt
        dt_min = max(self._ode_config.dt_min, 1.0e-6)
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

        # Initial state — the IC the caller just reset.
        if len(idx) > 0:
            timestamps.append(idx[0])
            for p in self._probes:
                trajectories[p.channel_id].append(self._pde.sample(p))
            # No interval has elapsed yet at the IC, so diagnostics are NaN.
            for key in diagnostics:
                diagnostics[key].append(float("nan"))
            _maybe_snapshot(idx[0])

        for ts_prev, ts_next in zip(idx[:-1], idx[1:]):
            elapsed_s = (ts_next - ts_prev).total_seconds()
            if elapsed_s <= 0:
                continue

            seg_evap, seg_transp = self._segment_flux_dicts(seg_et, ts_next)
            rain_flux = self._rain_flux(et_data, ts_next, elapsed_s)
            rates = FluxRates(
                seg_evap=seg_evap,
                seg_transp=seg_transp,
                flow_m3s=0.0,
                rain_flux=rain_flux,
            )
            clip_total = ClipDiagnostics()
            storage_before = self._total_water()

            # HYDRUS-style adaptive walk over this forecast interval —
            # snapshot the state, attempt ``sub_dt``, halve on
            # non-convergence (down to ``dt_min``), grow back after fast
            # convergence. Mirrors :meth:`SoilSimulation._walk`; the
            # per-substep ``ClipDiagnostics`` are accumulated for the
            # interval-level diagnostic computation below.
            sub_dt = dt_max
            t_offset = 0.0
            while t_offset < elapsed_s - 1.0e-9:
                attempted = min(sub_dt, elapsed_s - t_offset)
                snapshot = self._pde.snapshot()
                clip = self._pde.apply_source(
                    seg_evap=rates.seg_evap,
                    seg_transp=rates.seg_transp,
                    rain_flux=rates.rain_flux,
                    flow_m3s=rates.flow_m3s,
                    dt=attempted,
                )
                result = self._pde.solve(attempted, log_name=self.name)
                if not result.converged and attempted > dt_min:
                    self._pde.set_state(snapshot)
                    sub_dt = max(dt_min, attempted / 3.0)
                    continue
                t_offset += attempted
                clip_total.add(clip)
                if result.converged and result.sweeps <= 3 and sub_dt < dt_max:
                    sub_dt = min(dt_max, sub_dt * 1.5)

            timestamps.append(ts_next)
            for p in self._probes:
                trajectories[p.channel_id].append(self._pde.sample(p))
            delta_storage = self._total_water() - storage_before
            interval_diag = self._compute_diagnostics(
                rates, delta_storage, elapsed_s, clip_total,
            )
            for key in diagnostics:
                diagnostics[key].append(interval_diag.get(key, float("nan")))
            _maybe_snapshot(ts_next)

        # Always archive the final row so the horizon-end state is
        # available even when it falls < save_freq after the last save.
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
        return float(precip) / elapsed_s   # mm/s == kg/(m²·s)

    # The PDE state (state set/snapshot, apply_source, solve, sample) is
    # owned by :class:`SoilPDECore` (``self._pde``) and shared with
    # :class:`SoilSimulation` to keep the two solvers in lock-step.

    # =========================================================================
    # Channel publishing
    # =========================================================================

    def _publish_results(
        self,
        trajectories: dict[str, list[float]],
        probes: list[ProbeSpec],
        timestamps: list[pd.Timestamp],
        snapshots: dict[pd.Timestamp, np.ndarray],
        diagnostics: dict[str, list[float]],
        forecast_creation: pd.Timestamp,
    ) -> None:
        # Per-probe emission: one row per forecast target timestamp, every
        # row stamped with the same ``forecast_creation``. The composite
        # ``(timestamp, timestamp_creation)`` PK lets multiple forecasts
        # addressing the same target coexist as distinct rows.
        #
        # ``self.data`` on a Component is a ``DataAccess`` (a registry-like
        # facade over individual channels); it does not expose a
        # frame-level ``set_frame``. The lories pattern for multi-row
        # channel values is to hand each channel a ``pd.Series`` indexed
        # by the row timestamps — ``Channel.to_series`` then forwards the
        # Series straight to the logger, which lands N rows in one INSERT
        # (mirrors what WeatherForecast does after a Brightsky read).
        if not timestamps:
            return

        index = pd.DatetimeIndex(timestamps, name="timestamp")

        # ``timestamp_creation``: same value on every emitted row (composite
        # PK partner). Broadcasting the scalar through pandas gives us a
        # Series of identical Timestamps, indexed by the target timestamps.
        creation_series = pd.Series(forecast_creation, index=index)
        self.data[self._TIMESTAMP_CREATION_KEY].set(index[0], creation_series)

        for probe in probes:
            key = self._channel_keys.get(probe.channel_id)
            if key is None:
                continue
            traj = trajectories.get(probe.channel_id, [])
            if len(traj) != len(index):
                # Defensive: integration produced a different sample
                # count than the shared time axis. Skip rather than
                # broadcast or truncate silently.
                continue
            self.data[key].set(index[0], pd.Series(traj, index=index, dtype=float))

        # Per-interval flux-density diagnostics on the same time axis. The
        # IC row carries NaN (no interval to diagnose yet); pandas
        # serialises that as SQL NULL so the joined-on-timestamp logger
        # frame stays consistent and the dashboard table renders the
        # row as "—".
        for constant in _DIAGNOSTIC_CONSTANTS:
            values = diagnostics.get(constant.key, [])
            if len(values) != len(index):
                continue
            self.data[constant.key].set(
                index[0], pd.Series(values, index=index, dtype=float),
            )

        if not snapshots:
            return

        # Snapshot rows share timestamps with probe rows (we sample
        # save_freq-aligned forecast rows), so the timestamp_creation
        # Series above already covers them. Sort once so state and plot
        # rows line up to the same time axis.
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
                    # Disable for the rest of this tick — a failure here
                    # must not strand the state / probe channels we've
                    # already (or are about to) publish above.
                    logging.exception(
                        "%s: predict_plot render failed at %s; skipping "
                        "remaining plot snapshots this tick.",
                        self.name, t,
                    )
                    plot_values = []
                    break
            if plot_values:
                self.data[self._PLOT_CHANNEL_KEY].set(
                    save_index[0],
                    pd.Series(plot_values, index=save_index, dtype=object),
                )

    @staticmethod
    def _encode_state(rel_sat: np.ndarray) -> bytes:
        buf = io.BytesIO()
        np.savez(buf, rel_sat=rel_sat)
        return buf.getvalue()

    def _render_snapshot_png(self, rel_sat: np.ndarray, sim_t: pd.Timestamp) -> bytes:
        """Render one saturation snapshot to PNG bytes via the shared
        :mod:`plot_render` helpers. Builds the fig/ax/norm lazily on the
        first call and reuses them across the tick so 24 hourly frames
        don't allocate 24 figures."""
        if self._plot_fig is None:
            self._plot_fig, self._plot_ax, self._plot_norm = plot_render.init_rel_sat_figure(
                self._mesh_config.width, self._mesh_config.height,
            )
        return plot_render.render_rel_sat_png(
            self._plot_fig, self._plot_ax, self._plot_norm,
            self._pde.mesh, rel_sat, sim_t,
            title="Predicted relative saturation",
        )
