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
import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
from lories.components.weather import Weather
from lories.typing import Configurations
from lories.util import to_timedelta
from sparcs.components.agriculture.simulation.soil import (
    PDEConfig,
    ProbeSpec,
    SoilPDECore,
    SoilSimulation,
    resolve_probes,
)

from . import plot_render
from ._soil import ClipDiagnostics, FluxRates, MeshConfig, SoilBase

_DEFAULT_HORIZON: str = "24h"
_DEFAULT_SAVE_FREQ: str = "1h"

# Drip-flow derivation defaults; mirrors the [soil_simulation] default of a
# single already-per-metre line (see soil.py:186) when a field has no
# [soil_predictor.drip] block of its own.
_DEFAULT_NOZZLE_FLOW_LPH: float = 1.0
_DEFAULT_NOZZLE_COUNT: int = 1
_DEFAULT_DRIP_LINE_LENGTH_M: float = 1.0


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

    _horizon: pd.Timedelta
    _save_freq: pd.Timedelta
    _save_state: bool
    _save_plot: bool

    _mesh_config: MeshConfig
    _ode_config: PDEConfig
    _pde: SoilPDECore

    # Fixed design flow derived from the drip layout (m³/s per out-of-plane
    # metre of row); see _derive_flow_m3s.
    _flow_m3s: float

    _plot_fig: Any = None
    _plot_ax: Any = None
    _plot_norm: Any = None

    _channel_keys: dict[str, str]

    _probes: list[ProbeSpec]

    # Dedup gate: skip if (now, forecast_creation) was already published.
    _last_predicted_key: Optional[tuple[pd.Timestamp, pd.Timestamp]] = None

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

        model_block = self.context.configs.get_member("model", defaults={}, ensure_exists=True)
        soil_block = self.context.configs.get_member(SoilSimulation.TYPE, defaults={}, ensure_exists=True)
        soil_pde = PDEConfig(
            soil_block.get_member("pde", defaults={}, ensure_exists=True),
            model_configs=model_block,
        )

        drip_block = configs.get_member("drip", defaults={}, ensure_exists=True)
        nozzle_flow_lph = drip_block.get_float("nozzle_flow_lph", default=_DEFAULT_NOZZLE_FLOW_LPH)
        nozzle_count = drip_block.get_int("nozzle_count", default=_DEFAULT_NOZZLE_COUNT)
        total_drip_line_length_m = soil_block.get_float("total_drip_line_length_m", default=_DEFAULT_DRIP_LINE_LENGTH_M)
        self._flow_m3s = self._derive_flow_m3s(nozzle_count, nozzle_flow_lph, total_drip_line_length_m)

        if configs.has_member("pde"):
            self._ode_config = PDEConfig(configs.get_member("pde"), model_configs=model_block)
        else:
            self._ode_config = soil_pde

        self._pde = self._build_pde()

        probes_cfg = soil_block.get_member("probes", defaults={}, ensure_exists=True)
        self._probes = resolve_probes(
            probes_cfg,
            self._pde.mesh,
            self._mesh_config,
            log_name=self.name,
        )

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
                unit="-",
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
            timestamps, trajectories, snapshots, diagnostics = self._solve(ic_rel_sat, et_data, seg_et)
        except Exception:  # noqa: BLE001
            logging.exception("%s: integration failed; skipping tick.", self.name)
            return

        try:
            self._publish_results(
                trajectories,
                self._probes,
                timestamps,
                snapshots,
                diagnostics,
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
            len(timestamps),
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
            on_ts = horizon_start.replace(
                hour=window.start.hour,
                minute=window.start.minute,
                second=window.start.second,
                microsecond=window.start.microsecond,
            )
            if on_ts < horizon_start:
                on_ts += pd.Timedelta(days=1)
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
        interval at zero flow -- the behavior-identity guard for the current
        zero-flow roll.
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

    @staticmethod
    def _encode_state(rel_sat: np.ndarray) -> bytes:
        buf = io.BytesIO()
        np.savez(buf, rel_sat=rel_sat)
        return buf.getvalue()

    def _render_snapshot_png(self, rel_sat: np.ndarray, sim_t: pd.Timestamp) -> bytes:
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
            title="Predicted relative saturation",
        )
