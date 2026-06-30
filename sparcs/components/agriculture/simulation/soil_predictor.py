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

import io
import logging
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
from ._soil import FluxRates, MeshConfig, SoilBase

_DEFAULT_HORIZON: str = "24h"
_DEFAULT_SAVE_FREQ: str = "1h"

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
    ) -> tuple[
        list[pd.Timestamp],
        dict[str, list[float]],
        dict[pd.Timestamp, np.ndarray],
        dict[str, list[float]],
    ]:
        """Step the predictor PDE through every (t, t+Δt) forecast interval; no irrigation applied.

        Returns ``(timestamps, trajectories, snapshots, diagnostics)``.
        ``diagnostics`` values are kg/(m²·h) flux densities; NaN at the IC row (no interval yet).
        """
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
            rates = FluxRates(
                seg_evap=seg_evap,
                seg_transp=seg_transp,
                flow_m3s=0.0,
                rain_flux=rain_flux,
            )
            storage_before = self._total_water()

            walk = self._pde.walk_window(
                rates=rates,
                window_s=elapsed_s,
                accept_at_dt_min=True,
                log_name=self.name,
            )
            clip_total = walk.clip

            timestamps.append(ts_next)
            for p in self._probes:
                trajectories[p.channel_id].append(self._pde.sample(p))
            delta_storage = self._total_water() - storage_before
            interval_diag = self._compute_diagnostics(
                rates,
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
