# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.simulation.soil_predictor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Forecast-driven horizon predictor. For each watering strategy in
``[0%, 20%, ..., 100%]`` (where 100% defaults to 1 L/h), integrates the
Richards-equation soil PDE over the available weather forecast and
publishes the predicted relative saturation at every probe defined on the
live ``SoilSimulation``. The live solver is never touched: the predictor
owns its own FiPy mesh, equation, and ``CellVariable``s.

The expensive compute step is isolated in :meth:`SoilPredictor._solve_strategies`
so a future server-attached backend (sparcs running on an edge device,
strategy roll-outs offloaded to a server) can replace just that one
method with an HTTP / RPC call without touching forecast retrieval,
chain replay, or channel publishing.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from lories import Component
from lories.components.weather import Weather
from lories.typing import Configurations
from lories.util import to_timedelta

from ._soil import SoilBase

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


_DEFAULT_STRATEGIES: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
# 100 % strategy = 1 L/h. SoilSimulation expresses irrigation in L/min, so
# 1 L/h = 1/60 L/min. Override via [soil_predictor].flow_lpm_at_100.
_DEFAULT_FLOW_LPM_AT_100: float = 1.0 / 60.0
_DEFAULT_HORIZON: str = "24h"


@dataclass
class StrategyResult:
    """One strategy's prediction.

    ``end_rel_sat`` is the full saturation field at horizon end (kept so
    the end-state can be archived as a bytes channel and so a future
    remote backend has a single typed payload to return).
    ``probe_trajectories`` maps probe channel id → predicted
    volume-weighted mean rel_sat sampled at each forecast row. The time
    axis is shared across probes and strategies; the parent emits it on
    a separate ``predict_timestamps`` channel each tick.
    """

    strategy: float
    end_rel_sat: np.ndarray
    probe_trajectories: dict[str, list[float]]


class SoilPredictor(SoilBase):
    REL_SAT_NAME: str = "predictor relative saturation"

    TYPE: str = "soil_predictor"
    INCLUDES = ["pde"]

    _strategies: tuple[float, ...]
    _flow_lpm_at_100: float
    _horizon: pd.Timedelta
    _save_end_states: bool

    _mesh_config: object
    _ode_config: PDEConfig
    _pde: SoilPDECore

    # (probe_channel_id, strategy_fraction) → predict-channel id.
    # The literal key ``"__state__"`` is reserved for the optional
    # per-strategy end-state archive channel.
    _channel_keys: dict[tuple[str, float], str]

    _probes: list[ProbeSpec]

    # Last ``now`` we ran a prediction for. The simulation callback fires
    # once per input-channel update; multiple channels updating at the same
    # data timestamp (e.g. all weather columns at the top of a Brightsky
    # hour) re-fire the listener with identical ``now``. Without dedup,
    # ``set(now, value)`` lands twice on the same channel → CSV gets two
    # rows with identical index → ``validate_index`` rejects on read-back.
    _last_predicted_at: Optional[pd.Timestamp] = None

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

        self._strategies = tuple(
            float(s) for s in configs.get("strategies", default=list(_DEFAULT_STRATEGIES))
        )
        self._flow_lpm_at_100 = float(
            configs.get("flow_lpm_at_100", default=_DEFAULT_FLOW_LPM_AT_100)
        )
        self._horizon = to_timedelta(configs.get("horizon", default=_DEFAULT_HORIZON))
        self._save_end_states = configs.get_bool("save_end_states", default=False)

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

        # Channel layout: predict_<probe_id>_w<pct> per (probe, strategy),
        # plus optional predict_state_w<pct> for the saturation field.
        # Each per-probe channel holds a ``list[float]`` — the predicted
        # rel_sat trajectory sampled at each forecast row. The shared
        # ``predict_timestamps`` channel publishes the matching time axis.
        self._channel_keys = {}
        self._timestamps_key = "predict_timestamps"
        for probe in self._probes:
            for s in self._strategies:
                pct = int(round(s * 100))
                key = f"predict_{probe.channel_id}_w{pct:03d}"
                self._channel_keys[(probe.channel_id, s)] = key
                self.data.add(
                    key,
                    type=list,
                    name=f"Predicted {probe.name} @ {pct}% irrigation",
                    unit="-",
                    aggregate="last",
                    logger={"enabled": True},
                )
        if self._probes:
            self.data.add(
                self._timestamps_key,
                type=list,
                name="Predicted Trajectory Timestamps",
                aggregate="last",
                logger={"enabled": True},
            )
        if self._save_end_states:
            for s in self._strategies:
                pct = int(round(s * 100))
                key = f"predict_state_w{pct:03d}"
                self._channel_keys[("__state__", s)] = key
                self.data.add(
                    key,
                    type=bytes,
                    name=f"Predicted soil state @ {pct}% irrigation",
                    aggregate="last",
                    logger={"enabled": True},
                )

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

    def predict(self, now: pd.Timestamp) -> None:
        """One prediction tick. Silently skips when no forecast is available
        or the live soil solver hasn't produced a state yet — predictions
        are best-effort and must not break the live callback chain."""
        # Dedup: ``how="any"`` listener fires once per updated input channel,
        # so multiple channels updating at the same data timestamp would
        # re-enter ``predict()`` with identical ``now`` and write duplicate
        # CSV rows. Skip if we've already produced a prediction for this
        # timestamp (or an earlier one, which can happen if a logger replay
        # rolls the index backwards).
        if self._last_predicted_at is not None and now <= self._last_predicted_at:
            return

        forecast = self._fetch_forecast(now)
        if forecast is None or forecast.empty:
            return

        field = self.context
        soil = getattr(field, "soil_simulation", None)
        if soil is None:
            return

        # Skip until the live solver has actually produced a state — predicting
        # off the static IC during the cold-start spin-up wastes compute and
        # publishes trajectories that don't match the field. SoilSimulation
        # sets ``_last_simulated_at`` from ``_save_state`` after each successful
        # advance, so this gate clears exactly when the first real callback
        # has committed a saturation field.
        if getattr(soil, "_last_simulated_at", None) is None:
            logging.debug(
                "%s: live solver has no state yet (cold-start still running); "
                "skipping forecast tick at %s.", self.name, now,
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
            # Need at least two rows to define one (t, t+Δt) integration step.
            return

        ic_rel_sat = soil.get_rel_sat_snapshot()
        try:
            results = self._solve_strategies(ic_rel_sat, et_data, seg_et)
        except Exception:  # noqa: BLE001
            logging.exception(
                "%s: strategy integration failed; skipping tick.", self.name
            )
            return

        self._publish_results(now, results, self._probes)
        self._last_predicted_at = now

    # =========================================================================
    # Backend split-point — replace this method with a remote call to offload
    # =========================================================================

    def _solve_strategies(
        self,
        ic_rel_sat: np.ndarray,
        et_data: pd.DataFrame,
        seg_et: dict[str, pd.DataFrame],
    ) -> list[StrategyResult]:
        """Local PDE backend.

        A future server-attached deployment can swap this method for an
        RPC: the inputs (initial saturation field + ET-augmented forecast +
        per-segment ET decomposition) and the ``StrategyResult`` outputs
        are already a self-contained, pickle-friendly payload.
        """
        results: list[StrategyResult] = []
        # ``_integrate_horizon`` returns the same time axis for every strategy
        # (it's just the forecast row index); keep the last one so
        # ``_publish_results`` can emit it on the shared timestamps channel.
        self._last_trajectory_timestamps: list[pd.Timestamp] = []
        for strategy in self._strategies:
            flow_m3s = strategy * self._flow_lpm_at_100 / 60_000.0
            self._pde.set_state(ic_rel_sat)
            timestamps, trajectories = self._integrate_horizon(et_data, seg_et, flow_m3s)
            end_state = self._pde.snapshot()
            results.append(StrategyResult(strategy, end_state, trajectories))
            self._last_trajectory_timestamps = timestamps
        return results

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
        flow_m3s: float,
    ) -> tuple[list[pd.Timestamp], dict[str, list[float]]]:
        """Step the predictor PDE through every (t, t+Δt) interval covered
        by the forecast. Per-segment ET / rain values are read from the
        forecast row at each step; the irrigation flow is held constant
        over the horizon at the strategy's value.

        Returns ``(timestamps, trajectories)`` where ``timestamps`` is the
        list of forecast rows sampled (including the initial state) and
        ``trajectories[probe_id]`` is the matching list of probe
        rel_sat values.
        """
        dt_max = self._ode_config.dt
        dt_min = max(self._ode_config.dt_min, 1.0e-6)
        idx = et_data.index
        timestamps: list[pd.Timestamp] = []
        trajectories: dict[str, list[float]] = {p.channel_id: [] for p in self._probes}

        # Initial state — the IC the caller just reset.
        if len(idx) > 0:
            timestamps.append(idx[0])
            for p in self._probes:
                trajectories[p.channel_id].append(self._pde.sample(p))

        for ts_prev, ts_next in zip(idx[:-1], idx[1:]):
            elapsed_s = (ts_next - ts_prev).total_seconds()
            if elapsed_s <= 0:
                continue

            seg_evap, seg_transp = self._segment_flux_dicts(seg_et, ts_next)
            rain_flux = self._rain_flux(et_data, ts_next, elapsed_s)

            # HYDRUS-style adaptive walk over this forecast interval —
            # snapshot the state, attempt ``sub_dt``, halve on
            # non-convergence (down to ``dt_min``), grow back after fast
            # convergence. Mirrors :meth:`SoilSimulation._walk`.
            sub_dt = dt_max
            t_offset = 0.0
            while t_offset < elapsed_s - 1.0e-9:
                attempted = min(sub_dt, elapsed_s - t_offset)
                snapshot = self._pde.snapshot()
                self._pde.apply_source(
                    seg_evap=seg_evap,
                    seg_transp=seg_transp,
                    rain_flux=rain_flux,
                    flow_m3s=flow_m3s,
                    dt=attempted,
                )
                result = self._pde.solve(attempted, log_name=self.name)
                if not result.converged and attempted > dt_min:
                    self._pde.set_state(snapshot)
                    sub_dt = max(dt_min, attempted / 3.0)
                    continue
                t_offset += attempted
                if result.converged and result.sweeps <= 3 and sub_dt < dt_max:
                    sub_dt = min(dt_max, sub_dt * 1.5)

            timestamps.append(ts_next)
            for p in self._probes:
                trajectories[p.channel_id].append(self._pde.sample(p))

        return timestamps, trajectories

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
        now: pd.Timestamp,
        results: list[StrategyResult],
        probes: list[ProbeSpec],
    ) -> None:
        # Shared time axis — published once per tick so downstream readers
        # can zip ``predict_timestamps`` with each per-probe trajectory.
        ts_strs = [t.isoformat() for t in self._last_trajectory_timestamps]
        if ts_strs and self._timestamps_key in self.data:
            self.data[self._timestamps_key].set(now, ts_strs)

        for result in results:
            for probe in probes:
                key = self._channel_keys.get((probe.channel_id, result.strategy))
                if key is None:
                    continue
                traj = result.probe_trajectories.get(probe.channel_id, [])
                self.data[key].set(now, list(traj))
            if self._save_end_states:
                key = self._channel_keys.get(("__state__", result.strategy))
                if key is None:
                    continue
                buf = io.BytesIO()
                np.savez(buf, rel_sat=result.end_rel_sat)
                self.data[key].set(now, buf.getvalue())

