# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.fieldsim.core.planner
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Irrigation planning over a forecast horizon, on the shared engine and a state
snapshot. Candidate enumeration, scoring and roll-out are the live functions;
the frame builders keep the live forecast-table column vocabulary.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from lories.core import ConfigurationError
from sparcs.components.agriculture.simulation._predictor_candidates import (
    WateringWindow,
    build_candidate_grid,
    build_flow_schedule,
    check_candidate_cap,
    derive_flow_m3s,
    select_candidate,
    total_minutes,
)
from sparcs.components.agriculture.simulation._predictor_rollout import RolloutEngine
from sparcs.components.agriculture.simulation._predictor_tables import _merge_irrigation_intervals, forecast_ids
from sparcs.components.agriculture.simulation._soil import ProbeSpec

from .config import DripConfig, PlannerConfig
from .engine import SoilEngine
from .state import Plan, SoilState

logger = logging.getLogger(__name__)

Candidate = tuple  # tuple[pd.Timedelta, ...]
RawTrajectory = tuple  # tuple[list[pd.Timestamp], dict[str, list[float]]]

_HEADER_FORECAST_ID_KEY = "forecast_id"
_HEADER_IS_RECOMMENDED_KEY = "is_recommended"
_HEADER_TOTAL_MIN_KEY = "total_min"
_HEADER_WEATHER_CREATION_KEY = "weather_creation"
_DETAIL_TIMESTAMP_CREATION_SUFFIX = "_timestamp_creation"
_DETAIL_FORECAST_ID_SUFFIX = "_forecast_id"
_IRRIGATION_STATE_KEY = "irrigation_state"
_IRRIGATION_TIMESTAMP_CREATION_KEY = "irrigation_timestamp_creation"


def build_header_frame(
    ladder: Sequence[Candidate],
    chosen: Optional[Candidate],
    run_timestamp: pd.Timestamp,
    weather_creation: Optional[pd.Timestamp],
    windows: Sequence[WateringWindow],
    max_windows: int,
) -> pd.DataFrame:
    """One row per ladder candidate, indexed at ``run_timestamp``."""
    ids = forecast_ids(list(ladder))
    window_min_keys = [f"w{i}_min" for i in range(max_windows)]
    window_start_keys = [f"w{i}_start" for i in range(max_windows)]
    columns = [
        _HEADER_FORECAST_ID_KEY,
        *window_min_keys,
        *window_start_keys,
        _HEADER_IS_RECOMMENDED_KEY,
        _HEADER_TOTAL_MIN_KEY,
        _HEADER_WEATHER_CREATION_KEY,
    ]
    if not ladder:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    index: list[pd.Timestamp] = []
    for candidate in ladder:
        row: dict[str, Any] = {
            _HEADER_FORECAST_ID_KEY: ids[candidate],
            _HEADER_IS_RECOMMENDED_KEY: candidate == chosen,
            _HEADER_TOTAL_MIN_KEY: total_minutes(candidate),
            _HEADER_WEATHER_CREATION_KEY: weather_creation,
        }
        for i, key in enumerate(window_min_keys):
            row[key] = candidate[i].total_seconds() / 60.0 if i < len(candidate) else None
        for i, key in enumerate(window_start_keys):
            row[key] = windows[i].start.strftime("%H:%M") if i < len(windows) else None
        rows.append(row)
        index.append(run_timestamp)

    frame = pd.DataFrame.from_records(rows, index=pd.DatetimeIndex(index, name="timestamp"))
    return frame.loc[:, columns]


def build_detail_frame(
    ladder: Sequence[Candidate],
    ladder_trajectories: Mapping[Candidate, RawTrajectory],
    run_timestamp: pd.Timestamp,
    probe_ids: Sequence[str],
) -> pd.DataFrame:
    """One LONG row per candidate x forecast-timestamp x probe, populating only that probe's columns."""
    ids = forecast_ids(list(ladder))
    tension_keys = {p: f"traj_{p}" for p in probe_ids}
    creation_keys = {p: f"traj_{p}{_DETAIL_TIMESTAMP_CREATION_SUFFIX}" for p in probe_ids}
    forecast_id_keys = {p: f"traj_{p}{_DETAIL_FORECAST_ID_SUFFIX}" for p in probe_ids}
    columns: list[str] = []
    for probe_id in probe_ids:
        columns += [tension_keys[probe_id], creation_keys[probe_id], forecast_id_keys[probe_id]]

    rows: list[dict[str, Any]] = []
    index: list[pd.Timestamp] = []
    for candidate, (timestamps, probe_series) in ladder_trajectories.items():
        forecast_id = ids[candidate]
        for probe_id, tension_key in tension_keys.items():
            values = probe_series.get(probe_id)
            if not values:
                continue
            creation_key = creation_keys[probe_id]
            forecast_id_key = forecast_id_keys[probe_id]
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


def build_irrigation_frame(
    candidate: Candidate,
    windows: Sequence[WateringWindow],
    flow_m3s: float,
    horizon_start: pd.Timestamp,
    horizon_end: pd.Timestamp,
    run_timestamp: pd.Timestamp,
) -> pd.DataFrame:
    """An ``(on_ts, True)`` / ``(off_ts, False)`` row pair per merged on-interval."""
    columns = [_IRRIGATION_STATE_KEY, _IRRIGATION_TIMESTAMP_CREATION_KEY]
    schedule = build_flow_schedule(list(windows), list(candidate), flow_m3s, horizon_start, horizon_end)
    intervals = _merge_irrigation_intervals(schedule)
    if not intervals:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    index: list[pd.Timestamp] = []
    for on_ts, off_ts in intervals:
        rows.append({_IRRIGATION_STATE_KEY: True, _IRRIGATION_TIMESTAMP_CREATION_KEY: run_timestamp})
        index.append(on_ts)
        rows.append({_IRRIGATION_STATE_KEY: False, _IRRIGATION_TIMESTAMP_CREATION_KEY: run_timestamp})
        index.append(off_ts)

    frame = pd.DataFrame.from_records(rows, index=pd.DatetimeIndex(index, name="timestamp"))
    return frame.loc[:, columns]


class IrrigationPlanner:
    def __init__(
        self,
        config: PlannerConfig,
        engine: SoilEngine,
        *,
        probes: Sequence[ProbeSpec],
        drip: DripConfig,
        total_drip_line_length_m: float,
        name: str = "fieldsim.planner",
    ) -> None:
        self.config = config
        self.engine = engine
        self.name = name
        self._probes = list(probes)
        self._probe_ids = [p.channel_id for p in self._probes]

        self._flow_m3s = derive_flow_m3s(drip.nozzle_count, drip.nozzle_flow_lph, total_drip_line_length_m)

        self._windows: list[WateringWindow] = []
        self._window_durations: list[list[pd.Timedelta]] = []
        for key, window_cfg in (config.windows or {}).items():
            if "durations" not in window_cfg:
                raise ConfigurationError(f"{name}: [windows.{key}] is missing its 'durations' list.")
            start = pd.Timestamp(str(window_cfg["start"])).time()
            durations = sorted({pd.Timedelta(str(d)) for d in window_cfg["durations"]})
            if pd.Timedelta(0) not in durations:
                raise ValueError(
                    f"{name}: [windows.{key}] is missing a '0min' duration; "
                    "every window's durations list must include zero."
                )
            self._windows.append(WateringWindow(start=start))
            self._window_durations.append(durations)
        order = sorted(range(len(self._windows)), key=lambda i: self._windows[i].start)
        self._windows = [self._windows[i] for i in order]
        self._window_durations = [self._window_durations[i] for i in order]

        self._grid_mode = config.grid_mode
        self._ladder: list[Candidate] = build_candidate_grid(self._window_durations, self._grid_mode)
        check_candidate_cap(self._ladder, config.combo_cap, log_name=name)

        decision_probes = [str(p) for p in config.decision_probes]
        if not decision_probes:
            decision_probes = list(self._probe_ids)
            if self._probe_ids:
                logger.warning(
                    "%s: decision_probes not configured; using ALL probes (%s) for the "
                    "tension decision -- surface and deep probes may distort the result.",
                    name,
                    decision_probes,
                )
        self._decision_probes = decision_probes
        self._threshold_hpa = config.threshold_hpa
        self._max_windows = config.max_windows

        max_workers = config.max_workers
        self._max_workers = max_workers if max_workers is not None else max(1, (os.cpu_count() or 2) - 1)

    @property
    def has_windows(self) -> bool:
        return bool(self._windows)

    def plan(
        self,
        state: SoilState,
        weather: pd.DataFrame,
        seg_et: Mapping[str, pd.DataFrame],
        horizon_start: pd.Timestamp,
        horizon_end: pd.Timestamp,
        *,
        run_timestamp: pd.Timestamp,
        weather_creation: Optional[pd.Timestamp] = None,
    ) -> Plan:
        """Roll the candidates, select one against ``threshold_hpa`` and build the frames.

        Without configured windows the zero-flow roll is the plan (``chosen=None``).
        The engine's identity cache is always invalidated on the way out.
        """
        try:
            ic_se = np.asarray(state.se, dtype=float)
            zero_timestamps, zero_se = self._zero_flow_rollout(ic_se, weather, seg_et)
            zero_tension = self._to_tension(zero_se)

            if not self.has_windows:
                zero_candidate: Candidate = ()
                ladder = [zero_candidate]
                ladder_tension = {zero_candidate: (zero_timestamps, zero_tension)}
                trajectories = {zero_candidate: self._trajectory_frame(zero_timestamps, zero_tension)}
                header = build_header_frame(
                    ladder, None, run_timestamp, weather_creation, self._windows, self._max_windows
                )
                detail = build_detail_frame(ladder, ladder_tension, run_timestamp, self._probe_ids)
                irrigation = build_irrigation_frame(
                    zero_candidate, self._windows, self._flow_m3s, horizon_start, horizon_end, run_timestamp
                )
                return Plan(chosen=None, trajectories=trajectories, header=header, detail=detail, irrigation=irrigation)

            raw_trajectories = self.rollout(ic_se, weather, seg_et, horizon_start, horizon_end)
            ladder_tension = {
                candidate: (timestamps, self._to_tension(se_traj))
                for candidate, (timestamps, se_traj) in raw_trajectories.items()
            }
            trajectories = {
                candidate: self._trajectory_frame(timestamps, tension)
                for candidate, (timestamps, tension) in ladder_tension.items()
            }

            chosen = select_candidate(
                self._ladder, ladder_tension, self._decision_probes, self._threshold_hpa, self._grid_mode
            )

            header = build_header_frame(
                self._ladder, chosen, run_timestamp, weather_creation, self._windows, self._max_windows
            )
            detail = build_detail_frame(self._ladder, ladder_tension, run_timestamp, self._probe_ids)
            irrigation = build_irrigation_frame(
                chosen, self._windows, self._flow_m3s, horizon_start, horizon_end, run_timestamp
            )
            return Plan(chosen=chosen, trajectories=trajectories, header=header, detail=detail, irrigation=irrigation)
        finally:
            self.engine.invalidate()

    def rollout(
        self,
        ic_se: np.ndarray,
        weather: pd.DataFrame,
        seg_et: Mapping[str, pd.DataFrame],
        horizon_start: pd.Timestamp,
        horizon_end: pd.Timestamp,
    ) -> dict[Candidate, RawTrajectory]:
        """Sequential ladder or parallel process pool; a parallel failure degrades to the ladder."""
        engine = self._rollout_engine()
        if self.config.parallel:
            try:
                return engine.rollout_parallel(ic_se, weather, seg_et, horizon_start, horizon_end)
            except Exception:  # noqa: BLE001
                logger.exception(
                    "%s: parallel roll-out failed; falling back to the sequential caterpillar for this run.",
                    self.name,
                )
        return engine.rollout_ladder(ic_se, self._ladder, weather, seg_et, self._flow_m3s, horizon_start, horizon_end)

    def _zero_flow_rollout(
        self,
        ic_se: np.ndarray,
        weather: pd.DataFrame,
        seg_et: Mapping[str, pd.DataFrame],
    ) -> RawTrajectory:
        engine = self._rollout_engine()
        engine.pde.set_state(ic_se)
        return engine.roll_segment(weather.index, weather, seg_et, [])

    def _rollout_engine(self) -> RolloutEngine:
        return RolloutEngine(
            pde=self.engine.pde,
            probes=self._probes,
            windows=self._windows,
            window_durations=self._window_durations,
            flow_m3s=self._flow_m3s,
            grid_mode=self._grid_mode,
            ladder=self._ladder,
            max_workers=self._max_workers,
            name=self.name,
            mesh_config=self.engine.mesh_config,
            ode_config=self.engine.ode,
            rel_sat_name=getattr(self.engine.pde.rel_sat, "name", "relative saturation"),
        )

    def _to_tension(self, se_traj: Mapping[str, list[float]]) -> dict[str, list[float]]:
        """Se -> signed water tension (negative hPa) per probe."""
        out: dict[str, list[float]] = {}
        for probe_id, values in se_traj.items():
            tension = np.asarray(self.engine.model.psi_from_se(np.asarray(values, dtype=float)), dtype=float)
            out[probe_id] = [float(v) for v in tension]
        return out

    @staticmethod
    def _trajectory_frame(timestamps: Sequence[pd.Timestamp], tension: Mapping[str, list[float]]) -> pd.DataFrame:
        return pd.DataFrame(dict(tension), index=pd.DatetimeIndex(timestamps, name="timestamp"))
