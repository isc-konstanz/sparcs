# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.simulation.core.rollout
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Roll-out executor for the planner: the sequential prefix-sharing roll, the independent reference roll
and the parallel spawn-pool roll, whose workers rebuild a ``RolloutEngine`` in ``_worker_init``.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from .candidates import WateringWindow, build_flow_schedule, resolve_window_start, split_interval
from .pde import ClipDiagnostics, FluxRates, MeshConfig, PDEConfig, ProbeSpec, SoilPDECore, ensure_mesh
from .pde import rain_flux as _rain_flux
from .pde import segment_flux_dicts as _segment_flux_dicts

logger = logging.getLogger(__name__)


class RolloutEngine:
    """Per-call view of the PDE and the config fields the roll methods read; every field defaults to ``None``."""

    def __init__(
        self,
        *,
        pde: Optional[SoilPDECore] = None,
        probes: Optional[list[ProbeSpec]] = None,
        windows: Optional[list[WateringWindow]] = None,
        window_durations: Optional[list[list[pd.Timedelta]]] = None,
        flow_m3s: Optional[float] = None,
        grid_mode: Optional[str] = None,
        ladder: Optional[list[tuple[pd.Timedelta, ...]]] = None,
        max_workers: Optional[int] = None,
        name: Optional[str] = None,
        mesh_config: Optional[MeshConfig] = None,
        ode_config: Optional[PDEConfig] = None,
        rel_sat_name: Optional[str] = None,
    ) -> None:
        self.pde = pde
        self.probes = probes
        self.windows = windows
        self.window_durations = window_durations
        self.flow_m3s = flow_m3s
        self.grid_mode = grid_mode
        self.ladder = ladder
        self.max_workers = max_workers
        self.name = name
        self.mesh_config = mesh_config
        self.ode_config = ode_config
        self.rel_sat_name = rel_sat_name

    def roll_segment(
        self,
        idx: pd.DatetimeIndex,
        et_data: pd.DataFrame,
        seg_et: dict[str, pd.DataFrame],
        on_intervals: list[tuple[pd.Timestamp, pd.Timestamp]],
        snapshot_sink: Optional[Callable[[pd.Timestamp], None]] = None,
        interval_begin: Optional[Callable[[pd.Timestamp, pd.Timestamp, float], None]] = None,
        interval_end: Optional[Callable[..., None]] = None,
        sample_on_zero_dt: bool = True,
    ) -> tuple[list[pd.Timestamp], dict[str, list[float]]]:
        """Walk the PDE from its current state across ``idx``; returns Se per probe, ``idx[0]`` sampled as-is.
        ``snapshot_sink`` sees ``self.pde`` at each recorded timestamp; observers skip zero-dt intervals."""
        timestamps: list[pd.Timestamp] = [idx[0]]
        trajectories: dict[str, list[float]] = {p.channel_id: [self.pde.sample(p)] for p in self.probes}
        if snapshot_sink is not None:
            snapshot_sink(idx[0])

        for ts_prev, ts_next in zip(idx[:-1], idx[1:]):
            elapsed_s = (ts_next - ts_prev).total_seconds()
            if elapsed_s <= 0:
                if not sample_on_zero_dt:
                    continue
                timestamps.append(ts_next)
                for p in self.probes:
                    trajectories[p.channel_id].append(self.pde.sample(p))
                if snapshot_sink is not None:
                    snapshot_sink(ts_next)
                continue

            seg_evap, seg_transp = _segment_flux_dicts(seg_et, ts_next)
            rain_flux = _rain_flux(et_data, ts_next, elapsed_s)
            if interval_begin is not None:
                interval_begin(ts_prev, ts_next, elapsed_s)
            sub_segments = split_interval(on_intervals, ts_prev, ts_next, self.flow_m3s)
            clip_total = ClipDiagnostics() if interval_end is not None else None
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
                walk = self.pde.walk_window(
                    rates=sub_rates,
                    window_s=sub_window_s,
                    accept_at_dt_min=True,
                    log_name=self.name,
                )
                if clip_total is not None:
                    clip_total.add(walk.clip)
                    irrigated_mass += sub_flow_m3s * sub_window_s

            timestamps.append(ts_next)
            for p in self.probes:
                trajectories[p.channel_id].append(self.pde.sample(p))
            if interval_end is not None:
                interval_end(ts_next, elapsed_s, seg_evap, seg_transp, rain_flux, clip_total, irrigated_mass)
            if snapshot_sink is not None:
                snapshot_sink(ts_next)

        return timestamps, trajectories

    @staticmethod
    def extend_trajectory(
        base_timestamps: list[pd.Timestamp],
        base_trajectories: dict[str, list[float]],
        tail_timestamps: list[pd.Timestamp],
        tail_trajectories: dict[str, list[float]],
    ) -> tuple[list[pd.Timestamp], dict[str, list[float]]]:
        """Concatenate ``base`` and ``tail``, dropping the tail's first timestamp, which repeats the base's last."""
        timestamps = list(base_timestamps) + list(tail_timestamps[1:])
        trajectories = {
            channel_id: list(base_trajectories[channel_id]) + list(tail_trajectories[channel_id][1:])
            for channel_id in base_trajectories
        }
        return timestamps, trajectories

    def rollout_ladder(
        self,
        ic_rel_sat: np.ndarray,
        ladder: list[tuple[pd.Timedelta, ...]],
        et_data: pd.DataFrame,
        seg_et: dict[str, pd.DataFrame],
        flow_m3s: float,
        horizon_start: pd.Timestamp,
        horizon_end: pd.Timestamp,
    ) -> dict[tuple[pd.Timedelta, ...], tuple[list[pd.Timestamp], dict[str, list[float]]]]:
        """Roll the shared prefix once, then each window from a saved branch; grid_mode "full" rolls candidates alone.
        Branches use save_state_blob/load_state_blob because snapshot/set_state drop the surface_h ponds."""
        windows = self.windows
        idx = et_data.index
        results: dict[tuple[pd.Timedelta, ...], tuple[list[pd.Timestamp], dict[str, list[float]]]] = {}

        if self.grid_mode == "full" or not windows:
            # No caterpillar prefix-sharing for the full Cartesian product (or the
            # no-windows degenerate case): roll every candidate independently.
            for candidate in ladder:
                results[candidate] = self.rollout_independent(
                    ic_rel_sat, candidate, et_data, seg_et, flow_m3s, horizon_start, horizon_end
                )
            return results

        self.pde.set_state(ic_rel_sat)

        maxima = [max(durations) for durations in self.window_durations]
        window_starts = [resolve_window_start(w.start, horizon_start) for w in windows]

        def _floor_idx(ts: pd.Timestamp) -> pd.Timestamp:
            eligible = idx[idx <= ts]
            return eligible[-1] if len(eligible) > 0 else idx[0]

        segment_bounds = [_floor_idx(ts) for ts in window_starts] + [horizon_end if horizon_end in idx else idx[-1]]

        # Prefix sharing needs strictly increasing floored bounds; two windows in one forecast interval, or a window
        # rolled to the next day past a later one, would drop or misattribute water, so roll candidates alone.
        if not all(segment_bounds[k] < segment_bounds[k + 1] for k in range(len(segment_bounds) - 1)):
            logger.debug(
                "%s: caterpillar segment bounds not strictly increasing (%s); "
                "falling back to independent per-candidate rolls.",
                self.name,
                segment_bounds,
            )
            for candidate in ladder:
                results[candidate] = self.rollout_independent(
                    ic_rel_sat, candidate, et_data, seg_et, flow_m3s, horizon_start, horizon_end
                )
            return results

        prefix_idx = idx[idx <= segment_bounds[0]]
        prefix_timestamps, prefix_trajectories = self.roll_segment(prefix_idx, et_data, seg_et, [])
        prev_blob = self.pde.save_state_blob()

        for i, window in enumerate(windows):
            seg_start = segment_bounds[i]
            seg_end = segment_bounds[i + 1]
            seg_idx = idx[(idx >= seg_start) & (idx <= seg_end)]

            durations = self.window_durations[i]
            sweep = durations if i == 0 else [d for d in durations if d > pd.Timedelta(0)]
            max_duration = maxima[i]

            for d_i in sweep:
                self.pde.load_state_blob(prev_blob)
                on_intervals = build_flow_schedule([window], [d_i], flow_m3s, seg_start, horizon_end)
                tail_timestamps, tail_trajectories = self.roll_segment(
                    idx[idx >= seg_start], et_data, seg_et, on_intervals
                )
                full_timestamps, full_trajectories = self.extend_trajectory(
                    prefix_timestamps, prefix_trajectories, tail_timestamps, tail_trajectories
                )
                # Positions before i carry each earlier window's own max, which the state already holds
                # through the max-branch save below; this matches build_candidate_grid's keys.
                candidate = tuple(
                    maxima[j] if j < i else (d_i if j == i else pd.Timedelta(0)) for j in range(len(windows))
                )
                results[candidate] = (full_timestamps, full_trajectories)

                if d_i == max_duration and i + 1 < len(windows):
                    self.pde.load_state_blob(prev_blob)
                    on_intervals_seg = build_flow_schedule([window], [d_i], flow_m3s, seg_start, seg_end)
                    seg_timestamps, seg_trajectories = self.roll_segment(seg_idx, et_data, seg_et, on_intervals_seg)
                    prefix_timestamps, prefix_trajectories = self.extend_trajectory(
                        prefix_timestamps, prefix_trajectories, seg_timestamps, seg_trajectories
                    )
                    prev_blob = self.pde.save_state_blob()

            if not sweep and i + 1 < len(windows):
                # An all-zero window adds no rungs and skips the max-branch save; still advance the prefix
                # across its segment, or every later window would skip that segment's weather.
                self.pde.load_state_blob(prev_blob)
                seg_timestamps, seg_trajectories = self.roll_segment(seg_idx, et_data, seg_et, [])
                prefix_timestamps, prefix_trajectories = self.extend_trajectory(
                    prefix_timestamps, prefix_trajectories, seg_timestamps, seg_trajectories
                )
                prev_blob = self.pde.save_state_blob()

        return results

    def rollout_independent(
        self,
        ic_rel_sat: np.ndarray,
        candidate: tuple[pd.Timedelta, ...],
        et_data: pd.DataFrame,
        seg_et: dict[str, pd.DataFrame],
        flow_m3s: float,
        horizon_start: pd.Timestamp,
        horizon_end: pd.Timestamp,
    ) -> tuple[list[pd.Timestamp], dict[str, list[float]]]:
        """Reference roll for one candidate: reset to the IC and integrate the whole horizon in one pass.
        ``rollout_ladder``'s per-candidate trajectories must match it."""
        self.pde.set_state(ic_rel_sat)
        on_intervals = build_flow_schedule(self.windows, list(candidate), flow_m3s, horizon_start, horizon_end)
        idx = et_data.index
        return self.roll_segment(idx, et_data, seg_et, on_intervals)

    def rollout_parallel(
        self,
        ic_rel_sat: np.ndarray,
        et_data: pd.DataFrame,
        seg_et: dict[str, pd.DataFrame],
        horizon_start: pd.Timestamp,
        horizon_end: pd.Timestamp,
    ) -> dict[tuple[pd.Timedelta, ...], tuple[list[pd.Timestamp], dict[str, list[float]]]]:
        """Roll each ladder candidate with ``rollout_independent`` in a per-call spawn ``ProcessPoolExecutor``.
        Matches ``rollout_ladder`` within solver tolerance; raises on pool or worker failure."""
        ladder = self.ladder
        # No point spawning more workers than candidates; always at least one.
        n_workers = max(1, min(self.max_workers, len(ladder)))
        # Spawn children read these at numpy import, before _worker_init runs, so set them in the parent and
        # restore them in finally; the parent's numpy is already loaded. Not safe for concurrent rollouts.
        prior_omp_num_threads = os.environ.get("OMP_NUM_THREADS")
        prior_kmp_duplicate_lib_ok = os.environ.get("KMP_DUPLICATE_LIB_OK")
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        try:
            ctx = multiprocessing.get_context("spawn")
            # lories Constant column labels do not unpickle in a spawn worker; send plain str labels.
            et_data = _stringify_columns(et_data)
            seg_et = {name: _stringify_columns(frame) for name, frame in seg_et.items()}
            initargs = (
                self.mesh_config,
                self.ode_config,
                self.rel_sat_name,
                self.name,
                self.probes,
                self.windows,
                self.flow_m3s,
                self.grid_mode,
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


def _stringify_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Copy ``frame`` with plain ``str`` column labels; unpickling a lories ``Constant`` label calls it with key=None.
    A Constant equals its key str, so Constant-keyed lookups still match."""
    return frame.rename(columns=str)


# --- Parallel-executor workers: module level so the spawn start method can import them by name ---
# Each worker reuses one SoilPDECore across its candidates; _WORKER holds the shared inputs, so a task is the candidate.
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
    """ProcessPoolExecutor initializer, once per worker: one OpenMP thread, a PDE rebuilt from config,
    and the shared run inputs stashed in ``_WORKER``."""
    # rollout_parallel already set these before spawning; repeated for a pool built without it.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    ensure_mesh(mesh_config)
    engine = RolloutEngine(
        pde=SoilPDECore(mesh_config, ode_config, rel_sat_name=rel_sat_name),
        probes=probes,
        windows=windows,
        flow_m3s=flow_m3s,
        grid_mode=grid_mode,
        name=name,
    )

    _WORKER.clear()
    _WORKER["engine"] = engine
    _WORKER["ic_rel_sat"] = ic_rel_sat
    _WORKER["et_data"] = et_data
    _WORKER["seg_et"] = seg_et
    _WORKER["flow_m3s"] = flow_m3s
    _WORKER["horizon_start"] = horizon_start
    _WORKER["horizon_end"] = horizon_end


def _worker_roll(
    candidate: tuple[pd.Timedelta, ...],
) -> tuple[tuple[pd.Timedelta, ...], tuple[list[pd.Timestamp], dict[str, list[float]]]]:
    """ProcessPoolExecutor task: roll one candidate with ``rollout_independent`` on this worker's PDE.
    Returns the candidate with its result so the parent need not track submission order."""
    engine = _WORKER["engine"]
    result = engine.rollout_independent(
        _WORKER["ic_rel_sat"],
        candidate,
        _WORKER["et_data"],
        _WORKER["seg_et"],
        _WORKER["flow_m3s"],
        _WORKER["horizon_start"],
        _WORKER["horizon_end"],
    )
    return candidate, result
