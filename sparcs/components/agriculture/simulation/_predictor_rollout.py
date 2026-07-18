# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.simulation._predictor_rollout
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Roll-out executor extracted from ``SoilPredictor``: the sequential
prefix-shared caterpillar, the independent reference roll, the parallel
spawn-pool executor, and the module-level spawn workers. ``RolloutEngine``
is a per-call struct of the rollout inputs; ``SoilPredictor`` keeps
``_roll_segment`` / ``_rollout_ladder`` / ``_rollout_independent`` /
``_rollout_parallel`` as thin delegates that assemble an engine per call
from the instance's loose attributes, and ``_rollout_dispatch`` (the
parallel-vs-caterpillar routing) stays on the predictor. Nothing in this
module imports ``soil_predictor`` (would cycle); the spawn workers
therefore rebuild a ``RolloutEngine`` -- not a ``SoilPredictor`` -- in
``_worker_init``.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from lories.components.weather import Weather

from ._predictor_candidates import WateringWindow, build_flow_schedule, resolve_window_start, split_interval
from ._soil import FluxRates, MeshConfig, PDEConfig, ProbeSpec, SoilPDECore, ensure_mesh

logger = logging.getLogger(__name__)


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


def _rain_flux(et_data: pd.DataFrame, ts: pd.Timestamp, elapsed_s: float) -> float:
    col = Weather.PRECIPITATION
    if elapsed_s <= 0 or col not in et_data.columns or ts not in et_data.index:
        return 0.0
    precip = et_data.loc[ts, col]
    if pd.isna(precip) or precip <= 0:
        return 0.0
    return float(precip) / elapsed_s  # mm/s == kg/(m²·s)


class RolloutEngine:
    """Per-call view of the rollout inputs: the PDE plus the loose config
    fields the roll methods read. Every field defaults to ``None`` so a
    delegate can assemble an engine from a bare test predictor that only
    carries the attributes the invoked method actually reads.
    """

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
    ) -> tuple[list[pd.Timestamp], dict[str, list[float]]]:
        """Walk the PDE across ``idx`` (>=1 forecast timestamps; the live PDE state is
        already the state at ``idx[0]``), applying ``on_intervals`` inside each
        ``(t, t+dt)`` sub-interval via ``split_interval``. Returns per-forecast-
        timestamp Se at every probe, including ``idx[0]`` (sampled as-is, no walk).

        Shared by the prefix roll and every per-window sweep in ``rollout_ladder``,
        and by ``rollout_independent``'s single full-horizon roll.

        ``snapshot_sink``, when given, is called with each recorded forecast timestamp
        right after that state is reached -- the live ``self.pde`` is the field at
        that timestamp, so the sink can read ``self.pde.snapshot()``. Only the debug
        field-plot re-roll passes it; the forecast/recommendation paths leave it
        ``None`` (no per-step cost).
        """
        timestamps: list[pd.Timestamp] = [idx[0]]
        trajectories: dict[str, list[float]] = {p.channel_id: [self.pde.sample(p)] for p in self.probes}
        if snapshot_sink is not None:
            snapshot_sink(idx[0])

        for ts_prev, ts_next in zip(idx[:-1], idx[1:]):
            elapsed_s = (ts_next - ts_prev).total_seconds()
            if elapsed_s <= 0:
                timestamps.append(ts_next)
                for p in self.probes:
                    trajectories[p.channel_id].append(self.pde.sample(p))
                if snapshot_sink is not None:
                    snapshot_sink(ts_next)
                continue

            seg_evap, seg_transp = _segment_flux_dicts(seg_et, ts_next)
            rain_flux = _rain_flux(et_data, ts_next, elapsed_s)
            sub_segments = split_interval(on_intervals, ts_prev, ts_next, self.flow_m3s)
            for sub_window_s, sub_flow_m3s in sub_segments:
                if sub_window_s <= 0.0:
                    continue
                sub_rates = FluxRates(
                    seg_evap=seg_evap,
                    seg_transp=seg_transp,
                    flow_m3s=sub_flow_m3s,
                    rain_flux=rain_flux,
                )
                self.pde.walk_window(
                    rates=sub_rates,
                    window_s=sub_window_s,
                    accept_at_dt_min=True,
                    log_name=self.name,
                )

            timestamps.append(ts_next)
            for p in self.probes:
                trajectories[p.channel_id].append(self.pde.sample(p))
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
        """Caterpillar roll-out (``self.grid_mode == "fill_order"``): integrate the
        shared prefix once, then sweep each window's ladder-contributed durations
        from a save of the max-prefix branch, saving/restoring branch state with
        ``save_state_blob``/``load_state_blob`` (never ``snapshot``/``set_state`` --
        the latter drop the ``surface_h`` ponds that watering fills; see the module
        docstring / PRD).

        The fill-order chain's prefix-sharing only applies to the ``fill_order``
        candidate set: for ``self.grid_mode == "full"`` (the full Cartesian
        product, not a single chain) every candidate is rolled independently via
        ``rollout_independent`` instead, with no prefix sharing.

        ``ladder`` is ``build_candidate_grid``'s output; ``self.windows`` (ordered by
        ``start``) supplies the window clock times. Window starts are assumed to fall
        exactly on a forecast timestamp in ``et_data.index`` (the common on-the-hour
        case); if a window start does not land on a forecast timestamp, the nearest
        forecast timestamp at or before it is used as the segment boundary instead.

        Returns ``{candidate: (timestamps, {probe_id: [Se, ...]})}`` for every rung.
        """
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
                # Positions before i carry EACH earlier window's OWN max (maxima[j]),
                # not window i's max -- the state already reflects every earlier
                # window at its own max (via the max-branch save below), so the key
                # must label that accurately to match build_candidate_grid's candidates.
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
                # A window whose durations are only 0min contributes no rungs, so the
                # max-branch save above never ran; still advance the shared prefix
                # across its segment, or every later window's roll would silently
                # skip the weather in [bounds[i], bounds[i+1]].
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
        """Reference roll-out for one candidate: reset to the IC and integrate the
        whole horizon in a single pass with no prefix sharing. Ground truth that
        ``rollout_ladder``'s per-candidate trajectory must match.
        """
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
        """Roll every ladder candidate as an independent parallel roll across an
        in-component spawn ``ProcessPoolExecutor``. Each worker rebuilds the PDE once
        from the pickled ``MeshConfig`` + ``PDEConfig`` (spawn-safe) and rolls its
        assigned candidates via ``rollout_independent``; the parent gathers the
        ``{candidate: (timestamps, trajectories)}`` map and does every downstream
        step (select, frame, write) serially. Same candidate set and same stored
        trajectories as the caterpillar within solver tolerance (``docs/adr/0005-...``).

        The pool is created and torn down per call -- daily cadence makes the setup
        cost negligible. Raises on pool/worker failure so ``_rollout_dispatch`` can
        degrade to the caterpillar.
        """
        ladder = self.ladder
        # No point spawning more workers than candidates; always at least one.
        n_workers = max(1, min(self.max_workers, len(ladder)))
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
    # Belt-and-braces: rollout_parallel already exported these in the parent
    # before spawning (they must be in the environment before the child's numpy
    # import for OpenMP/OpenBLAS to honor them); restated here for any caller
    # that builds a pool without going through rollout_parallel.
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
    """ProcessPoolExecutor task: roll one candidate on this worker's rebuilt PDE
    via the reference ``rollout_independent``. The payload is only the candidate
    tuple; every other input comes from the worker globals set by ``_worker_init``.
    Returns ``(candidate, (timestamps, trajectories))`` so the parent can key the
    gathered map without tracking submission order.
    """
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
