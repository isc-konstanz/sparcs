# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.simulation.core.candidates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pure candidate-grid, flow-schedule and scoring functions for the planner. In the glossary "ladder" means only
the front-loaded ``fill_order`` subset, so these functions say "candidate".
"""

from __future__ import annotations

import datetime
import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .pde import design_flow_lpm, flow_m3s_per_m
from .schedule import slot_floor

__all__ = [
    "WateringWindow",
    "current_boundary",
    "derive_flow_m3s",
    "build_flow_schedule",
    "split_interval",
    "build_candidate_grid",
    "check_candidate_cap",
    "resolve_window_start",
    "score_candidate",
    "select_candidate",
    "total_minutes",
]


@dataclass(frozen=True)
class WateringWindow:
    """One configured watering window: a site-local clock time the emitters start at."""

    start: datetime.time


def current_boundary(now: pd.Timestamp, tz, interval_min: int, offset_min: int) -> pd.Timestamp:
    """Most recent run boundary at or before ``now``, site-local."""
    return slot_floor(now, tz, interval_min, offset_min)


def derive_flow_m3s(
    nozzle_count: int,
    nozzle_flow_lph: float,
    total_drip_line_length_m: float,
) -> float:
    """Design flow from the drip layout (nozzle output x count) in m³/s per out-of-plane metre of row."""
    return flow_m3s_per_m(design_flow_lpm(nozzle_count, nozzle_flow_lph), total_drip_line_length_m)


def build_flow_schedule(
    windows: list[WateringWindow],
    durations: list[pd.Timedelta],
    flow_m3s: float,
    horizon_start: pd.Timestamp,
    horizon_end: pd.Timestamp,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """One candidate's on intervals from its per-window durations; zero durations add none, ends clamp to horizon_end.
    ``flow_m3s`` is not stored; callers apply it during every returned interval and zero elsewhere."""
    intervals: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for window, duration in zip(windows, durations):
        if duration <= pd.Timedelta(0):
            continue
        on_ts = resolve_window_start(window.start, horizon_start)
        off_ts = min(on_ts + duration, horizon_end)
        intervals.append((on_ts, off_ts))
    return intervals


def split_interval(
    on_intervals: list[tuple[pd.Timestamp, pd.Timestamp]],
    ts_prev: pd.Timestamp,
    ts_next: pd.Timestamp,
    flow_m3s: float,
) -> list[tuple[float, float]]:
    """Split ``[ts_prev, ts_next]`` at every on/off edge strictly inside it into ``(sub_window_s, flow)`` pairs.
    Flow is ``flow_m3s`` inside an on-interval, else 0.0; empty ``on_intervals`` give one zero-flow segment."""
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


def build_candidate_grid(
    window_durations: list[list[pd.Timedelta]],
    grid_mode: str,
) -> list[tuple[pd.Timedelta, ...]]:
    """Candidates (one duration per window) as the ``fill_order`` ladder or the ``full`` Cartesian product.
    ``fill_order``: window 0 gives all its durations, each later window its non-zero ones after every earlier max."""
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


def check_candidate_cap(
    ladder: list[tuple[pd.Timedelta, ...]],
    combo_cap: int,
    log_name: str = "",
) -> None:
    """Raise ``ValueError`` when the ladder has more than ``combo_cap`` candidates."""
    if len(ladder) > combo_cap:
        raise ValueError(
            f"{log_name}: ladder has {len(ladder)} candidates, exceeding "
            f"combo_cap={combo_cap}; reduce the per-window durations lists, "
            "raise combo_cap, or drop windows."
        )


def resolve_window_start(start: datetime.time, horizon_start: pd.Timestamp) -> pd.Timestamp:
    """Resolve a window's clock time onto ``horizon_start``'s date, or the next calendar day if already elapsed.
    The next day's wall-clock fields are set again rather than adding 24h, so the local time holds across DST."""
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


def score_candidate(
    trajectory: tuple[list[pd.Timestamp], dict[str, list[float]]],
    decision_probes: list[str],
    threshold_hpa: float,
) -> float:
    """RMS distance of the decision probes' tension from the ``threshold_hpa`` setpoint; lower is better.
    The setpoint is a target, not a ceiling; returns ``+inf`` when no decision probe is in the trajectory."""
    _timestamps, probe_series = trajectory
    deviations: list[np.ndarray] = []
    for channel_id in decision_probes:
        tension_values = probe_series.get(channel_id)
        if not tension_values:
            continue
        # Trajectories are signed negative hPa; compare their magnitude against the positive setpoint.
        deviations.append(np.abs(np.asarray(tension_values, dtype=float)) - threshold_hpa)
    if not deviations:
        return float("inf")
    stacked = np.concatenate(deviations)
    return float(np.sqrt(np.mean(np.square(stacked))))


def select_candidate(
    ladder: list[tuple[pd.Timedelta, ...]],
    trajectories: dict[tuple[pd.Timedelta, ...], tuple[list[pd.Timestamp], dict[str, list[float]]]],
    decision_probes: list[str],
    threshold_hpa: float,
    grid_mode: str,
) -> tuple[pd.Timedelta, ...]:
    """Lowest ``score_candidate``, ties to the fewest ``total_minutes``; raises ``ValueError`` on an empty ladder.
    ``fill_order`` searches only the ladder and may miss the optimum; ``grid_mode = "full"`` is exact."""
    if not ladder:
        raise ValueError("_select requires a non-empty ladder.")
    if grid_mode not in ("fill_order", "full"):
        raise ValueError(f"Unknown grid_mode {grid_mode!r}; expected 'fill_order' or 'full'.")

    scores = {c: score_candidate(trajectories[c], decision_probes, threshold_hpa) for c in ladder}
    return min(ladder, key=lambda c: (scores[c], total_minutes(c)))


def total_minutes(candidate: tuple[pd.Timedelta, ...]) -> float:
    """Total watering minutes across a candidate's per-window durations."""
    return sum((d.total_seconds() / 60.0 for d in candidate), 0.0)
