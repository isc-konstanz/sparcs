# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.simulation._predictor_candidates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pure candidate-grid/schedule/scoring statics extracted from
``SoilPredictor``, renamed to glossary-correct
names -- ``context/sparcs.md`` reserves "ladder" for the strictly
front-loaded ``fill_order`` subset, not the full candidate space, so the
grid-building/selection statics live here as ``build_candidate_grid`` /
``select_candidate`` etc. ``SoilPredictor`` keeps every pinned ``_x`` name
as a ``staticmethod`` alias onto the matching function here; nothing in
this module imports ``soil_predictor`` (would cycle).
"""

from __future__ import annotations

import copy
import datetime
import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd
from lories.typing import Configurations

from ._schedule import slot_floor
from ._soil import (
    PDEConfig,
    apply_surface_forcing,
    design_flow_lpm,
    flow_m3s_per_m,
    resolve_pde_config,
)

__all__ = [
    "WateringWindow",
    "resolve_ode_config",
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
    """One configured watering window: a site-local clock time the emitters start at.

    The candidate duration for a given ladder rung is passed alongside, not stored
    here, so the same window definition is reused across every candidate.
    """

    start: datetime.time


def resolve_ode_config(
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
        return resolve_pde_config(configs, model_configs, inherit_forcing_from=soil_pde)
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


def current_boundary(now: pd.Timestamp, tz, interval_min: int, offset_min: int) -> pd.Timestamp:
    """Most-recent run boundary at or before ``now``, site-local. Mirrors the
    interval/offset pattern of lories ``WeatherForecast`` (forecast.py).

    Thin wrapper over ``_schedule.slot_floor`` -- see that module's
    docstring for why the slot math has one home.
    """
    return slot_floor(now, tz, interval_min, offset_min)


def derive_flow_m3s(
    nozzle_count: int,
    nozzle_flow_lph: float,
    total_drip_line_length_m: float,
) -> float:
    """Fixed design flow from the drip layout: nozzle output x count, normalized
    per out-of-plane metre of row.

    The l/min core is the shared ``design_flow_lpm`` (also fed to the live sim
    when its physical meter is unavailable); here it is DERIVED from the layout
    instead of read from the meter, then normalized to m³/s per metre of row via
    the shared ``flow_m3s_per_m`` (also fed by ``SoilSimulation._compute_flux_rates``).
    """
    return flow_m3s_per_m(design_flow_lpm(nozzle_count, nozzle_flow_lph), total_drip_line_length_m)


def build_flow_schedule(
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


def build_candidate_grid(
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


def check_candidate_cap(
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


def resolve_window_start(start: datetime.time, horizon_start: pd.Timestamp) -> pd.Timestamp:
    """Resolve a window's clock time onto ``horizon_start``'s date, rolling
    forward a **calendar** day if that time already elapsed before
    ``horizon_start``. The roll-forward re-resolves the wall-clock fields on the
    next calendar day rather than adding a fixed ``Timedelta(days=1)``, so the
    result stays at the intended local clock time across a DST transition (a
    fixed 24h add would land an hour off on the spring-forward / fall-back night).
    The single canonical resolver; ``build_flow_schedule`` calls it too.
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


def score_candidate(
    trajectory: tuple[list[pd.Timestamp], dict[str, list[float]]],
    decision_probes: list[str],
    threshold_hpa: float,
) -> float:
    """RMS distance of a candidate's water tension from the setpoint
    ``threshold_hpa``, over the whole horizon, pooled across the decision
    probes. Lower is better; ``select_candidate`` takes the argmin.

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


def select_candidate(
    ladder: list[tuple[pd.Timedelta, ...]],
    trajectories: dict[tuple[pd.Timedelta, ...], tuple[list[pd.Timestamp], dict[str, list[float]]]],
    decision_probes: list[str],
    threshold_hpa: float,
    grid_mode: str,
) -> tuple[pd.Timedelta, ...]:
    """Select the recommended candidate: the rung whose water-tension
    trajectory tracks the ``threshold_hpa`` setpoint most closely, scored by
    ``score_candidate`` (RMS-to-setpoint, lower is better).

    Both grid modes reduce to the same rule -- score every candidate and take
    the argmin, breaking ties by least total watering (``total_minutes``) for a
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

    scores = {c: score_candidate(trajectories[c], decision_probes, threshold_hpa) for c in ladder}
    return min(ladder, key=lambda c: (scores[c], total_minutes(c)))


def total_minutes(candidate: tuple[pd.Timedelta, ...]) -> float:
    """Total watering minutes across a candidate's per-window durations."""
    return sum((d.total_seconds() / 60.0 for d in candidate), 0.0)
