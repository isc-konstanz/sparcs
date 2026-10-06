"""Pure objective metric for the soil-tuning bench: per-probe RMSE/bias
between a job's modeled tension series and the corresponding measured
tension series, plus a pooled aggregate.

Imports pandas/numpy/stdlib only -- never soil_tuning, dash, sparcs.*,
lories.*, or fipy (see ENVIRONMENT.md). No I/O, no logging of data values;
every function here is deterministic given its arguments.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_TENSION_SUFFIX = "__tension"


def modeled_tension_series(rows: List[dict]) -> Dict[str, pd.Series]:
    """Split a job's ``rows`` (list of dicts, one per sim step, each with a
    ``timestamp`` key and ``{probe_id}__tension`` columns) into one float
    series per probe, indexed by timestamp, keyed by the bare ``probe_id``
    (the ``__tension`` suffix stripped). NaNs are dropped."""
    if not rows:
        return {}

    frame = pd.DataFrame(rows)
    if "timestamp" not in frame.columns:
        return {}

    # utc=True matches _build_figure's cast of the same job rows: a tz-naive
    # index would inner-join to zero overlap against tz-aware measured data
    # and silently read as "insufficient_overlap" instead of failing loudly.
    frame = frame.set_index(pd.DatetimeIndex(pd.to_datetime(frame["timestamp"], utc=True)))

    out: Dict[str, pd.Series] = {}
    for col in frame.columns:
        col_str = str(col)
        if not col_str.endswith(_TENSION_SUFFIX):
            continue
        probe_id = col_str[: -len(_TENSION_SUFFIX)]
        series = frame[col].astype(float).dropna()
        out[probe_id] = series
    return out


def _utc_index(series: pd.Series) -> pd.Series:
    """Return ``series`` with its DatetimeIndex normalized to tz-aware UTC
    (naive indices are assumed to be UTC wall clock, matching main()'s own
    ``tz_localize("UTC")`` guard on the replay window). Both sides must land
    on the same tz-awareness or the inner join silently yields zero overlap."""
    index = pd.DatetimeIndex(pd.to_datetime(series.index))
    index = index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")
    series = series.copy()
    series.index = index
    return series


def _normalize_convention(series: pd.Series) -> pd.Series:
    """Negate ``series`` at most once so its median is <= 0 (the negative
    matric-potential convention), then drop the remaining positive samples
    as glitches. Never uses ``-abs()`` -- that would fold positive-tension
    glitches (real anomalies) into the signal instead of excluding them."""
    if series.empty:
        return series
    if float(series.median()) > 0:
        series = -series
    return series[series <= 0]


def tension_objective(
    modeled: Dict[str, pd.Series],
    measured: Dict[str, pd.Series],
    *,
    freq: str = "1h",
    min_samples: int = 3,
) -> dict:
    """Per-probe RMSE/bias/n between ``modeled`` and ``measured`` tension
    series, plus a pooled aggregate over every included probe's aligned
    samples.

    See ``.scratch/soil-tuning-api/issues/04-objective-metric.md`` section 4
    for the pinned normalization/exclusion/alignment/pooling steps. Units are
    never rescaled: hPa in, hPa out.
    """
    probes: Dict[str, dict] = {}
    skipped: List[dict] = []
    pooled_err: List[np.ndarray] = []

    probe_ids = sorted(set(modeled) | set(measured))
    for probe_id in probe_ids:
        modeled_series = modeled.get(probe_id)
        measured_series = measured.get(probe_id)

        if modeled_series is None or measured_series is None:
            skipped.append(
                {
                    "probe": probe_id,
                    "reason": "missing_side",
                }
            )
            continue

        modeled_norm = _normalize_convention(_utc_index(modeled_series.astype(float)))
        measured_norm = _normalize_convention(_utc_index(measured_series.astype(float)))

        if modeled_norm.empty or measured_norm.empty:
            skipped.append(
                {
                    "probe": probe_id,
                    "reason": "no_usable_samples",
                    "n": 0,
                }
            )
            continue

        modeled_resampled = modeled_norm.resample(freq).mean()
        measured_resampled = measured_norm.resample(freq).mean()

        aligned = pd.concat(
            [modeled_resampled.rename("modeled"), measured_resampled.rename("measured")],
            axis=1,
            join="inner",
        ).dropna()

        n = len(aligned)
        if n < min_samples:
            skipped.append(
                {
                    "probe": probe_id,
                    "reason": "insufficient_overlap",
                    "n": int(n),
                }
            )
            continue

        err = (aligned["modeled"] - aligned["measured"]).to_numpy(dtype=float)
        rmse = float(np.sqrt(np.mean(err**2)))
        bias = float(np.mean(err))
        probes[probe_id] = {"rmse": rmse, "bias": bias, "n": int(n)}
        pooled_err.append(err)

    if pooled_err:
        all_err = np.concatenate(pooled_err)
        aggregate: Optional[dict] = {
            "rmse": float(np.sqrt(np.mean(all_err**2))),
            "bias": float(np.mean(all_err)),
            "n": int(all_err.size),
        }
    else:
        aggregate = None

    return {
        "probes": probes,
        "aggregate": aggregate,
        "skipped": skipped,
        "freq": freq,
    }
