# -*- coding: utf-8 -*-
"""Objective metric: modeled_tension_series row-splitting and tension_objective's
normalization/exclusion/alignment/pooling steps (issue 04), against synthetic
series only -- no FiPy, no measured data, no network. ``soil_tuning_objective``
imports pandas/numpy/stdlib only, so ``importorskip`` is a formality here (kept
for consistency with the project's optional-dep test convention).
"""

from __future__ import annotations

import json

import pytest

import numpy as np
import pandas as pd

soil_tuning_objective = pytest.importorskip("soil_tuning_objective")

from soil_tuning_objective import modeled_tension_series, tension_objective  # noqa: E402

RNG_SEED = 20260702


def _series(start: str, n: int, freq: str, values) -> pd.Series:
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    return pd.Series(list(values), index=idx, dtype=float)


def _constant(start: str, n: int, freq: str, value: float) -> pd.Series:
    return _series(start, n, freq, [value] * n)


# ---------------------------------------------------------------------------
# modeled_tension_series
# ---------------------------------------------------------------------------


def test_modeled_tension_series_splits_rows_by_probe_and_drops_nan():
    rows = [
        {"timestamp": "2026-06-01T00:00:00Z", "a__tension": -100.0, "b__tension": -50.0, "a__se": 0.4},
        {"timestamp": "2026-06-01T01:00:00Z", "a__tension": -110.0, "b__tension": float("nan")},
        {"timestamp": "2026-06-01T02:00:00Z", "a__tension": -120.0, "b__tension": -55.0},
    ]
    out = modeled_tension_series(rows)
    assert set(out) == {"a", "b"}
    assert len(out["a"]) == 3
    assert len(out["b"]) == 2  # the NaN row dropped
    assert out["a"].dtype == float


def test_modeled_tension_series_empty_rows():
    assert modeled_tension_series([]) == {}


# ---------------------------------------------------------------------------
# tension_objective: DoD bullets
# ---------------------------------------------------------------------------


def test_constant_offset_gives_exact_bias_and_rmse():
    measured = {"p1": _constant("2026-06-01", 10, "1h", -200.0)}
    offset = 37.5
    modeled = {"p1": _constant("2026-06-01", 10, "1h", -200.0 + offset)}

    result = tension_objective(modeled, measured)
    probe = result["probes"]["p1"]
    assert probe["bias"] == pytest.approx(offset, abs=1e-9)
    assert probe["rmse"] == pytest.approx(abs(offset), abs=1e-9)
    assert probe["n"] == 10


def test_seeded_gaussian_noise_rmse_matches_sigma():
    rng = np.random.default_rng(RNG_SEED)
    n = 500
    sigma = 15.0
    base = _constant("2026-06-01", n, "1min", -300.0)
    noise = rng.normal(loc=0.0, scale=sigma, size=n)
    measured = {"p1": base}
    modeled = {"p1": pd.Series(base.values + noise, index=base.index)}

    # freq="1min" keeps the resample a no-op so the injected noise survives intact.
    result = tension_objective(modeled, measured, freq="1min", min_samples=3)
    probe = result["probes"]["p1"]
    assert probe["rmse"] == pytest.approx(sigma, rel=0.15)


def test_positive_spike_glitches_excluded_matches_clean_series():
    measured_clean = _constant("2026-06-01", 12, "1h", -150.0)
    modeled = {"p1": _constant("2026-06-01", 12, "1h", -140.0)}

    clean_result = tension_objective(modeled, {"p1": measured_clean})

    spiked_values = measured_clean.values.copy()
    spiked_values[3] = 250.0  # positive-tension glitch
    spiked_values[7] = 999.0
    measured_spiked = pd.Series(spiked_values, index=measured_clean.index)

    spiked_result = tension_objective(modeled, {"p1": measured_spiked})

    # The 2 spiked timestamps are excluded outright (not folded into the
    # signal), so rmse/bias match the clean series exactly even though n is
    # smaller by exactly the spike count.
    clean_probe = clean_result["probes"]["p1"]
    spiked_probe = spiked_result["probes"]["p1"]
    assert spiked_probe["rmse"] == pytest.approx(clean_probe["rmse"], abs=1e-9)
    assert spiked_probe["bias"] == pytest.approx(clean_probe["bias"], abs=1e-9)
    assert spiked_probe["n"] == clean_probe["n"] - 2


def test_positive_magnitude_convention_matches_negative_convention_not_double_negated():
    measured_negative = _constant("2026-06-01", 8, "1h", -180.0)
    measured_positive = _constant("2026-06-01", 8, "1h", 180.0)  # magnitude convention
    modeled = {"p1": _constant("2026-06-01", 8, "1h", -170.0)}

    result_neg = tension_objective(modeled, {"p1": measured_negative})
    result_pos = tension_objective(modeled, {"p1": measured_positive})

    assert result_neg["probes"]["p1"] == result_pos["probes"]["p1"]
    # modeled(-170) - measured(-180) = +10: bias must be positive, not
    # flipped by an accidental double negation of the positive-convention input.
    assert result_pos["probes"]["p1"]["bias"] == pytest.approx(10.0, abs=1e-9)


def test_misaligned_timestamps_align_via_freq():
    # Modeled: minutely for 6 hours, covering hourly bins 00:00..05:00.
    modeled = {"p1": _constant("2026-06-01T00:00:00", 360, "1min", -200.0)}
    # Measured: hourly, offset 20 minutes from modeled's hour marks, but each
    # timestamp still falls in the same [HH:00, HH:00+1h) resample bin.
    measured = {"p1": _constant("2026-06-01T00:20:00", 6, "1h", -190.0)}

    result = tension_objective(modeled, measured, freq="1h")
    probe = result["probes"]["p1"]
    assert probe["n"] >= 3
    assert probe["bias"] == pytest.approx(-10.0, abs=1e-6)


def test_overlap_below_min_samples_is_skipped_not_reported_as_probe():
    modeled = {"p1": _constant("2026-06-01", 2, "1h", -200.0)}
    measured = {"p1": _constant("2026-06-01", 2, "1h", -190.0)}

    result = tension_objective(modeled, measured, min_samples=3)
    assert "p1" not in result["probes"]
    reasons = {s["probe"]: s["reason"] for s in result["skipped"]}
    assert reasons["p1"] == "insufficient_overlap"
    assert result["aggregate"] is None


def test_probe_on_one_side_only_is_skipped():
    modeled = {"p1": _constant("2026-06-01", 5, "1h", -200.0)}
    measured = {"p2": _constant("2026-06-01", 5, "1h", -190.0)}

    result = tension_objective(modeled, measured)
    assert result["probes"] == {}
    reasons = {s["probe"]: s["reason"] for s in result["skipped"]}
    assert reasons["p1"] == "missing_side"
    assert reasons["p2"] == "missing_side"
    assert result["aggregate"] is None


def test_aggregate_equals_pooled_computation_over_concatenated_errors():
    modeled = {
        "p1": _constant("2026-06-01", 6, "1h", -200.0),
        "p2": _constant("2026-06-01", 10, "1h", -300.0),
    }
    measured = {
        "p1": _constant("2026-06-01", 6, "1h", -190.0),
        "p2": _constant("2026-06-01", 10, "1h", -280.0),
    }

    result = tension_objective(modeled, measured)

    # Hand-rolled reference: concatenate the two probes' aligned errors, then
    # compute rmse/bias/n over the pooled array (NOT a mean of per-probe values).
    err1 = np.full(6, -200.0 - (-190.0))
    err2 = np.full(10, -300.0 - (-280.0))
    all_err = np.concatenate([err1, err2])
    expected_rmse = float(np.sqrt(np.mean(all_err**2)))
    expected_bias = float(np.mean(all_err))
    expected_n = int(all_err.size)

    agg = result["aggregate"]
    assert agg["n"] == expected_n
    assert agg["rmse"] == pytest.approx(expected_rmse, abs=1e-9)
    assert agg["bias"] == pytest.approx(expected_bias, abs=1e-9)

    # A naive mean-of-per-probe-values would give a different (wrong) number
    # here since the two probes have different sample counts.
    naive_mean_bias = (result["probes"]["p1"]["bias"] + result["probes"]["p2"]["bias"]) / 2
    assert agg["bias"] == pytest.approx(-16.25, abs=1e-9)
    assert naive_mean_bias == pytest.approx(-15.0, abs=1e-9)
    assert agg["bias"] != pytest.approx(naive_mean_bias, abs=1e-6)


def test_json_dumps_succeeds():
    modeled = {"p1": _constant("2026-06-01", 5, "1h", -200.0)}
    measured = {"p1": _constant("2026-06-01", 5, "1h", -190.0)}
    result = tension_objective(modeled, measured)
    # Must not raise, and must round-trip numpy scalars as plain float/int.
    dumped = json.dumps(result)
    reloaded = json.loads(dumped)
    assert reloaded["aggregate"]["n"] == 5


def test_empty_inputs_give_none_aggregate_no_exception():
    result = tension_objective({}, {})
    assert result["aggregate"] is None
    assert result["probes"] == {}
    assert result["skipped"] == []
    assert json.dumps(result)  # does not raise


def test_mixed_tz_awareness_still_aligns():
    # A tz-naive side must be treated as UTC wall clock, not silently
    # inner-joined to zero overlap (which would masquerade as
    # "insufficient_overlap").
    naive_idx = pd.date_range("2026-07-01", periods=24, freq="1h")
    modeled = {"p": pd.Series([-100.0] * 24, index=naive_idx)}
    measured = {"p": _constant("2026-07-01", 24, "1h", -110.0)}
    result = tension_objective(modeled, measured)
    assert result["skipped"] == []
    assert result["probes"]["p"]["n"] == 24
    assert result["probes"]["p"]["bias"] == pytest.approx(10.0)


def test_spike_inside_populated_bin_excluded_before_resample():
    # Six 10-min samples per hourly bin; one +500 glitch inside a bin must be
    # excluded BEFORE the bin mean is taken, so the bin mean stays the clean
    # mean instead of a spike-contaminated one.
    measured_values = [-100.0] * 12
    measured_values[3] = 500.0
    measured = {"p": _series("2026-07-01", 12, "10min", measured_values)}
    modeled = {"p": _constant("2026-07-01", 2, "1h", -100.0)}
    result = tension_objective(modeled, measured, min_samples=2)
    assert result["probes"]["p"]["rmse"] == pytest.approx(0.0)
    assert result["probes"]["p"]["bias"] == pytest.approx(0.0)
