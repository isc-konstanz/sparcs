# -*- coding: utf-8 -*-
"""Bench job API HTTP boundary: blueprint-scoped bearer auth, the job
serialization envelope, validation order, batch atomicity, lazy objective
computation, and the pinned routes from issue 03.

Exercises the real route code in ``soil_tuning_api.register_api`` against a
fake runner (FakeRunner/FakeJob defined below) with the Flask test client --
no Dash, no FiPy, no real worker pool. ``soil_tuning_api`` imports flask +
soil_tuning_auth + stdlib only, so ``importorskip`` is a formality here (kept
for consistency with the project's optional-dep test convention).
"""

from __future__ import annotations

import os
import subprocess
import sys
import uuid
from typing import Any, Callable, Dict, List, Optional

import pytest

flask = pytest.importorskip("flask")
soil_tuning_api = pytest.importorskip("soil_tuning_api")

from flask import Flask  # noqa: E402

from soil_tuning_api import register_api  # noqa: E402

TOKEN = "s3cr3t-token"
KNOWN_PARAMS = {"theta_r", "theta_s", "alpha", "n", "k_s", "dt", "dt_min", "ic_water_table_depth"}


class FakeJob:
    def __init__(self, params: Dict[str, float], label: str = ""):
        self.job_id = uuid.uuid4().hex[:8]
        self.params = dict(params)
        self.label = label or "auto"
        self.status = "pending"  # pending | running | done | failed | cancelled
        self.progress = 0.0
        self.error: Optional[str] = None
        self.objective: Optional[dict] = None
        self.submitted_at = _FakeTimestamp("2026-07-02T00:00:00+00:00")


class _FakeTimestamp:
    """Minimal stand-in for pd.Timestamp exposing only .isoformat()."""

    def __init__(self, iso: str):
        self._iso = iso

    def isoformat(self) -> str:
        return self._iso


class FakeRunner:
    """Duck-typed stand-in for TuningRunner: submit/cancel/cancel_all/jobs."""

    def __init__(self):
        self.submitted: List[Dict[str, Any]] = []
        self._jobs: "Dict[str, FakeJob]" = {}
        self.cancel_calls: List[str] = []
        self.cancel_all_calls = 0

    def submit(self, params: Dict[str, float], label: str = "") -> FakeJob:
        self.submitted.append({"params": dict(params), "label": label})
        job = FakeJob(params, label)
        self._jobs[job.job_id] = job
        return job

    def cancel(self, job_id: str) -> None:
        self.cancel_calls.append(job_id)
        job = self._jobs.get(job_id)
        if job is not None and job.status in ("pending", "running"):
            job.status = "cancelled"

    def cancel_all(self) -> None:
        self.cancel_all_calls += 1
        for job in self._jobs.values():
            if job.status in ("pending", "running"):
                job.status = "cancelled"

    def jobs(self) -> List[FakeJob]:
        return list(self._jobs.values())

    def add_job(self, job: FakeJob) -> FakeJob:
        self._jobs[job.job_id] = job
        return job


def _param_exists(key: str) -> bool:
    return key in KNOWN_PARAMS


def _boot_info() -> dict:
    return {
        "project": "demo",
        "replay_window": {"start": "2026-06-25T00:00:00+00:00", "end": "2026-07-02T00:00:00+00:00"},
        "max_workers": 4,
        "started_at": "2026-07-02T00:00:00+00:00",
    }


def _make_app(
    runner: FakeRunner,
    *,
    png_lookup: Optional[Callable[[str], Optional[bytes]]] = None,
    objective_fn: Optional[Callable[[Any], Optional[dict]]] = None,
    dt_ceiling_s: float = 10.0,
    token: str = TOKEN,
):
    server = Flask(__name__)
    server.config["TESTING"] = True

    if png_lookup is None:

        def png_lookup(_job_id: str) -> Optional[bytes]:
            return None

    register_api(
        server,
        runner,
        token=token,
        boot_info=_boot_info(),
        png_lookup=png_lookup,
        param_exists=_param_exists,
        objective_fn=objective_fn,
        dt_ceiling_s=dt_ceiling_s,
    )
    return server


@pytest.fixture
def runner():
    return FakeRunner()


@pytest.fixture
def client(runner):
    server = _make_app(runner)
    with server.test_client() as c:
        yield c


def _auth(token: str = TOKEN) -> dict:
    return {"Authorization": f"Bearer {token}"}


# --- no-heavy-import guard -------------------------------------------------


def test_importing_soil_tuning_api_does_not_pull_in_dash():
    # In-process sys.modules is pre-polluted by dash's pytest entry-point
    # plugin, so the no-heavy-import guarantee must be proven in a fresh
    # interpreter without pytest.
    script = (
        "import sys; import soil_tuning_api; "
        "banned = [m for m in ('dash', 'fipy', 'soil_tuning') if m in sys.modules]; "
        "assert not banned, banned"
    )
    sparcs_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    subprocess.run([sys.executable, "-c", script], check=True, cwd=sparcs_root)


def test_non_api_routes_stay_open_without_token(runner):
    server = _make_app(runner)

    @server.route("/ui")
    def ui():
        return "ui ok"

    with server.test_client() as c:
        assert c.get("/ui").status_code == 200
        assert c.get("/api/v1/jobs").status_code == 401


# --- auth --------------------------------------------------------------


def test_untokened_request_is_401(client):
    resp = client.get("/api/v1/jobs")
    assert resp.status_code == 401
    assert resp.get_json()["error"]["code"] == "unauthorized"


def test_wrong_token_is_401(client):
    resp = client.get("/api/v1/jobs", headers=_auth("wrong"))
    assert resp.status_code == 401


# --- submit single -------------------------------------------------------


def test_submit_happy_path_201_and_shape(client, runner):
    resp = client.post(
        "/api/v1/jobs",
        json={"params": {"alpha": 0.02, "n": 1.14}, "label": "trial-1"},
        headers=_auth(),
    )
    assert resp.status_code == 201
    body = resp.get_json()
    assert set(body.keys()) == {
        "id",
        "label",
        "status",
        "progress",
        "params",
        "submitted_at",
        "error",
        "objective",
        "has_plot",
    }
    assert body["label"] == "trial-1"
    assert body["params"] == {"alpha": 0.02, "n": 1.14}
    assert body["status"] == "pending"
    assert body["objective"] is None
    assert body["has_plot"] is False
    assert len(runner.submitted) == 1
    assert runner.submitted[0]["params"] == {"alpha": 0.02, "n": 1.14}
    assert runner.submitted[0]["label"] == "trial-1"


def test_submit_unknown_param_400(client, runner):
    resp = client.post("/api/v1/jobs", json={"params": {"bogus_key": 1.0}}, headers=_auth())
    assert resp.status_code == 400
    body = resp.get_json()
    assert body["error"]["code"] == "unknown_param"
    assert body["error"]["detail"]["key"] == "bogus_key"
    assert runner.submitted == []


def test_submit_nan_value_400(client, runner):
    resp = client.post(
        "/api/v1/jobs",
        json={"params": {"alpha": float("nan")}},
        headers=_auth(),
    )
    assert resp.status_code == 400
    assert resp.get_json()["error"]["code"] == "invalid_param_value"
    assert runner.submitted == []


def test_submit_non_object_params_400(client, runner):
    resp = client.post("/api/v1/jobs", json={"params": [1, 2, 3]}, headers=_auth())
    assert resp.status_code == 400
    assert resp.get_json()["error"]["code"] == "invalid_request"
    assert runner.submitted == []


def test_submit_dt_at_ceiling_passes(client, runner):
    resp = client.post("/api/v1/jobs", json={"params": {"dt": 10.0}}, headers=_auth())
    assert resp.status_code == 201
    assert len(runner.submitted) == 1


def test_submit_dt_above_ceiling_400(client, runner):
    resp = client.post("/api/v1/jobs", json={"params": {"dt": 10.1}}, headers=_auth())
    assert resp.status_code == 400
    body = resp.get_json()
    assert body["error"]["code"] == "dt_ceiling_exceeded"
    assert body["error"]["detail"] == {"key": "dt", "value": 10.1, "ceiling": 10.0}
    assert runner.submitted == []


def test_submit_dt_min_above_ceiling_400(client, runner):
    resp = client.post("/api/v1/jobs", json={"params": {"dt_min": 10.1}}, headers=_auth())
    assert resp.status_code == 400
    assert resp.get_json()["error"]["code"] == "dt_ceiling_exceeded"
    assert resp.get_json()["error"]["detail"]["key"] == "dt_min"
    assert runner.submitted == []


# --- submit batch --------------------------------------------------------


def test_batch_submit_happy_path(client, runner):
    resp = client.post(
        "/api/v1/jobs/batch",
        json=[{"params": {"alpha": 0.01}}, {"params": {"alpha": 0.02}, "label": "b"}],
        headers=_auth(),
    )
    assert resp.status_code == 201
    body = resp.get_json()
    assert len(body) == 2
    assert len(runner.submitted) == 2


def test_batch_atomicity_one_bad_item_rejects_whole_batch(client, runner):
    resp = client.post(
        "/api/v1/jobs/batch",
        json=[{"params": {"alpha": 0.01}}, {"params": {"bogus": 1.0}}],
        headers=_auth(),
    )
    assert resp.status_code == 400
    body = resp.get_json()
    assert body["error"]["code"] == "unknown_param"
    assert body["error"]["detail"]["index"] == 1
    assert runner.submitted == []


def test_batch_non_array_body_400(client, runner):
    resp = client.post("/api/v1/jobs/batch", json={"params": {"alpha": 0.01}}, headers=_auth())
    assert resp.status_code == 400
    assert resp.get_json()["error"]["code"] == "invalid_request"
    assert runner.submitted == []


# --- list ------------------------------------------------------------


def test_list_jobs(client, runner):
    runner.add_job(FakeJob({"alpha": 0.01}, "a"))
    runner.add_job(FakeJob({"alpha": 0.02}, "b"))
    resp = client.get("/api/v1/jobs", headers=_auth())
    assert resp.status_code == 200
    body = resp.get_json()
    assert len(body["jobs"]) == 2


# --- poll ------------------------------------------------------------


def test_poll_unknown_id_404(client):
    resp = client.get("/api/v1/jobs/doesnotexist", headers=_auth())
    assert resp.status_code == 404
    assert resp.get_json()["error"]["code"] == "job_not_found"


def test_poll_known_job_200(client, runner):
    job = runner.add_job(FakeJob({"alpha": 0.01}, "a"))
    resp = client.get(f"/api/v1/jobs/{job.job_id}", headers=_auth())
    assert resp.status_code == 200
    assert resp.get_json()["id"] == job.job_id


def test_lazy_objective_computed_once_for_done_job(runner):
    job = runner.add_job(FakeJob({"alpha": 0.01}, "a"))
    job.status = "done"

    calls = []

    def objective_fn(j):
        calls.append(j.job_id)
        return {"rmse": 1.23}

    server = _make_app(runner, objective_fn=objective_fn)
    with server.test_client() as c:
        resp1 = c.get(f"/api/v1/jobs/{job.job_id}", headers=_auth())
        resp2 = c.get(f"/api/v1/jobs/{job.job_id}", headers=_auth())

    assert resp1.get_json()["objective"] == {"rmse": 1.23}
    assert resp2.get_json()["objective"] == {"rmse": 1.23}
    assert calls == [job.job_id]  # computed exactly once


def test_lazy_objective_not_computed_for_running_job(runner):
    job = runner.add_job(FakeJob({"alpha": 0.01}, "a"))
    job.status = "running"

    calls = []

    def objective_fn(j):
        calls.append(j.job_id)
        return {"rmse": 1.23}

    server = _make_app(runner, objective_fn=objective_fn)
    with server.test_client() as c:
        resp = c.get(f"/api/v1/jobs/{job.job_id}", headers=_auth())

    assert resp.get_json()["objective"] is None
    assert calls == []


def test_objective_fn_exception_yields_null_objective_not_500(runner):
    job = runner.add_job(FakeJob({"alpha": 0.01}, "a"))
    job.status = "done"

    def objective_fn(_j):
        raise ValueError("boom")

    server = _make_app(runner, objective_fn=objective_fn)
    with server.test_client() as c:
        resp = c.get(f"/api/v1/jobs/{job.job_id}", headers=_auth())

    assert resp.status_code == 200
    assert resp.get_json()["objective"] is None


# --- cancel ------------------------------------------------------------


def test_cancel_unknown_404(client):
    resp = client.delete("/api/v1/jobs/doesnotexist", headers=_auth())
    assert resp.status_code == 404
    assert resp.get_json()["error"]["code"] == "job_not_found"


def test_cancel_known_200(client, runner):
    job = runner.add_job(FakeJob({"alpha": 0.01}, "a"))
    resp = client.delete(f"/api/v1/jobs/{job.job_id}", headers=_auth())
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "cancelled"
    assert runner.cancel_calls == [job.job_id]


def test_cancel_all_returns_count_of_active_jobs(client, runner):
    j1 = runner.add_job(FakeJob({"alpha": 0.01}, "a"))
    j1.status = "running"
    j2 = runner.add_job(FakeJob({"alpha": 0.02}, "b"))
    j2.status = "pending"
    j3 = runner.add_job(FakeJob({"alpha": 0.03}, "c"))
    j3.status = "done"

    resp = client.post("/api/v1/jobs/cancel_all", headers=_auth())
    assert resp.status_code == 200
    assert resp.get_json() == {"cancelled": 2}
    assert runner.cancel_all_calls == 1


# --- plot ------------------------------------------------------------


def test_plot_bytes_and_content_type(runner):
    job = runner.add_job(FakeJob({"alpha": 0.01}, "a"))
    png_bytes = b"\x89PNG\r\n\x1a\nfakepngdata"

    def png_lookup(job_id: str) -> Optional[bytes]:
        return png_bytes if job_id == job.job_id else None

    server = _make_app(runner, png_lookup=png_lookup)
    with server.test_client() as c:
        resp = c.get(f"/api/v1/jobs/{job.job_id}/plot.png", headers=_auth())

    assert resp.status_code == 200
    assert resp.content_type == "image/png"
    assert resp.data == png_bytes


def test_plot_unknown_job_404(client):
    resp = client.get("/api/v1/jobs/doesnotexist/plot.png", headers=_auth())
    assert resp.status_code == 404
    assert resp.get_json()["error"]["code"] == "job_not_found"


def test_plot_known_job_no_plot_yet_404(client, runner):
    job = runner.add_job(FakeJob({"alpha": 0.01}, "a"))
    resp = client.get(f"/api/v1/jobs/{job.job_id}/plot.png", headers=_auth())
    assert resp.status_code == 404
    assert resp.get_json()["error"]["code"] == "plot_not_available"


# --- health ------------------------------------------------------------


def test_health_shape_and_job_counts(client, runner):
    j1 = runner.add_job(FakeJob({"alpha": 0.01}, "a"))
    j1.status = "running"
    j2 = runner.add_job(FakeJob({"alpha": 0.02}, "b"))
    j2.status = "done"
    j3 = runner.add_job(FakeJob({"alpha": 0.03}, "c"))
    j3.status = "done"

    resp = client.get("/api/v1/health", headers=_auth())
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] == "ok"
    assert body["project"] == "demo"
    assert body["replay_window"] == _boot_info()["replay_window"]
    assert body["max_workers"] == 4
    assert body["started_at"] == "2026-07-02T00:00:00+00:00"
    assert isinstance(body["uptime_s"], (int, float))
    assert body["jobs"] == {"pending": 0, "running": 1, "done": 2, "failed": 0, "cancelled": 0}
