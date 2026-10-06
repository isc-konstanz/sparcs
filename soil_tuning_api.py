"""Token-guarded ``/api/v1`` job API attached to the soil-tuning bench's
existing Flask server (the one Dash already runs). Wraps ``TuningRunner``
unchanged -- the runner and its jobs are duck-typed here, never imported.

Imports flask / soil_tuning_auth / stdlib only -- never soil_tuning, dash,
sparcs.*, lories.*, or fipy (see ENVIRONMENT.md). The Dash UI and the
existing ``/job-png`` route stay reachable without a token: auth is scoped
to the ``api_v1`` blueprint's ``before_request``, not the whole server."""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Callable, Optional

import flask
from flask import Blueprint, jsonify, request

from soil_tuning_auth import error_response, require_bearer

logger = logging.getLogger(__name__)

_BLUEPRINT_NAME = "api_v1"
_DT_KEYS = ("dt", "dt_min")


def _serialize_job(job: Any) -> dict:
    """Job envelope shared by every route that returns a job: {id, label,
    status, progress, params, submitted_at (ISO UTC), error, objective,
    has_plot}. ``has_plot`` is a static field name; a real bool is filled in
    by the caller where a png_lookup is available."""
    submitted_at = getattr(job, "submitted_at", None)
    if submitted_at is not None and hasattr(submitted_at, "isoformat"):
        submitted_at = submitted_at.isoformat()
    return {
        "id": job.job_id,
        "label": job.label,
        "status": job.status,
        "progress": job.progress,
        "params": dict(job.params),
        "submitted_at": submitted_at,
        "error": job.error,
        "objective": job.objective,
        "has_plot": False,
    }


def _validate_params(
    params: Any,
    *,
    param_exists: Callable[[str], bool],
    dt_ceiling_s: float,
) -> Optional[dict]:
    """Validate one item's ``params`` dict. Returns None if valid, else an
    error dict {code, message, detail} for the first failing key (fail-fast,
    validation order per the pinned contract: invalid_request ->
    invalid_param_value -> unknown_param -> dt_ceiling_exceeded)."""
    if not isinstance(params, dict):
        return {
            "code": "invalid_request",
            "message": "params must be a JSON object",
            "detail": {},
        }
    for key, value in params.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return {
                "code": "invalid_param_value",
                "message": f"param {key!r} must be a finite number",
                "detail": {"key": key},
            }
        if not math.isfinite(value):
            return {
                "code": "invalid_param_value",
                "message": f"param {key!r} must be a finite number",
                "detail": {"key": key},
            }
        if not param_exists(key):
            return {
                "code": "unknown_param",
                "message": f"unknown parameter {key!r}",
                "detail": {"key": key},
            }
        if key in _DT_KEYS and value > dt_ceiling_s:
            return {
                "code": "dt_ceiling_exceeded",
                "message": f"{key} {value} exceeds ceiling {dt_ceiling_s}",
                "detail": {"key": key, "value": value, "ceiling": dt_ceiling_s},
            }
    return None


def _validate_item(item: Any, *, param_exists: Callable[[str], bool], dt_ceiling_s: float) -> Optional[dict]:
    """Validate one submit item's whole body: {params, label?}."""
    if not isinstance(item, dict):
        return {
            "code": "invalid_request",
            "message": "each item must be a JSON object",
            "detail": {},
        }
    label = item.get("label")
    if label is not None and not isinstance(label, str):
        return {
            "code": "invalid_request",
            "message": "label must be a string",
            "detail": {},
        }
    return _validate_params(item.get("params"), param_exists=param_exists, dt_ceiling_s=dt_ceiling_s)


def register_api(
    server: flask.Flask,
    runner: Any,
    *,
    token: str,
    boot_info: dict,
    png_lookup: Callable[[str], Optional[bytes]],
    param_exists: Callable[[str], bool],
    objective_fn: Optional[Callable[[Any], Optional[dict]]] = None,
    dt_ceiling_s: float = 10.0,
) -> None:
    """Register the ``api_v1`` blueprint at ``/api/v1`` on ``server``, wrapping
    ``runner`` (duck-typed: submit/cancel/cancel_all/jobs). Blueprint-scoped
    bearer auth via ``before_request`` -- routes outside the blueprint (the
    Dash UI, ``/job-png``) stay reachable without a token."""
    bp = Blueprint(_BLUEPRINT_NAME, __name__)
    bp.before_request(require_bearer(token))

    started_at = time.time()

    def _serialize_with_plot(job: Any) -> dict:
        payload = _serialize_job(job)
        payload["has_plot"] = png_lookup(job.job_id) is not None
        return payload

    def _maybe_compute_objective(job: Any) -> None:
        if job.status != "done" or job.objective is not None or objective_fn is None:
            return
        try:
            job.objective = objective_fn(job)
        except Exception:
            logger.warning("objective_fn raised for job %s", getattr(job, "job_id", "?"), exc_info=True)

    def _find_job(job_id: str) -> Optional[Any]:
        for job in runner.jobs():
            if job.job_id == job_id:
                return job
        return None

    @bp.route("/jobs", methods=["POST"])
    def _submit_job():
        body = request.get_json(silent=True)
        err = _validate_item(body, param_exists=param_exists, dt_ceiling_s=dt_ceiling_s)
        if err is not None:
            return error_response(err["code"], err["message"], 400, **err["detail"])
        label = body.get("label") or ""
        job = runner.submit(body["params"], label)
        return jsonify(_serialize_with_plot(job)), 201

    @bp.route("/jobs/batch", methods=["POST"])
    def _submit_batch():
        body = request.get_json(silent=True)
        if not isinstance(body, list):
            return error_response("invalid_request", "batch body must be a JSON array", 400)
        for index, item in enumerate(body):
            err = _validate_item(item, param_exists=param_exists, dt_ceiling_s=dt_ceiling_s)
            if err is not None:
                detail = dict(err["detail"])
                detail["index"] = index
                return error_response(err["code"], err["message"], 400, **detail)
        jobs = [runner.submit(item["params"], item.get("label") or "") for item in body]
        return jsonify([_serialize_with_plot(job) for job in jobs]), 201

    @bp.route("/jobs", methods=["GET"])
    def _list_jobs():
        return jsonify({"jobs": [_serialize_with_plot(job) for job in runner.jobs()]}), 200

    @bp.route("/jobs/<job_id>", methods=["GET"])
    def _poll_job(job_id: str):
        job = _find_job(job_id)
        if job is None:
            return error_response("job_not_found", f"no job {job_id!r}", 404)
        _maybe_compute_objective(job)
        return jsonify(_serialize_with_plot(job)), 200

    @bp.route("/jobs/<job_id>", methods=["DELETE"])
    def _cancel_job(job_id: str):
        job = _find_job(job_id)
        if job is None:
            return error_response("job_not_found", f"no job {job_id!r}", 404)
        runner.cancel(job_id)
        job = _find_job(job_id) or job
        return jsonify(_serialize_with_plot(job)), 200

    @bp.route("/jobs/cancel_all", methods=["POST"])
    def _cancel_all():
        cancelled = sum(1 for job in runner.jobs() if job.status in ("pending", "running"))
        runner.cancel_all()
        return jsonify({"cancelled": cancelled}), 200

    @bp.route("/jobs/<job_id>/plot.png", methods=["GET"])
    def _job_plot(job_id: str):
        job = _find_job(job_id)
        if job is None:
            return error_response("job_not_found", f"no job {job_id!r}", 404)
        png_bytes = png_lookup(job_id)
        if png_bytes is None:
            return error_response("plot_not_available", f"no plot for job {job_id!r}", 404)
        return flask.Response(png_bytes, mimetype="image/png")

    @bp.route("/health", methods=["GET"])
    def _health():
        counts = {"pending": 0, "running": 0, "done": 0, "failed": 0, "cancelled": 0}
        for job in runner.jobs():
            if job.status in counts:
                counts[job.status] += 1
        return (
            jsonify(
                {
                    "status": "ok",
                    "project": boot_info.get("project"),
                    "replay_window": boot_info.get("replay_window"),
                    "max_workers": boot_info.get("max_workers"),
                    "started_at": boot_info.get("started_at"),
                    "uptime_s": time.time() - started_at,
                    "jobs": counts,
                }
            ),
            200,
        )

    server.register_blueprint(bp, url_prefix="/api/v1")
