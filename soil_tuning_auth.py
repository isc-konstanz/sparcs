"""Shared bearer-token auth and JSON error envelope for the soil-tuning HTTP
surfaces (supervisor service and bench job API). Imports flask + stdlib only."""

from __future__ import annotations

import hmac
import logging
import os
from typing import Callable, Optional

import flask
from flask import jsonify
from werkzeug.exceptions import HTTPException

logger = logging.getLogger(__name__)

TOKEN_ENV_VAR = "SOIL_TUNING_API_TOKEN"
TOKEN_FILE_ENV_VAR = "SOIL_TUNING_API_TOKEN_FILE"


def load_token() -> str:
    """Resolve the bearer token from env, else from the file it points at.

    Precedence: ``SOIL_TUNING_API_TOKEN`` wins if set (non-empty); otherwise the
    stripped content of the file named by ``SOIL_TUNING_API_TOKEN_FILE``. Raises
    ``RuntimeError`` if neither is set or the file cannot be read.
    """
    env_token = os.environ.get(TOKEN_ENV_VAR)
    if env_token:
        return env_token

    token_file = os.environ.get(TOKEN_FILE_ENV_VAR)
    if not token_file:
        raise RuntimeError(
            f"no API token configured: set {TOKEN_ENV_VAR} or {TOKEN_FILE_ENV_VAR} (path to a token file)"
        )
    try:
        with open(token_file, "r", encoding="utf-8") as f:
            content = f.read().strip()
    except OSError as e:
        raise RuntimeError(f"could not read token file {token_file!r}: {e}") from e
    if not content:
        raise RuntimeError(f"token file {token_file!r} is empty")
    return content


def error_response(code: str, message: str, status: int, **detail) -> flask.Response:
    """Build the pinned JSON error envelope: {"error": {"code", "message", "detail"}}."""
    payload = {"error": {"code": code, "message": message, "detail": detail}}
    response = jsonify(payload)
    response.status_code = status
    return response


def require_bearer(token: str) -> Callable[[], Optional[flask.Response]]:
    """Return a Flask before-request hook enforcing ``Authorization: Bearer <token>``.

    The returned callable takes no arguments (bind ``token`` via closure) so it
    can be registered directly with ``app.before_request``. Returns ``None`` to
    let the request proceed, or the 401 ``unauthorized`` envelope to short-circuit it.
    """
    if not token:
        raise ValueError("token must be non-empty")

    def _check() -> Optional[flask.Response]:
        header = flask.request.headers.get("Authorization", "")
        prefix = "Bearer "
        if not header.startswith(prefix):
            return error_response("unauthorized", "missing or malformed Authorization header", 401)
        candidate = header[len(prefix) :]
        if not hmac.compare_digest(candidate, token):
            return error_response("unauthorized", "invalid token", 401)
        return None

    return _check


def install_json_errors(app: flask.Flask) -> None:
    """Register error handlers so 404/405/500 and any exception return the
    JSON envelope, never an HTML body."""

    @app.errorhandler(404)
    def _not_found(_e):
        return error_response("not_found", "resource not found", 404)

    @app.errorhandler(405)
    def _method_not_allowed(_e):
        return error_response("method_not_allowed", "method not allowed", 405)

    @app.errorhandler(HTTPException)
    def _http_exception(e: HTTPException):
        code = e.code or 500
        return error_response(_code_for_status(code), e.description or "request error", code)

    @app.errorhandler(Exception)
    def _internal_error(e: Exception):
        logger.exception("unhandled exception in request")
        return error_response("internal_error", "internal server error", 500)


def _code_for_status(status: int) -> str:
    """Map a generic HTTP status to a stable snake_case error code."""
    if status == 401:
        return "unauthorized"
    if status == 404:
        return "not_found"
    if status == 405:
        return "method_not_allowed"
    return "internal_error"
