"""Supervisor service for the soil-tuning bench: a small always-on Flask app
that starts/stops/restarts the bench process and reports its health and
resource usage. Process control goes through the ``ProcessController`` seam
so the HTTP layer is testable against a fake; ``SubprocessController`` is the
real ``psutil``-backed implementation, exercised only in integration tests.

Imports flask / psutil / stdlib only -- never soil_tuning, sparcs.*, lories.*,
dash, or fipy (see ENVIRONMENT.md).

The app is served single-threaded (Flask's default): route handlers mutate
shared state (recorded boot args, the controller) without locks and rely on
that. Do not enable threaded serving without adding locking around the
start/stop/restart read-modify-write paths."""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import flask
import psutil
from flask import Flask, jsonify, request

from soil_tuning_auth import error_response, install_json_errors, load_token, require_bearer

logger = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8052
DEFAULT_BENCH_HOST = "127.0.0.1"
DEFAULT_BENCH_PORT = 8051
DEFAULT_STARTUP_GRACE_S = 300.0
DEFAULT_STOP_TIMEOUT_S = 30.0
BENCH_LOG_FILENAME = "soil_tuning_bench.log"
HEALTH_PROBE_TIMEOUT_S = 2.0
KILL_WAIT_S = 5.0
DEFAULT_LOG_TAIL_LINES = 200
MAX_LOG_TAIL_LINES = 10000
LOG_READ_CHUNK = 8192

# /start and /restart body fields that become CLI flags on the bench
# (positional `project` is handled separately). Order matches the pinned
# bench CLI (soil_tuning.py:1285-1360).
_START_FIELDS = ("project", "start", "end", "conf_dir", "data_dir", "port", "host")


class ProcessController:
    """The process-control seam: spawn/poll/signal a child process and report
    its resource stats. Plain base class (not ABC) so a test fake can satisfy
    it by simple duck typing without importing this module's internals."""

    pid: Optional[int] = None

    def spawn(
        self,
        cmd: List[str],
        *,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        log_path: Optional[str] = None,
    ) -> int:
        """Start the process; return its pid."""
        raise NotImplementedError

    def is_running(self) -> bool:
        raise NotImplementedError

    def terminate(self) -> None:
        """Ask the process to stop gracefully (the signal it already handles)."""
        raise NotImplementedError

    def kill(self) -> None:
        """Hard-kill the process."""
        raise NotImplementedError

    def wait(self, timeout: float) -> Optional[int]:
        """Wait up to ``timeout`` seconds for exit; return the returncode, or
        ``None`` if it is still running when the wait expires."""
        raise NotImplementedError

    def stats(self) -> dict:
        """psutil-backed resource stats, shaped for ``/metrics``: ``{state,
        total: {cpu_percent, rss_bytes, n_processes}, processes: [{pid, name,
        cpu_percent, rss_bytes}]}``."""
        raise NotImplementedError


class SubprocessController(ProcessController):
    """Real ``ProcessController`` backed by ``subprocess.Popen`` + ``psutil``.
    The HTTP-layer tests exercise the fake only; this class is proven against
    a real trivial subprocess by the slow integration test in
    test_soil_supervisor_process.py, and is what ``main()`` hands to
    ``create_app``."""

    def __init__(self) -> None:
        self._popen: Optional[subprocess.Popen] = None
        self._log_file = None
        self.pid: Optional[int] = None

    def spawn(
        self,
        cmd: List[str],
        *,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        log_path: Optional[str] = None,
    ) -> int:
        if self.is_running():
            raise RuntimeError("a process is already running under this controller")
        log_file = None
        stdout_target: Any = subprocess.DEVNULL
        stderr_target: Any = subprocess.DEVNULL
        if log_path:
            log_file = open(log_path, "ab")
            stdout_target = log_file
            stderr_target = subprocess.STDOUT
        popen_env = dict(os.environ) if env is None else env
        try:
            self._popen = subprocess.Popen(
                cmd,
                cwd=cwd,
                env=popen_env,
                stdout=stdout_target,
                stderr=stderr_target,
            )
        except Exception:
            if log_file is not None:
                log_file.close()
            raise
        self._log_file = log_file
        self.pid = self._popen.pid
        return self.pid

    def is_running(self) -> bool:
        if self._popen is None:
            return False
        return self._popen.poll() is None

    def terminate(self) -> None:
        if self._popen is not None and self.is_running():
            self._popen.terminate()

    def kill(self) -> None:
        if self._popen is not None and self.is_running():
            self._popen.kill()

    def wait(self, timeout: float) -> Optional[int]:
        if self._popen is None:
            return None
        try:
            return self._popen.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return None
        finally:
            if not self.is_running() and self._log_file is not None:
                self._log_file.close()
                self._log_file = None

    def stats(self) -> dict:
        """psutil-backed resource stats for the bench process and its worker
        children, shaped for ``/metrics``: ``{state, total: {cpu_percent,
        rss_bytes, n_processes}, processes: [{pid, name, cpu_percent,
        rss_bytes}]}``. A child that exits mid-iteration (``NoSuchProcess``)
        is skipped rather than raised; the whole call never raises.
        """
        empty = {
            "state": "running" if self.is_running() else "stopped",
            "total": {"cpu_percent": 0.0, "rss_bytes": 0, "n_processes": 0},
            "processes": [],
        }
        if self.pid is None or not self.is_running():
            return empty
        try:
            proc = psutil.Process(self.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            return empty

        try:
            targets = [proc] + proc.children(recursive=True)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            targets = [proc]

        processes: List[dict] = []
        for p in targets:
            try:
                cpu_percent = p.cpu_percent(interval=0.05)
                rss_bytes = p.memory_info().rss
                name = p.name()
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
            processes.append({"pid": p.pid, "name": name, "cpu_percent": cpu_percent, "rss_bytes": rss_bytes})

        total_cpu = sum(p["cpu_percent"] for p in processes)
        total_rss = sum(p["rss_bytes"] for p in processes)
        return {
            "state": "running",
            "total": {"cpu_percent": total_cpu, "rss_bytes": total_rss, "n_processes": len(processes)},
            "processes": processes,
        }


def _build_boot_args(body: dict, bench: dict, prior: Optional[dict]) -> dict:
    """Merge a request body over ``prior`` boot args (for /restart) or bare
    bench defaults (for /start), and validate ``project`` is present.

    Only ``conf_dir``/``data_dir``/``port``/``host`` fall back to the bench
    launch config; ``start``/``end`` have no bench-level default and are
    simply omitted when absent. Raises ``ValueError`` if no ``project`` is
    resolvable.
    """
    base: Dict[str, Any] = dict(prior) if prior else {}
    for key in _START_FIELDS:
        if key in body and body[key] is not None:
            base[key] = body[key]
    if not base.get("project"):
        raise ValueError("project is required")
    base.setdefault("conf_dir", bench.get("conf_dir"))
    base.setdefault("data_dir", bench.get("data_dir"))
    base.setdefault("port", bench.get("port"))
    base.setdefault("host", bench.get("host"))
    base.setdefault("start", None)
    base.setdefault("end", None)
    port = base.get("port")
    if port is not None and (isinstance(port, bool) or not isinstance(port, int)):
        raise ValueError("port must be an integer")
    for key in ("project", "start", "end", "conf_dir", "data_dir", "host"):
        value = base.get(key)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"{key} must be a string")
    return {key: base.get(key) for key in _START_FIELDS}


def _build_bench_cmd(boot_args: dict, bench: dict) -> List[str]:
    """Assemble ``[bench.python, bench.script, project, ...flags]`` per the
    pinned bench CLI (soil_tuning.py:1285-1360): only flags that are set are
    included."""
    cmd = [bench["python"], bench["script"], boot_args["project"]]
    if boot_args.get("conf_dir"):
        cmd += ["-c", str(boot_args["conf_dir"])]
    if boot_args.get("data_dir"):
        cmd += ["--data-dir", str(boot_args["data_dir"])]
    if boot_args.get("start"):
        cmd += ["--start", str(boot_args["start"])]
    if boot_args.get("end"):
        cmd += ["--end", str(boot_args["end"])]
    if boot_args.get("port"):
        cmd += ["--port", str(boot_args["port"])]
    if boot_args.get("host"):
        cmd += ["--host", str(boot_args["host"])]
    return cmd


def _probe_bench_health(bench: dict, token: str) -> Dict[str, Any]:
    """Probe the bench's ``/api/v1/health`` endpoint. Never raises: any
    connection error, HTTP error, timeout, or unparsable body is reported as
    ``reachable: false`` with a short ``detail`` string."""
    host = bench.get("host")
    if host is None:
        host = DEFAULT_BENCH_HOST
    port = bench.get("port")
    if port is None:
        port = DEFAULT_BENCH_PORT
    url = f"http://{host}:{port}/api/v1/health"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, timeout=HEALTH_PROBE_TIMEOUT_S) as resp:
            raw = resp.read()
            status_code = resp.getcode()
    except urllib.error.HTTPError as e:
        return {"reachable": False, "detail": f"HTTP {e.code}"}
    except (urllib.error.URLError, OSError, ValueError) as e:
        return {"reachable": False, "detail": str(e)}

    if status_code != 200:
        return {"reachable": False, "detail": f"HTTP {status_code}"}
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        return {"reachable": False, "detail": f"unparsable health body: {e}"}
    return {"reachable": True, "detail": payload}


def _tail_lines(path: str, n: int) -> List[str]:
    """Return the last ``n`` lines of the file at ``path``, reading only the
    tail (seek from end in chunks) so large log files stay cheap. Tolerates
    the file growing concurrently (reads a snapshot as of the seek)."""
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        block = b""
        newline_count = 0
        pos = file_size
        while pos > 0 and newline_count <= n:
            read_size = min(LOG_READ_CHUNK, pos)
            pos -= read_size
            f.seek(pos)
            chunk = f.read(read_size)
            block = chunk + block
            newline_count = block.count(b"\n")
    text = block.decode("utf-8", errors="replace")
    lines = text.splitlines()
    return lines[-n:] if n > 0 else []


def create_app(controller: ProcessController, *, token: str, bench: dict) -> Flask:
    """App factory: builds the supervisor Flask app bound to ``controller``.

    ``bench`` is the launch config (keys: python, script, conf_dir, data_dir,
    host, port, log_path, startup_grace_s, stop_timeout_s). Routes: /start,
    /stop, /restart, /status, /metrics, /logs. Auth is enforced on every
    route via before_request; JSON error handlers are installed so no HTML
    error body can escape.
    """
    app = Flask(__name__)
    app.config["SOIL_TUNING_CONTROLLER"] = controller
    app.config["SOIL_TUNING_BENCH"] = dict(bench)
    app.config["SOIL_TUNING_STARTED_AT"] = None
    app.config["SOIL_TUNING_BOOT_ARGS"] = None
    app.config["SOIL_TUNING_TOKEN"] = token

    app.before_request(require_bearer(token))
    install_json_errors(app)

    def _bench() -> dict:
        return app.config["SOIL_TUNING_BENCH"]

    def _startup_grace_s() -> float:
        grace = _bench().get("startup_grace_s")
        return float(DEFAULT_STARTUP_GRACE_S if grace is None else grace)

    def _stop_timeout_default() -> float:
        timeout = _bench().get("stop_timeout_s")
        return float(DEFAULT_STOP_TIMEOUT_S if timeout is None else timeout)

    @app.route("/start", methods=["POST"])
    def start() -> flask.Response:
        if controller.is_running():
            return error_response("already_running", "the bench is already running", 409)
        body = request.get_json(silent=True) or {}
        if not isinstance(body, dict) or not body.get("project"):
            return error_response("invalid_request", "project is required", 400)
        try:
            boot_args = _build_boot_args(body, _bench(), prior=None)
        except ValueError as e:
            return error_response("invalid_request", str(e), 400)

        cmd = _build_bench_cmd(boot_args, _bench())
        pid = controller.spawn(cmd, log_path=_bench().get("log_path"))
        app.config["SOIL_TUNING_STARTED_AT"] = time.time()
        app.config["SOIL_TUNING_BOOT_ARGS"] = boot_args
        response = jsonify({"state": "starting", "pid": pid, "boot_args": boot_args})
        response.status_code = 202
        return response

    @app.route("/stop", methods=["POST"])
    def stop() -> flask.Response:
        if not controller.is_running():
            return error_response("not_running", "the bench is not running", 409)
        body = request.get_json(silent=True) or {}
        force = bool(body.get("force", False))
        try:
            timeout = float(body.get("timeout", _stop_timeout_default()))
        except (TypeError, ValueError):
            return error_response("invalid_request", "timeout must be a number", 400)

        controller.terminate()
        returncode = controller.wait(timeout)
        if not controller.is_running():
            return jsonify({"stopped": True, "forced": False, "returncode": returncode})

        if not force:
            return error_response(
                "stop_timeout",
                f"bench did not exit within {timeout}s of terminate()",
                409,
            )

        controller.kill()
        returncode = controller.wait(KILL_WAIT_S)
        return jsonify({"stopped": True, "forced": True, "returncode": returncode})

    @app.route("/restart", methods=["POST"])
    def restart() -> flask.Response:
        body = request.get_json(silent=True) or {}
        if not isinstance(body, dict):
            body = {}
        prior = app.config.get("SOIL_TUNING_BOOT_ARGS")
        try:
            boot_args = _build_boot_args(body, _bench(), prior=prior)
        except ValueError as e:
            return error_response("invalid_request", str(e), 400)

        force = bool(body.get("force", True))
        try:
            timeout = float(body.get("timeout", _stop_timeout_default()))
        except (TypeError, ValueError):
            return error_response("invalid_request", "timeout must be a number", 400)
        if controller.is_running():
            controller.terminate()
            controller.wait(timeout)
            if controller.is_running():
                if not force:
                    return error_response(
                        "stop_timeout",
                        f"bench did not exit within {timeout}s of terminate()",
                        409,
                    )
                controller.kill()
                controller.wait(KILL_WAIT_S)

        cmd = _build_bench_cmd(boot_args, _bench())
        pid = controller.spawn(cmd, log_path=_bench().get("log_path"))
        app.config["SOIL_TUNING_STARTED_AT"] = time.time()
        app.config["SOIL_TUNING_BOOT_ARGS"] = boot_args
        response = jsonify({"state": "starting", "pid": pid, "boot_args": boot_args})
        response.status_code = 202
        return response

    @app.route("/status", methods=["GET"])
    def status() -> flask.Response:
        boot_args = app.config.get("SOIL_TUNING_BOOT_ARGS")
        if not controller.is_running():
            return jsonify(
                {
                    "state": "stopped",
                    "pid": None,
                    "uptime_s": None,
                    "started_at": None,
                    "boot_args": boot_args,
                    "health": {"reachable": False, "detail": "not running"},
                }
            )

        pid = controller.pid
        started_at_epoch = app.config.get("SOIL_TUNING_STARTED_AT")
        uptime_s = (time.time() - started_at_epoch) if started_at_epoch is not None else None
        started_at_iso = (
            datetime.fromtimestamp(started_at_epoch, tz=timezone.utc).isoformat()
            if started_at_epoch is not None
            else None
        )

        health = _probe_bench_health(_bench(), app.config["SOIL_TUNING_TOKEN"])
        if health["reachable"]:
            state = "running"
        elif uptime_s is not None and uptime_s < _startup_grace_s():
            state = "starting"
        else:
            state = "wedged"

        return jsonify(
            {
                "state": state,
                "pid": pid,
                "uptime_s": uptime_s,
                "started_at": started_at_iso,
                "boot_args": boot_args,
                "health": health,
            }
        )

    @app.route("/metrics", methods=["GET"])
    def metrics() -> flask.Response:
        return jsonify(controller.stats())

    @app.route("/logs", methods=["GET"])
    def logs() -> flask.Response:
        try:
            tail = int(request.args.get("tail", DEFAULT_LOG_TAIL_LINES))
        except (TypeError, ValueError):
            return error_response("invalid_request", "tail must be an integer", 400)
        tail = max(0, min(tail, MAX_LOG_TAIL_LINES))

        log_path = _bench().get("log_path")
        if not log_path or not os.path.isfile(log_path):
            return error_response("log_not_found", "bench log file not found", 404, path=log_path)

        lines = _tail_lines(log_path, tail)
        return jsonify({"path": log_path, "lines": lines})

    return app


def _build_arg_parser() -> argparse.ArgumentParser:
    default_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "soil_tuning.py")
    parser = argparse.ArgumentParser(description="Soil-tuning bench supervisor service")
    parser.add_argument("--host", default=DEFAULT_HOST, help="bind address for the supervisor (default: loopback)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--bench-python", default=sys.executable, help="interpreter used to launch the bench")
    parser.add_argument("--bench-script", default=default_script, help="path to soil_tuning.py")
    parser.add_argument("--conf-dir", default=None, help="app config dir passed to the bench")
    parser.add_argument("--data-dir", default=None, help="project data dir passed to the bench")
    parser.add_argument("--bench-host", default=DEFAULT_BENCH_HOST)
    parser.add_argument("--bench-port", type=int, default=DEFAULT_BENCH_PORT)
    parser.add_argument("--log-dir", default=".", help="directory the bench log is written under")
    parser.add_argument("--startup-grace", type=float, default=DEFAULT_STARTUP_GRACE_S, dest="startup_grace_s")
    parser.add_argument("--stop-timeout", type=float, default=DEFAULT_STOP_TIMEOUT_S, dest="stop_timeout_s")
    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    try:
        token = load_token()
    except RuntimeError as e:
        logger.error("%s", e)
        return 2

    bench = {
        "python": args.bench_python,
        "script": args.bench_script,
        "conf_dir": args.conf_dir,
        "data_dir": args.data_dir,
        "host": args.bench_host,
        "port": args.bench_port,
        "log_path": os.path.join(args.log_dir, BENCH_LOG_FILENAME),
        "startup_grace_s": args.startup_grace_s,
        "stop_timeout_s": args.stop_timeout_s,
    }

    controller = SubprocessController()
    app = create_app(controller, token=token, bench=bench)
    logger.info("starting soil-tuning supervisor on %s:%s", args.host, args.port)
    app.run(host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
