"""Supervisor service for the soil-tuning bench: a small always-on Flask app
that starts/stops/restarts the bench process and reports its health and
resource usage. Process control goes through the ``ProcessController`` seam
so the HTTP layer is testable against a fake; ``SubprocessController`` is the
real ``psutil``-backed implementation, exercised only in integration tests.

Imports flask / psutil / stdlib only -- never soil_tuning, sparcs.*, lories.*,
dash, or fipy (see ENVIRONMENT.md)."""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

import flask
import psutil
from flask import Flask, jsonify

from soil_tuning_auth import install_json_errors, load_token, require_bearer

logger = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8052
DEFAULT_BENCH_HOST = "127.0.0.1"
DEFAULT_BENCH_PORT = 8051
DEFAULT_STARTUP_GRACE_S = 300.0
DEFAULT_STOP_TIMEOUT_S = 30.0
BENCH_LOG_FILENAME = "soil_tuning_bench.log"


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
        """psutil-backed resource stats; shape pinned in unit 02."""
        raise NotImplementedError


class SubprocessController(ProcessController):
    """Real ``ProcessController`` backed by ``subprocess.Popen`` + ``psutil``.
    Not exercised by this unit's tests (no process is spawned here); it exists
    so ``main()`` has a real implementation to hand to ``create_app``."""

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
        if self.pid is None or not self.is_running():
            return {"pid": None, "cpu_percent": None, "memory_rss": None, "children": []}
        try:
            proc = psutil.Process(self.pid)
            return {
                "pid": self.pid,
                "cpu_percent": proc.cpu_percent(interval=None),
                "memory_rss": proc.memory_info().rss,
                "children": [c.pid for c in proc.children(recursive=True)],
            }
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            return {"pid": self.pid, "cpu_percent": None, "memory_rss": None, "children": []}


def create_app(controller: ProcessController, *, token: str, bench: dict) -> Flask:
    """App factory: builds the supervisor Flask app bound to ``controller``.

    ``bench`` is the launch config (keys: python, script, conf_dir, data_dir,
    host, port, log_path, startup_grace_s, stop_timeout_s); this unit only
    stores it (start/stop/restart land in unit 02). Auth is enforced on every
    route via before_request; JSON error handlers are installed so no HTML
    error body can escape.
    """
    app = Flask(__name__)
    app.config["SOIL_TUNING_CONTROLLER"] = controller
    app.config["SOIL_TUNING_BENCH"] = dict(bench)
    app.config["SOIL_TUNING_STARTED_AT"] = None

    app.before_request(require_bearer(token))
    install_json_errors(app)

    @app.route("/status", methods=["GET"])
    def status() -> flask.Response:
        if not controller.is_running():
            return jsonify({"state": "stopped", "pid": None, "uptime_s": None, "boot_args": None})
        pid = controller.pid
        started_at = app.config.get("SOIL_TUNING_STARTED_AT")
        uptime_s = (time.time() - started_at) if started_at is not None else None
        return jsonify(
            {
                "state": "running",
                "pid": pid,
                "uptime_s": uptime_s,
                "boot_args": app.config.get("SOIL_TUNING_BOOT_ARGS"),
            }
        )

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
