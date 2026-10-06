# -*- coding: utf-8 -*-
"""Supervisor HTTP boundary: bearer auth, JSON error envelope, and the
start/stop/restart/status/metrics/logs contract pinned in issue 02.

Exercises the real Flask app from ``create_app`` against a fake process
controller with the Flask test client -- no real process is spawned. The
graceful-vs-forced stop distinction is proven here, against the fake, because
``Popen.terminate()`` is SIGTERM on POSIX but a hard TerminateProcess call on
Windows -- only the fake can assert "terminate() alone was insufficient, kill()
was required" in a way that holds on both platforms. Also covers
``load_token``'s env/file precedence. ``soil_tuning_supervisor`` imports
flask/psutil/stdlib only, so ``importorskip`` is a formality here (kept for
consistency with the project's optional-dep test convention).
"""

from typing import Dict, List, Optional

import pytest

flask = pytest.importorskip("flask")
soil_tuning_supervisor = pytest.importorskip("soil_tuning_supervisor")
soil_tuning_auth = pytest.importorskip("soil_tuning_auth")

from soil_tuning_auth import load_token  # noqa: E402
from soil_tuning_supervisor import ProcessController, create_app  # noqa: E402

TOKEN = "s3cr3t-token"


class FakeProcessController(ProcessController):
    """Minimal ``ProcessController`` fake: no real process is ever spawned.

    ``survives_terminate`` simulates a wedged process that ignores
    ``terminate()`` (stays running until ``kill()``), so tests can exercise
    the stop_timeout / force-kill escalation without a real subprocess.
    """

    def __init__(
        self,
        running: bool = False,
        pid: Optional[int] = None,
        survives_terminate: bool = False,
    ):
        self.pid = pid
        self._running = running
        self.survives_terminate = survives_terminate
        self.spawn_calls: List[dict] = []
        self.terminate_calls = 0
        self.kill_calls = 0
        self.terminated = False
        self.killed = False

    def spawn(
        self,
        cmd: List[str],
        *,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        log_path: Optional[str] = None,
    ) -> int:
        self.spawn_calls.append({"cmd": cmd, "cwd": cwd, "env": env, "log_path": log_path})
        self.pid = 4242
        self._running = True
        return self.pid

    def is_running(self) -> bool:
        return self._running

    def terminate(self) -> None:
        self.terminate_calls += 1
        self.terminated = True
        if not self.survives_terminate:
            self._running = False

    def kill(self) -> None:
        self.kill_calls += 1
        self.killed = True
        self._running = False

    def wait(self, timeout: float) -> Optional[int]:
        return None if self._running else 0

    def stats(self) -> dict:
        if not self._running:
            return {
                "state": "stopped",
                "total": {"cpu_percent": 0.0, "rss_bytes": 0, "n_processes": 0},
                "processes": [],
            }
        processes = [{"pid": self.pid, "name": "bench", "cpu_percent": 1.5, "rss_bytes": 1024}]
        return {
            "state": "running",
            "total": {"cpu_percent": 1.5, "rss_bytes": 1024, "n_processes": 1},
            "processes": processes,
        }


@pytest.fixture
def bench_config():
    return {
        "python": "python",
        "script": "soil_tuning.py",
        "conf_dir": None,
        "data_dir": None,
        "host": "127.0.0.1",
        "port": 8051,
        "log_path": "soil_tuning_bench.log",
        "startup_grace_s": 300,
        "stop_timeout_s": 30,
    }


@pytest.fixture
def client(bench_config):
    controller = FakeProcessController()
    app = create_app(controller, token=TOKEN, bench=bench_config)
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def _auth_header(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def test_missing_auth_header_returns_401_unauthorized_envelope(client):
    resp = client.get("/status")
    assert resp.status_code == 401
    body = resp.get_json()
    assert body["error"]["code"] == "unauthorized"
    assert "message" in body["error"]


def test_wrong_token_returns_401(client):
    resp = client.get("/status", headers=_auth_header("wrong-token"))
    assert resp.status_code == 401
    assert resp.get_json()["error"]["code"] == "unauthorized"


def test_correct_token_returns_200(client):
    resp = client.get("/status", headers=_auth_header(TOKEN))
    assert resp.status_code == 200


def test_unknown_path_with_token_returns_404_json_envelope(client):
    resp = client.get("/no/such/route", headers=_auth_header(TOKEN))
    assert resp.status_code == 404
    assert resp.content_type.startswith("application/json")
    body = resp.get_json()
    assert body["error"]["code"] == "not_found"


def test_unknown_path_without_token_is_401_not_404(client):
    # Auth runs before routing via before_request: an unauthenticated caller
    # must not learn whether a path exists.
    resp = client.get("/no/such/route")
    assert resp.status_code == 401
    assert resp.get_json()["error"]["code"] == "unauthorized"


def test_status_stopped_shape(client):
    resp = client.get("/status", headers=_auth_header(TOKEN))
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["state"] == "stopped"
    assert body["pid"] is None
    assert body["uptime_s"] is None
    assert body["started_at"] is None
    assert body["boot_args"] is None
    assert body["health"]["reachable"] is False


def test_status_running_and_healthy(bench_config, monkeypatch):
    controller = FakeProcessController(running=True, pid=1234)
    app = create_app(controller, token=TOKEN, bench=bench_config)
    monkeypatch.setattr(
        soil_tuning_supervisor,
        "_probe_bench_health",
        lambda bench, token: {"reachable": True, "detail": {"ok": True}},
    )
    with app.test_client() as c:
        resp = c.get("/status", headers=_auth_header(TOKEN))
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["state"] == "running"
    assert body["pid"] == 1234
    assert body["health"] == {"reachable": True, "detail": {"ok": True}}


def test_status_starting_when_unreachable_within_grace(bench_config, monkeypatch):
    controller = FakeProcessController(running=True, pid=1234)
    app = create_app(controller, token=TOKEN, bench=bench_config)
    app.config["SOIL_TUNING_STARTED_AT"] = soil_tuning_supervisor.time.time()
    monkeypatch.setattr(
        soil_tuning_supervisor,
        "_probe_bench_health",
        lambda bench, token: {"reachable": False, "detail": "connection refused"},
    )
    with app.test_client() as c:
        resp = c.get("/status", headers=_auth_header(TOKEN))
    body = resp.get_json()
    assert body["state"] == "starting"


def test_status_wedged_when_unreachable_past_grace(bench_config, monkeypatch):
    controller = FakeProcessController(running=True, pid=1234)
    bench_config = dict(bench_config, startup_grace_s=0.0)
    app = create_app(controller, token=TOKEN, bench=bench_config)
    app.config["SOIL_TUNING_STARTED_AT"] = soil_tuning_supervisor.time.time() - 10.0
    monkeypatch.setattr(
        soil_tuning_supervisor,
        "_probe_bench_health",
        lambda bench, token: {"reachable": False, "detail": "connection refused"},
    )
    with app.test_client() as c:
        resp = c.get("/status", headers=_auth_header(TOKEN))
    body = resp.get_json()
    assert body["state"] == "wedged"


def test_status_stopped_after_crash_survives_recorded_boot_args(bench_config):
    # A bench crash means is_running() goes False; the supervisor must still
    # report "stopped" (not raise) and keep the last boot_args around so
    # /restart still works.
    controller = FakeProcessController(running=False, pid=None)
    app = create_app(controller, token=TOKEN, bench=bench_config)
    app.config["SOIL_TUNING_BOOT_ARGS"] = {
        "project": "demo",
        "start": None,
        "end": None,
        "conf_dir": None,
        "data_dir": None,
        "port": 8051,
        "host": "127.0.0.1",
    }
    with app.test_client() as c:
        resp = c.get("/status", headers=_auth_header(TOKEN))
    body = resp.get_json()
    assert body["state"] == "stopped"
    assert body["boot_args"]["project"] == "demo"


def test_fake_process_controller_satisfies_the_seam():
    fake = FakeProcessController()
    assert isinstance(fake, ProcessController)
    assert fake.is_running() is False
    pid = fake.spawn(["echo", "hi"], cwd=None, env=None, log_path=None)
    assert pid == 4242
    assert fake.is_running() is True
    fake.terminate()
    assert fake.terminated is True
    assert fake.is_running() is False
    assert fake.wait(timeout=1.0) == 0
    stats = fake.stats()
    assert stats["state"] == "stopped"


def test_start_happy_path_assembles_cmd_and_returns_202(client, bench_config):
    body = {"project": "demo", "start": "2026-01-01T00:00:00Z", "port": 9000}
    resp = client.post("/start", json=body, headers=_auth_header(TOKEN))
    assert resp.status_code == 202
    payload = resp.get_json()
    assert payload["state"] == "starting"
    assert payload["pid"] == 4242
    assert payload["boot_args"]["project"] == "demo"
    assert payload["boot_args"]["port"] == 9000


def test_start_assembles_argv_exactly_as_pinned(client, bench_config):
    body = {
        "project": "demo",
        "start": "2026-01-01T00:00:00Z",
        "end": "2026-02-01T00:00:00Z",
        "conf_dir": "/etc/soil/conf",
        "data_dir": "/data/soil",
        "port": 9000,
        "host": "0.0.0.0",
    }
    resp = client.post("/start", json=body, headers=_auth_header(TOKEN))
    assert resp.status_code == 202
    fake = client.application.config["SOIL_TUNING_CONTROLLER"]
    assert fake.spawn_calls[-1]["cmd"] == [
        "python",
        "soil_tuning.py",
        "demo",
        "-c",
        "/etc/soil/conf",
        "--data-dir",
        "/data/soil",
        "--start",
        "2026-01-01T00:00:00Z",
        "--end",
        "2026-02-01T00:00:00Z",
        "--port",
        "9000",
        "--host",
        "0.0.0.0",
    ]
    assert fake.spawn_calls[-1]["log_path"] == bench_config["log_path"]


def test_start_missing_project_returns_400(client):
    resp = client.post("/start", json={}, headers=_auth_header(TOKEN))
    assert resp.status_code == 400
    assert resp.get_json()["error"]["code"] == "invalid_request"


def test_start_while_running_returns_409_already_running(bench_config):
    controller = FakeProcessController(running=True, pid=99)
    app = create_app(controller, token=TOKEN, bench=bench_config)
    with app.test_client() as c:
        resp = c.post("/start", json={"project": "demo"}, headers=_auth_header(TOKEN))
    assert resp.status_code == 409
    assert resp.get_json()["error"]["code"] == "already_running"


def test_stop_graceful_returns_200(bench_config):
    controller = FakeProcessController(running=True, pid=99)
    app = create_app(controller, token=TOKEN, bench=bench_config)
    with app.test_client() as c:
        resp = c.post("/stop", json={}, headers=_auth_header(TOKEN))
    assert resp.status_code == 200
    body = resp.get_json()
    assert body == {"stopped": True, "forced": False, "returncode": 0}
    assert controller.terminate_calls == 1
    assert controller.kill_calls == 0


def test_stop_while_not_running_returns_409_not_running(client):
    resp = client.post("/stop", json={}, headers=_auth_header(TOKEN))
    assert resp.status_code == 409
    assert resp.get_json()["error"]["code"] == "not_running"


def test_stop_timeout_without_force_returns_409(bench_config):
    controller = FakeProcessController(running=True, pid=99, survives_terminate=True)
    app = create_app(controller, token=TOKEN, bench=bench_config)
    with app.test_client() as c:
        resp = c.post("/stop", json={"timeout": 0.01}, headers=_auth_header(TOKEN))
    assert resp.status_code == 409
    assert resp.get_json()["error"]["code"] == "stop_timeout"
    assert controller.kill_calls == 0
    assert controller.is_running() is True


def test_stop_escalates_to_kill_when_forced(bench_config):
    controller = FakeProcessController(running=True, pid=99, survives_terminate=True)
    app = create_app(controller, token=TOKEN, bench=bench_config)
    with app.test_client() as c:
        resp = c.post("/stop", json={"force": True, "timeout": 0.01}, headers=_auth_header(TOKEN))
    assert resp.status_code == 200
    body = resp.get_json()
    assert body == {"stopped": True, "forced": True, "returncode": 0}
    assert controller.terminate_calls == 1
    assert controller.kill_calls == 1


def test_restart_merges_args_over_prior_boot_args(bench_config):
    controller = FakeProcessController(running=True, pid=99)
    app = create_app(controller, token=TOKEN, bench=bench_config)
    app.config["SOIL_TUNING_BOOT_ARGS"] = {
        "project": "demo",
        "start": "2026-01-01T00:00:00Z",
        "end": "2026-02-01T00:00:00Z",
        "conf_dir": None,
        "data_dir": None,
        "port": 8051,
        "host": "127.0.0.1",
    }
    with app.test_client() as c:
        resp = c.post("/restart", json={"end": "2026-03-01T00:00:00Z"}, headers=_auth_header(TOKEN))
    assert resp.status_code == 202
    body = resp.get_json()
    assert body["boot_args"]["project"] == "demo"
    assert body["boot_args"]["start"] == "2026-01-01T00:00:00Z"
    assert body["boot_args"]["end"] == "2026-03-01T00:00:00Z"
    assert controller.terminate_calls == 1
    assert controller.spawn_calls  # a new process was spawned


def test_restart_without_prior_args_or_project_returns_400(client):
    resp = client.post("/restart", json={}, headers=_auth_header(TOKEN))
    assert resp.status_code == 400
    assert resp.get_json()["error"]["code"] == "invalid_request"


def test_restart_when_stopped_just_starts(client):
    resp = client.post("/restart", json={"project": "demo"}, headers=_auth_header(TOKEN))
    assert resp.status_code == 202
    body = resp.get_json()
    assert body["boot_args"]["project"] == "demo"


def test_metrics_running_shape(bench_config):
    controller = FakeProcessController(running=True, pid=99)
    app = create_app(controller, token=TOKEN, bench=bench_config)
    with app.test_client() as c:
        resp = c.get("/metrics", headers=_auth_header(TOKEN))
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["state"] == "running"
    assert body["total"]["n_processes"] == 1
    assert body["processes"][0]["pid"] == 99


def test_metrics_stopped_shape_is_zeros(client):
    resp = client.get("/metrics", headers=_auth_header(TOKEN))
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["state"] == "stopped"
    assert body["total"] == {"cpu_percent": 0.0, "rss_bytes": 0, "n_processes": 0}
    assert body["processes"] == []


def test_logs_tail_returns_last_n_lines(client, bench_config, tmp_path):
    log_file = tmp_path / "bench.log"
    log_file.write_text("\n".join(f"line {i}" for i in range(1, 11)) + "\n", encoding="utf-8")
    client.application.config["SOIL_TUNING_BENCH"]["log_path"] = str(log_file)

    resp = client.get("/logs?tail=3", headers=_auth_header(TOKEN))
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["path"] == str(log_file)
    assert body["lines"] == ["line 8", "line 9", "line 10"]


def test_logs_missing_file_returns_404_log_not_found(client):
    resp = client.get("/logs", headers=_auth_header(TOKEN))
    assert resp.status_code == 404
    assert resp.get_json()["error"]["code"] == "log_not_found"


def test_load_token_env_takes_precedence_over_file(monkeypatch, tmp_path):
    token_file = tmp_path / "token.txt"
    token_file.write_text("file-token", encoding="utf-8")
    monkeypatch.setenv("SOIL_TUNING_API_TOKEN", "env-token")
    monkeypatch.setenv("SOIL_TUNING_API_TOKEN_FILE", str(token_file))
    assert load_token() == "env-token"


def test_load_token_falls_back_to_file(monkeypatch, tmp_path):
    token_file = tmp_path / "token.txt"
    token_file.write_text("  file-token  \n", encoding="utf-8")
    monkeypatch.delenv("SOIL_TUNING_API_TOKEN", raising=False)
    monkeypatch.setenv("SOIL_TUNING_API_TOKEN_FILE", str(token_file))
    assert load_token() == "file-token"


def test_load_token_raises_when_neither_set(monkeypatch):
    monkeypatch.delenv("SOIL_TUNING_API_TOKEN", raising=False)
    monkeypatch.delenv("SOIL_TUNING_API_TOKEN_FILE", raising=False)
    with pytest.raises(RuntimeError):
        load_token()


def test_load_token_raises_when_file_missing(monkeypatch, tmp_path):
    monkeypatch.delenv("SOIL_TUNING_API_TOKEN", raising=False)
    monkeypatch.setenv("SOIL_TUNING_API_TOKEN_FILE", str(tmp_path / "does-not-exist.txt"))
    with pytest.raises(RuntimeError):
        load_token()


def test_require_bearer_rejects_empty_token():
    with pytest.raises(ValueError):
        soil_tuning_auth.require_bearer("")


def test_stop_malformed_timeout_is_400(bench_config):
    controller = FakeProcessController(running=True, pid=1234)
    app = create_app(controller, token=TOKEN, bench=bench_config)
    with app.test_client() as c:
        resp = c.post("/stop", json={"timeout": "soon"}, headers=_auth_header(TOKEN))
    assert resp.status_code == 400
    assert resp.get_json()["error"]["code"] == "invalid_request"
    assert controller.terminated is False


def test_start_malformed_port_is_400(client):
    resp = client.post("/start", json={"project": "kob", "port": "eight"}, headers=_auth_header(TOKEN))
    assert resp.status_code == 400
    assert resp.get_json()["error"]["code"] == "invalid_request"


def test_status_wedged_at_grace_boundary(bench_config, monkeypatch):
    controller = FakeProcessController(running=True, pid=1234)
    bench_config = dict(bench_config, startup_grace_s=10.0)
    app = create_app(controller, token=TOKEN, bench=bench_config)
    app.config["SOIL_TUNING_STARTED_AT"] = soil_tuning_supervisor.time.time() - 10.0
    monkeypatch.setattr(
        soil_tuning_supervisor,
        "_probe_bench_health",
        lambda bench, token: {"reachable": False, "detail": "connection refused"},
    )
    with app.test_client() as c:
        resp = c.get("/status", headers=_auth_header(TOKEN))
    assert resp.get_json()["state"] == "wedged"
