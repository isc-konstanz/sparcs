# -*- coding: utf-8 -*-
"""Supervisor HTTP boundary: bearer auth, JSON error envelope, /status shape.

Exercises the real Flask app from ``create_app`` against a fake process
controller with the Flask test client -- no real process is spawned. Also
covers ``load_token``'s env/file precedence. ``soil_tuning_supervisor``
imports flask/psutil/stdlib only, so ``importorskip`` is a formality here
(kept for consistency with the project's optional-dep test convention).
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
    """Minimal ``ProcessController`` fake: no real process is ever spawned."""

    def __init__(self, running: bool = False, pid: Optional[int] = None):
        self.pid = pid
        self._running = running
        self.spawn_calls: List[dict] = []
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
        self.terminated = True
        self._running = False

    def kill(self) -> None:
        self.killed = True
        self._running = False

    def wait(self, timeout: float) -> Optional[int]:
        return None if self._running else 0

    def stats(self) -> dict:
        return {"pid": self.pid, "cpu_percent": 0.0, "memory_rss": 0, "children": []}


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
    assert resp.get_json() == {
        "state": "stopped",
        "pid": None,
        "uptime_s": None,
        "boot_args": None,
    }


def test_status_running_reports_pid(bench_config):
    controller = FakeProcessController(running=True, pid=1234)
    app = create_app(controller, token=TOKEN, bench=bench_config)
    with app.test_client() as c:
        resp = c.get("/status", headers=_auth_header(TOKEN))
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["state"] == "running"
    assert body["pid"] == 1234


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
    assert stats["pid"] == 4242


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
