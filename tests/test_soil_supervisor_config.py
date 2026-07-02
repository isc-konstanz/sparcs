# -*- coding: utf-8 -*-
"""Tier-2 config API: allowlisted whole-file GET/PUT with TOML syntax
validation, a subprocess dry-run validator, byte-identical rollback on any
failure, and a one-commit-per-accepted-write git audit trail (issue 05).

Exercises ``ConfigStore`` directly and through the real Flask app from
``create_app`` (Flask test client, no real process spawned). Every test
builds its own throwaway git repo under ``tmp_path`` -- an assertion in the
fixture pins the repo root under ``tmp_path`` so no test can ever touch the
sparcs/lories/parent repos. ``soil_tuning_configfiles``/``soil_tuning_supervisor``
import flask/psutil/stdlib only, so ``importorskip`` is a formality here
(kept for consistency with the project's optional-dep test convention).
"""

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pytest

flask = pytest.importorskip("flask")
soil_tuning_configfiles = pytest.importorskip("soil_tuning_configfiles")
soil_tuning_supervisor = pytest.importorskip("soil_tuning_supervisor")

from soil_tuning_configfiles import ConfigStore  # noqa: E402
from soil_tuning_supervisor import ProcessController, create_app  # noqa: E402

TOKEN = "s3cr3t-token"
DEMO_CONTENT = 'name = "demo"\nvalue = 1\n'

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is not on PATH")


def _run_git(args: List[str], cwd: Path) -> subprocess.CompletedProcess:
    result = subprocess.run(["git"] + args, cwd=str(cwd), capture_output=True, text=True)
    assert result.returncode == 0, f"git {args} failed: {result.stderr}"
    return result


def _commit_count(repo_dir: Path) -> int:
    result = _run_git(["rev-list", "--count", "HEAD"], cwd=repo_dir)
    return int(result.stdout.strip())


@pytest.fixture
def repo(tmp_path) -> Path:
    """A throwaway git repo under ``tmp_path`` with one committed demo.conf."""
    repo_dir = tmp_path / "config_repo"
    repo_dir.mkdir()
    assert repo_dir.is_relative_to(tmp_path)  # never touch a real repo

    _run_git(["init"], cwd=repo_dir)
    _run_git(["config", "user.name", "Test User"], cwd=repo_dir)
    _run_git(["config", "user.email", "test@example.invalid"], cwd=repo_dir)

    demo = repo_dir / "demo.conf"
    demo.write_text(DEMO_CONTENT, encoding="utf-8")
    _run_git(["add", "--", "demo.conf"], cwd=repo_dir)
    _run_git(["commit", "-m", "initial demo.conf"], cwd=repo_dir)

    return repo_dir


@pytest.fixture
def demo_path(repo) -> Path:
    return repo / "demo.conf"


def _passing_validator() -> List[str]:
    return [sys.executable, "-c", "import sys; sys.exit(0)"]


def _failing_validator() -> List[str]:
    return [sys.executable, "-c", "import sys; sys.exit(1)"]


def _marker_validator(marker_path: Path) -> List[str]:
    """A validator that writes a marker file when invoked, so a test can
    assert it was (or was not) run."""
    return [sys.executable, "-c", f"open(r'{marker_path}', 'w').close()"]


class FakeProcessController(ProcessController):
    """Minimal ``ProcessController`` fake: no real process is ever spawned.
    The /config tests don't exercise process control, but ``create_app``
    requires a controller."""

    pid: Optional[int] = None

    def is_running(self) -> bool:
        return False

    def spawn(self, cmd, *, cwd=None, env=None, log_path=None) -> int:  # pragma: no cover - unused here
        raise NotImplementedError

    def terminate(self) -> None:  # pragma: no cover - unused here
        pass

    def kill(self) -> None:  # pragma: no cover - unused here
        pass

    def wait(self, timeout: float):  # pragma: no cover - unused here
        return None

    def stats(self) -> dict:
        return {"state": "stopped", "total": {"cpu_percent": 0.0, "rss_bytes": 0, "n_processes": 0}, "processes": []}


@pytest.fixture
def bench_config() -> Dict:
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


def _client(bench_config, config_store):
    controller = FakeProcessController()
    app = create_app(controller, token=TOKEN, bench=bench_config, config_store=config_store)
    app.config["TESTING"] = True
    return app.test_client()


def _auth_header() -> dict:
    return {"Authorization": f"Bearer {TOKEN}"}


# --------------------------------------------------------------------------
# ConfigStore unit tests
# --------------------------------------------------------------------------


def test_names_lists_allowlisted_names(demo_path):
    store = ConfigStore({"demo": demo_path})
    assert store.names() == ["demo"]


def test_read_returns_name_path_content(demo_path):
    store = ConfigStore({"demo": demo_path})
    result = store.read("demo")
    assert result == {"name": "demo", "path": str(demo_path), "content": DEMO_CONTENT}


def test_read_unknown_name_raises_config_not_allowed(demo_path):
    store = ConfigStore({"demo": demo_path})
    with pytest.raises(soil_tuning_configfiles.ConfigNotAllowed):
        store.read("nope")


def test_write_accepted_with_passing_validator_commits_once(repo, demo_path):
    before_count = _commit_count(repo)
    store = ConfigStore({"demo": demo_path}, validate_cmd=_passing_validator())

    new_content = 'name = "demo"\nvalue = 2\n'
    result = store.write("demo", new_content, author="alice", message="bump value")

    assert result["name"] == "demo"
    assert result["path"] == str(demo_path)
    assert result["committed"]
    assert result["note"] == "restart required to apply"

    assert demo_path.read_text(encoding="utf-8") == new_content
    after_count = _commit_count(repo)
    assert after_count - before_count == 1

    log = _run_git(["log", "-1", "--pretty=%B"], cwd=repo).stdout
    assert "alice" in log
    assert "Co-Authored-By" not in log

    status = _run_git(["status", "--porcelain"], cwd=repo).stdout
    assert status.strip() == ""

    head_sha = _run_git(["rev-parse", "HEAD"], cwd=repo).stdout.strip()
    assert result["committed"] == head_sha


def test_write_failing_validator_rolls_back_byte_identical(repo, demo_path):
    original_bytes = demo_path.read_bytes()
    before_count = _commit_count(repo)
    store = ConfigStore({"demo": demo_path}, validate_cmd=_failing_validator())

    with pytest.raises(soil_tuning_configfiles.ConfigValidationFailed):
        store.write("demo", 'name = "demo"\nvalue = 999\n')

    assert demo_path.read_bytes() == original_bytes
    assert _commit_count(repo) == before_count
    status = _run_git(["status", "--porcelain"], cwd=repo).stdout
    assert status.strip() == ""


def test_write_invalid_toml_never_invokes_validator(repo, demo_path, tmp_path):
    marker = tmp_path / "validator_ran.marker"
    original_bytes = demo_path.read_bytes()
    before_count = _commit_count(repo)
    store = ConfigStore({"demo": demo_path}, validate_cmd=_marker_validator(marker))

    with pytest.raises(soil_tuning_configfiles.ConfigValidationFailed):
        store.write("demo", "this is not [ valid toml")

    assert not marker.exists()
    assert demo_path.read_bytes() == original_bytes
    assert _commit_count(repo) == before_count


def test_write_validator_timeout_rolls_back(repo, demo_path):
    original_bytes = demo_path.read_bytes()
    before_count = _commit_count(repo)
    slow_validator = [sys.executable, "-c", "import time; time.sleep(5)"]
    store = ConfigStore({"demo": demo_path}, validate_cmd=slow_validator, timeout_s=0.2)

    with pytest.raises(soil_tuning_configfiles.ConfigValidationFailed):
        store.write("demo", 'name = "demo"\nvalue = 3\n')

    assert demo_path.read_bytes() == original_bytes
    assert _commit_count(repo) == before_count


def test_write_dirty_target_raises_config_dirty(repo, demo_path):
    demo_path.write_text('name = "demo"\nvalue = "dirty"\n', encoding="utf-8")  # uncommitted local edit
    store = ConfigStore({"demo": demo_path}, validate_cmd=_passing_validator())

    with pytest.raises(soil_tuning_configfiles.ConfigDirty):
        store.write("demo", 'name = "demo"\nvalue = 4\n')


def test_write_non_git_dir_raises_config_not_versioned(tmp_path):
    plain_dir = tmp_path / "not_a_repo"
    plain_dir.mkdir()
    plain_file = plain_dir / "demo.conf"
    plain_file.write_text(DEMO_CONTENT, encoding="utf-8")
    store = ConfigStore({"demo": plain_file}, validate_cmd=_passing_validator())

    with pytest.raises(soil_tuning_configfiles.ConfigNotVersioned):
        store.write("demo", 'name = "demo"\nvalue = 5\n')


# --------------------------------------------------------------------------
# HTTP route tests
# --------------------------------------------------------------------------


def test_get_config_lists_allowlisted_names(bench_config, demo_path):
    store = ConfigStore({"demo": demo_path})
    client = _client(bench_config, store)
    resp = client.get("/config", headers=_auth_header())
    assert resp.status_code == 200
    assert resp.get_json() == {"configs": ["demo"]}


def test_get_config_name_reads_content(bench_config, demo_path):
    store = ConfigStore({"demo": demo_path})
    client = _client(bench_config, store)
    resp = client.get("/config/demo", headers=_auth_header())
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["name"] == "demo"
    assert body["content"] == DEMO_CONTENT


def test_get_config_unknown_name_returns_404_config_not_allowlisted(bench_config, demo_path):
    store = ConfigStore({"demo": demo_path})
    client = _client(bench_config, store)
    resp = client.get("/config/nope", headers=_auth_header())
    assert resp.status_code == 404
    assert resp.get_json()["error"]["code"] == "config_not_allowlisted"


def test_get_config_traversal_shaped_name_returns_404(bench_config, demo_path):
    store = ConfigStore({"demo": demo_path})
    client = _client(bench_config, store)
    resp = client.get("/config/../demo.conf", headers=_auth_header())
    assert resp.status_code == 404


def test_put_config_accepted_returns_200_with_sha(bench_config, repo, demo_path):
    store = ConfigStore({"demo": demo_path}, validate_cmd=_passing_validator())
    client = _client(bench_config, store)
    before_count = _commit_count(repo)

    new_content = 'name = "demo"\nvalue = 42\n'
    resp = client.put(
        "/config/demo",
        json={"content": new_content, "author": "bob", "message": "set to 42"},
        headers=_auth_header(),
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["committed"]

    assert demo_path.read_text(encoding="utf-8") == new_content
    assert _commit_count(repo) - before_count == 1
    log = _run_git(["log", "-1", "--pretty=%B"], cwd=repo).stdout
    assert "bob" in log
    status = _run_git(["status", "--porcelain"], cwd=repo).stdout
    assert status.strip() == ""


def test_put_config_failing_validator_returns_422_and_rolls_back(bench_config, repo, demo_path):
    original_bytes = demo_path.read_bytes()
    before_count = _commit_count(repo)
    store = ConfigStore({"demo": demo_path}, validate_cmd=_failing_validator())
    client = _client(bench_config, store)

    resp = client.put("/config/demo", json={"content": 'name = "demo"\nvalue = 7\n'}, headers=_auth_header())
    assert resp.status_code == 422
    assert resp.get_json()["error"]["code"] == "validation_failed"

    assert demo_path.read_bytes() == original_bytes
    assert _commit_count(repo) == before_count
    status = _run_git(["status", "--porcelain"], cwd=repo).stdout
    assert status.strip() == ""


def test_put_config_dirty_returns_409_config_dirty(bench_config, repo, demo_path):
    demo_path.write_text('name = "demo"\nvalue = "dirty"\n', encoding="utf-8")
    store = ConfigStore({"demo": demo_path}, validate_cmd=_passing_validator())
    client = _client(bench_config, store)

    resp = client.put("/config/demo", json={"content": 'name = "demo"\nvalue = 8\n'}, headers=_auth_header())
    assert resp.status_code == 409
    assert resp.get_json()["error"]["code"] == "config_dirty"


def test_put_config_non_versioned_returns_409(bench_config, tmp_path):
    plain_dir = tmp_path / "not_a_repo2"
    plain_dir.mkdir()
    plain_file = plain_dir / "demo.conf"
    plain_file.write_text(DEMO_CONTENT, encoding="utf-8")
    store = ConfigStore({"demo": plain_file})
    client = _client(bench_config, store)

    resp = client.put("/config/demo", json={"content": 'name = "demo"\nvalue = 9\n'}, headers=_auth_header())
    assert resp.status_code == 409
    assert resp.get_json()["error"]["code"] == "config_not_versioned"


def test_put_config_without_content_returns_400(bench_config, demo_path):
    store = ConfigStore({"demo": demo_path})
    client = _client(bench_config, store)

    resp = client.put("/config/demo", json={}, headers=_auth_header())
    assert resp.status_code == 400
    assert resp.get_json()["error"]["code"] == "invalid_request"


def test_put_config_non_str_content_returns_400(bench_config, demo_path):
    store = ConfigStore({"demo": demo_path})
    client = _client(bench_config, store)

    resp = client.put("/config/demo", json={"content": 123}, headers=_auth_header())
    assert resp.status_code == 400
    assert resp.get_json()["error"]["code"] == "invalid_request"


def test_config_routes_require_auth(bench_config, demo_path):
    store = ConfigStore({"demo": demo_path})
    client = _client(bench_config, store)

    resp = client.get("/config")
    assert resp.status_code == 401
    assert resp.get_json()["error"]["code"] == "unauthorized"


def test_write_commit_failure_unstages_and_next_write_succeeds(repo, demo_path):
    # A rejecting pre-commit hook makes `git commit` fail after `git add`
    # succeeded. The rollback must unstage the candidate: leftover staged
    # content would make every later write on this name false-positive as
    # config_dirty until someone unwedges the repo by hand.
    hooks_dir = repo / ".git" / "hooks"
    hooks_dir.mkdir(exist_ok=True)
    _run_git(["config", "core.hooksPath", str(hooks_dir)], cwd=repo)
    hook = hooks_dir / "pre-commit"
    hook.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    hook.chmod(0o755)

    original_bytes = demo_path.read_bytes()
    before_count = _commit_count(repo)
    store = ConfigStore({"demo": demo_path}, validate_cmd=_passing_validator())

    with pytest.raises(soil_tuning_configfiles.ConfigValidationFailed):
        store.write("demo", 'name = "demo"\nvalue = 3\n')

    assert demo_path.read_bytes() == original_bytes
    assert _commit_count(repo) == before_count
    status = _run_git(["status", "--porcelain"], cwd=repo).stdout
    assert status.strip() == ""

    hook.unlink()
    result = store.write("demo", 'name = "demo"\nvalue = 3\n')
    assert result["committed"]
    assert _commit_count(repo) == before_count + 1
