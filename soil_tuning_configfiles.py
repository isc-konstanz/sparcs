"""Allowlisted whole-file config read/write with TOML validation, a
subprocess dry-run validator, and a git audit trail.

``ConfigStore`` is the only way the soil-tuning supervisor touches config
files on disk: names are resolved through a fixed dict (no path is ever
built from user input, so traversal is impossible by construction), every
write is validated before it lands, and any failure leaves the target file's
bytes and the repo's git history exactly as they were.

Git side effects are narrow on purpose: only ``git add <path>``,
``git commit -m ... -- <path>``, and (during rollback of a failed commit) a
pathspec-scoped ``git reset -- <path>`` that touches nothing but the one
file's index entry -- never ``-A``, never ``commit -a``, no bare/hard reset
or other destructive git command, and no ``Co-Authored-By`` trailer in
generated messages.

Imports stdlib only -- never flask, soil_tuning, sparcs.*, lories.*, dash,
or fipy (see ENVIRONMENT.md)."""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

try:
    import tomllib as toml
except ModuleNotFoundError:
    import tomli as toml

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 120.0
_TOML_SUFFIXES = (".conf", ".toml")
_TAIL_CHARS = 2000
_GIT_TIMEOUT_S = 30.0

_write_lock = threading.Lock()


class ConfigError(Exception):
    """Base class for ``ConfigStore`` failures. ``code``/``status``/``detail``
    map directly onto the supervisor's JSON error envelope."""

    code = "config_error"
    status = 500

    def __init__(self, message: str, **detail):
        super().__init__(message)
        self.message = message
        self.detail = detail


class ConfigNotAllowed(ConfigError):
    code = "config_not_allowlisted"
    status = 404


class ConfigNotVersioned(ConfigError):
    code = "config_not_versioned"
    status = 409


class ConfigDirty(ConfigError):
    code = "config_dirty"
    status = 409


class ConfigValidationFailed(ConfigError):
    code = "validation_failed"
    status = 422


def _run_git(args: List[str], *, cwd: Path) -> subprocess.CompletedProcess:
    """Run a git subcommand against ``cwd``; never raises on nonzero exit
    (callers inspect ``returncode``)."""
    return subprocess.run(
        ["git"] + args,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=_GIT_TIMEOUT_S,
    )


def _tail(text: str, n: int = _TAIL_CHARS) -> str:
    return text[-n:] if text else ""


def _unstage(path: Path, repo_root: Path) -> None:
    """Best-effort pathspec-scoped unstage after a failed add/commit: without
    it the index keeps the staged candidate, ``git status --porcelain`` stays
    non-empty, and every later write on this name false-positives as
    ``config_dirty`` until someone unwedges the box by hand."""
    try:
        result = _run_git(["reset", "--", str(path)], cwd=repo_root)
        if result.returncode != 0:
            logger.warning("rollback unstage failed for %s: %s", path, _tail(result.stderr, 200))
    except subprocess.TimeoutExpired:
        logger.warning("rollback unstage timed out for %s", path)


class ConfigStore:
    """Allowlisted whole-file config read/write, git-audited.

    ``allowlist`` maps opaque names to absolute file paths; lookups are
    dict-only, so no user-supplied path is ever joined onto a base
    directory. ``validate_cmd`` (if given) is run as a dry-run validator
    after each candidate write, with ``timeout_s`` and cwd set to the
    target file's git repo root.
    """

    def __init__(
        self,
        allowlist: Dict[str, Union[str, Path]],
        *,
        validate_cmd: Optional[List[str]] = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        self._allowlist = {name: Path(path) for name, path in allowlist.items()}
        self._validate_cmd = list(validate_cmd) if validate_cmd else None
        self._timeout_s = timeout_s

    def names(self) -> List[str]:
        return list(self._allowlist.keys())

    def _resolve(self, name: str) -> Path:
        path = self._allowlist.get(name)
        if path is None:
            raise ConfigNotAllowed(f"config {name!r} is not allowlisted", name=name)
        return path

    def read(self, name: str) -> dict:
        path = self._resolve(name)
        # Shares the write lock so a read can never observe a candidate that
        # step 5 has written but not yet validated/committed.
        with _write_lock:
            content = path.read_text(encoding="utf-8")
        return {"name": name, "path": str(path), "content": content}

    def write(
        self,
        name: str,
        content: str,
        *,
        author: str = "unknown",
        message: str = "",
    ) -> dict:
        # Step 1: allowlist lookup.
        path = self._resolve(name)

        with _write_lock:
            # Step 2: the file must live inside a git work tree.
            repo_dir = path.parent
            inside = _run_git(["rev-parse", "--is-inside-work-tree"], cwd=repo_dir)
            if inside.returncode != 0 or inside.stdout.strip() != "true":
                raise ConfigNotVersioned(f"{path} is not inside a git work tree", name=name, path=str(path))
            toplevel = _run_git(["rev-parse", "--show-toplevel"], cwd=repo_dir)
            if toplevel.returncode != 0:
                raise ConfigNotVersioned(f"{path} is not inside a git work tree", name=name, path=str(path))
            repo_root = Path(toplevel.stdout.strip())

            # Step 3: the target file must have no uncommitted diff.
            status = _run_git(["status", "--porcelain", "--", str(path)], cwd=repo_root)
            if status.returncode != 0:
                raise ConfigNotVersioned(f"{path} is not inside a git work tree", name=name, path=str(path))
            if status.stdout.strip():
                raise ConfigDirty(f"{path} has uncommitted local changes", name=name, path=str(path))

            # Step 4: if a TOML-suffixed file, parse the CANDIDATE content
            # in-memory before touching disk; file untouched on failure and
            # validate_cmd is never run.
            if path.suffix in _TOML_SUFFIXES:
                try:
                    toml.loads(content)
                except toml.TOMLDecodeError as e:
                    raise ConfigValidationFailed(
                        f"invalid TOML in {name!r}: {e}", name=name, path=str(path), parser_message=str(e)
                    ) from e

            # Step 5: write candidate bytes, run the validator, roll back on
            # any failure (nonzero exit or timeout).
            original_bytes = path.read_bytes()
            path.write_text(content, encoding="utf-8")
            if self._validate_cmd:
                try:
                    result = subprocess.run(
                        self._validate_cmd,
                        cwd=str(repo_root),
                        capture_output=True,
                        text=True,
                        timeout=self._timeout_s,
                    )
                except subprocess.TimeoutExpired:
                    path.write_bytes(original_bytes)
                    raise ConfigValidationFailed(
                        f"validation of {name!r} timed out after {self._timeout_s}s",
                        name=name,
                        path=str(path),
                        timeout_s=self._timeout_s,
                    )
                if result.returncode != 0:
                    path.write_bytes(original_bytes)
                    raise ConfigValidationFailed(
                        f"validation of {name!r} failed (exit {result.returncode})",
                        name=name,
                        path=str(path),
                        returncode=result.returncode,
                        stdout=_tail(result.stdout),
                        stderr=_tail(result.stderr),
                    )

            # Step 6: commit exactly the one file. Any failure (including a
            # git timeout, e.g. index.lock contention) must unstage AND
            # restore bytes, or the polluted index wedges this name behind a
            # false config_dirty forever.
            commit_message = message or "update"
            try:
                add_result = _run_git(["add", "--", str(path)], cwd=repo_root)
                if add_result.returncode != 0:
                    _unstage(path, repo_root)
                    path.write_bytes(original_bytes)
                    raise ConfigValidationFailed(
                        f"git add failed for {name!r}",
                        name=name,
                        path=str(path),
                        stderr=_tail(add_result.stderr),
                    )
                commit_result = _run_git(
                    [
                        "commit",
                        "-m",
                        f"config-api: {name}: {commit_message}",
                        "-m",
                        f"Author: {author}",
                        "-m",
                        "Via: soil-tuning supervisor API",
                        "--",
                        str(path),
                    ],
                    cwd=repo_root,
                )
                if commit_result.returncode != 0:
                    _unstage(path, repo_root)
                    path.write_bytes(original_bytes)
                    raise ConfigValidationFailed(
                        f"git commit failed for {name!r}",
                        name=name,
                        path=str(path),
                        stderr=_tail(commit_result.stderr),
                        stdout=_tail(commit_result.stdout),
                    )
                sha_result = _run_git(["rev-parse", "HEAD"], cwd=repo_root)
            except subprocess.TimeoutExpired:
                _unstage(path, repo_root)
                path.write_bytes(original_bytes)
                raise ConfigValidationFailed(
                    f"git operation for {name!r} timed out after {_GIT_TIMEOUT_S}s",
                    name=name,
                    path=str(path),
                    timeout_s=_GIT_TIMEOUT_S,
                )
            sha = sha_result.stdout.strip() if sha_result.returncode == 0 else None

        return {
            "name": name,
            "path": str(path),
            "committed": sha,
            "note": "restart required to apply",
        }


def parse_config_allow(spec: str) -> Tuple[str, str]:
    """Parse one ``--config-allow NAME=PATH`` CLI value into ``(name, path)``.
    Raises ``ValueError`` on a malformed spec (no ``=``, or an empty name)."""
    if "=" not in spec:
        raise ValueError(f"--config-allow value {spec!r} must be NAME=PATH")
    name, _, path = spec.partition("=")
    name = name.strip()
    path = path.strip()
    if not name or not path:
        raise ValueError(f"--config-allow value {spec!r} must be NAME=PATH")
    return name, path


def parse_validate_cmd(spec: Optional[str]) -> Optional[List[str]]:
    """Split a ``--config-validate-cmd`` string into an argv list with
    ``shlex``. Returns ``None`` for an unset/blank spec. POSIX lexing would
    silently strip the backslashes out of Windows paths, so it is disabled
    on Windows."""
    if not spec:
        return None
    return shlex.split(spec, posix=os.name != "nt")
