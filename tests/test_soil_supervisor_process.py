# -*- coding: utf-8 -*-
"""Integration test for ``SubprocessController``: proves the real
terminate/kill escalation against a trivial subprocess.

Deliberately narrow in what it asserts. ``Popen.terminate()`` sends SIGTERM
on POSIX (the signal the bench's ``_cleanup_and_exit`` handler catches) but
maps to a hard ``TerminateProcess`` call on Windows -- there is no graceful
signal delivered to a plain Windows process by ``terminate()``. Since the
deploy target is Linux, this test asserts only "process ends" semantics that
hold on both platforms (terminate, then escalate to kill if still alive, then
confirm psutil no longer sees the pid). It does NOT assert that terminate()
alone was sufficient, and it does NOT assert graceful-vs-forced signal
distinctions -- those are proven against ``FakeProcessController`` in
test_soil_supervisor.py, which is platform-independent by construction. Do
not treat a green run of this test on Windows as proof of graceful shutdown.

Marked slow per house convention: runs in the suite, not skipped by default
selection.
"""

import sys
import time

import pytest

psutil = pytest.importorskip("psutil")
soil_tuning_supervisor = pytest.importorskip("soil_tuning_supervisor")

pytestmark = pytest.mark.slow

from soil_tuning_supervisor import SubprocessController  # noqa: E402

WAIT_S = 10.0
POLL_INTERVAL_S = 0.1


def _wait_until(predicate, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(POLL_INTERVAL_S)
    return predicate()


def test_terminate_then_kill_escalation_ends_a_real_sleeper():
    controller = SubprocessController()
    cmd = [sys.executable, "-c", "import time; time.sleep(60)"]
    pid = controller.spawn(cmd)

    assert controller.is_running() is True
    assert psutil.pid_exists(pid)

    controller.terminate()
    controller.wait(WAIT_S)

    if controller.is_running():
        # terminate() alone did not end it (expected on Windows; possible on
        # POSIX too under load) -- escalate exactly like the /stop route does.
        controller.kill()
        controller.wait(WAIT_S)

    assert controller.is_running() is False
    # Popen.wait() reaps the child, so psutil must no longer see it as a live
    # process at this pid (allow a short grace period for OS bookkeeping).
    gone = _wait_until(lambda: not _is_alive(pid), WAIT_S)
    assert gone


def _is_alive(pid: int) -> bool:
    try:
        proc = psutil.Process(pid)
        return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False
