# -*- coding: utf-8 -*-
"""tests.test_fieldsim_ticker
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Ticker``: the lories scheduler it wraps, the consecutive-failure tally, the
pre-tick handoff of that tally to the runner, and the cancel signal ``stop``
raises. Every test drives ``_on_slot`` directly; no thread is started.
"""

import datetime as dt
import logging

import pandas as pd
import pytz
from lories.scheduler import TickScheduler
from sparcs.components.agriculture.fieldsim import components
from sparcs.components.agriculture.fieldsim.core.config import FieldConfig, FieldSetup, SoilConfig
from sparcs.components.agriculture.fieldsim.runtime import Ticker

SCHEDULER_LOGGER = "sparcs.components.agriculture.fieldsim.runtime.scheduler"


def _setup(interval: int = 30, offset: int = 10) -> FieldSetup:
    return FieldSetup(
        field=FieldConfig.from_dict({"interval": interval, "offset": offset}),
        soil=SoilConfig.from_dict({"mesh": {}}),
        shading=components.ShadingConfig.from_dict(),
    )


class _StubRunner:
    """Records the pre-tick ``tick_failures`` it was handed and the cancel
    callable it was called with; raises for its first ``failures`` ticks."""

    def __init__(self, failures: int = 0):
        self.failures_left = failures
        self.seen_failures: list = []
        self.cancels: list = []

    def run_tick(self, now, cancel=None) -> bool:
        self.seen_failures.append(self.tick_failures)
        self.cancels.append(cancel)
        if self.failures_left > 0:
            self.failures_left -= 1
            raise RuntimeError("boom")
        return True


def test_ticker_wraps_a_lories_tick_scheduler():
    ticker = Ticker(_setup(interval=30, offset=10), _StubRunner())

    assert isinstance(ticker.scheduler, TickScheduler)
    assert ticker.scheduler._interval == pd.Timedelta(minutes=30)
    assert ticker.scheduler._offset == pd.Timedelta(minutes=10)
    assert ticker.scheduler._on_slot == ticker._on_slot


def test_ticker_normalises_the_timezone():
    assert Ticker(_setup(), _StubRunner(), tz=None).timezone is pytz.UTC
    assert Ticker(_setup(), _StubRunner(), tz=dt.timezone.utc).timezone is pytz.UTC
    assert Ticker(_setup(), _StubRunner(), tz="Europe/Berlin").timezone == pytz.timezone("Europe/Berlin")


def test_ticker_counts_consecutive_failures_and_resets_on_success():
    ticker = Ticker(_setup(), _StubRunner(failures=2))

    ticker._on_slot()
    assert ticker.failed_ticks == 1
    ticker._on_slot()
    assert ticker.failed_ticks == 2
    ticker._on_slot()
    assert ticker.failed_ticks == 0


def test_ticker_hands_the_pre_tick_failure_tally_to_the_runner():
    runner = _StubRunner(failures=2)
    ticker = Ticker(_setup(), runner)

    for _ in range(3):
        ticker._on_slot()

    assert runner.seen_failures == [0, 1, 2]
    assert runner.tick_failures == 2


def test_ticker_failure_log_carries_the_consecutive_count(caplog):
    """A one-off transient and a deterministic every-tick crash must be
    distinguishable in the log, not two identical tracebacks."""
    ticker = Ticker(_setup(), _StubRunner(failures=2), name="field")

    with caplog.at_level(logging.ERROR, logger=SCHEDULER_LOGGER):
        ticker._on_slot()
        ticker._on_slot()

    failures = [r for r in caplog.records if "tick failed" in r.getMessage()]
    assert [r.levelno for r in failures] == [logging.ERROR, logging.ERROR]
    assert "(1 consecutive)" in failures[0].getMessage()
    assert "(2 consecutive)" in failures[1].getMessage()


def test_ticker_stop_raises_the_cancel_signal_run_tick_reads():
    runner = _StubRunner()
    ticker = Ticker(_setup(), runner)

    ticker._on_slot()
    cancel = runner.cancels[0]
    assert cancel() is False

    ticker.stop()
    assert cancel() is True
