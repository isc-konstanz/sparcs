# -*- coding: utf-8 -*-
"""
tests.test_fieldsim_ticker
~~~~~~~~~~~~~~~~~~~~~~~~~~

``Ticker``: the lories scheduler it wraps and what it hands the runner --
a UTC clock, the scheduler's consecutive-failure count before each tick and
the scheduler's interrupt as the cancel signal. Failures are counted and
logged once by ``TickScheduler._run_slot``, which the tests drive directly;
no thread is started.
"""

import datetime as dt
import logging

import pandas as pd
import pytz
from lories.scheduler import TickScheduler
from sparcs.components.agriculture.fieldsim import components
from sparcs.components.agriculture.fieldsim.core.config import FieldConfig, FieldSetup, SoilConfig
from sparcs.components.agriculture.fieldsim.runtime import Ticker

LORIES_SCHEDULER_LOGGER = "lories.scheduler"


def _setup(interval: int = 30, offset: int = 10) -> FieldSetup:
    return FieldSetup(
        field=FieldConfig.from_dict({"interval": interval, "offset": offset}),
        soil=SoilConfig.from_dict({"mesh": {}}),
        shading=components.ShadingConfig.from_dict(),
    )


class _StubRunner:
    """Records the clock, the pre-tick ``tick_failures`` and the cancel callable
    of every tick; raises for its first ``failures`` ticks."""

    def __init__(self, failures: int = 0):
        self.failures_left = failures
        self.nows: list = []
        self.seen_failures: list = []
        self.cancels: list = []

    def run_tick(self, now, cancel=None) -> bool:
        self.nows.append(now)
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


def test_ticker_schedules_in_the_site_timezone():
    assert Ticker(_setup(), _StubRunner(), tz=None).scheduler._timezone is pytz.UTC
    assert Ticker(_setup(), _StubRunner(), tz="Europe/Berlin").scheduler._timezone == pytz.timezone("Europe/Berlin")


def test_ticker_hands_the_runner_a_utc_clock():
    runner = _StubRunner()
    ticker = Ticker(_setup(), runner, tz="Europe/Berlin")

    ticker._on_slot()

    assert runner.nows[0].utcoffset() == dt.timedelta(0)


def test_scheduler_counts_consecutive_failures_and_resets_on_success():
    ticker = Ticker(_setup(), _StubRunner(failures=2))

    ticker.scheduler._run_slot()
    assert ticker.scheduler.failures == 1
    ticker.scheduler._run_slot()
    assert ticker.scheduler.failures == 2
    ticker.scheduler._run_slot()
    assert ticker.scheduler.failures == 0


def test_ticker_hands_the_pre_tick_failure_count_to_the_runner():
    runner = _StubRunner(failures=2)
    ticker = Ticker(_setup(), runner)

    for _ in range(3):
        ticker.scheduler._run_slot()

    assert runner.seen_failures == [0, 1, 2]
    assert runner.tick_failures == 2


def test_a_failed_tick_is_logged_once_with_the_consecutive_count(caplog):
    ticker = Ticker(_setup(), _StubRunner(failures=2), name="field")

    with caplog.at_level(logging.ERROR):
        ticker.scheduler._run_slot()
        ticker.scheduler._run_slot()

    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert [r.name for r in errors] == [LORIES_SCHEDULER_LOGGER, LORIES_SCHEDULER_LOGGER]
    assert "(1 consecutive)" in errors[0].getMessage()
    assert "(2 consecutive)" in errors[1].getMessage()


def test_ticker_stop_raises_the_cancel_signal_run_tick_reads():
    runner = _StubRunner()
    ticker = Ticker(_setup(), runner)

    ticker._on_slot()
    cancel = runner.cancels[0]
    assert cancel() is False

    ticker.stop()
    assert cancel() is True
