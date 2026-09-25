# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.fieldsim.runtime.scheduler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The tick thread, delegated to lories. ``Ticker`` wraps
``lories.scheduler.TickScheduler`` and adds only what is field-specific: the
cancel signal handed to ``FieldRunner.run_tick``, the consecutive-failure
tally mirrored down to the runner, and the slot-overrun warning.
"""

from __future__ import annotations

import datetime as dt
import logging
import threading
from typing import Any

import pandas as pd
import pytz
from lories.scheduler import TickScheduler

from ..core.config import FieldSetup
from .runner import FieldRunner

logger = logging.getLogger(__name__)


def _as_timezone(tz: Any) -> pytz.BaseTzInfo:
    """Normalise ``None``, a zone name or any ``tzinfo`` into a pytz zone."""
    if tz is None:
        return pytz.UTC
    if isinstance(tz, pytz.BaseTzInfo):
        return tz
    if isinstance(tz, dt.tzinfo):
        name = getattr(tz, "zone", None) or getattr(tz, "key", None) or str(tz)
    else:
        name = str(tz)
    if name.upper() == "UTC":
        return pytz.UTC
    return pytz.timezone(name)


class Ticker:
    """Run ``FieldRunner.run_tick`` on the field's aligned cadence.

    lories aligns slots on epoch seconds, which lands on the same instant as
    the live ``slot_ceil`` (a floor in the site timezone) for every interval
    that divides 60 minutes; a longer, non-dividing interval would drift
    against the site's wall clock.
    """

    def __init__(
        self,
        setup: FieldSetup,
        runner: FieldRunner,
        *,
        tz: Any = pytz.UTC,
        name: str = "fieldsim",
    ) -> None:
        self.setup = setup
        self.runner = runner
        self.timezone = _as_timezone(tz)
        self.failed_ticks = 0
        self._name = name
        self._interval = pd.Timedelta(minutes=setup.field.interval)
        # Not lories' interrupt: run_tick reads this one, so a stop lands
        # inside the running tick, not only on the next slot.
        self._cancel = threading.Event()
        self._scheduler = TickScheduler(
            self._on_slot,
            self._interval,
            pd.Timedelta(minutes=setup.field.offset),
            name=name,
            timezone=self.timezone,
        )

    @property
    def scheduler(self) -> TickScheduler:
        """The wrapped lories scheduler."""
        return self._scheduler

    def start(self) -> None:
        self._cancel.clear()
        self._scheduler.start()

    def stop(self) -> None:
        self._cancel.set()
        self._scheduler.stop()

    def _on_slot(self) -> None:
        now = pd.Timestamp.now(tz=self.timezone)
        # Pre-tick tally: a failed tick commits no row, so the rows this one
        # commits carry the count of the failures that preceded them.
        self.runner.tick_failures = self.failed_ticks
        try:
            self.runner.run_tick(now, cancel=self._cancel.is_set)
        except Exception:
            self.failed_ticks += 1
            # Swallowed: lories' _run_slot would log a second traceback.
            logger.exception("%s: tick failed (%d consecutive).", self._name, self.failed_ticks)
        else:
            self.failed_ticks = 0
        self._log_overrun(now)

    def _log_overrun(self, start: pd.Timestamp) -> None:
        """WARN for a tick whose duration crossed one or more slot boundaries."""
        duration = pd.Timestamp.now(tz=self.timezone) - start
        skipped = int(duration // self._interval)
        if skipped < 1:
            return
        logger.warning("%s: tick overran its slot (duration=%s, slots_skipped=%d).", self._name, duration, skipped)
