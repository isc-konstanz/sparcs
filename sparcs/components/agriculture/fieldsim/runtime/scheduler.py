# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.fieldsim.runtime.scheduler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The tick thread, delegated to lories. ``Ticker`` wraps
``lories.scheduler.TickScheduler``, which counts consecutive failures, logs
a failed or overrunning slot and raises the interrupt ``stop`` sets; the
ticker hands the runner a UTC clock, that failure count and the interrupt
as its cancel signal.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytz
from lories.scheduler import TickScheduler
from lories.util import to_timezone

from ..core.config import FieldSetup
from .runner import FieldRunner


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
        tz: Any = None,
        name: str = "fieldsim",
    ) -> None:
        self.setup = setup
        self.runner = runner
        self._scheduler = TickScheduler(
            self._on_slot,
            pd.Timedelta(minutes=setup.field.interval),
            pd.Timedelta(minutes=setup.field.offset),
            name=name,
            timezone=to_timezone(tz) or pytz.UTC,
        )

    @property
    def scheduler(self) -> TickScheduler:
        """The wrapped lories scheduler."""
        return self._scheduler

    def start(self) -> None:
        self._scheduler.start()

    def stop(self) -> None:
        self._scheduler.stop()

    def _on_slot(self) -> None:
        # The rows this tick commits carry the failures that preceded it.
        self.runner.tick_failures = float(self._scheduler.failures)
        self.runner.run_tick(pd.Timestamp.now(tz="UTC"), cancel=self._scheduler.is_interrupted)
