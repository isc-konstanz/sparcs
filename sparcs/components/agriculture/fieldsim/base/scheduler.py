# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.fieldsim.base.scheduler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The tick thread and nothing else: slot alignment, intake delay, stall and
failure counters, watchdog, clean stop. Calls ``FieldRunner.run_tick`` and
reads its bool. Today this is interleaved with the tick logic in
``FieldSimulation._tick_loop`` / ``_tick`` / ``_on_tick``.
"""

from __future__ import annotations

import datetime as dt
import logging
import threading

from ..core.config import FieldConfig
from .runner import FieldRunner

logger = logging.getLogger(__name__)

STALL_ERROR_TICKS = 3
FAILURE_ESCALATE_AT = 2


class TickScheduler:
    def __init__(self, config: FieldConfig, runner: FieldRunner) -> None:
        self.config = config
        self.runner = runner
        self._interrupt = threading.Event()
        self._thread: threading.Thread | None = None
        self.stalled_ticks = 0
        self.failed_ticks = 0

    def start(self) -> None:
        self._interrupt.clear()
        self._thread = threading.Thread(target=self._loop, name="fieldsim-tick", daemon=True)
        self._thread.start()

    def stop(self, timeout_s: float = 30.0) -> None:
        self._interrupt.set()
        if self._thread is not None:
            self._thread.join(timeout_s)

    def _loop(self) -> None:
        while not self._interrupt.is_set():
            self._interrupt.wait(self._seconds_to_next_slot())
            if self._interrupt.is_set():
                break
            self._tick(dt.datetime.now(dt.timezone.utc))

    def _tick(self, now: dt.datetime) -> None:
        try:
            processed = self.runner.run_tick(now)
        except Exception:
            self.failed_ticks += 1
            log = logger.error if self.failed_ticks >= FAILURE_ESCALATE_AT else logger.warning
            log("tick failed (%d consecutive)", self.failed_ticks, exc_info=True)
            return
        self.failed_ticks = 0
        if processed:
            self.stalled_ticks = 0
            return
        self.stalled_ticks += 1
        log = logger.error if self.stalled_ticks == STALL_ERROR_TICKS else logger.warning
        log("simulation stalled for %d ticks", self.stalled_ticks)

    def _seconds_to_next_slot(self) -> float:
        """Wall-clock alignment to ``interval`` + ``offset`` (today ``_schedule``)."""
        raise NotImplementedError
