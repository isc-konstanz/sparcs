# -*- coding: utf-8 -*-
"""
sparcs.components.camera.inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""

from __future__ import annotations

import json
import queue
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from lories.components import Component, register_component_type
from lories.components.cameras._core import _Camera
from lories.components.cameras.camera import Camera
from lories.connectors.models import Model
from lories.core import Configurations, Constant, ResourceError

from sparcs.components.camera.roi_pipeline import (
    CLASS_NAMES,
    ROI_COLUMNS,
    TRANSFORM_MAX_SIZE,
    TRANSFORM_MIN_SIZE,
    WARP_QUADS,
    draw_on_frame,
    filter_season,
    run_roi,
)

_TABLE_NAME = "predictions"


class OrchardInference(Component):
    TYPE: str = "inference"

    ROI_COUNT: int = len(WARP_QUADS)
    ROIS = tuple(Constant(str, f"roi_{i + 1}", f"Plant ROI {i + 1} detections") for i in range(ROI_COUNT))
    TOTAL = Constant(int, "total_count", "Total detections")
    PREVIEW = Constant(bytes, "preview", "Live annotated preview frame")

    _model: Optional[Model] = None
    _pad_hw: Optional[Tuple[int, int]] = None
    _db: Optional[Any] = None
    _pool: Optional[ThreadPoolExecutor] = None

    _frame_queue: Optional[queue.Queue] = None
    _stop_event: Optional[threading.Event] = None
    _worker: Optional[threading.Thread] = None
    _sync_total: Optional[int] = None
    _sync_done: int = 0

    workers: int = 8
    score_threshold: float = 0.35
    min_count: int = 2

    sync_predictions: bool = False
    sync_schedule: str = "1D"
    sync_at: Optional[str] = None

    log_live_predictions: bool = True

    @classmethod
    def _assert_context(cls, context: _Camera) -> _Camera:
        if context is None or not isinstance(context, _Camera):
            raise ResourceError(f"Invalid '{cls.__name__}' context: {type(context)}")
        return super()._assert_context(context)

    def configure(self, configs: Configurations) -> None:
        super().configure(configs)

        weights = configs.get("weights", default=None)
        if not weights:
            self._logger.warning("inference.weights not set in camera.conf; inference disabled")
            return

        self.workers = configs.get_int("workers", default=OrchardInference.workers)
        self.score_threshold = configs.get_float("score_threshold", default=OrchardInference.score_threshold)
        self.min_count = configs.get_int("min_count", default=OrchardInference.min_count)
        self.sync_predictions = configs.get_bool("sync_predictions", default=OrchardInference.sync_predictions)
        self.sync_schedule = configs.get("sync_schedule", default=OrchardInference.sync_schedule)
        self.sync_at = configs.get("sync_at", default=OrchardInference.sync_at)
        self.log_live_predictions = configs.get_bool(
            "log_live_predictions", default=OrchardInference.log_live_predictions
        )
        device = configs.get("device", default="auto")
        num_classes = configs.get_int("num_classes", default=len(CLASS_NAMES))
        nms_iou_threshold = configs.get_float("nms_iou_threshold", default=0.5)
        model_builder = configs.get("model_builder", default="sparcs.components.camera.model_builder.build_fasterrcnn")

        try:
            self._model = Model(
                weights,
                device=device,
                model_builder=model_builder,
                build_args={
                    "num_classes": num_classes,
                    "nms_thresh": nms_iou_threshold,
                    "transform_min_size": TRANSFORM_MIN_SIZE,
                    "transform_max_size": TRANSFORM_MAX_SIZE,
                },
            )
        except FileNotFoundError:
            self._logger.warning("Apple/flower detection weights not found: %s; inference disabled", weights)
            return
        except Exception as exc:
            self._logger.error("Failed to load apple/flower detection model: %s", exc, exc_info=True)
            return

        self._pad_hw = tuple(self._model.meta["input_hw"]) if self._model.meta.get("input_hw") else None

        logger_configs = configs.get_member("logger", defaults={"connector": "mariadb", "table": _TABLE_NAME})
        logger_connector = logger_configs.get("connector", default="mariadb")
        logger_table = logger_configs.get("table", default=_TABLE_NAME)

        for roi, column in zip(OrchardInference.ROIS, ROI_COLUMNS):
            self.data.add(
                roi,
                aggregate="last",
                freq=None,
                length=65535,
                logger={"connector": logger_connector, "table": logger_table, "column": column},
            )
        self.data.add(
            OrchardInference.TOTAL,
            aggregate="last",
            freq=None,
            logger={"connector": logger_connector, "table": logger_table, "column": "total_count"},
        )

        self.data.add(OrchardInference.PREVIEW, aggregate="last", freq=None, stream=True)

    def activate(self) -> None:
        super().activate()

        if self._model is None:
            return

        self._db = self.data.get(OrchardInference.TOTAL).logger._connector
        if self._db is None:
            self._logger.warning("No 'mariadb' connector available; apple/flower predictions will not be persisted")

        self._pool = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix=f"{self.id}-roi")
        self._frame_queue = queue.Queue()
        self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._worker_loop, name=f"{self.id}-inference", daemon=True)
        self._worker.start()

        camera = self.context
        if _Camera.FRAME not in camera.data:
            raise ResourceError(
                f"OrchardInference '{self.id}' requires camera-level 'frame = true' "
                f"to enable the {_Camera.FRAME!s} channel"
            )
        camera.data.register(self._on_frame, _Camera.FRAME, how="any", unique=False)

        if self.sync_predictions and self._db is not None:
            threading.Thread(target=self._sync_loop, name=f"{self.id}-sync", daemon=True).start()

    def deactivate(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        if self._pool is not None:
            self._pool.shutdown(wait=False)
        super().deactivate()

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                timestamp, jpeg_bytes, publish_preview, persist = self._frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                self._run_inference(timestamp, jpeg_bytes, publish_preview, persist)
            except Exception as exc:
                self._logger.error("Inference pipeline error at %s: %s", timestamp, exc, exc_info=True)
            finally:
                self._frame_queue.task_done()
            if not publish_preview:
                self._sync_done += 1
                self._logger.info(self._sync_progress())

    def _on_frame(self, data: pd.DataFrame) -> None:
        camera = self.context
        frame_key = camera.data.frame.key
        if data is None or data.empty or frame_key not in data.columns:
            return
        jpeg_bytes = data[frame_key].iloc[-1]
        if jpeg_bytes is None:
            return
        timestamp = data.index[-1]
        self._frame_queue.put_nowait((timestamp, jpeg_bytes, True, self.log_live_predictions))

    def _sync_loop(self) -> None:
        # Own timer instead of a connector-read channel, so a failing read elsewhere cannot stall the sync
        interval = 30 if self.sync_at else pd.Timedelta(self.sync_schedule).total_seconds()
        while not self._stop_event.is_set():
            if not self.sync_at or pd.Timestamp.now(tz="UTC").strftime("%H:%M") == self.sync_at:
                try:
                    self._sync_predictions()
                except Exception as exc:
                    self._logger.error("Scheduled prediction backfill error: %s", exc, exc_info=True)
            self._stop_event.wait(interval)

    def _sync_progress(self, width: int = 30) -> str:
        done = self._sync_done
        total = max(self._sync_total, done)
        filled = width * done // total if total else width
        return (
            f"Syncing predictions [{'#' * filled}{'-' * (width - filled)}] "
            f"{done}/{total} frames, {total - done} left"
        )

    _BACKLOG_WINDOW: pd.Timedelta = pd.Timedelta(minutes=30)
    _BACKLOG_EPOCH: pd.Timestamp = pd.Timestamp("2020-01-01", tz="UTC")

    def _count(self, channel, start: pd.Timestamp, end: pd.Timestamp) -> int:
        return channel.logger._connector.count(channel.from_logger().to_list(), start=start, end=end)

    def _unsynced_windows(self, lo: pd.Timestamp, hi: pd.Timestamp) -> Iterator[Tuple[pd.Timestamp, pd.Timestamp]]:
        frame = self.context.data.frame
        total = self.data.get(OrchardInference.TOTAL)
        ranges = [(lo, hi)]
        while ranges and not self._stop_event.is_set():
            lo, hi = ranges.pop()
            if self._count(frame, lo, hi) <= self._count(total, lo, hi):
                continue
            if hi - lo <= OrchardInference._BACKLOG_WINDOW:
                yield lo, hi
                continue
            mid = (lo + (hi - lo) / 2).floor("s")
            ranges.append((mid, hi))
            ranges.append((lo, mid - pd.Timedelta(microseconds=1)))

    def _sync_predictions(self) -> None:
        if not self._frame_queue.empty():
            self._logger.debug("Previous prediction batch still processing; deferring sync pass")
            return

        camera = self.context
        frame_channels = camera.data.frame.to_list()
        lo = OrchardInference._BACKLOG_EPOCH
        now = pd.Timestamp.now(tz="UTC")

        self._sync_done = 0
        self._sync_total = max(
            0, self._count(camera.data.frame, lo, now) - self._count(self.data.get(OrchardInference.TOTAL), lo, now)
        )
        if self._sync_total == 0:
            return
        self._logger.info("Prediction sync started: ~%d frames without predictions", self._sync_total)

        for start, end in self._unsynced_windows(lo, now):
            frames = camera.data.read_logged(channels=frame_channels, start=start, end=end)
            predicted = self.data.read_logged(channels=[OrchardInference.TOTAL], start=start, end=end).index
            frames = frames[~frames.index.isin(predicted)]
            for timestamp, jpeg_bytes in frames.iloc[:, 0].items():
                if jpeg_bytes is not None:
                    self._frame_queue.put_nowait((timestamp, jpeg_bytes, False, True))
            self._frame_queue.join()

        if self._sync_done > 0:
            self._logger.info("Prediction sync complete: %d frames", self._sync_done)

    def _run_inference(
            self,
            timestamp: pd.Timestamp,
            jpeg_bytes: bytes,
            publish_preview: bool,
            persist: bool,
    ) -> None:
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            self._logger.warning("Could not decode frame at %s", timestamp)
            return

        roi_boxes: List[List[List[float]]] = [[] for _ in range(OrchardInference.ROI_COUNT)]
        futures = {
            self._pool.submit(run_roi, i, frame, self._model, self._pad_hw, self.score_threshold): i
            for i in range(OrchardInference.ROI_COUNT)
        }
        for future in as_completed(futures):
            roi_idx = futures[future]
            try:
                idx, boxes = future.result()
                roi_boxes[idx] = boxes
            except Exception as exc:
                self._logger.error("ROI %d inference error: %s", roi_idx, exc, exc_info=True)

        roi_boxes = filter_season(roi_boxes, self.min_count)
        total = sum(len(b) for b in roi_boxes)
        season = next((b[0][4] for b in roi_boxes if b), "none")
        self._logger.debug(
            "Orchard inference ts=%s class=%s apples=%d flowers=%d per_roi=%s",
            timestamp,
            season,
            total if season == "apples" else 0,
            total if season == "flowers" else 0,
            [len(b) for b in roi_boxes],
        )

        if publish_preview:
            self._publish_preview(timestamp, frame, roi_boxes, total)
        if persist:
            self._write_predictions(timestamp, roi_boxes, total)

    def _publish_preview(
            self,
            timestamp: pd.Timestamp,
            frame: np.ndarray,
            roi_boxes: List[List[List[float]]],
            total: int,
    ) -> None:
        annotated = draw_on_frame(frame, roi_boxes, total, target_size=self._stream_target_size())
        ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            self.data.get(OrchardInference.PREVIEW).set(timestamp, buf.tobytes())

    def _stream_target_size(self) -> Optional[Tuple[int, int]]:
        camera = self.context
        if _Camera.STREAM not in camera.data:
            return None
        stream = camera.data.stream
        if not stream.is_valid():
            return None
        arr = np.frombuffer(stream.value, dtype=np.uint8)
        sub_frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if sub_frame is None:
            return None
        h, w = sub_frame.shape[:2]
        return w, h

    def _write_predictions(
            self,
            timestamp: pd.Timestamp,
            roi_boxes: List[List[List[float]]],
            total: int,
    ) -> None:
        if self._db is None:
            return
        row: Dict[str, Any] = {
            self.data.get(roi).id: json.dumps(boxes) for roi, boxes in zip(OrchardInference.ROIS, roi_boxes)
        }
        row[self.data.get(OrchardInference.TOTAL).id] = total
        data = pd.DataFrame([row], index=[timestamp])
        self._db.write(data)


# noinspection SpellCheckingInspection
@register_component_type("camera", replace=True)
class OrchardCamera(Camera):
    inference: Optional[OrchardInference] = None

    def configure(self, configs: Configurations) -> None:
        frame_configs = configs.get_member("data").get_member("channels").get_member(_Camera.FRAME, ensure_exists=True)
        inference_enabled = frame_configs.has_member(OrchardInference.TYPE, includes=True)
        inference_configs = frame_configs.get_member(OrchardInference.TYPE) if inference_enabled else None

        super().configure(configs)

        if inference_enabled:
            self.inference = OrchardInference(
                self,
                name=f"{self.name} Inference",
                configs=inference_configs,
            )
            self.components.add(self.inference)
