# -*- coding: utf-8 -*-
"""
sparcs.components.camera.inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""

from __future__ import annotations

import json
import logging
import queue
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from lories.components import Component, register_component_type
from lories.components.cameras._core import _Camera
from lories.components.cameras.camera import Camera
from lories.connectors.models import Model
from lories.core import Configurations, Constant, ResourceError

from sparcs.components.camera.roi_pipeline import (
    ROI_COLUMNS,
    TRANSFORM_MAX_SIZE,
    TRANSFORM_MIN_SIZE,
    WARP_QUADS,
    draw_on_frame,
    run_roi,
)

_logger = logging.getLogger(__name__)

_TABLE_NAME = "apples_predictions"


class AppleInference(Component):
    TYPE: str = "inference"

    ROI_COUNT: int = len(WARP_QUADS)
    ROIS = tuple(Constant(str, f"roi_{i + 1}", f"Plant ROI {i + 1} detections") for i in range(ROI_COUNT))
    TOTAL = Constant(int, "total_count", "Total apple detections")
    PREVIEW = Constant(bytes, "preview", "Live annotated preview frame")

    SYNC_TRIGGER = Constant(bool, "sync_trigger", "Prediction sync-check trigger")
    PREDICT_TRIGGER = Constant(bool, "predict_trigger", "Stored-frame prediction trigger")

    _model: Optional[Model] = None
    _pad_hw: Optional[Tuple[int, int]] = None
    _db: Optional[Any] = None
    _pool: Optional[ThreadPoolExecutor] = None

    _frame_queue: Optional[queue.Queue] = None
    _stop_event: Optional[threading.Event] = None
    _worker: Optional[threading.Thread] = None

    live_stream: bool = True
    predict_freq: str = "1min"
    sync_freq: str = "5min"
    workers: int = 8
    score_threshold: float = 0.35

    _synchronized: bool = False

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

        self.live_stream = configs.get_bool("live_stream", default=AppleInference.live_stream)
        self.predict_freq = configs.get("predict_freq", default=AppleInference.predict_freq)
        self.sync_freq = configs.get("sync_freq", default=AppleInference.sync_freq)
        self.workers = configs.get_int("workers", default=AppleInference.workers)
        self.score_threshold = configs.get_float("score_threshold", default=AppleInference.score_threshold)
        device = configs.get("device", default="auto")
        num_classes = configs.get_int("num_classes", default=2)
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
            self._logger.warning("Fruit detection weights not found: %s; inference disabled", weights)
            return
        except Exception as exc:
            self._logger.error("Failed to load fruit detection model: %s", exc, exc_info=True)
            return

        self._pad_hw = tuple(self._model.meta["input_hw"]) if self._model.meta.get("input_hw") else None

        logger_configs = configs.get_member("logger", defaults={"connector": "mariadb", "table": _TABLE_NAME})
        logger_connector = logger_configs.get("connector", default="mariadb")
        logger_table = logger_configs.get("table", default=_TABLE_NAME)

        for roi, column in zip(AppleInference.ROIS, ROI_COLUMNS):
            self.data.add(
                roi,
                aggregate="last",
                freq=None,
                length=1024,
                logger={"connector": logger_connector, "table": logger_table, "column": column},
            )
        self.data.add(
            AppleInference.TOTAL,
            aggregate="last",
            freq=None,
            logger={"connector": logger_connector, "table": logger_table, "column": "total_count"},
        )

        if self.live_stream:
            self.data.add(AppleInference.PREVIEW, aggregate="last", freq=None, stream=True)
        else:
            self.data.add(
                AppleInference.PREDICT_TRIGGER,
                aggregate="last",
                freq=self.predict_freq,
                connector="dummy",
                default=False,
            )
            self.data.add(
                AppleInference.SYNC_TRIGGER,
                aggregate="last",
                freq=self.sync_freq,
                connector="dummy",
                default=False,
            )

    def activate(self) -> None:
        super().activate()

        if self._model is None:
            return

        self._db = self.data.get(AppleInference.TOTAL).logger._connector
        if self._db is None:
            self._logger.warning("No 'mariadb' connector available; fruit predictions will not be persisted")

        self._pool = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix=f"{self.id}-roi")
        self._frame_queue = queue.Queue()
        self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._worker_loop, name=f"{self.id}-inference", daemon=True)
        self._worker.start()

        camera = self.context
        if self.live_stream:
            if _Camera.FRAME not in camera.data:
                raise ResourceError(
                    f"AppleInference '{self.id}' requires camera-level 'frame = true' "
                    f"to enable the {_Camera.FRAME!s} channel"
                )
            camera.data.register(self._on_frame, _Camera.FRAME, how="any", unique=False)
        else:
            self.data.register(self._on_sync_check, AppleInference.SYNC_TRIGGER, how="any", unique=False)
            self.data.register(self._on_predict_trigger, AppleInference.PREDICT_TRIGGER, how="any", unique=False)

    def deactivate(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        if self._pool is not None:
            self._pool.shutdown(wait=False)
        super().deactivate()

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                timestamp, jpeg_bytes = self._frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                self._run_inference(timestamp, jpeg_bytes)
            except Exception as exc:
                self._logger.error("Inference pipeline error at %s: %s", timestamp, exc, exc_info=True)

    def _on_frame(self, data: pd.DataFrame) -> None:
        camera = self.context
        frame_id = camera.data.frame.id
        if data is None or data.empty or frame_id not in data.columns:
            return
        jpeg_bytes = data[frame_id].iloc[-1]
        if jpeg_bytes is None:
            return
        timestamp = data.index[-1]
        self._frame_queue.put_nowait((timestamp, jpeg_bytes))

    def _on_sync_check(self, data: pd.DataFrame) -> None:
        threading.Thread(target=self._run_check_sync, name=f"{self.id}-sync", daemon=True).start()

    def _run_check_sync(self) -> None:
        try:
            self._check_sync()
        except Exception as exc:
            self._logger.error("Frame/prediction sync check error: %s", exc, exc_info=True)

    def _check_sync(self) -> None:
        camera = self.context
        latest_frame = self._latest_timestamp(camera.data.read_logged(channels=camera.data.frame.to_list()))
        latest_prediction = self._latest_timestamp(self.data.read_logged(channels=[AppleInference.TOTAL]))
        self._synchronized = (
            latest_frame is None or (latest_prediction is not None and latest_prediction >= latest_frame)
        )
        self._logger.debug("Frame/prediction sync check: synchronized=%s", self._synchronized)

    def _on_predict_trigger(self, data: pd.DataFrame) -> None:
        if self._synchronized:
            self._logger.debug("Frames and predictions synchronized; skipping prediction pass")
            return
        threading.Thread(target=self._run_predict_from_database, name=f"{self.id}-scan", daemon=True).start()

    def _run_predict_from_database(self) -> None:
        try:
            self._predict_from_database()
        except Exception as exc:
            self._logger.error("Backlog prediction scan error: %s", exc, exc_info=True)

    _BACKLOG_WINDOW: pd.Timedelta = pd.Timedelta(minutes=30)
    _BACKLOG_EPOCH: pd.Timestamp = pd.Timestamp("2020-01-01", tz="UTC")

    def _predict_from_database(self) -> None:
        if not self._frame_queue.empty():
            self._logger.debug("Previous prediction batch still processing; skipping this trigger")
            return

        camera = self.context
        frame_channels = camera.data.frame.to_list()
        start = self._latest_timestamp(self.data.read_logged(channels=[AppleInference.TOTAL]))
        now = pd.Timestamp.now(tz="UTC")

        lo = start + pd.Timedelta(milliseconds=1) if start is not None else AppleInference._BACKLOG_EPOCH
        window_start = self._find_earliest_frame(camera, frame_channels, lo, now)
        if window_start is None:
            self._synchronized = True
            return

        frames = camera.data.read_logged(
            channels=frame_channels,
            start=window_start,
            end=window_start + AppleInference._BACKLOG_WINDOW,
        )
        if start is not None:
            frames = frames[frames.index > start]

        if frames.empty:
            self._synchronized = True
            return

        for timestamp, jpeg_bytes in frames.iloc[:, 0].items():
            if jpeg_bytes is not None:
                self._frame_queue.put_nowait((timestamp, jpeg_bytes))
        self._synchronized = False

    @staticmethod
    def _find_earliest_frame(
        camera: _Camera,
        channels: List[str],
        lo: pd.Timestamp,
        hi: pd.Timestamp,
    ) -> Optional[pd.Timestamp]:
        if lo >= hi or not camera.data.has_logged(channels=channels, start=lo, end=hi):
            return None
        while hi - lo > AppleInference._BACKLOG_WINDOW:
            mid = lo + (hi - lo) / 2
            if camera.data.has_logged(channels=channels, start=lo, end=mid):
                hi = mid
            else:
                lo = mid
        return lo

    @staticmethod
    def _latest_timestamp(data: pd.DataFrame) -> Optional[pd.Timestamp]:
        if data.empty:
            return None
        return data.index.max()

    def _run_inference(self, timestamp: pd.Timestamp, jpeg_bytes: bytes) -> None:
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            self._logger.warning("Could not decode frame at %s", timestamp)
            return

        roi_boxes: List[List[List[float]]] = [[] for _ in range(AppleInference.ROI_COUNT)]
        futures = {
            self._pool.submit(run_roi, i, frame, self._model, self._pad_hw, self.score_threshold): i
            for i in range(AppleInference.ROI_COUNT)
        }
        for future in as_completed(futures):
            roi_idx = futures[future]
            try:
                idx, boxes = future.result()
                roi_boxes[idx] = boxes
            except Exception as exc:
                self._logger.error("ROI %d inference error: %s", roi_idx, exc, exc_info=True)

        total = sum(len(b) for b in roi_boxes)
        self._logger.info(
            "fruit inference ts=%s detections_per_roi=%s total=%d",
            timestamp,
            [len(b) for b in roi_boxes],
            total,
        )

        if self.live_stream:
            self._publish_preview(timestamp, frame, roi_boxes, total)

        self._write_predictions(timestamp, roi_boxes, total)

    def _publish_preview(
        self,
        timestamp: pd.Timestamp,
        frame: np.ndarray,
        roi_boxes: List[List[List[float]]],
        total: int,
    ) -> None:
        annotated = draw_on_frame(frame, roi_boxes, total)
        ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            self.data.get(AppleInference.PREVIEW).set(timestamp, buf.tobytes())

    def _write_predictions(
        self,
        timestamp: pd.Timestamp,
        roi_boxes: List[List[List[float]]],
        total: int,
    ) -> None:
        if self._db is None:
            return
        row: Dict[str, Any] = {
            self.data.get(roi).id: json.dumps(boxes) for roi, boxes in zip(AppleInference.ROIS, roi_boxes)
        }
        row[self.data.get(AppleInference.TOTAL).id] = total
        data = pd.DataFrame([row], index=[timestamp])
        self._db.write(data)


# noinspection SpellCheckingInspection
@register_component_type("camera", replace=True)
class CameraWithInference(Camera):

    inference: Optional[AppleInference] = None

    def configure(self, configs: Configurations) -> None:
        frame_configs = configs.get_member("data").get_member("channels").get_member(_Camera.FRAME, ensure_exists=True)
        inference_enabled = frame_configs.has_member(AppleInference.TYPE, includes=True)
        inference_configs = frame_configs.get_member(AppleInference.TYPE) if inference_enabled else None

        super().configure(configs)

        if inference_enabled:
            self.inference = AppleInference(
                self,
                name=f"{self.name} Inference",
                configs=inference_configs,
            )
            self.components.add(self.inference)
