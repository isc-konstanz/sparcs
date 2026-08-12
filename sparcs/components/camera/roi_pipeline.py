# -*- coding: utf-8 -*-
"""
sparcs.components.camera.roi_pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np
import torch
from lories.connectors.models import Model
from lories.core import ResourceError

# Perspective warp quads for each of the 8 plant ROIs (source pixel coordinates).
WARP_QUADS: List[List[Tuple[int, int]]] = [
    [(2087, 357),  (2458, 255),  (2607, 1739), (2313, 1671)],
    [(2439, 249),  (2792, 194),  (2933, 1800), (2605, 1755)],
    [(2775, 194),  (3291, 138),  (3379, 1906), (2931, 1835)],
    [(3269, 140),  (3850, 128),  (3838, 1909), (3345, 1908)],
    [(4128, 143),  (4810, 248),  (4673, 1950), (4046, 1947)],
    [(4788, 246),  (5236, 379),  (5018, 1923), (4639, 1940)],
    [(5214, 376),  (5586, 488),  (5270, 1871), (4994, 1920)],
    [(5560, 481),  (5832, 569),  (5576, 1808), (5248, 1864)],
]

ROI_COLUMNS: List[str] = [f"plant_roi_{i + 1}" for i in range(len(WARP_QUADS))]

SHORT_SIDE_TARGET: int = 256
ALIGN_TO: int = 32

DISPLAY_SCALE: float = 0.35
BOX_COLOR_BGR: Tuple[int, int, int] = (40, 40, 210)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def quad_target_size(quad: List[Tuple[int, int]], min_size: int = 16) -> Tuple[int, int]:
    tl, tr, br, bl = (np.array(p, dtype=np.float32) for p in quad)
    w = int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
    h = int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
    return max(min_size, w), max(min_size, h)


def compute_target_size(
    orig_w: int,
    orig_h: int,
    short_side: int = SHORT_SIDE_TARGET,
    align: int = ALIGN_TO,
) -> Tuple[int, int]:
    scale = short_side / min(orig_w, orig_h)
    new_w = max(int(round(orig_w * scale / align) * align), align)
    new_h = max(int(round(orig_h * scale / align) * align), align)
    return new_w, new_h


def enhance_image(bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    blur = cv2.GaussianBlur(bgr, (0, 0), sigmaX=3)
    bgr = cv2.addWeighted(bgr, 1.5, blur, -0.5, 0)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.25, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


@dataclass(frozen=True)
class _RoiGeometry:
    dst_w: int
    dst_h: int
    content_w: int
    content_h: int
    M: np.ndarray
    M_inv: np.ndarray


def _build_roi_geometry() -> List[_RoiGeometry]:
    geometry = []
    for quad in WARP_QUADS:
        dst_w, dst_h = quad_target_size(quad)
        content_w, content_h = compute_target_size(dst_w, dst_h)
        src = np.array(quad, dtype=np.float32)
        dst = np.array([[0, 0], [dst_w - 1, 0], [dst_w - 1, dst_h - 1], [0, dst_h - 1]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(src, dst)
        M_inv = cv2.getPerspectiveTransform(dst, src)
        geometry.append(_RoiGeometry(dst_w, dst_h, content_w, content_h, M, M_inv))
    return geometry


ROI_GEOMETRY: List[_RoiGeometry] = _build_roi_geometry()

TRANSFORM_MIN_SIZE: int = min(g.content_w for g in ROI_GEOMETRY)
TRANSFORM_MAX_SIZE: int = max(g.content_h for g in ROI_GEOMETRY)


def preprocess_frame(
    frame: np.ndarray,
    roi_index: int,
    pad_hw: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    geom = ROI_GEOMETRY[roi_index]
    warped = cv2.warpPerspective(frame, geom.M, (geom.dst_w, geom.dst_h), flags=cv2.INTER_LINEAR)
    enhanced = enhance_image(warped)
    rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (geom.content_w, geom.content_h), interpolation=cv2.INTER_LINEAR)
    img_f = rgb.astype(np.float32) / 255.0

    if pad_hw is not None:
        pad_h, pad_w = pad_hw
        if geom.content_h > pad_h or geom.content_w > pad_w:
            raise ResourceError(
                f"ROI content {geom.content_w}x{geom.content_h} exceeds the standardized canvas "
                f"{pad_w}x{pad_h} this model was exported for. Re-export after changing "
                f"WARP_QUADS/ROI geometry."
            )
        canvas = np.zeros((pad_h, pad_w, 3), dtype=np.float32)
        canvas[: geom.content_h, : geom.content_w] = img_f
        img_f = canvas

    return torch.from_numpy(img_f.transpose(2, 0, 1))


def _unpack_detections(raw: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(raw, (list, tuple)) and len(raw) > 0 and isinstance(raw[0], dict):
        pred = raw[0]
        return pred["boxes"], pred["scores"], pred["labels"]
    boxes, scores, labels = raw
    return boxes, scores, labels


def postprocess_detections(
    raw: Any,
    roi_index: int,
    padded: bool,
    score_threshold: float,
) -> List[List[float]]:
    geom = ROI_GEOMETRY[roi_index]
    boxes, scores, _labels = _unpack_detections(raw)
    if len(boxes) == 0:
        return []
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)

    if padded:
        keep_content = (boxes[:, 0] < geom.content_w) & (boxes[:, 1] < geom.content_h)
        boxes, scores = boxes[keep_content], scores[keep_content]
        if len(boxes) == 0:
            return []
        boxes[:, 2] = np.minimum(boxes[:, 2], geom.content_w)
        boxes[:, 3] = np.minimum(boxes[:, 3], geom.content_h)

    keep = scores >= score_threshold
    boxes = boxes[keep]
    if len(boxes) == 0:
        return []

    sx = geom.dst_w / geom.content_w
    sy = geom.dst_h / geom.content_h
    boxes = boxes.copy()
    boxes[:, [0, 2]] *= sx
    boxes[:, [1, 3]] *= sy

    result: List[List[float]] = []
    for x1, y1, x2, y2 in boxes:
        corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32).reshape(1, -1, 2)
        corners_orig = cv2.perspectiveTransform(corners, geom.M_inv).reshape(-1, 2)
        xo, yo = corners_orig[:, 0], corners_orig[:, 1]
        result.append([
            round(float(xo.min()), 2),
            round(float(yo.min()), 2),
            round(float(xo.max()), 2),
            round(float(yo.max()), 2),
        ])
    return result


def draw_on_frame(frame: np.ndarray, roi_boxes: List[List[List[float]]], total: int) -> np.ndarray:
    fh, fw = frame.shape[:2]
    out_w, out_h = int(fw * DISPLAY_SCALE), int(fh * DISPLAY_SCALE)
    canvas = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

    for roi_index, boxes in enumerate(roi_boxes):
        pts = (np.array(WARP_QUADS[roi_index], dtype=np.float32) * DISPLAY_SCALE).astype(np.int32)
        cv2.polylines(canvas, [pts], isClosed=True, color=BOX_COLOR_BGR, thickness=2, lineType=cv2.LINE_AA)
        for box in boxes:
            x1 = int(max(0, round(box[0] * DISPLAY_SCALE)))
            y1 = int(max(0, round(box[1] * DISPLAY_SCALE)))
            x2 = int(min(out_w - 1, round(box[2] * DISPLAY_SCALE)))
            y2 = int(min(out_h - 1, round(box[3] * DISPLAY_SCALE)))
            if x2 <= x1 or y2 <= y1:
                continue
            cv2.rectangle(canvas, (x1, y1), (x2, y2), BOX_COLOR_BGR, 2)

    banner = f"Total Fruits: {total}"
    (bw, bh), bbl = cv2.getTextSize(banner, FONT, 1.0, 2)
    cv2.rectangle(canvas, (0, 0), (bw + 20, bh + bbl + 20), (20, 20, 20), cv2.FILLED)
    cv2.putText(canvas, banner, (10, bh + 10), FONT, 1.0, (0, 220, 255), 2, cv2.LINE_AA)
    return canvas


def run_roi(
    roi_index: int,
    frame: np.ndarray,
    model: Model,
    pad_hw: Optional[Tuple[int, int]],
    score_threshold: float,
) -> Tuple[int, List[List[float]]]:
    tensor = preprocess_frame(frame, roi_index, pad_hw)
    raw = model.predict(tensor)
    boxes = postprocess_detections(raw, roi_index, pad_hw is not None, score_threshold)
    return roi_index, boxes
