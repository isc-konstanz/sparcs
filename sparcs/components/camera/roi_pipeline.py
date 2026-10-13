# -*- coding: utf-8 -*-
"""
sparcs.components.camera.roi_pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""

from __future__ import annotations

from collections import Counter
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np
import torch
from lories.connectors.models import Model

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

CROP_SIZE: Tuple[int, int] = (600, 1800)
SHORT_SIDE_TARGET: int = 256
ALIGN_TO: int = 32

DISPLAY_SCALE: float = 0.35
BOX_COLOR_BGR: Tuple[int, int, int] = (40, 40, 210)
# Smallest/largest annotated box (w, h) per class in CROP_SIZE pixels, from fruits_detection_kob/data/*/annotations.
# Predictions outside this range are dropped as false detections.
MIN_BOX_WH = {"apples": (20.5, 31.0), "flowers": (29.6, 30.9)}
MAX_BOX_WH = {"apples": (96.3, 113.0), "flowers": (398.2, 376.3)}
CLASS_NAMES: List[str] = ["__background__", "apples", "flowers"]
CLASS_COLORS_BGR = {"apples": (40, 40, 210), "flowers": (230, 120, 240)}
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


def _roi_warps() -> List[Tuple[np.ndarray, np.ndarray]]:
    w, h = CROP_SIZE
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    warps = []
    for quad in WARP_QUADS:
        src = np.array(quad, dtype=np.float32)
        warps.append((cv2.getPerspectiveTransform(src, dst), cv2.getPerspectiveTransform(dst, src)))
    return warps


ROI_WARPS: List[Tuple[np.ndarray, np.ndarray]] = _roi_warps()
CONTENT_SIZE: Tuple[int, int] = compute_target_size(*CROP_SIZE)

_NATURAL_SIZES = [compute_target_size(*quad_target_size(quad)) for quad in WARP_QUADS]
TRANSFORM_MIN_SIZE: int = min(w for w, _ in _NATURAL_SIZES)
TRANSFORM_MAX_SIZE: int = max(h for _, h in _NATURAL_SIZES)


def content_size(pad_hw: Optional[Tuple[int, int]] = None) -> Tuple[int, int]:
    content_w, content_h = CONTENT_SIZE
    if pad_hw is None:
        return content_w, content_h
    pad_h, pad_w = pad_hw
    scale = min(1.0, pad_w / content_w, pad_h / content_h)
    return max(1, int(content_w * scale)), max(1, int(content_h * scale))


def preprocess_frame(
    frame: np.ndarray,
    roi_index: int,
    pad_hw: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    warped = cv2.warpPerspective(frame, ROI_WARPS[roi_index][0], CROP_SIZE, flags=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    content_w, content_h = content_size(pad_hw)
    rgb = cv2.resize(rgb, (content_w, content_h), interpolation=cv2.INTER_LINEAR)
    img_f = rgb.astype(np.float32) / 255.0

    if pad_hw is not None:
        pad_h, pad_w = pad_hw
        canvas = np.zeros((pad_h, pad_w, 3), dtype=np.float32)
        canvas[:content_h, :content_w] = img_f
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
    pad_hw: Optional[Tuple[int, int]],
    score_threshold: float,
    class_names: Optional[List[str]] = None,
) -> List[List[Any]]:
    content_w, content_h = content_size(pad_hw)
    boxes, scores, labels = _unpack_detections(raw)
    if len(boxes) == 0:
        return []
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)

    if pad_hw is not None:
        keep_content = (boxes[:, 0] < content_w) & (boxes[:, 1] < content_h)
        boxes, scores, labels = boxes[keep_content], scores[keep_content], labels[keep_content]
        if len(boxes) == 0:
            return []
        boxes[:, 2] = np.minimum(boxes[:, 2], content_w)
        boxes[:, 3] = np.minimum(boxes[:, 3], content_h)

    keep = scores >= score_threshold
    boxes, labels = boxes[keep], labels[keep]
    if len(boxes) == 0:
        return []

    sx = CROP_SIZE[0] / content_w
    sy = CROP_SIZE[1] / content_h
    boxes = boxes.copy()
    boxes[:, [0, 2]] *= sx
    boxes[:, [1, 3]] *= sy

    result: List[List[Any]] = []
    for (x1, y1, x2, y2), label in zip(boxes, labels):
        name = class_names[label] if class_names and label < len(class_names) else int(label)
        min_w, min_h = MIN_BOX_WH.get(str(name).lower(), (0, 0))
        max_w, max_h = MAX_BOX_WH.get(str(name).lower(), (np.inf, np.inf))
        if not (min_w <= x2 - x1 <= max_w and min_h <= y2 - y1 <= max_h):
            continue
        corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32).reshape(1, -1, 2)
        corners_orig = cv2.perspectiveTransform(corners, ROI_WARPS[roi_index][1]).reshape(-1, 2)
        xo, yo = corners_orig[:, 0], corners_orig[:, 1]
        result.append([
            round(float(xo.min()), 2),
            round(float(yo.min()), 2),
            round(float(xo.max()), 2),
            round(float(yo.max()), 2),
            name,
        ])
    return result


def draw_on_frame(
    frame: np.ndarray,
    roi_boxes: List[List[List[Any]]],
    total: int,
    target_size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    fh, fw = frame.shape[:2]
    out_w, out_h = target_size if target_size is not None else (int(fw * DISPLAY_SCALE), int(fh * DISPLAY_SCALE))
    sx, sy = out_w / fw, out_h / fh
    canvas = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

    for roi_index, boxes in enumerate(roi_boxes):
        pts = (np.array(WARP_QUADS[roi_index], dtype=np.float32) * [sx, sy]).astype(np.int32)
        cv2.polylines(canvas, [pts], isClosed=True, color=BOX_COLOR_BGR, thickness=2, lineType=cv2.LINE_AA)
        for box in boxes:
            x1 = int(max(0, round(box[0] * sx)))
            y1 = int(max(0, round(box[1] * sy)))
            x2 = int(min(out_w - 1, round(box[2] * sx)))
            y2 = int(min(out_h - 1, round(box[3] * sy)))
            if x2 <= x1 or y2 <= y1:
                continue
            color = CLASS_COLORS_BGR.get(str(box[4]).lower(), BOX_COLOR_BGR)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

    counts = Counter(box[4] for boxes in roi_boxes for box in boxes)
    banner = "  ".join(f"{name}: {n}" for name, n in sorted(counts.items())) or f"Total: {total}"
    (bw, bh), bbl = cv2.getTextSize(banner, FONT, 1.0, 2)
    cv2.rectangle(canvas, (0, 0), (bw + 20, bh + bbl + 20), (20, 20, 20), cv2.FILLED)
    cv2.putText(canvas, banner, (10, bh + 10), FONT, 1.0, (0, 220, 255), 2, cv2.LINE_AA)
    return canvas


def filter_season(roi_boxes: List[List[List[Any]]], min_count: int = 2) -> List[List[List[Any]]]:
    counts = Counter(box[4] for boxes in roi_boxes for box in boxes)
    ranked = counts.most_common(2)
    if not ranked or ranked[0][1] < min_count or (len(ranked) > 1 and ranked[0][1] == ranked[1][1]):
        return [[] for _ in roi_boxes]
    season = ranked[0][0]
    kept = [[box for box in boxes if box[4] == season] for boxes in roi_boxes]
    # A single apple/flower on a plant is treated as a false detection
    return [boxes if len(boxes) >= min_count else [] for boxes in kept]


def run_roi(
    roi_index: int,
    frame: np.ndarray,
    model: Model,
    pad_hw: Optional[Tuple[int, int]],
    score_threshold: float,
) -> Tuple[int, List[List[Any]]]:
    tensor = preprocess_frame(frame, roi_index, pad_hw)
    raw = model.predict(tensor)
    boxes = postprocess_detections(raw, roi_index, pad_hw, score_threshold, model.meta.get("class_names", CLASS_NAMES))
    return roi_index, boxes


if __name__ == "__main__":
    a, f = [0, 0, 1, 1, "apples"], [0, 0, 1, 1, "flowers"]
    assert filter_season([[a, a, f], [a]]) == [[a, a], []]  # single apple on plant 2 dropped
    assert filter_season([[f, f], [a]]) == [[f, f], []]
    assert filter_season([[a, a], [a, a, a]]) == [[a, a], [a, a, a]]
    assert filter_season([[a], [f]]) == [[], []]  # tie: ambiguous
    assert filter_season([[a], []]) == [[], []]  # single instance
    assert filter_season([[], []]) == [[], []]
    sx, sy = CONTENT_SIZE[0] / CROP_SIZE[0], CONTENT_SIZE[1] / CROP_SIZE[1]  # crop px -> model px
    sized = postprocess_detections(
        ([[0, 0, 500 * sx, 500 * sy], [0, 0, 50 * sx, 50 * sy], [0, 0, 50 * sx, 50 * sy],
          [0, 0, 10 * sx, 10 * sy], [0, 0, 10 * sx, 10 * sy]], [1] * 5, [1, 0, 1, 0, 1]),
        0, None, 0.5, ["apples", "flowers"])
    assert [b[4] for b in sized] == ["apples", "flowers"]  # oversized flower + tiny apple/flower dropped
    print("ok")
