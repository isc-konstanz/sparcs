# -*- coding: utf-8 -*-
"""
sparcs.components.camera.model_builder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from __future__ import annotations

from typing import Optional

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_fasterrcnn(
    num_classes: int = 2,
    nms_thresh: Optional[float] = None,
    transform_min_size: Optional[int] = None,
    transform_max_size: Optional[int] = None,
) -> torch.nn.Module:
    model = fasterrcnn_resnet50_fpn_v2(weights=None)
    in_feats = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feats, num_classes)

    if nms_thresh is not None:
        model.roi_heads.nms_thresh = nms_thresh
    if transform_min_size is not None:
        model.transform.min_size = (transform_min_size,)
    if transform_max_size is not None:
        model.transform.max_size = transform_max_size
    return model
