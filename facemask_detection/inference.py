from __future__ import annotations

import json
import sys
import types
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from facemask_detection.config import FaceMaskConfig
except ModuleNotFoundError:  # pragma: no cover - convenience for direct script execution
    from config import FaceMaskConfig


ID_TO_CLASS = {0: "Mask", 1: "NoMask"}
FEATURE_MAP_SIZES = [(45, 45), (23, 23), (12, 12), (6, 6), (4, 4)]
ANCHOR_SIZES = [(0.04, 0.056), (0.08, 0.11), (0.16, 0.22), (0.32, 0.45), (0.64, 0.72)]
ANCHOR_RATIOS = (1.0, 0.62, 0.42)
CENTER_VARIANCE = 0.1
SIZE_VARIANCE = 0.2


@dataclass
class FaceMaskDetection:
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: Tuple[int, int, int, int]


@dataclass
class FaceMaskInferenceEngine:
    model: nn.Module
    device: str
    input_size: int
    confidence_threshold: float
    iou_threshold: float
    anchors: np.ndarray


def validate_device(device: str) -> str:
    normalized_device = device.strip()
    if not normalized_device:
        raise ValueError("Device must not be empty.")
    if normalized_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested for facemask detection but is not available."
        )
    return normalized_device


def _reshape_cls(prediction: torch.Tensor) -> torch.Tensor:
    batch_size, _, _, _ = prediction.shape
    return prediction.view(batch_size, 2, -1)


def _reshape_loc(prediction: torch.Tensor) -> torch.Tensor:
    batch_size, _, _, _ = prediction.shape
    return prediction.view(batch_size, 4, -1)


class LegacyKitModel(nn.Module):
    """Forward shim for the legacy AIZOO model serialized in facemask.pth."""

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        conv2d_0 = self.conv2d_0(F.pad(x, (1, 1, 1, 1)))
        conv2d_0_activation = F.relu(self.conv2d_0_bn(conv2d_0))
        maxpool2d_0 = F.max_pool2d(conv2d_0_activation, kernel_size=2, stride=2)

        conv2d_1 = self.conv2d_1(F.pad(maxpool2d_0, (1, 1, 1, 1)))
        conv2d_1_activation = F.relu(self.conv2d_1_bn(conv2d_1))
        maxpool2d_1 = F.max_pool2d(conv2d_1_activation, kernel_size=2, stride=2)

        conv2d_2 = self.conv2d_2(F.pad(maxpool2d_1, (1, 1, 1, 1)))
        conv2d_2_activation = F.relu(self.conv2d_2_bn(conv2d_2))
        maxpool2d_2 = F.max_pool2d(conv2d_2_activation, kernel_size=2, stride=2)

        conv2d_3 = self.conv2d_3(F.pad(maxpool2d_2, (1, 1, 1, 1)))
        conv2d_3_activation = F.relu(self.conv2d_3_bn(conv2d_3))
        maxpool2d_3 = F.max_pool2d(
            F.pad(conv2d_3_activation, (0, 1, 0, 1), value=float("-inf")),
            kernel_size=2,
            stride=2,
        )

        cls_0_insert = self.cls_0_insert_conv2d(F.pad(conv2d_3_activation, (1, 1, 1, 1)))
        loc_0_insert = self.loc_0_insert_conv2d(F.pad(conv2d_3_activation, (1, 1, 1, 1)))
        conv2d_4 = self.conv2d_4(F.pad(maxpool2d_3, (1, 1, 1, 1)))
        cls_0_activation = F.relu(self.cls_0_insert_conv2d_bn(cls_0_insert))
        loc_0_activation = F.relu(self.loc_0_insert_conv2d_bn(loc_0_insert))
        conv2d_4_activation = F.relu(self.conv2d_4_bn(conv2d_4))
        cls_0_conv = self.cls_0_conv(F.pad(cls_0_activation, (1, 1, 1, 1)))
        loc_0_conv = self.loc_0_conv(F.pad(loc_0_activation, (1, 1, 1, 1)))

        maxpool2d_4 = F.max_pool2d(
            F.pad(conv2d_4_activation, (0, 1, 0, 1), value=float("-inf")),
            kernel_size=2,
            stride=2,
        )
        cls_1_insert = self.cls_1_insert_conv2d(F.pad(conv2d_4_activation, (1, 1, 1, 1)))
        loc_1_insert = self.loc_1_insert_conv2d(F.pad(conv2d_4_activation, (1, 1, 1, 1)))
        conv2d_5 = self.conv2d_5(F.pad(maxpool2d_4, (1, 1, 1, 1)))
        cls_1_activation = F.relu(self.cls_1_insert_conv2d_bn(cls_1_insert))
        loc_1_activation = F.relu(self.loc_1_insert_conv2d_bn(loc_1_insert))
        conv2d_5_activation = F.relu(self.conv2d_5_bn(conv2d_5))
        cls_1_conv = self.cls_1_conv(F.pad(cls_1_activation, (1, 1, 1, 1)))
        loc_1_conv = self.loc_1_conv(F.pad(loc_1_activation, (1, 1, 1, 1)))

        maxpool2d_5 = F.max_pool2d(conv2d_5_activation, kernel_size=2, stride=2)
        cls_2_insert = self.cls_2_insert_conv2d(F.pad(conv2d_5_activation, (1, 1, 1, 1)))
        loc_2_insert = self.loc_2_insert_conv2d(F.pad(conv2d_5_activation, (1, 1, 1, 1)))
        conv2d_6 = self.conv2d_6(F.pad(maxpool2d_5, (1, 1, 1, 1)))
        cls_2_activation = F.relu(self.cls_2_insert_conv2d_bn(cls_2_insert))
        loc_2_activation = F.relu(self.loc_2_insert_conv2d_bn(loc_2_insert))
        conv2d_6_activation = F.relu(self.conv2d_6_bn(conv2d_6))
        cls_2_conv = self.cls_2_conv(F.pad(cls_2_activation, (1, 1, 1, 1)))
        loc_2_conv = self.loc_2_conv(F.pad(loc_2_activation, (1, 1, 1, 1)))

        conv2d_7 = self.conv2d_7(conv2d_6_activation)
        cls_3_insert = self.cls_3_insert_conv2d(F.pad(conv2d_6_activation, (1, 1, 1, 1)))
        loc_3_insert = self.loc_3_insert_conv2d(F.pad(conv2d_6_activation, (1, 1, 1, 1)))
        conv2d_7_activation = F.relu(self.conv2d_7_bn(conv2d_7))
        cls_3_activation = F.relu(self.cls_3_insert_conv2d_bn(cls_3_insert))
        loc_3_activation = F.relu(self.loc_3_insert_conv2d_bn(loc_3_insert))
        cls_3_conv = self.cls_3_conv(F.pad(cls_3_activation, (1, 1, 1, 1)))
        loc_3_conv = self.loc_3_conv(F.pad(loc_3_activation, (1, 1, 1, 1)))

        cls_4_insert = self.cls_4_insert_conv2d(F.pad(conv2d_7_activation, (1, 1, 1, 1)))
        loc_4_insert = self.loc_4_insert_conv2d(F.pad(conv2d_7_activation, (1, 1, 1, 1)))
        cls_4_activation = F.relu(self.cls_4_insert_conv2d_bn(cls_4_insert))
        loc_4_activation = F.relu(self.loc_4_insert_conv2d_bn(loc_4_insert))
        cls_4_conv = self.cls_4_conv(F.pad(cls_4_activation, (1, 1, 1, 1)))
        loc_4_conv = self.loc_4_conv(F.pad(loc_4_activation, (1, 1, 1, 1)))

        loc_branch_concat = torch.cat(
            (
                _reshape_loc(loc_0_conv),
                _reshape_loc(loc_1_conv),
                _reshape_loc(loc_2_conv),
                _reshape_loc(loc_3_conv),
                _reshape_loc(loc_4_conv),
            ),
            dim=2,
        )
        cls_branch_concat = torch.cat(
            (
                torch.sigmoid(_reshape_cls(cls_0_conv)),
                torch.sigmoid(_reshape_cls(cls_1_conv)),
                torch.sigmoid(_reshape_cls(cls_2_conv)),
                torch.sigmoid(_reshape_cls(cls_3_conv)),
                torch.sigmoid(_reshape_cls(cls_4_conv)),
            ),
            dim=2,
        )
        return loc_branch_concat.transpose(1, 2), cls_branch_concat.transpose(1, 2)


LegacyKitModel.__module__ = "MainModel"


def _register_legacy_module() -> None:
    legacy_module = types.ModuleType("MainModel")
    legacy_module.KitModel = LegacyKitModel
    legacy_module.MainModel = LegacyKitModel
    sys.modules["MainModel"] = legacy_module


def generate_anchors() -> np.ndarray:
    anchors: List[List[float]] = []
    for (feature_h, feature_w), (min_size, max_size) in zip(
        FEATURE_MAP_SIZES, ANCHOR_SIZES
    ):
        for y_index in range(feature_h):
            for x_index in range(feature_w):
                center_x = (x_index + 0.5) / feature_w
                center_y = (y_index + 0.5) / feature_h
                anchors.append([center_x, center_y, min_size, min_size])
                anchors.append([center_x, center_y, max_size, max_size])
                for ratio in ANCHOR_RATIOS[1:]:
                    sqrt_ratio = np.sqrt(ratio)
                    anchors.append(
                        [center_x, center_y, min_size * sqrt_ratio, min_size / sqrt_ratio]
                    )
    return np.clip(np.asarray(anchors, dtype=np.float32), 0.0, 1.0)


def decode_bboxes(anchors: np.ndarray, raw_bboxes: np.ndarray) -> np.ndarray:
    anchor_centers = anchors[:, :2]
    anchor_sizes = anchors[:, 2:]
    decoded_centers = raw_bboxes[:, :2] * CENTER_VARIANCE * anchor_sizes + anchor_centers
    decoded_sizes = np.exp(raw_bboxes[:, 2:] * SIZE_VARIANCE) * anchor_sizes
    decoded_xyxy = np.concatenate(
        (decoded_centers - decoded_sizes / 2.0, decoded_centers + decoded_sizes / 2.0),
        axis=1,
    )
    return np.clip(decoded_xyxy, 0.0, 1.0)


def compute_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    inter_x1 = np.maximum(box[0], boxes[:, 0])
    inter_y1 = np.maximum(box[1], boxes[:, 1])
    inter_x2 = np.minimum(box[2], boxes[:, 2])
    inter_y2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    box_area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    boxes_area = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
        0.0, boxes[:, 3] - boxes[:, 1]
    )
    union_area = box_area + boxes_area - inter_area
    union_area = np.maximum(union_area, 1e-6)
    return inter_area / union_area


def single_class_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    confidence_threshold: float,
    iou_threshold: float,
) -> List[int]:
    valid_indices = np.where(scores >= confidence_threshold)[0]
    if valid_indices.size == 0:
        return []

    sorted_indices = valid_indices[np.argsort(scores[valid_indices])[::-1]]
    keep_indices: List[int] = []

    while sorted_indices.size > 0:
        current_index = int(sorted_indices[0])
        keep_indices.append(current_index)
        if sorted_indices.size == 1:
            break

        remaining_indices = sorted_indices[1:]
        ious = compute_iou(boxes[current_index], boxes[remaining_indices])
        sorted_indices = remaining_indices[ious <= iou_threshold]

    return keep_indices


def load_facemask_inference_engine(config: FaceMaskConfig) -> FaceMaskInferenceEngine:
    model_path = config.model_path.strip()
    if not model_path:
        raise ValueError("Model path must not be empty.")
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Facemask model not found: {model_path}")

    device = validate_device(config.device)
    _register_legacy_module()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="You are using `torch.load` with `weights_only=False`.*",
            category=FutureWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="Couldn't retrieve source code for container of type .*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="source code of class '.*' has changed.*",
            category=Warning,
        )
        model = torch.load(model_path, map_location=device)
    if not isinstance(model, nn.Module):
        raise TypeError("Loaded facemask checkpoint is not a torch.nn.Module.")

    model = model.to(device)
    model.eval()
    return FaceMaskInferenceEngine(
        model=model,
        device=device,
        input_size=config.input_size,
        confidence_threshold=config.confidence_threshold,
        iou_threshold=config.iou_threshold,
        anchors=generate_anchors(),
    )


def run_facemask_inference(
    engine: FaceMaskInferenceEngine,
    image: np.ndarray,
) -> List[FaceMaskDetection]:
    resized_image = cv2.resize(image, (engine.input_size, engine.input_size))
    normalized_image = resized_image.astype(np.float32) / 255.0
    nchw_image = np.transpose(normalized_image[None, ...], (0, 3, 1, 2))

    with torch.no_grad():
        input_tensor = torch.from_numpy(nchw_image).to(engine.device)
        raw_bboxes_tensor, raw_classes_tensor = engine.model(input_tensor)

    raw_bboxes = raw_bboxes_tensor.detach().cpu().numpy()[0]
    raw_classes = raw_classes_tensor.detach().cpu().numpy()[0]
    if raw_bboxes.shape[0] != engine.anchors.shape[0]:
        raise ValueError(
            "Anchor count does not match model output. "
            f"Expected {engine.anchors.shape[0]} anchors, got {raw_bboxes.shape[0]}. "
            "Please verify the configured input_size and anchor settings."
        )
    decoded_bboxes = decode_bboxes(engine.anchors, raw_bboxes)
    max_scores = np.max(raw_classes, axis=1)
    max_classes = np.argmax(raw_classes, axis=1)
    keep_indices = single_class_nms(
        decoded_bboxes,
        max_scores,
        confidence_threshold=engine.confidence_threshold,
        iou_threshold=engine.iou_threshold,
    )

    image_height, image_width = image.shape[:2]
    detections: List[FaceMaskDetection] = []
    for index in keep_indices:
        bbox = decoded_bboxes[index]
        x1 = max(0, int(round(bbox[0] * image_width)))
        y1 = max(0, int(round(bbox[1] * image_height)))
        x2 = min(image_width, int(round(bbox[2] * image_width)))
        y2 = min(image_height, int(round(bbox[3] * image_height)))
        class_id = int(max_classes[index])
        detections.append(
            FaceMaskDetection(
                class_id=class_id,
                class_name=ID_TO_CLASS.get(class_id, f"class_{class_id}"),
                confidence=float(max_scores[index]),
                bbox_xyxy=(x1, y1, x2, y2),
            )
        )

    detections.sort(key=lambda item: item.confidence, reverse=True)
    return detections


def detections_to_json(detections: List[FaceMaskDetection]) -> str:
    payload: List[Dict[str, object]] = []
    for detection in detections:
        payload.append(
            {
                "class_id": detection.class_id,
                "class_name": detection.class_name,
                "confidence": round(detection.confidence, 6),
                "bbox_xyxy": list(detection.bbox_xyxy),
            }
        )
    return json.dumps(payload, indent=2)
