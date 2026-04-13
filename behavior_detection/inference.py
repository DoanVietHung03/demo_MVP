from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import supervision as sv
import torch

try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover - import error is environment-specific
    raise ImportError(
        "Missing dependency 'ultralytics'. Install it with `pip install ultralytics`."
    ) from exc


@dataclass
class YOLODetectionInference:
    model: YOLO
    device: str
    confidence_threshold: float
    iou_threshold: float
    input_resize_scale: float
    class_name_to_id: Dict[str, int]


@dataclass
class PosePerson:
    bbox_xyxy: Tuple[int, int, int, int]
    confidence: float
    keypoints_xy: np.ndarray
    keypoints_conf: np.ndarray


@dataclass
class YOLOPoseInference:
    model: YOLO
    device: str
    confidence_threshold: float
    iou_threshold: float
    input_resize_scale: float


def validate_device(device: str) -> str:
    normalized_device = device.strip()
    if not normalized_device:
        raise ValueError("Device must not be empty.")
    if normalized_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is not available. "
            "Please check your NVIDIA driver, CUDA runtime, and PyTorch installation."
        )
    return normalized_device


def resolve_class_name_to_id(model: YOLO) -> Dict[str, int]:
    names = getattr(model.model, "names", None)
    mapping: Dict[str, int] = {}
    if isinstance(names, dict):
        for class_id, class_name in names.items():
            mapping[str(class_name).lower()] = int(class_id)
    elif isinstance(names, list):
        for class_id, class_name in enumerate(names):
            mapping[str(class_name).lower()] = class_id
    return mapping


def load_yolo_detection_inference(
    model_path: str,
    device: str,
    confidence_threshold: float,
    iou_threshold: float,
    input_resize_scale: float,
) -> YOLODetectionInference:
    normalized_model_path = model_path.strip()
    if not normalized_model_path:
        raise ValueError("Model path must not be empty.")
    if input_resize_scale <= 0:
        raise ValueError("Input resize scale must be greater than 0.")

    normalized_device = validate_device(device)
    model = YOLO(normalized_model_path).to(normalized_device)
    return YOLODetectionInference(
        model=model,
        device=normalized_device,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        input_resize_scale=input_resize_scale,
        class_name_to_id=resolve_class_name_to_id(model),
    )


def load_yolo_pose_inference(
    model_path: str,
    device: str,
    confidence_threshold: float,
    iou_threshold: float,
    input_resize_scale: float,
) -> YOLOPoseInference:
    normalized_model_path = model_path.strip()
    if not normalized_model_path:
        raise ValueError("Model path must not be empty.")
    if input_resize_scale <= 0:
        raise ValueError("Input resize scale must be greater than 0.")

    normalized_device = validate_device(device)
    model = YOLO(normalized_model_path).to(normalized_device)
    return YOLOPoseInference(
        model=model,
        device=normalized_device,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        input_resize_scale=input_resize_scale,
    )


def _resize_frame_for_inference(
    frame: np.ndarray,
    resize_scale: float,
) -> Tuple[np.ndarray, float]:
    if resize_scale == 1.0:
        return frame, 1.0

    resized_width = max(1, int(frame.shape[1] * resize_scale))
    resized_height = max(1, int(frame.shape[0] * resize_scale))
    resized_frame = cv2.resize(frame, (resized_width, resized_height))
    return resized_frame, resize_scale


def infer_filtered_detections(
    inference_engine: YOLODetectionInference,
    frame: np.ndarray,
    target_class_names: Sequence[str],
) -> sv.Detections:
    inference_frame, resize_scale = _resize_frame_for_inference(
        frame,
        float(inference_engine.input_resize_scale),
    )
    result = inference_engine.model.predict(
        source=inference_frame,
        device=inference_engine.device,
        conf=inference_engine.confidence_threshold,
        iou=inference_engine.iou_threshold,
        verbose=False,
    )[0]
    detections = sv.Detections.from_ultralytics(result)
    if len(detections) == 0 or detections.class_id is None:
        return detections

    if resize_scale != 1.0:
        detections.xyxy = detections.xyxy / resize_scale

    allowed_ids: List[int] = []
    for class_name in target_class_names:
        class_id = inference_engine.class_name_to_id.get(class_name.lower())
        if class_id is not None:
            allowed_ids.append(class_id)
    if not allowed_ids:
        return detections[np.array([], dtype=int)]

    class_mask = np.isin(detections.class_id, np.asarray(allowed_ids))
    return detections[class_mask]


def infer_pose_people(
    inference_engine: YOLOPoseInference,
    frame: np.ndarray,
) -> List[PosePerson]:
    inference_frame, resize_scale = _resize_frame_for_inference(
        frame,
        float(inference_engine.input_resize_scale),
    )
    result = inference_engine.model.predict(
        source=inference_frame,
        device=inference_engine.device,
        conf=inference_engine.confidence_threshold,
        iou=inference_engine.iou_threshold,
        verbose=False,
    )[0]

    boxes = getattr(result, "boxes", None)
    keypoints = getattr(result, "keypoints", None)
    if boxes is None or keypoints is None or boxes.xyxy is None:
        return []

    xyxy_tensor = boxes.xyxy.detach().cpu().numpy()
    cls_tensor = boxes.cls.detach().cpu().numpy() if boxes.cls is not None else None
    conf_tensor = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else None
    keypoints_xy_tensor = keypoints.xy.detach().cpu().numpy()
    keypoints_conf_tensor = (
        keypoints.conf.detach().cpu().numpy()
        if getattr(keypoints, "conf", None) is not None
        else None
    )

    pose_people: List[PosePerson] = []
    for index, bbox_xyxy in enumerate(xyxy_tensor):
        if cls_tensor is not None and int(cls_tensor[index]) != 0:
            continue

        scaled_bbox = bbox_xyxy / resize_scale
        x1, y1, x2, y2 = scaled_bbox.astype(int)
        keypoints_xy = keypoints_xy_tensor[index] / resize_scale
        if keypoints_conf_tensor is None:
            keypoints_conf = np.ones(keypoints_xy.shape[0], dtype=np.float32)
        else:
            keypoints_conf = keypoints_conf_tensor[index].astype(np.float32)
        pose_people.append(
            PosePerson(
                bbox_xyxy=(int(x1), int(y1), int(x2), int(y2)),
                confidence=float(conf_tensor[index]) if conf_tensor is not None else 0.0,
                keypoints_xy=keypoints_xy.astype(np.float32),
                keypoints_conf=keypoints_conf,
            )
        )

    return pose_people
